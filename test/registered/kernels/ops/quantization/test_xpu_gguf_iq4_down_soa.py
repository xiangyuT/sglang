"""Lossless IQ4 down storage, live paging, graph replay and native prefill."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache, encode_host_experts
from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts
from sglang.kernels.ops.quantization.gguf_prefill_layer import apply_layer
from sglang.kernels.ops.quantization.gguf_sycl import _library, dequantize_gguf
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=40, suite="xpu-unit-test")

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="Requires B70 and the optional GGUF SYCL library",
)


def raw_experts(count=6):
    rng = np.random.default_rng(372)
    fields = []
    for size, block in [(704000, 110), (704000, 110), (921600, 18)]:
        data = rng.integers(0, 256, (count, size), dtype=np.uint8)
        data.reshape(-1, block)[:, :2] = np.array([1/128], dtype=np.float16).view(np.uint8)
        fields.append(torch.from_numpy(data))
    return fields


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_permutation_and_dequant(dtype):
    lib = _library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"])
    raw = raw_experts(3)[2]
    packed = encode_host_experts(raw, 21, 2, down_kind=20, down_soa=True)
    source, expected = raw.to("xpu"), packed.to("xpu")
    actual, restored = torch.empty_like(source), torch.empty_like(source)
    queue = torch.xpu.current_stream().sycl_queue
    assert lib.xpu_gguf_iq4_down_permute(queue, source.data_ptr(), actual.data_ptr(), 3, 0) == 0
    assert torch.equal(actual, expected)
    assert lib.xpu_gguf_iq4_down_permute(queue, actual.data_ptr(), restored.data_ptr(), 3, 1) == 0
    assert torch.equal(restored, source)
    reference = dequantize_gguf(source.flatten(), 20, dtype=dtype)
    decoded = torch.empty_like(reference)
    assert lib.xpu_gguf_iq4_cache_dequant(
        queue, actual.data_ptr(), decoded.data_ptr(), 20, decoded.numel(), int(dtype == torch.float32),
    ) == 0
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    assert torch.equal(decoded.view(bits), reference.view(bits))


def make_cache(raw, soa):
    encoded = [encode_host_experts(t, 21, i, down_kind=20, down_soa=soa) for i, t in enumerate(raw)]
    return GPUExpertCache(
        21, range(2), [t[:2].to("xpu") for t in encoded], range(2, 6),
        [t[2:].pin_memory() for t in encoded], layout="original", down_kind=20, down_soa=soa,
    )


def test_paging_growth_and_graph():
    raw = raw_experts()
    baseline, candidate = make_cache(raw, False), make_cache(raw, True)
    graphs = []
    try:
        x = (torch.randn(1, 2560, device="xpu") * .02).bfloat16()
        weights = torch.full((1, 10), .1, device="xpu", dtype=torch.float32)
        ids = torch.tensor([[0, 1]*5], device="xpu", dtype=torch.int32)
        outputs = []
        for cache, flag in [(baseline, "0"), (candidate, "1")]:
            workspaces = {}

            def run(cache=cache, workspaces=workspaces):
                route = cache.prepare(ids)
                return fused_q8_experts(
                    x, *cache.resident_chunks[0], 21, route.slots, weights, workspaces,
                    grouping=route.grouping, slot_addresses=cache.slot_addresses, down_kind=20,
                )

            with patch.dict(os.environ, {"SGLANG_XPU_GGUF_MIXED_PACKETS": "1", "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": flag}):
                run()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = run()
            graphs.append(graph)
            outputs.append(output)
        for experts in [[2, 3], [4, 5], [0, 5], [1, 4]]:
            ids.copy_(torch.tensor([experts*5], device="xpu", dtype=torch.int32))
            x.mul_(.97)
            for graph in graphs:
                graph.replay()
            assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            staged = [torch.empty(t.shape, dtype=torch.uint8, device="xpu") for t in raw]
            candidate.stage_original(range(6), staged)
            for source, actual in zip(raw, staged):
                assert torch.equal(source, actual.cpu())
        cold = int((candidate.expert_slots.cpu() < 0).nonzero()[0])
        candidate.grow([torch.empty((1, t.shape[1]), dtype=torch.uint8, device="xpu") for t in raw], [cold])
        before = candidate.mirror_addresses.data_ptr()
        candidate.compact_mirrors()
        assert candidate.mirror_addresses.data_ptr() == before
        ids.fill_(cold)
        for graph in graphs:
            graph.replay()
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    finally:
        graphs.clear()
        baseline.close()
        candidate.close()


@pytest.mark.parametrize("path", ["single", "copy", "packed"])
@pytest.mark.parametrize("wg", ["0", "1"])
def test_native_prefill(path, wg):
    if not os.environ.get("SGLANG_XPU_GGUF_PREFILL_I8_LIB"):
        pytest.skip("Requires the native INT8 prefill library")
    raw = raw_experts()
    caches = [make_cache(raw, False), make_cache(raw, True)]
    try:
        x = (torch.randn(9, 2560, device="xpu") * .02).bfloat16()
        weights = torch.full((9, 10), .1, device="xpu", dtype=torch.float32)
        ids = np.arange(90, dtype=np.int32).reshape(9, 10) % 6
        results = []
        for cache, flag in zip(caches, ["0", "1"]):
            method = SimpleNamespace(_gpu_cache=cache, num_experts=6)
            with patch.dict(os.environ, {
                "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": flag,
                "SGLANG_XPU_GGUF_PREFILL_COPY_ENGINE": "0" if path == "single" else "1",
                "SGLANG_XPU_GGUF_PREFILL_HOST_PACK": "1" if path == "packed" else "0",
                "SGLANG_GGUF_PF_I8_WG": wg,
            }):
                results.append(apply_layer(method, x, weights, ids, list(range(6))))
        assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
    finally:
        for cache in caches:
            cache.close()
