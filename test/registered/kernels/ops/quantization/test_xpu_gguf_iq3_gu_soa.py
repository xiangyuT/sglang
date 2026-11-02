"""Lossless IQ3 gate/up storage across decode, paging and prefill readers."""
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

register_xpu_ci(est_time=60, suite="xpu-unit-test")
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="Requires B70 and the optional GGUF SYCL library",
)


def raw_experts(down_kind=20):
    rng = np.random.default_rng(413)
    fields = []
    down_size, down_block = (921600, 18) if down_kind == 20 else (1740800, 34)
    for size, block in [(704000, 110), (704000, 110), (down_size, down_block)]:
        data = rng.integers(0, 256, (6, size), dtype=np.uint8)
        headers = data.reshape(-1, block)
        scales = np.ldexp(np.ones(len(headers), dtype=np.float32),
                          np.arange(len(headers)) % 5 - 9).astype(np.float16)
        headers[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        fields.append(torch.from_numpy(data))
    return fields


def make_cache(raw, soa, down_kind=20):
    fields = [encode_host_experts(t, 21, i, down_kind=down_kind,
                                 down_soa=down_kind == 20, gu_soa=soa)
              for i, t in enumerate(raw)]
    return GPUExpertCache(
        21, range(2), [t[:2].to("xpu") for t in fields], range(2, 6),
        [t[2:].pin_memory() for t in fields], layout="original",
        down_kind=down_kind, down_soa=down_kind == 20, gu_soa=soa,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_permutation_and_dequant(dtype):
    lib = _library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"])
    queue = torch.xpu.current_stream().sycl_queue
    for field, raw in enumerate(raw_experts()[:2]):
        packed = encode_host_experts(raw, 21, field, down_kind=20, gu_soa=True)
        source, expected = raw.to("xpu"), packed.to("xpu")
        actual, restored = torch.empty_like(source), torch.empty_like(source)
        assert lib.xpu_gguf_iq3_gu_permute(queue, source.data_ptr(), actual.data_ptr(), 6, 0) == 0
        assert torch.equal(actual, expected)
        assert lib.xpu_gguf_iq3_gu_permute(queue, actual.data_ptr(), restored.data_ptr(), 6, 1) == 0
        assert torch.equal(restored, source)
        reference = dequantize_gguf(source.flatten(), 21, dtype=dtype)
        decoded = torch.empty_like(reference)
        assert lib.xpu_gguf_iq3_cache_dequant(
            queue, actual.data_ptr(), decoded.data_ptr(), 21, decoded.numel(), int(dtype == torch.float32),
        ) == 0
        assert torch.equal(decoded.view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 4, 14])
def test_paging_growth_and_graph(rows):
    raw = raw_experts()
    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_EXPERT_WAVE_ROWS": "16"}):
        caches = [make_cache(raw, False), make_cache(raw, True)]
    graphs = []
    try:
        torch.manual_seed(413 + rows)
        x = (torch.randn(rows, 2560, device="xpu") * .002).bfloat16()
        weights = torch.full((rows, 10), .1, device="xpu", dtype=torch.float32)
        ids = torch.tensor([[0, 1] * 5] * rows, device="xpu", dtype=torch.int32)
        outputs = []
        workspaces = []
        for cache, flag in zip(caches, ["0", "1"]):
            workspace = {}
            workspaces.append(workspace)
            def run(cache=cache, workspace=workspace):
                route = cache.prepare(ids)
                return fused_q8_experts(
                    x, *cache.resident_chunks[0], 21, route.slots, weights, workspace,
                    grouping=route.grouping, slot_addresses=cache.slot_addresses, down_kind=20,
                )
            with patch.dict(os.environ, {"SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
                                         "SGLANG_XPU_GGUF_IQ3_GU_SOA": flag,
                                         "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "1"}):
                run()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = run()
            graphs.append(graph)
            outputs.append(output)
        for experts in [[2, 3], [4, 5], [0, 5], [1, 4]]:
            ids.copy_(torch.tensor([experts * 5] * rows, device="xpu", dtype=torch.int32))
            x.mul_(.97)
            for graph in graphs:
                graph.replay()
            assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            a = next(iter(workspaces[0].values()))
            b = next(iter(workspaces[1].values()))
            for name in ["gate", "up", "activation", "down"]:
                assert torch.equal(a[name].view(torch.int32), b[name].view(torch.int32)), name
            staged = [torch.empty(t.shape, device="xpu", dtype=torch.uint8) for t in raw]
            caches[1].stage_original(range(6), staged)
            for expected, actual in zip(raw, staged):
                assert torch.equal(expected, actual.cpu())
        for cache in caches:
            cold = int((cache.expert_slots.cpu() < 0).nonzero()[0])
            cache.grow([torch.empty((1, t.shape[1]), device="xpu", dtype=torch.uint8)
                        for t in raw], [cold])
            before = cache.mirror_addresses.data_ptr()
            cache.compact_mirrors()
            assert cache.mirror_addresses.data_ptr() == before
        ids.fill_(cold)
        for graph in graphs:
            graph.replay()
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    finally:
        graphs.clear()
        for cache in caches:
            cache.close()


@pytest.mark.parametrize("path", ["single", "copy", "packed"])
@pytest.mark.parametrize("wg", ["0", "1"])
def test_native_int8_prefill(path, wg):
    caches = [make_cache(raw_experts(), flag) for flag in [False, True]]
    try:
        torch.manual_seed(413)
        x = (torch.randn(9, 2560, device="xpu") * .002).bfloat16()
        weights = torch.full((9, 10), .1, device="xpu")
        ids = np.arange(90, dtype=np.int32).reshape(9, 10) % 6
        results = []
        for cache, flag in zip(caches, ["0", "1"]):
            with patch.dict(os.environ, {
                "SGLANG_XPU_GGUF_IQ3_GU_SOA": flag, "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "1",
                "SGLANG_XPU_GGUF_PREFILL_COPY_ENGINE": "0" if path == "single" else "1",
                "SGLANG_XPU_GGUF_PREFILL_HOST_PACK": "1" if path == "packed" else "0",
                "SGLANG_GGUF_PF_I8_WG": wg,
            }):
                method = SimpleNamespace(_gpu_cache=cache, num_experts=6)
                results.append(apply_layer(method, x, weights, ids, list(range(6)),
                                           output_dtype=torch.float32))
        assert torch.equal(results[0].view(torch.int32), results[1].view(torch.int32))
    finally:
        for cache in caches:
            cache.close()


@pytest.mark.parametrize("path", ["single", "copy", "packed"])
def test_native_fp16_prefill_callback(path):
    from sglang.kernels.ops.quantization.gguf_prefill_f16_layer import apply_layer as fp16_layer
    caches = [make_cache(raw_experts(8), flag, 8) for flag in [False, True]]
    try:
        torch.manual_seed(413)
        x = (torch.randn(9, 2560, device="xpu") * .002).bfloat16()
        weights = torch.full((9, 10), .1, device="xpu")
        ids = np.arange(90, dtype=np.int32).reshape(9, 10) % 6
        results = []
        for cache in caches:
            with patch.dict(os.environ, {
                "SGLANG_XPU_GGUF_PREFILL_COPY_ENGINE": "0" if path == "single" else "1",
                "SGLANG_XPU_GGUF_PREFILL_F16_HOST_PACK": "1" if path == "packed" else "0",
            }):
                method = SimpleNamespace(_gpu_cache=cache, num_experts=6)
                results.append(fp16_layer(method, x, weights, ids, list(range(6)),
                                          output_dtype=torch.float32))
        assert torch.equal(results[0].view(torch.int32), results[1].view(torch.int32))
    finally:
        for cache in caches:
            cache.close()
