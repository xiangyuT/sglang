"""Partial EP2 expert unions retain bytes, route math and wide output ownership."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache, encode_host_experts
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=60, suite="xpu-unit-test")
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="Requires B70 and the optional GGUF SYCL library",
)


def raw_fields(kind, down_kind):
    rng = np.random.default_rng(1343 + kind)
    fields = []
    for quant in [kind, kind, down_kind]:
        block, size = BLOCK_GEOMETRY[GGML_TYPES[quant]]
        data = rng.integers(0, 256, (257, 640 * 2560 // block * size), dtype=np.uint8)
        data.reshape(-1, size)[:, :2] = np.array([1 / 256], dtype=np.float16).view(np.uint8)
        data[-1] = 0
        fields.append(torch.from_numpy(data))
    return fields


def make_method(fields, kind, down_kind, union):
    resident = list(range(151)) + [256]
    cold = list(range(151, 256))
    with patch.dict(os.environ, {
        "SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW": str(int(union)),
        "SGLANG_XPU_GGUF_EXPERT_WAVE_ROWS": "32" if union else "16",
    }):
        cache = GPUExpertCache(kind, resident, [t[resident].to("xpu") for t in fields],
            cold, [t[cold].pin_memory() for t in fields], layout="original",
            down_kind=down_kind, gu_soa=kind == 21, down_soa=down_kind == 20)
    assert cache.mirror_addresses.shape == (257, 3)
    assert cache.slot_addresses.shape == (512 if union else 257, 3)
    method = CompressedGGUFMoEMethod(152, decode_q8=True)
    method.num_experts, method.hidden_size, method.intermediate = 257, 2560, 640
    method.sources = tuple(SimpleNamespace(kind=q) for q in [kind, kind, down_kind])
    method._gpu_cache, method.fused_q8, method._paged_zero_expert = cache, True, 256
    layer = SimpleNamespace(**dict(zip(["gguf_gate", "gguf_up", "gguf_down"],
                                       cache.resident_chunks[0])))
    return method, layer


def forward(method, layer, x, ids, weights, union, arena=False):
    with patch.dict(os.environ, {
        "SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW": str(int(union)),
        "SGLANG_XPU_GGUF_EXPERT_WORKSPACE_ARENA": str(int(arena)),
    }):
        batch = StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, None))
        return method.apply(layer, batch).hidden_states


@pytest.mark.parametrize("kind,down_kind", [(21, 20), (21, 8), (23, 8)])
@pytest.mark.parametrize("arena", [False, True])
@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("masked", [False, True])
def test_union_graph_and_retained_output(kind, down_kind, rows, masked, arena):
    raw = raw_fields(kind, down_kind)
    fields = [encode_host_experts(t, kind, i, down_kind=down_kind,
        gu_soa=kind == 21, down_soa=down_kind == 20) for i, t in enumerate(raw)]
    before, layer0 = make_method(fields, kind, down_kind, False)
    after, layer1 = make_method(fields, kind, down_kind, True)
    mode = ForwardMode.TARGET_VERIFY
    flags = {
        "SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
        "SGLANG_XPU_GGUF_Q8_DOWN_PACKETS": "1",
        "SGLANG_XPU_GGUF_IQ3_GU_SOA": str(int(kind == 21)),
        "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": str(int(down_kind == 20)),
        "SGLANG_XPU_GGUF_PARALLEL_PLAN": "64",
        "SGLANG_XPU_GGUF_PAGED_PREP": "1",
        "SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8": "1",
        "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
        "SGLANG_XPU_GGUF_PACKET_REUSE": "8",
    }
    graph = None
    stream = torch.xpu.Stream()
    try:
        with torch.xpu.stream(stream), patch.dict(os.environ, flags), patch(
            "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
            return_value=SimpleNamespace(forward_mode=mode),
        ):
            torch.manual_seed(1343)
            x = (torch.randn(rows, 2560, device="xpu") * .002).bfloat16()
            weights = torch.full((rows, 10), .1, device="xpu")
            def routes(shift=0):
                value = ((torch.arange(rows * 10, device="xpu", dtype=torch.int32)
                          + shift) % 256).reshape(rows, 10)
                if masked:
                    value[:, 5:] = 256
                return value
            # Both full route unions and the usual masked EP2 route layout.
            ids = routes()
            expected = forward(before, layer0, x, ids, weights, False)
            output = forward(after, layer1, x, ids, weights, True, arena)
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            retained = output.clone()
            # Legacy multi-wave outputs survive another eager invocation.
            forward(after, layer1, x * .7, ids, weights, True, arena)
            assert torch.equal(output.view(torch.int16), retained.view(torch.int16))
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                output = forward(after, layer1, x, ids, weights, True, arena)
            for shift in [5, 17, 0]:
                ids.copy_(routes(shift))
                x.mul_(.97)
                expected = forward(before, layer0, x, ids, weights, False)
                graph.replay()
                assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
                retained = output.clone()
                forward(after, layer1, x * .5, ids, weights, True, arena)
                assert torch.equal(output.view(torch.int16), retained.view(torch.int16))
                for method in [before, after]:
                    method._gpu_cache.residency_snapshot(zero_expert=256)
            staged = [torch.empty_like(t, device="xpu") for t in raw]
            after._gpu_cache.stage_original(range(257), staged)
            assert all(torch.equal(a, b.cpu()) for a, b in zip(raw, staged))
    finally:
        graph = None
        before._gpu_cache.close()
        after._gpu_cache.close()


@pytest.mark.parametrize("rows,union", [(16, False), (16, True), (32, True)])
@pytest.mark.parametrize("masked", [False, True])
def test_post_capture_growth_keeps_short_output_ownership(rows, union, masked):
    kind, down_kind = 21, 20
    raw = raw_fields(kind, down_kind)
    fields = [encode_host_experts(t, kind, i, down_kind=down_kind,
              gu_soa=True, down_soa=True) for i, t in enumerate(raw)]
    method, layer = make_method(fields, kind, down_kind, union)
    flags = {
        "SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
        "SGLANG_XPU_GGUF_Q8_DOWN_PACKETS": "1",
        "SGLANG_XPU_GGUF_IQ3_GU_SOA": "1",
        "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "1",
        "SGLANG_XPU_GGUF_PARALLEL_PLAN": "64",
        "SGLANG_XPU_GGUF_PAGED_PREP": "1",
        "SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8": "1",
        "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
        "SGLANG_XPU_GGUF_PACKET_REUSE": "8",
        "SGLANG_XPU_GGUF_EXPERT_PROFILE": "",
    }
    graph = None
    try:
        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream), patch.dict(os.environ, flags), patch(
            "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
            return_value=SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
        ):
            torch.manual_seed(1375)
            x = (torch.randn(rows, 2560, device="xpu") * .002).bfloat16()
            ids = (torch.arange(rows * 10, device="xpu", dtype=torch.int32) % 256).reshape(rows, 10)
            if masked:
                ids[:, 5:] = 256
            weights = torch.full((rows, 10), .1, device="xpu")
            expected = forward(method, layer, x, ids, weights, union, True).clone()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                captured = forward(method, layer, x, ids, weights, union, True)
            method.grow_cache(layer, 160)
            method._gpu_cache.compact_mirrors()
            graph.replay()
            assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
            output = forward(method, layer, x, ids, weights, union, True)
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            retained = output.clone()
            forward(method, layer, x * .7, ids, weights, union, True)
            assert torch.equal(output.view(torch.int16), retained.view(torch.int16))
            method._gpu_cache.residency_snapshot(zero_expert=256)
    finally:
        graph = None
        method._gpu_cache.close()
