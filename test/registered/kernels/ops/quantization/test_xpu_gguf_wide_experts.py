"""Bounded mixed expert waves preserve route math and live paging ownership."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang.kernels.ops.quantization import gguf_moe_q8
from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache, encode_host_experts
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=60, suite="xpu-unit-test")
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="Requires B70 and the mixed expert SYCL library",
)


def raw_fields(kind, down_kind):
    rng = np.random.default_rng(396 + kind)
    fields = []
    for quant in [kind, kind, down_kind]:
        block, size = BLOCK_GEOMETRY[GGML_TYPES[quant]]
        data = rng.integers(0, 256, (257, 640*2560//block*size), dtype=np.uint8)
        data.reshape(-1, size)[:, :2] = np.array([1/256], dtype=np.float16).view(np.uint8)
        data[-1] = 0  # The final logical expert is the EP dummy.
        fields.append(torch.from_numpy(data))
    return fields


def make_method(raw, kind, down_kind, soa, rows):
    fields = [encode_host_experts(t, kind, i, down_kind=down_kind, down_soa=soa)
              for i, t in enumerate(raw)]
    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_EXPERT_WAVE_ROWS": str(rows)}):
        cache = GPUExpertCache(
            kind, range(144), [t[:144].to("xpu") for t in fields], range(144, 257),
            [t[144:].pin_memory() for t in fields], layout="original",
            down_kind=down_kind, down_soa=soa,
        )
    method = CompressedGGUFMoEMethod(144, decode_q8=True)
    method.num_experts, method.hidden_size, method.intermediate = 257, 2560, 640
    method.sources = tuple(SimpleNamespace(kind=q) for q in [kind, kind, down_kind])
    method._gpu_cache, method.fused_q8, method._paged_zero_expert = cache, True, 256
    layer = SimpleNamespace(**dict(zip(
        ["gguf_gate", "gguf_up", "gguf_down"], cache.resident_chunks[0],
    )))
    return method, layer


def local_ids(rows, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(512, 10, replace=False) for _ in range(rows)]).astype(np.int32)
    ids[ids >= 256] = 256
    return torch.from_numpy(ids).to("xpu")


def forward(method, layer, x, ids, weights):
    batch = StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, None))
    return method.apply(layer, batch).hidden_states


def intermediates(method, layer, x, ids, weights):
    original = gguf_moe_q8.fused_q8_experts
    parts = []

    def observed(*args, **kwargs):
        out = original(*args, **kwargs)
        ws = next(v for v in args[7].values() if v["output"] is out)
        parts.append({k: ws[k].clone() for k in ["gate", "up", "activation", "down"]})
        return out

    with patch.object(gguf_moe_q8, "fused_q8_experts", observed):
        out = forward(method, layer, x, ids, weights)
    return out.clone(), {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


@pytest.mark.parametrize("kind", [21, 23])
def test_q8_down_packets_preserve_paged_intermediates_and_graph(kind):
    raw = raw_fields(kind, 8)
    method, layer = make_method(raw, kind, 8, False, 16)
    mode = ForwardMode.TARGET_VERIFY
    flags = {
        "SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
        "SGLANG_XPU_GGUF_Q8_DOWN_PACKETS": "0",
        "SGLANG_XPU_GGUF_IQ3_GU_SOA": "0",
        "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "0",
        "SGLANG_XPU_GGUF_PARALLEL_PLAN": "128",
        "SGLANG_XPU_GGUF_PAGED_PREP": "1",
        "SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8": "1",
        "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
    }
    graph = None
    try:
        with patch.dict(os.environ, flags), patch(
            "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
            return_value=SimpleNamespace(forward_mode=mode),
        ):
            for rows in (4, 8, 12, 20, 24, 28, 32):
                x = (torch.randn(rows, 2560, device="xpu") * .002).bfloat16()
                ids = local_ids(rows, 731 + rows)
                weights = torch.rand(rows, 10, device="xpu")
                weights /= weights.sum(dim=1, keepdim=True)
                expected, before = intermediates(method, layer, x, ids, weights)
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_Q8_DOWN_PACKETS": "1"}):
                    actual, after = intermediates(method, layer, x, ids, weights)
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                for name in before:
                    assert torch.equal(before[name].view(torch.int32),
                                       after[name].view(torch.int32)), name
            with patch.dict(os.environ, {"SGLANG_XPU_GGUF_Q8_DOWN_PACKETS": "1"}):
                forward(method, layer, x, ids, weights)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = forward(method, layer, x, ids, weights)
            # GPU planner and slot exchange stay live after capture. Force
            # cold experts and EP dummy routes through the same address table.
            for seed in (751, 753, 757):
                x.copy_((torch.randn_like(x) * .001).bfloat16())
                ids.copy_(local_ids(32, seed))
                weights.copy_(torch.rand_like(weights))
                weights.div_(weights.sum(dim=1, keepdim=True))
                expected = forward(method, layer, x, ids, weights).clone()
                graph.replay()
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    finally:
        del graph
        method._gpu_cache.close()


@pytest.mark.parametrize("kind,down_kind,soa,planner", [
    (21, 20, True, "128"), (21, 20, False, "0"),
    (21, 8, False, "128"), (23, 8, False, "128"),
])
def test_wide_model_waves_paging_growth_and_fp32(kind, down_kind, soa, planner):
    raw = raw_fields(kind, down_kind)
    baseline, layer0 = make_method(raw, kind, down_kind, soa, 4)
    candidate, layer1 = make_method(raw, kind, down_kind, soa, 16)
    mode = ForwardMode.TARGET_VERIFY
    graphs = []
    flags = {
        "SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
        "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "1" if soa else "0",
        "SGLANG_XPU_GGUF_PARALLEL_PLAN": planner,
        "SGLANG_XPU_GGUF_PAGED_PREP": "1",
        "SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8": "1",
        "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
    }
    try:
        with patch.dict(os.environ, flags), patch(
            "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
            return_value=SimpleNamespace(forward_mode=mode),
        ):
            x = (torch.randn(32, 2560, device="xpu")*.002).bfloat16()
            ids = local_ids(32, 1)
            weights = torch.full((32, 10), .1, device="xpu")
            ref, a = intermediates(baseline, layer0, x, ids, weights)
            got, b = intermediates(candidate, layer1, x, ids, weights)
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16))
            for key in a:
                assert torch.equal(a[key].view(torch.int32), b[key].view(torch.int32)), key
            results = []
            shapes = []
            original = candidate._gpu_cache.prepare

            def recorded(value):
                if torch.xpu.is_current_stream_capturing():
                    shapes.append(value.shape[0])
                return original(value)

            for method, layer in [(baseline, layer0), (candidate, layer1)]:
                graph = torch.xpu.XPUGraph()
                with patch.object(candidate._gpu_cache, "prepare", recorded):
                    with torch.xpu.graph(graph):
                        out = forward(method, layer, x, ids, weights)
                graphs.append(graph)
                results.append(out)
            assert sum(shapes) == 32 and max(shapes)*10 <= 144
            assert len(shapes) < 8
            for seed in [2, 3, 0]:
                ids.copy_(local_ids(32, seed) if seed else torch.full_like(ids, 256))
                x.mul_(.97)
                weights.mul_(.99)
                for graph in graphs:
                    graph.replay()
                assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))

            cache = candidate._gpu_cache
            # More than capacity distinct routes must reject without eviction.
            full = torch.arange(160, dtype=torch.int32, device="xpu").view(16, 10)
            before = cache.expert_slots.clone()
            old_mirror = cache.mirror_addresses.clone()
            invalid = cache.prepare(full)
            assert invalid.error.item() == 2
            assert torch.equal(before, cache.expert_slots)
            assert torch.equal(old_mirror, cache.mirror_addresses)
            cold = (cache.expert_slots.cpu() < 0).nonzero().flatten()[:16].tolist()
            cache.grow([torch.empty((16, t.shape[1]), device="xpu", dtype=torch.uint8)
                        for t in raw], cold)
            candidate.cache_slots = 160
            valid = cache.prepare(full)
            assert valid.error.item() == 0
            ids.copy_(local_ids(32, 8))
            for graph in graphs:
                graph.replay()
            ref = forward(baseline, layer0, x, ids, weights).clone()
            eager = forward(candidate, layer1, x, ids, weights)
            assert torch.equal(ref.view(torch.int16), eager.view(torch.int16))
            assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))
            # Every encoded expert is still available after paging and growth.
            staged = [torch.empty(t.shape, device="xpu", dtype=torch.uint8) for t in raw]
            cache.stage_original(range(257), staged)
            for expected, actual in zip(raw, staged):
                assert torch.equal(expected, actual.cpu())
    finally:
        graphs.clear()
        baseline._gpu_cache.close()
        candidate._gpu_cache.close()


@pytest.mark.parametrize("rows", [12, 20, 24, 28])
def test_remaining_target_verify_batch_shapes(rows):
    """BS3/5/6/7 exercise the remaining captured wave and tail lengths."""
    torch.manual_seed(396 + rows)
    raw = raw_fields(21, 20)
    baseline, layer0 = make_method(raw, 21, 20, True, 4)
    candidate, layer1 = make_method(raw, 21, 20, True, 16)
    mode = ForwardMode.TARGET_VERIFY
    flags = {
        "SGLANG_XPU_GGUF_MIXED_PACKETS": "1",
        "SGLANG_XPU_GGUF_IQ4_DOWN_SOA": "1",
        "SGLANG_XPU_GGUF_PARALLEL_PLAN": "128",
        "SGLANG_XPU_GGUF_PAGED_PREP": "1",
        "SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8": "1",
        "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
    }
    graphs = []
    try:
        with patch.dict(os.environ, flags), patch(
            "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
            return_value=SimpleNamespace(forward_mode=mode),
        ):
            x = (torch.randn(rows, 2560, device="xpu") * .002).bfloat16()
            ids = local_ids(rows, 11)
            weights = torch.full((rows, 10), .1, device="xpu")
            ref, a = intermediates(baseline, layer0, x, ids, weights)
            got, b = intermediates(candidate, layer1, x, ids, weights)
            assert torch.equal(ref.view(torch.int16), got.view(torch.int16))
            for key in a:
                assert torch.equal(a[key].view(torch.int32), b[key].view(torch.int32)), key
            outputs = []
            for method, layer in [(baseline, layer0), (candidate, layer1)]:
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = forward(method, layer, x, ids, weights)
                graphs.append(graph)
                outputs.append(output)
            for seed in [12, 13]:
                ids.copy_(local_ids(rows, seed))
                x.mul_(.97)
                for graph in graphs:
                    graph.replay()
                assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
    finally:
        graphs.clear()
        baseline._gpu_cache.close()
        candidate._gpu_cache.close()
