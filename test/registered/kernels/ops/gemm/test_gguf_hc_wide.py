"""Wider HC keeps original native arithmetic and dynamic graph inputs."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_XPU_GGUF_HC_WIDE_LIB"), reason="Build the HC wide library"
)


@pytest.fixture(autouse=True)
def restore_current_device():
    device = torch.xpu.current_device()
    try:
        yield
    finally:
        torch.xpu.set_device(device)


@torch.inference_mode()
def test_native4_bits_graph_live_inputs_and_shape_guard(monkeypatch):
    from sglang.kernels.ops.gemm.gguf_hc_wide_sycl import eligible, warm_from_env
    from sglang.kernels.ops.gemm.gguf_hc_xpu import (
        _native_down_low, _native_mix_from_scores, mix,
    )

    torch.set_num_threads(1)
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_WIDE_MIX", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_UP_MIX_TILE", "1")
    warm_from_env()
    for rank in range(torch.xpu.device_count()):
        if torch.xpu.get_device_properties(rank).device_id != 0xE223:
            continue
        device = torch.device("xpu", rank)
        torch.xpu.set_device(device)
        torch.manual_seed(1233)
        down = (torch.randn(324, 10240) * .01).bfloat16().to(device)
        up = (torch.randn(10240, 320) * .01).bfloat16().to(device)
        for rows in (8, 12, 16, 20, 24):
            x = torch.randn(rows, 10240, dtype=torch.bfloat16, device=device)
            mix(x, down, up)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=torch.xpu.Stream(device=device)):
                actual = mix(x, down, up)
            for scale in (.125, -8.):
                x.mul_(scale)
                graph.replay()
                outputs, injections = [], []
                for first in range(0, rows, 4):
                    part = x[first:first + 4]
                    scores, low = _native_down_low(part, down, os.environ["SGLANG_XPU_GGUF_HC_UP_MIX_LIB"])
                    value, injection = _native_mix_from_scores(part, scores, up, os.environ["SGLANG_XPU_GGUF_HC_UP_MIX_LIB"], low)
                    outputs.append(value); injections.append(injection)
                for result, reference in zip(actual, (torch.cat(outputs), torch.cat(injections))):
                    assert torch.equal(result.cpu().view(torch.int16), reference.cpu().view(torch.int16))
            del graph, actual
        for rows in (1, 4, 5, 7, 28, 32, 1024):
            assert not eligible(torch.empty(rows, 10240, dtype=torch.bfloat16, device=device))
        monkeypatch.setenv("SGLANG_XPU_GGUF_HC_WIDE_MIX", "0")
        assert not eligible(torch.empty(8, 10240, dtype=torch.bfloat16, device=device))
        monkeypatch.setenv("SGLANG_XPU_GGUF_HC_WIDE_MIX", "1")


@torch.inference_mode()
def test_module_mix_injection_and_combine_contract(monkeypatch):
    from sglang.kernels.ops.gemm.gguf_hc_xpu import (
        _native_down_low, _native_mix_from_scores, combine, install,
    )
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_WIDE_MIX", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_UP_MIX_TILE", "1")
    for rank in range(torch.xpu.device_count()):
        device = torch.device("xpu", rank)
        torch.xpu.set_device(device)
        with set_default_torch_dtype(torch.bfloat16), torch.device(device):
            module = GatedResidual(HyperConnectionConfig(
                hc_count=4, hidden_size=2560, params_dtype=torch.bfloat16,
                hc_lowrank=320, hc_per_branch_norm=True,
            )).requires_grad_(False)
        install(module)
        x = torch.randn(24, 10240, device=device, dtype=torch.bfloat16)
        block = torch.randn(24, 2560, device=device, dtype=torch.bfloat16)
        normed = module.hc_norm(x)
        expected, injections = [], []
        for chunk in normed.split(4):
            scores, low = _native_down_low(chunk, module._xpu_gguf_hc_down_gate, os.environ["SGLANG_XPU_GGUF_HC_UP_MIX_LIB"])
            value, injection = _native_mix_from_scores(chunk, scores, module.input_mix_weight_up.weight, os.environ["SGLANG_XPU_GGUF_HC_UP_MIX_LIB"], low)
            expected.append(value); injections.append(injection)
        actual, state = module.mix(x)
        assert state[2] is not None and state[2].stride(0) == 324
        assert torch.equal(actual.view(torch.int16), torch.cat(expected).view(torch.int16))
        assert torch.equal(module.combine(block, state).view(torch.int16), combine(block, x, torch.cat(injections)).view(torch.int16))
