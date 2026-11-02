"""BF16 router operands, FP32 logits and graph replay on Intel XPU."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.gguf_router_fp32 import GGUFRouterFP32Method


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 32, 1024])
def test_fp32_router_reference_and_graph_replay(rows):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    generator = torch.Generator().manual_seed(735 + rows)
    x_cpu = torch.randn(rows, 2560, generator=generator).bfloat16()
    w_cpu = torch.randn(512, 2560, generator=generator).bfloat16() * .01
    x = x_cpu.to("xpu")
    gate = SimpleNamespace(weight=w_cpu.to("xpu"))
    method = GGUFRouterFP32Method()
    actual = method.apply(gate, x)
    assert actual.dtype == torch.float32
    reference = x_cpu.double() @ w_cpu.double().t()
    torch.testing.assert_close(actual.cpu().double(), reference, atol=3e-6, rtol=3e-5)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = method.apply(gate, x)
    # Change graph inputs; replay must not reuse logits from capture/warmup.
    x.mul_(-1)
    graph.replay()
    torch.xpu.synchronize()
    torch.testing.assert_close(captured.cpu().double(), -reference, atol=3e-6, rtol=3e-5)
