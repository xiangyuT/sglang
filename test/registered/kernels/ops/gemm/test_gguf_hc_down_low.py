"""Fused HC reduction/SiLU keeps the split order and BF16 score boundary."""
import os

import pytest
import torch

from sglang.kernels.ops.gemm.gguf_hc_xpu import (
    _low_from_scores,
    _native_down_low,
    _native_mix_from_scores,
    mix,
)
from sglang.kernels.ops.quantization.gguf_sycl import _library

PATH = os.environ.get("SGLANG_XPU_GGUF_HC_UP_MIX_LIB")
pytestmark = pytest.mark.skipif(
    not PATH or not torch.xpu.is_available(), reason="requires B70 HC down/low library"
)


@torch.inference_mode()
def test_down_low_dynamic_graph_and_dispatch(monkeypatch):
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 HC down/low")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(4, 10240).bfloat16().to("xpu")
    w = (torch.randn(324, 10240) * 0.01).bfloat16().to("xpu")
    up = (torch.randn(10240, 320) * 0.01).bfloat16().to("xpu")
    native = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    expected = torch.empty((4, 324), dtype=torch.bfloat16, device="xpu")
    partial = torch.empty((10, 4, 324), dtype=torch.float32, device="xpu")

    def reference():
        stream = torch.xpu.current_stream()
        assert native.xpu_bf16_raw_2da_gemm(
            stream.sycl_queue, x.data_ptr(), w.data_ptr(), expected.data_ptr(),
            partial.data_ptr(), None, 4, 324, 10240, 10,
        ) == 0
        return _low_from_scores(expected)

    _native_down_low(x, w, PATH)
    reference()
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_DOWN_DPAS", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_DOWN_LOW", "1")
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        scores, low = _native_down_low(x, w, PATH)
        result, injection = mix(x, w, up)
    for seed in (3, 7, 11):
        torch.manual_seed(seed)
        x.normal_()
        graph.replay()
        expected_low = reference()
        expected_result, expected_injection = _native_mix_from_scores(x, expected, up, PATH)
        for actual, ref in (
            (scores, expected), (low, expected_low),
            (result, expected_result), (injection, expected_injection),
        ):
            assert torch.equal(actual.cpu().view(torch.int16), ref.cpu().view(torch.int16))


@torch.inference_mode()
def test_down_low_rejects_unbounded_geometry():
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 HC down/low")
    with pytest.raises(ValueError, match="geometry"):
        _native_down_low(
            torch.empty(3, 10240, dtype=torch.bfloat16, device="xpu"),
            torch.empty(324, 10240, dtype=torch.bfloat16, device="xpu"), PATH,
        )
