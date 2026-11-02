"""Original BF16 HC up weights and live four-token graph inputs."""
import os

import pytest
import torch

from sglang.kernels.ops.gemm.gguf_hc_xpu import (
    _mix_from_scores,
    _native_mix_from_scores,
)

PATH = os.environ.get("SGLANG_XPU_GGUF_HC_UP_MIX_LIB")
pytestmark = pytest.mark.skipif(
    not PATH or not torch.xpu.is_available(), reason="requires B70 HC up/mix library"
)


@torch.inference_mode()
@pytest.mark.parametrize("tile", [1, 2, 4])
def test_native_up_mix_live_graph_and_injection_view(monkeypatch, tile):
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 HC up/mix")
    torch.set_num_threads(1)
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_UP_MIX_TILE", str(tile))
    torch.manual_seed(42)
    normed = torch.randn(4, 10240).bfloat16().to("xpu")
    scores = torch.randn(4, 324).bfloat16().to("xpu")
    weight = (torch.randn(10240, 320) * 0.01).bfloat16().to("xpu")
    _native_mix_from_scores(normed, scores, weight, PATH)
    _mix_from_scores(normed, scores, weight)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual, injection = _native_mix_from_scores(normed, scores, weight, PATH)
    for seed in (3, 7, 11):
        torch.manual_seed(seed)
        normed.normal_()
        scores.normal_()
        graph.replay()
        expected, old_injection = _mix_from_scores(normed, scores, weight)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
        assert torch.equal(injection.cpu().view(torch.int16), old_injection.cpu().view(torch.int16))
        assert injection.data_ptr() == scores[:, 320:324].data_ptr()


@torch.inference_mode()
def test_native_up_mix_rejects_unbounded_rows():
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 HC up/mix")
    with pytest.raises(ValueError, match="geometry"):
        _native_mix_from_scores(
            torch.empty(5, 10240, dtype=torch.bfloat16, device="xpu"),
            torch.empty(5, 324, dtype=torch.bfloat16, device="xpu"),
            torch.empty(10240, 320, dtype=torch.bfloat16, device="xpu"),
            PATH,
        )
