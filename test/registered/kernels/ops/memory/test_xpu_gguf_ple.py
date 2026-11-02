# SPDX-License-Identifier: Apache-2.0
"""Pinned compressed PLE row lookup, exact rounding and live graph indices."""
import os
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.memory.gguf_ple_sycl import PinnedPLE
from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_PLE_LIB"),
    reason="Requires B70 and the separately built PLE library",
)


def table():
    torch.manual_seed(42)
    data = torch.empty((65, 90), dtype=torch.uint8, pin_memory=True)
    data.random_(0, 256)
    blocks = data.view(-1, 18)
    scales = (torch.randn(blocks.shape[0]) * 7).half().view(torch.uint8).view(-1, 2)
    blocks[:, :2].copy_(scales)
    return data


def reference(data, indices):
    raw = data[indices.cpu().long()].to("xpu").reshape(-1)
    return dequantize_gguf(raw, 20, dtype=torch.float32).bfloat16().view(*indices.shape, 160)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("chunked", [False, True])
def test_exact_original_dequant_and_out(dtype, chunked):
    data = table()
    before = data.clone()
    native = PinnedPLE([data[i:i+16] for i in range(0, 65, 16)] if chunked else data)
    ids = torch.tensor([[64, 0, 17, 17], [1, 2, 3, 4]], dtype=dtype, device="xpu")
    expected = reference(data, ids)
    out = torch.empty_like(expected)
    assert native.gather(ids, out) is out
    assert torch.equal(out.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    assert torch.equal(data, before)
    with pytest.raises(ValueError):
        native.gather(ids.float())
    with pytest.raises(ValueError):
        native.gather(ids, out.float())


@torch.inference_mode()
def test_graph_uses_live_ids_without_cpu_readback():
    data = table()
    native = PinnedPLE(data)
    ids = torch.tensor([0, 1, 2, 3], device="xpu")
    native.gather(ids)
    graph = torch.xpu.XPUGraph()
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU index readback")):
        with torch.xpu.graph(graph):
            out = native.gather(ids)
    for values in [[64, 63, 2, 0], [17, 17, 21, 1]]:
        ids.copy_(torch.tensor(values, device="xpu"))
        graph.replay()
        expected = reference(data, ids)
        assert torch.equal(out.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    ids.copy_(torch.tensor([-1, 65, 1, 0], device="xpu"))
    graph.replay()
    assert out[:2].count_nonzero().item() == 0


def test_actual_gguf_rows():
    path = os.environ.get("SGLANG_XPU_GGUF_PLE_TEST_MODEL")
    if not path:
        pytest.skip("Set the local original GGUF path to check real scales and codes")
    from sglang.srt.model_loader.gguf_native_reader import open_gguf_shards
    from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows

    matches = [(s, t) for s in open_gguf_shards(path) for t in s.tensors
               if t.name == "per_layer_token_embd.weight"]
    assert len(matches) == 1
    source = CompressedGGUFRows(*matches[0])
    actual_rows = [0, 1, 1024, source.rows // 2, source.rows - 1]
    data = source.packed_rows(actual_rows).pin_memory()
    native = PinnedPLE(data)
    ids = torch.arange(len(actual_rows), device="xpu", dtype=torch.int64)
    expected = source.decode_rows(actual_rows, device="xpu", dtype=torch.bfloat16)
    out = native.gather(ids)
    assert torch.equal(out.cpu().view(torch.int16), expected.cpu().view(torch.int16))
