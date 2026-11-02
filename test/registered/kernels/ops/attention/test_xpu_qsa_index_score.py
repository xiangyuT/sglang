"""Paged BF16 index scoring, padding and live graph inputs on B70."""
import os

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import (
    _library,
    try_decode_scores,
    try_decode_topk,
)
from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_decode

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_QSA_SYCL_LIB"),
    reason="requires the B70 QSA SYCL library",
)


@pytest.fixture(autouse=True)
def native(monkeypatch):
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 native index scorer")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    monkeypatch.setenv("SGLANG_XPU_QSA_INDEX_SCORE", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    lib = _library(os.environ["SGLANG_XPU_QSA_SYCL_LIB"])
    assert hasattr(lib, "xpu_qsa_index_score")


def inputs(rows, dtype=torch.int32):
    q = torch.randn(rows, 4, 128).bfloat16().to("xpu")
    cache = torch.randn(80, 16, 1, 128).bfloat16().to("xpu")
    table = torch.stack([torch.randperm(80)[:65] for _ in range(rows)])
    return q, cache, table.to(device="xpu", dtype=dtype)


def same_scores(actual, expected):
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-6, rtol=1e-5)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 32, 64])
@pytest.mark.parametrize("width", [1, 15, 16, 17, 1024, 1025])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_paged_scores_and_partial_last_tile(rows, width, dtype):
    q, cache, table = inputs(rows, dtype)
    for n in [0, 1, 511, 512, 513, 1040]:
        lengths = torch.full((rows,), n, device="xpu", dtype=dtype)
        actual = try_decode_scores(q, cache, table, lengths, width)
        expected = torch_qsa_mqa_decode(q, cache, table, lengths, width)
        same_scores(actual, expected)
        if width == 1024:
            a, b = (try_decode_topk(v, lengths, 512) for v in (actual, expected))
            assert a is not None and b is not None
            assert torch.equal(a, b)


@torch.inference_mode()
def test_graph_replay_reads_updated_queries_pages_and_lengths():
    q, cache, table = inputs(4)
    lengths = torch.zeros(4, device="xpu", dtype=torch.int32)
    try_decode_scores(q, cache, table, lengths, 1025)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = try_decode_scores(q, cache, table, lengths, 1025)
    for seed in [7, 11, 42]:
        torch.manual_seed(seed)
        q.normal_()
        cache.normal_()
        table.copy_(torch.stack([torch.randperm(80)[:65] for _ in range(4)]))
        lengths.copy_(torch.tensor([0, 1, 257, 513]))
        graph.replay()
        same_scores(actual, torch_qsa_mqa_decode(q, cache, table, lengths, 1025))


@pytest.mark.parametrize("rows", [4, 32])
@pytest.mark.parametrize("width", [65536, 65537])
@torch.inference_mode()
def test_256k_compressed_context(rows, width):
    # 256K raw tokens produce 64K compressed keys. Include a partial last
    # compressed page and ragged lengths in a full eight-request verify batch.
    pages = (width + 15) // 16
    q = torch.randn(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
    cache = torch.randn(pages + 7, 16, 1, 128, device="xpu", dtype=torch.bfloat16)
    table = torch.stack([torch.randperm(pages + 7)[:pages] for _ in range(rows)]).to("xpu")
    lengths = (torch.arange(rows, device="xpu") * 4093 % width).to(torch.int32)
    lengths[-1] = width
    actual = try_decode_scores(q, cache, table, lengths, width)
    assert actual is not None
    same_scores(actual, torch_qsa_mqa_decode(q, cache, table, lengths, width))


@torch.inference_mode()
def test_nonfinite_propagation_and_flag_fallback(monkeypatch):
    q, cache, table = inputs(1)
    lengths = torch.tensor([17], device="xpu", dtype=torch.int32)
    q[0, 0, 0] = float("nan")
    same_scores(
        try_decode_scores(q, cache, table, lengths, 17),
        torch_qsa_mqa_decode(q, cache, table, lengths, 17),
    )
    assert try_decode_scores(q.float(), cache, table, lengths, 17) is None
    monkeypatch.delenv("SGLANG_XPU_QSA_INDEX_SCORE")
    assert try_decode_scores(q, cache, table, lengths, 17) is None
