"""Exact B70 QSA indexer order, BF16 rounding and dynamic graph inputs."""

import os
from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import (
    can_capture_mtp_indexer,
    can_capture_verify_indexer,
    try_decode_topk,
    try_indexer_rope,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_QSA_SYCL_LIB"),
    reason="requires the separately built B70 QSA SYCL library",
)


@pytest.fixture(scope="module", autouse=True)
def _warm_qsa_library_before_capture():
    from sglang.kernels.ops.attention.xpu_qsa_sycl import _library

    _library(os.environ["SGLANG_XPU_QSA_SYCL_LIB"])


def _rotary(dtype, interleaved=True, max_position=4096):
    with torch.device("xpu"):
        return get_rope(
            head_size=256,
            rotary_dim=256,
            max_position=max_position,
            base=10000000,
            is_neox_style=True,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": interleaved,
            },
            dtype=dtype,
            partial_rotary_factor=0.25,
        )


@pytest.mark.parametrize("rows,heads", [(1, 1), (1, 4), (4, 1), (4, 4), (128, 4)])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("position_axes", [1, 3])
@pytest.mark.parametrize("interleaved", [True, False])
def test_rope_per_operation_rounding_and_strides(
    monkeypatch, rows, heads, cache_dtype, position_axes, interleaved
):
    # The model normally publishes these runtime settings before using MRoPE.
    monkeypatch.setattr(
        "sglang.srt.layers.rotary_embedding.mrope.attention_backends",
        lambda: ("intel_xpu", "intel_xpu"),
    )
    rotary = _rotary(cache_dtype, interleaved)
    g = torch.Generator().manual_seed(42)
    # Trailing rows, strided head/token storage and nonzero storage offset.
    data = torch.randn((rows + 2, heads * 2, 256), generator=g).bfloat16()
    data = data.to("xpu")[1:-1, ::2, 31:159]
    positions = (torch.arange(rows * 2 + 4) + 19).to("xpu")[2:-2:2]
    if position_axes == 3:
        positions = (
            torch.stack((positions, positions + 301, positions + 101)).T.contiguous().T
        )
    indexer = SimpleNamespace(rotary_emb=rotary)
    monkeypatch.delenv("SGLANG_XPU_QSA_ROPE_SYCL", raising=False)
    expected = QSAIndexer.apply_rope(indexer, positions, data)
    monkeypatch.setenv("SGLANG_XPU_QSA_ROPE_SYCL", "1")
    actual = QSAIndexer.apply_rope(indexer, positions, data)
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    assert torch.equal(actual[..., 64:].cpu(), data[..., 64:].cpu())


def test_rope_dynamic_graph_and_fallback(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_QSA_ROPE_SYCL", "1")
    rotary = _rotary(torch.bfloat16)
    data = (
        torch.linspace(-100, 100, 4 * 4 * 128).reshape(4, 4, 128).bfloat16().to("xpu")
    )
    positions = torch.arange(4, dtype=torch.int64, device="xpu")
    assert try_indexer_rope(data.float(), positions, rotary) is None
    assert try_indexer_rope(data, positions.int(), rotary) is None
    output = try_indexer_rope(data, positions, rotary)
    assert output is not None
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = try_indexer_rope(data, positions, rotary)
    for offset in (0, 128, 4092):
        positions.copy_(torch.arange(4, device="xpu") + offset)
        data.mul_(-0.5)
        graph.replay()
        monkeypatch.delenv("SGLANG_XPU_QSA_ROPE_SYCL")
        expected = QSAIndexer.apply_rope(
            SimpleNamespace(rotary_emb=rotary), positions, data
        )
        monkeypatch.setenv("SGLANG_XPU_QSA_ROPE_SYCL", "1")
        assert torch.equal(
            output.cpu().view(torch.int16), expected.cpu().view(torch.int16)
        )
    # A violated fixed-cache contract is bounds-safe and visibly nonfinite.
    positions.fill_(4096)
    invalid = try_indexer_rope(data, positions, rotary)
    assert bool(invalid[..., :64].isnan().all())
    assert torch.equal(invalid[..., 64:].cpu(), data[..., 64:].cpu())


@pytest.mark.parametrize("columns", [127, 512, 1024, 4096, 8192, 32768, 65536])
@pytest.mark.parametrize(
    "distribution", ["random", "zero", "ties", "relu", "nonfinite"]
)
def test_batched_topk_exact_order_and_padding(monkeypatch, columns, distribution):
    monkeypatch.setenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH", "1")
    g = torch.Generator().manual_seed(41)
    logits = torch.randn((8, columns), generator=g)
    lengths = torch.tensor([0, 1, 31, 128, 288, 511, 512, columns]).clamp_max(columns)
    if distribution == "zero":
        logits.zero_()
    elif distribution == "ties":
        logits = logits.round()
    elif distribution == "relu":
        logits.relu_()
    elif distribution == "nonfinite":
        logits[:, ::5] = -float("inf")
        logits[:, 1::11] = float("nan")
        logits[:, 2::17] = float("inf")
    logits.masked_fill_(torch.arange(columns)[None] >= lengths[:, None], -float("inf"))
    logits, lengths = logits.to("xpu"), lengths.int().to("xpu")
    expected = qsa_fast_topk(logits, torch.zeros_like(lengths), lengths, 512)
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    actual = try_decode_topk(logits, lengths, 512)
    assert torch.equal(actual.cpu(), expected.cpu())


def test_batched_topk_graph_reads_updated_lengths(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    logits = torch.zeros((4, 1024), device="xpu")
    lengths = torch.tensor([288, 512, 513, 1024], device="xpu", dtype=torch.int32)
    columns = torch.arange(1024, device="xpu")[None]
    logits.masked_fill_(columns >= lengths[:, None], -float("inf"))
    try_decode_topk(logits, lengths, 512)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = try_decode_topk(logits, lengths, 512)
    for values in ([0, 1, 257, 511], [1024, 768, 256, 32]):
        lengths.copy_(torch.tensor(values, device="xpu", dtype=torch.int32))
        logits.copy_(torch.arange(1024, device="xpu").float()[None].expand(4, -1))
        logits.masked_fill_(columns >= lengths[:, None], -float("inf"))
        graph.replay()
        expected = qsa_fast_topk(logits, torch.zeros_like(lengths), lengths, 512)
        assert torch.equal(actual.cpu(), expected.cpu())


class _IndexerPool:
    qsa_compressed_dtype = torch.bfloat16
    qsa_compressed_page_size = 16
    qsa_index_kv_heads = 1
    qsa_index_head_dim = 128
    set_qsa_key_state_buffer = QSATokenToKVPool.set_qsa_key_state_buffer
    set_qsa_rope_position_buffer = QSATokenToKVPool.set_qsa_rope_position_buffer
    set_qsa_compressed_k_buffer = QSATokenToKVPool.set_qsa_compressed_k_buffer

    def __init__(self):
        self.pending = torch.randn(16, 1, 128, device="xpu", dtype=torch.bfloat16)
        self.compressed = torch.randn(1040, 1, 128, device="xpu", dtype=torch.bfloat16)
        self.qsa_rope_position_buffer = torch.zeros(
            16, 3, device="xpu", dtype=torch.int64
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.pending

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed


class _IndexerProjection(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


@torch.inference_mode()
def test_causal_ring_rejects_oversized_window_before_cache_writes():
    pool = _IndexerPool()
    pool.qsa_pending_ring_size = 16
    before = pool.pending.clone()
    indexer = SimpleNamespace(layer_id=0, compress_ratio=4)
    metadata = SimpleNamespace(token_to_kv_pool=pool, compress_member_rows=None)
    with pytest.raises(ValueError, match="causal pending-ring capacity"):
        QSAIndexer.update_key_state_and_compress(
            indexer,
            torch.zeros(14, 1, 128, dtype=torch.bfloat16, device="xpu"),
            torch.arange(14, device="xpu"),
            torch.arange(14, device="xpu"),
            metadata,
        )
    assert torch.equal(pool.pending.view(torch.int16), before.view(torch.int16))


@pytest.mark.parametrize("start", [1024, 1025, 1026, 1027])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
@torch.inference_mode()
def test_causal_ring_preserves_prefix_across_rejected_lookahead(monkeypatch, start, accepted):
    """Future/rejected keys cannot replace a completed group's prefix members."""
    monkeypatch.setenv("SGLANG_XPU_QSA_ROPE_SYCL", "1")
    pool = _IndexerPool()
    pool.pending = torch.zeros(32, 1, 128, device="xpu", dtype=torch.bfloat16)
    pool.qsa_rope_position_buffer = torch.zeros(32, 3, device="xpu", dtype=torch.int64)
    pool.qsa_pending_ring_size = 16
    indexer = SimpleNamespace(
        layer_id=0, compress_ratio=4,
        _use_fused_compress=lambda pool: False,
        _rope_from_matrix=lambda positions: positions[:, 0],
        normalize_compressed_keys=lambda keys, positions: keys,
    )
    indexer._compress_decode_cuda_graph = MethodType(QSAIndexer._compress_decode_cuda_graph, indexer)
    positions = torch.arange(4, device="xpu", dtype=torch.int64) + start
    requests = torch.ones(1, device="xpu", dtype=torch.int64)
    mapping = torch.zeros(4, device="xpu", dtype=torch.int32)
    slots = torch.zeros(4, device="xpu", dtype=torch.int64)
    groups = torch.zeros(4, 4, device="xpu", dtype=torch.int32)
    writes = torch.zeros(4, device="xpu", dtype=torch.int32)
    metadata = SimpleNamespace(token_to_kv_pool=pool, compress_member_rows=None,
        is_cuda_graph=True, graph_ring_group_locs=groups, graph_write_locs=writes)
    keys = torch.zeros(4, 1, 128, device="xpu", dtype=torch.bfloat16)

    def refresh(first):
        positions.copy_(torch.arange(first, first+4, device="xpu"))
        lengths = positions + 1
        slots.copy_(build_pending_ring_slots(token_to_batch_idx=mapping,
            req_pool_indices=requests, sequence_lengths=lengths,
            logical_positions=positions, compress_ratio=4, is_extend=False, ring_size=16))
        groups.copy_(build_group_ring_slots(req_pool_indices=requests,
            group_end_positions=positions, sequence_ids=mapping.long(), compress_ratio=4, ring_size=16))
        writes.copy_(torch.where(lengths % 4 == 0, lengths // 4, 0).int())

    def key_for(p):
        return ((torch.arange(128, device="xpu") % 13 - 6) * 0.125 + (p % 7) * 0.25).bfloat16().view(1, 128)

    refresh(start)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        QSAIndexer.update_key_state_and_compress(indexer, keys, positions, positions,
            metadata, state_slots=slots)
    results = []
    for poison in [8.0, -16.0]:
        pool.pending.zero_()
        pool.compressed.zero_()
        for p in range(start - 3, start):
            pool.pending[16 + p % 16] = key_for(p)
        refresh(start)
        keys.copy_(torch.stack([key_for(p) if p < start + accepted else key_for(p) + poison
            for p in range(start, start + 4)]))
        graph.replay()
        # The accepted prefix survives; new target inputs replace rejected drafts.
        second = start + accepted
        refresh(second)
        keys.copy_(torch.stack([key_for(p) for p in range(second, second + 4)]))
        graph.replay()
        end = next(p for p in range(second, second + 4) if (p + 1) % 4 == 0)
        actual = pool.compressed[(end + 1) // 4].cpu()
        expected = torch.stack([key_for(p) for p in range(end - 3, end + 1)]).float().mean(0).bfloat16().cpu()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        results.append(actual)
    assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16))


@pytest.mark.parametrize("phase", ["verify", "draft_extend"])
@pytest.mark.parametrize("compressed_columns", [1024, 65536])
@torch.inference_mode()
def test_verify_indexer_graph_replays_updated_metadata_and_pool(
    monkeypatch, phase, compressed_columns
):
    for flag in ("VERIFY_GRAPH", "ROPE_SYCL", "BATCHED_TOPK"):
        monkeypatch.setenv("SGLANG_XPU_QSA_" + flag, "1")
    monkeypatch.setattr(
        "sglang.srt.layers.rotary_embedding.mrope.attention_backends",
        lambda: ("intel_xpu", "intel_xpu"),
    )
    monkeypatch.setenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_INDEX_SCORE", "1")
    torch.manual_seed(42)
    # Exercise the actual orchestration with a small explicit pool. Avoid a
    # distributed model constructor; projection/norms and all QSA methods run.
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.index_n_heads, indexer.index_kv_heads, indexer.index_head_dim = 4, 1, 128
    indexer.compress_ratio, indexer.token_topk, indexer.block_topk = 4, 2048, 512
    indexer.rotary_emb = _rotary(torch.bfloat16, max_position=compressed_columns * 4)
    indexer.index_qk_proj = _IndexerProjection(
        2560, 640, bias=False, device="xpu", dtype=torch.bfloat16
    )
    indexer.q_layernorm = GemmaRMSNorm(128).to(device="xpu", dtype=torch.bfloat16)
    indexer.k_layernorm = GemmaRMSNorm(128).to(device="xpu", dtype=torch.bfloat16)
    indexer.q_layernorm.weight.normal_(0, 0.1)
    indexer.k_layernorm.weight.normal_(0, 0.1)
    pool = _IndexerPool()
    pool.compressed = torch.randn(
        compressed_columns + 16, 1, 128, device="xpu", dtype=torch.bfloat16
    )
    lengths = torch.full((4,), 1024, device="xpu", dtype=torch.int32)
    positions = torch.arange(4, device="xpu", dtype=torch.int64)
    metadata = QSAIndexerMetadata(
        sequence_lengths=lengths,
        token_to_batch_idx=torch.arange(4, device="xpu", dtype=torch.int32),
        token_slot_table=torch.zeros(4, 1, device="xpu", dtype=torch.int32),
        out_cache_loc=torch.zeros(4, device="xpu", dtype=torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=4,
        block_topk=512,
        is_cuda_graph=True,
        decode_logical_positions=torch.zeros(4, device="xpu", dtype=torch.int32),
        pending_ring_slots=torch.arange(4, 8, device="xpu", dtype=torch.int64),
        graph_write_locs=torch.zeros(4, device="xpu", dtype=torch.int64),
        graph_ring_group_locs=torch.arange(4, 8, device="xpu", dtype=torch.int32)
        .expand(4, -1)
        .contiguous(),
        graph_compressed_page_table=torch.arange(
            compressed_columns // 16, device="xpu", dtype=torch.int32
        )
        .expand(4, -1)
        .contiguous(),
        graph_compressed_lengths=torch.full((4,), 256, device="xpu", dtype=torch.int32),
    )
    hidden = torch.randn(4, 2560, device="xpu", dtype=torch.bfloat16)
    mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: phase == "verify",
        is_draft_extend_v2=lambda: phase == "draft_extend",
    )
    batch = SimpleNamespace(forward_mode=mode)
    if compressed_columns > 4096:
        monkeypatch.delenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH")
        assert not can_capture_verify_indexer(indexer, hidden, positions, metadata)
        monkeypatch.setenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH", "1")
        monkeypatch.delenv("SGLANG_XPU_QSA_INDEX_SCORE")
        assert not can_capture_verify_indexer(indexer, hidden, positions, metadata)
        monkeypatch.setenv("SGLANG_XPU_QSA_INDEX_SCORE", "1")
    assert can_capture_verify_indexer(indexer, hidden, positions, metadata)
    monkeypatch.delenv("SGLANG_XPU_QSA_VERIFY_GRAPH")
    assert not can_capture_verify_indexer(indexer, hidden, positions, metadata)
    monkeypatch.setenv("SGLANG_XPU_QSA_VERIFY_GRAPH", "1")
    assert not can_capture_verify_indexer(indexer, hidden.float(), positions, metadata)
    for _ in range(2):
        indexer._forward_impl(hidden, positions, batch, metadata)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = indexer._forward_impl(hidden, positions, batch, metadata)
    positions_to_check = (
        (1024, 2047, 3072, 4093)
        if compressed_columns == 1024
        else (1024, 32767, 131072, 262141)
    )
    for start in positions_to_check:
        lengths.copy_(torch.arange(start, start + 4, device="xpu", dtype=torch.int32))
        positions.copy_(lengths.long() - 1)
        metadata.decode_logical_positions.copy_(lengths - 1)
        metadata.graph_compressed_lengths.copy_(lengths // 4)
        metadata.graph_write_locs.copy_(torch.where(lengths % 4 == 0, lengths // 4, 0))
        hidden.normal_()
        pool.pending.normal_()
        pool.compressed.normal_()
        initial = [
            v.clone()
            for v in (pool.pending, pool.compressed, pool.qsa_rope_position_buffer)
        ]
        graph.replay()
        actual_values = [
            v.clone()
            for v in (
                actual,
                pool.pending,
                pool.compressed,
                pool.qsa_rope_position_buffer,
            )
        ]
        for dest, value in zip(
            (pool.pending, pool.compressed, pool.qsa_rope_position_buffer), initial
        ):
            dest.copy_(value)
        expected = indexer._forward_impl(hidden, positions, batch, metadata)
        for got, want in zip(
            actual_values,
            (expected, pool.pending, pool.compressed, pool.qsa_rope_position_buffer),
        ):
            if want.dtype == torch.bfloat16:
                got, want = got.view(torch.int16), want.view(torch.int16)
            assert torch.equal(got.cpu(), want.cpu())


@torch.inference_mode()
def test_mtp_shared_indices_graph_reads_live_acceptance_and_request(monkeypatch):
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QSAMTPSharedSparseIndices,
        QwenSparseAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    monkeypatch.setenv("SGLANG_XPU_QSA_MTP_GRAPH", "1")
    state = QSAMTPSharedSparseIndices(
        layer_ids=[48], num_requests=3, token_topk=2051, tail_width=4, device="xpu"
    )
    req = torch.ones(1, device="xpu", dtype=torch.int64)
    accept = torch.ones(1, device="xpu", dtype=torch.int32)
    extend = torch.full((1,), 4, device="xpu", dtype=torch.int32)
    positions = torch.zeros(1, device="xpu", dtype=torch.int32)
    lengths = torch.arange(1024, 1028, device="xpu", dtype=torch.int32)
    indices = torch.arange(4 * 2051, device="xpu", dtype=torch.int32).view(4, 2051)
    metadata = SimpleNamespace(get_seqlens_expanded=lambda: lengths)
    batch = SimpleNamespace(
        batch_size=1, req_pool_indices=req, _original_forward_mode=None,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        spec_info=SimpleNamespace(extend_seq_lens_tensor=extend,
                                  num_accept_tokens=accept, num_front_tokens=0),
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._mtp_shared_sparse_indices = state
    assert can_capture_mtp_indexer(backend, batch)
    monkeypatch.delenv("SGLANG_XPU_QSA_MTP_GRAPH")
    assert not can_capture_mtp_indexer(backend, batch)
    monkeypatch.setenv("SGLANG_XPU_QSA_MTP_GRAPH", "1")
    batch.batch_size = 2
    assert not can_capture_mtp_indexer(backend, batch)
    batch.batch_size = 1
    backend.forward_metadata = SimpleNamespace(indexer_metadata=SimpleNamespace(
        decode_logical_positions=positions))
    decode_batch = SimpleNamespace(batch_size=1, req_pool_indices=req,
                                   forward_mode=ForwardMode.DECODE)
    assert can_capture_mtp_indexer(backend, decode_batch)
    backend._capture_mtp_sparse_indices_from_extend_lens(indices, batch, metadata, 48)
    backend.lookup_mtp_sparse_indices(decode_batch, 48)
    capture_graph, lookup_graph = torch.xpu.XPUGraph(), torch.xpu.XPUGraph()
    with torch.xpu.graph(capture_graph):
        backend._capture_mtp_sparse_indices_from_extend_lens(indices, batch, metadata, 48)
    with torch.xpu.graph(lookup_graph):
        output = backend.lookup_mtp_sparse_indices(decode_batch, 48)
    for request, count in [(1, 4), (2, 1), (1, 2), (2, 3)]:
        req.fill_(request)
        accept.fill_(count)
        indices.add_(100)
        base = 1024 + count - 1
        positions.fill_(base + 1)
        capture_graph.replay()
        lookup_graph.replay()
        expected = torch.cat((indices[count - 1], torch.tensor(
            [base, base + 1, -1, -1], device="xpu", dtype=torch.int32)))
        assert torch.equal(output[0].cpu(), expected.cpu())
    # Padding writes the trash row rather than overwriting a real request.
    before = state.indices[:, :3].clone()
    req.zero_()
    extend.zero_()
    capture_graph.replay()
    assert torch.equal(state.indices[:, :3].cpu(), before.cpu())


@pytest.mark.parametrize("rows", [1, 4, 32])
def test_long_topk_graph_matches_existing_selector_on_replay(monkeypatch, rows):
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH", "1")
    columns = 65536
    logits = torch.zeros((rows, columns), device="xpu", dtype=torch.float32)
    lengths = torch.full((rows,), columns, device="xpu", dtype=torch.int32)
    indices = torch.arange(columns, device="xpu")[None, :]
    try_decode_topk(logits, lengths, 512)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = try_decode_topk(logits, lengths, 512)
    for length in (0, 511, 4097, 65536):
        lengths.fill_(length)
        logits.copy_((indices % 7).float().expand(rows, -1))
        logits.masked_fill_(indices >= lengths[:, None], -torch.inf)
        graph.replay()
        expected = qsa_fast_topk(logits, torch.zeros_like(lengths), lengths, 512)
        assert torch.equal(output, expected)
    monkeypatch.delenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH")
    assert try_decode_topk(logits, lengths, 512) is None
