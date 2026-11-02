"""Compare SYCL replay metadata against the unchanged Torch XPU fallback."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import try_refresh_graph_metadata
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_QSA_SYCL_LIB"),
    reason="requires the separately built B70 QSA SYCL library",
)


def make_case(mode, bs, dtype, capacity=4096):
    rows = bs * (
        4 if mode.is_target_verify() else 8 if mode.is_draft_extend_v2() else 1
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.device, backend.compress_ratio = torch.device("xpu"), 4
    # Nontrivial physical page addresses and a padded row stride.
    stride = capacity + 32
    raw = torch.arange(10 * stride, device="xpu", dtype=torch.int64).reshape(10, stride)
    backend.req_to_token = (((raw // 64 * 37) % 1024) * 64 + raw % 64).to(dtype)[
        :, :capacity
    ]
    pool = SimpleNamespace(qsa_compress_ratio=4, qsa_compressed_page_size=16)
    lengths = torch.zeros(rows, device="xpu", dtype=torch.int32)
    requests = torch.zeros_like(lengths)
    mapping = torch.arange(rows, device="xpu", dtype=torch.int32)
    table = torch.zeros(rows, 1, device="xpu", dtype=torch.int32)
    indexer = QSAIndexerMetadata(
        sequence_lengths=lengths,
        token_to_batch_idx=mapping,
        token_slot_table=table,
        out_cache_loc=torch.zeros(rows, device="xpu", dtype=torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=4,
        block_topk=512,
        req_pool_indices=requests,
        is_cuda_graph=True,
        graph_prefix_lengths=torch.zeros_like(lengths),
        graph_compressed_lengths=torch.zeros_like(lengths),
        graph_write_locs=torch.zeros_like(lengths),
        decode_logical_positions=torch.zeros_like(lengths),
        pending_ring_slots=torch.zeros(rows, device="xpu", dtype=torch.int64),
        graph_ring_group_locs=torch.zeros(rows, 4, device="xpu", dtype=torch.int32),
        graph_compressed_page_table=torch.full(
            (rows, max(73, capacity // 64)), -97, device="xpu", dtype=torch.int32
        ),
    )
    metadata = QwenSparseAttnMetadata(
        sequence_lengths=lengths,
        token_to_batch_idx=mapping,
        token_slot_table=table,
        indexer_metadata=indexer,
        row_req_pool_indices=requests,
        is_cuda_graph=True,
    )
    backend._cuda_graph_metadata = {(mode, bs): metadata}
    backend._graph_extend_lens = torch.zeros(bs, device="xpu", dtype=torch.int32)
    backend._graph_extend_lens_pin = [
        torch.zeros(bs, dtype=torch.int32, pin_memory=True) for _ in range(2)
    ]
    backend._extend_lens_pin_idx = 0
    seq = torch.zeros(bs, device="xpu", dtype=dtype)
    req = torch.tensor([3, 1, 4, 2, 7, 5, 8, 6][:bs], device="xpu", dtype=dtype)
    ext = torch.tensor([1, 4, 7, 2, 3, 5, 6, 8][:bs], device="xpu", dtype=dtype)
    spec = SimpleNamespace(
        topk=1,
        draft_token_num=4,
        extend_seq_lens_cpu=ext.cpu(),
        extend_seq_lens_tensor=ext,
    )
    return backend, metadata, seq, req, spec


def output_tensors(metadata):
    i = metadata.indexer_metadata
    return (
        metadata.sequence_lengths,
        metadata.row_req_pool_indices,
        i.graph_prefix_lengths,
        i.graph_compressed_lengths,
        i.graph_write_locs,
        i.decode_logical_positions,
        i.pending_ring_slots,
        i.graph_ring_group_locs,
        i.graph_compressed_page_table,
    )


def refresh(backend, metadata, mode, bs, seq, req, spec, padding):
    backend._replay_cuda_graph_metadata(
        bs=bs,
        req_pool_indices=req,
        seq_lens=seq,
        forward_mode=mode,
        spec_info=spec,
        seq_lens_cpu=seq.cpu(),
        num_padding=padding,
    )


@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2]
)
@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("ring_size", [4, 16])
@torch.inference_mode()
def test_metadata_matches_legacy(monkeypatch, mode, bs, dtype, padding, ring_size):
    backend, metadata, seq, req, spec = make_case(mode, bs, dtype)
    metadata.indexer_metadata.token_to_kv_pool.qsa_pending_ring_size = ring_size
    for base in (0, 1024, 3073):
        seq.copy_(torch.arange(base, base + bs, device="xpu", dtype=dtype))
        spec.extend_seq_lens_cpu = spec.extend_seq_lens_tensor.cpu()
        monkeypatch.delenv("SGLANG_XPU_QSA_METADATA_SYCL", raising=False)
        refresh(backend, metadata, mode, bs, seq, req, spec, padding)
        expected = [v.clone() for v in output_tensors(metadata)]
        for value in output_tensors(metadata):
            value.fill_(-97)
        monkeypatch.setenv("SGLANG_XPU_QSA_METADATA_SYCL", "1")
        assert try_refresh_graph_metadata(
            backend, metadata, bs, req, seq, mode, spec, padding
        )
        refresh(backend, metadata, mode, bs, seq, req, spec, padding)
        for got, want in zip(output_tensors(metadata), expected):
            assert torch.equal(got.cpu(), want.cpu())
        spec.extend_seq_lens_tensor.copy_((spec.extend_seq_lens_tensor + 3) % 8)


@torch.inference_mode()
@pytest.mark.parametrize("ring_size", [4, 16])
def test_metadata_graph_reads_changed_inputs(monkeypatch, ring_size):
    mode, bs = ForwardMode.TARGET_VERIFY, 2
    backend, metadata, seq, req, spec = make_case(mode, bs, torch.int32)
    metadata.indexer_metadata.token_to_kv_pool.qsa_pending_ring_size = ring_size
    seq.fill_(1024)
    monkeypatch.setenv("SGLANG_XPU_QSA_METADATA_SYCL", "1")
    assert try_refresh_graph_metadata(backend, metadata, bs, req, seq, mode, spec, 0)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        assert try_refresh_graph_metadata(
            backend, metadata, bs, req, seq, mode, spec, 0
        )
    for base in (3, 2047, 4089):
        seq.copy_(torch.tensor([base, base + 1], device="xpu", dtype=seq.dtype))
        req.copy_(req.flip(0))
        graph.replay()
        actual = [v.clone() for v in output_tensors(metadata)]
        monkeypatch.delenv("SGLANG_XPU_QSA_METADATA_SYCL")
        refresh(backend, metadata, mode, bs, seq, req, spec, 0)
        for got, want in zip(actual, output_tensors(metadata)):
            assert torch.equal(got.cpu(), want.cpu())
        monkeypatch.setenv("SGLANG_XPU_QSA_METADATA_SYCL", "1")


@pytest.mark.parametrize("mode", [ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2])
@torch.inference_mode()
def test_256k_metadata(monkeypatch, mode):
    bs = 8
    backend, metadata, seq, req, spec = make_case(mode, bs, torch.int32, 262144)
    metadata.indexer_metadata.token_to_kv_pool.qsa_pending_ring_size = 16
    for base in (131071, 262129):
        seq.copy_(torch.arange(base, base + bs, device="xpu", dtype=seq.dtype))
        monkeypatch.delenv("SGLANG_XPU_QSA_METADATA_SYCL", raising=False)
        refresh(backend, metadata, mode, bs, seq, req, spec, 0)
        expected = [value.clone() for value in output_tensors(metadata)]
        monkeypatch.setenv("SGLANG_XPU_QSA_METADATA_SYCL", "1")
        assert try_refresh_graph_metadata(backend, metadata, bs, req, seq, mode, spec, 0)
        for got, want in zip(output_tensors(metadata), expected):
            assert torch.equal(got.cpu(), want.cpu())
