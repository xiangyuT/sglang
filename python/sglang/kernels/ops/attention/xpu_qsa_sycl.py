# SPDX-License-Identifier: Apache-2.0
"""BF16 physical-slot QSA using Strata's chunk/merge algorithm on B70."""

import ctypes
import math
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _library(path):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SGLANG_XPU_QSA_SYCL_LIB before graph capture")
    library = ctypes.CDLL(path)
    for name in ("xpu_qsa_bf16", "xpu_qsa_bf16_tp2", "xpu_qsa_bf16_tp4",
                 "xpu_qsa_int8", "xpu_qsa_int8_tp2", "xpu_qsa_int8_tp4"):
        if hasattr(library, name):
            function = getattr(library, name)
            function.argtypes = (
                [ctypes.c_void_p] * 9
                + [ctypes.c_int] * 3
                + [ctypes.c_int64] * 2
                + [ctypes.c_float]
            )
            function.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_int8_store"):
        library.xpu_qsa_int8_store.argtypes = (
            [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_int64] * 4
        )
        library.xpu_qsa_int8_store.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_expand"):
        library.xpu_qsa_expand.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
        library.xpu_qsa_expand.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_store_rows"):
        library.xpu_qsa_store_rows.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_int64] * 2
        )
        library.xpu_qsa_store_rows.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_rope"):
        library.xpu_qsa_rope.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_int64] * 5
        )
        library.xpu_qsa_rope.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_graph_metadata"):
        library.xpu_qsa_graph_metadata.argtypes = [ctypes.c_void_p] * 14 + [
            ctypes.c_int
        ] * 15
        library.xpu_qsa_graph_metadata.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_graph_metadata_causal"):
        library.xpu_qsa_graph_metadata_causal.argtypes = (
            [ctypes.c_void_p] * 14 + [ctypes.c_int] * 16
        )
        library.xpu_qsa_graph_metadata_causal.restype = ctypes.c_int
    if hasattr(library, "xpu_qsa_index_score"):
        library.xpu_qsa_index_score.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float]
        )
        library.xpu_qsa_index_score.restype = ctypes.c_int
        library.xpu_qsa_index_score_error.restype = ctypes.c_char_p
    if os.environ.get("SGLANG_XPU_QSA_NATIVE_TOPK") == "1":
        from sglang.kernels.ops.attention.xpu_qsa_topk_sycl import load_library

        load_library(os.environ.get("SGLANG_XPU_QSA_TOPK_LIB") or path)
    return library


def try_decode_scores(q, cache, table, lengths, width, score_scale=None):
    """Score paged BF16 keys directly, retaining the caller's top-k path.

    FP32 DPAS and head reduction may round differently from the Torch scoring
    fallback. This is opt-in; it does not skip scoring short contexts.
    """
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    if not (
        os.environ.get("SGLANG_XPU_QSA_INDEX_SCORE") == "1"
        and path
        and q.device.type == "xpu"
        and q.dtype == cache.dtype == torch.bfloat16
        and q.ndim == 3 and 1 <= q.shape[0] <= 64 and q.shape[1:] == (4, 128)
        and cache.ndim == 4 and cache.shape[0] > 0 and cache.shape[1:] == (16, 1, 128)
        and cache.is_contiguous()
        and table.ndim == 2 and table.shape[0] == q.shape[0] and table.shape[1] > 0
        and lengths.ndim == 1 and lengths.numel() == q.shape[0]
        and table.dtype in (torch.int32, torch.int64)
        and lengths.dtype in (torch.int32, torch.int64)
        and q.device == cache.device == table.device == lengths.device
        and 1 <= width <= 262144
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    ):
        return None
    scale = math.sqrt(128) if score_scale is None else float(score_scale)
    if not math.isfinite(scale) or scale <= 0:
        return None
    library = _library(path)
    if not hasattr(library, "xpu_qsa_index_score"):
        raise RuntimeError("Rebuild the QSA library for native index scoring")
    q = q.contiguous()
    table = table.to(torch.int32).contiguous()
    lengths = lengths.to(torch.int32).contiguous()
    out = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    stream = torch.xpu.current_stream(q.device)
    status = library.xpu_qsa_index_score(
        stream.sycl_queue, q.data_ptr(), cache.data_ptr(), table.data_ptr(),
        lengths.data_ptr(), out.data_ptr(), q.shape[0], width,
        table.shape[1], cache.shape[0], 1.0 / scale,
    )
    if status:
        raise RuntimeError(library.xpu_qsa_index_score_error().decode())
    for value in (q, cache, table, lengths, out):
        value.record_stream(stream)
    return out


def try_indexer_rope(tensor, positions, rotary):
    """Optional B70 BF16 index RoPE with a fully preallocated position cache.

    SGLANG_XPU_QSA_ROPE_SYCL=1 selects a fixed-cache contract: every supplied
    coordinate must be within cos_sin_cache. The scheduler/model configuration
    must preallocate the complete usable coordinate range. This avoids reading
    GPU positions back to Python on each indexer invocation; ordinary callers
    retain the portable path with dynamic cache growth. Unsupported geometry
    and old native libraries also retain that path.
    """
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    cache = getattr(rotary, "cos_sin_cache", None)
    if not (
        os.environ.get("SGLANG_XPU_QSA_ROPE_SYCL") == "1"
        and path
        and tensor.device.type == "xpu"
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 3
        and tensor.shape[0] > 0
        and tensor.shape[1] in (1, 4)
        and tensor.shape[2] == 128
        and tensor.stride(2) == 1
        and rotary.rotary_dim == 64
        and rotary.is_neox_style
        and not getattr(rotary, "mrope_interleaved_glm", False)
        and cache is not None
        and cache.ndim == 2
        and cache.shape[0] > 0
        and cache.shape[1] == 64
        and cache.stride(1) == 1
        and cache.dtype in (torch.bfloat16, torch.float32)
        and cache.device == positions.device == tensor.device
        and positions.dtype == torch.int64
        and positions.ndim in (1, 2)
        and positions.shape[-1] == tensor.shape[0]
        and (positions.ndim == 1 or positions.shape[0] == 3)
        and torch.xpu.get_device_properties(tensor.device).device_id == 0xE223
    ):
        return None
    if torch.is_grad_enabled() and tensor.requires_grad:
        return None
    axes = getattr(rotary, "axis_map", None) if positions.ndim == 2 else None
    if positions.ndim == 2 and not (
        axes is not None
        and axes.device == tensor.device
        and axes.dtype == torch.int64
        and axes.shape == (32,)
        and axes.is_contiguous()
    ):
        return None
    library = _library(path)
    if not hasattr(library, "xpu_qsa_rope"):
        return None
    output = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    stream = torch.xpu.current_stream(tensor.device)
    status = library.xpu_qsa_rope(
        stream.sycl_queue,
        tensor.data_ptr(),
        cache.data_ptr(),
        positions.data_ptr(),
        axes.data_ptr() if axes is not None else None,
        output.data_ptr(),
        tensor.shape[0],
        tensor.shape[1],
        cache.shape[0],
        cache.element_size(),
        1 if positions.ndim == 1 else 3,
        tensor.stride(0),
        tensor.stride(1),
        cache.stride(0),
        0 if positions.ndim == 1 else positions.stride(0),
        positions.stride(-1),
    )
    if status:
        raise RuntimeError(f"Native QSA RoPE submission failed: status={status}")
    for value in (tensor, cache, positions, axes, output):
        if value is not None:
            value.record_stream(stream)
    return output


def try_prefill_topk(logits, starts, ends, topk):
    """Batch short ragged prefill rows without a GPU scalar read per query.

    All valid entries fit the budget. Stable descending score order and stable
    compaction retain low-ID ties and valid -inf entries ahead of padding.
    Longer compressed contexts retain the existing selector.
    """
    if not (os.environ.get("SGLANG_XPU_QSA_PREFILL_BATCHED_TOPK") == "1"
            and logits.device.type == "xpu" and logits.dtype == torch.float32
            and logits.ndim == 2 and logits.shape[0] > 4
            and 0 < logits.shape[1] <= topk and topk in (512, 2048)
            and starts.device == ends.device == logits.device
            and starts.shape == ends.shape == (logits.shape[0],)
            and starts.dtype in (torch.int32, torch.int64)
            and ends.dtype in (torch.int32, torch.int64)
            and torch.xpu.get_device_properties(logits.device).device_id == 0xE223):
        return None
    rows, width = logits.shape
    column = torch.arange(width, device=logits.device)
    valid = (column[None, :] >= starts[:, None]) & (column[None, :] < ends[:, None])
    selected = torch.argsort(logits.masked_fill(~valid, -torch.inf), dim=-1,
                             descending=True, stable=True)
    good = valid.gather(1, selected)
    rank = column[None, :]
    compact = torch.argsort(torch.where(good, rank, rank + width), dim=-1, stable=True)
    selected = selected.gather(1, compact)
    good = good.gather(1, compact)
    selected = (selected - starts[:, None]).to(torch.int32).masked_fill(~good, -1)
    if width == topk:
        return selected
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    output[:, :width].copy_(selected)
    return output


def _decode_topk_column_limit():
    # 256K model context / four keys per compressed position. Keep widening
    # opt-in until long graph and retrieval checks pass on this device/runtime.
    return 65536 if os.environ.get("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH") == "1" else 4096


def try_decode_topk(logits, lengths, topk):
    """Batch the zero-start decode rows, retaining descending score/low-ID order.

    The installed XPU top-k uses low IDs for equal scores in this supported
    range. Stable sorting also keeps valid -inf entries ahead of the masked
    suffix, so padding cannot remove valid selections. No GPU scalar is read
    by Python. The generic ragged-start/prefill route is unchanged.
    """
    if not (
        os.environ.get("SGLANG_XPU_QSA_BATCHED_TOPK") == "1"
        and logits.device.type == "xpu"
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 0 < logits.shape[1] <= _decode_topk_column_limit()
        and lengths.device == logits.device
        and lengths.ndim == 1
        and lengths.shape[0] == logits.shape[0]
        and lengths.dtype in (torch.int32, torch.int64)
        and topk == 512
        and torch.xpu.get_device_properties(logits.device).device_id == 0xE223
    ):
        return None
    if os.environ.get("SGLANG_XPU_QSA_NATIVE_TOPK") == "1":
        from sglang.kernels.ops.attention.xpu_qsa_topk_sycl import (
            supports,
            topk as native_topk,
        )

        if supports(logits, lengths):
            path = os.environ.get("SGLANG_XPU_QSA_TOPK_LIB") or os.environ.get(
                "SGLANG_XPU_QSA_SYCL_LIB"
            )
            if not path:
                raise RuntimeError("Set SGLANG_XPU_QSA_TOPK_LIB for native TopK512")
            return native_topk(logits, lengths, path)
    width = min(topk, logits.shape[1])
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)
    selected = selected[:, :width].to(torch.int32)
    selected.masked_fill_(selected >= lengths[:, None], -1)
    if width < topk:
        output = torch.full(
            (logits.shape[0], topk), -1, device=logits.device, dtype=torch.int32
        )
        output[:, :width].copy_(selected)
        return output
    return selected


def can_capture_verify_indexer(indexer, hidden_states, positions, metadata):
    """Whether the opt-in B70 indexer consumes only replay-updated GPU state.

    The caller restricts this to TARGET_VERIFY, or separately opts in a guarded
    MTP phase. Prefill still reads host lengths and retains its eager boundary.
    Graph metadata owns stable buffers updated before every replay. The fixed
    RoPE cache and bounded batched top-k avoid the portable scalar readbacks.
    """
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    if not (
        os.environ.get("SGLANG_XPU_QSA_VERIFY_GRAPH") == "1"
        and os.environ.get("SGLANG_XPU_QSA_ROPE_SYCL") == "1"
        and os.environ.get("SGLANG_XPU_QSA_BATCHED_TOPK") == "1"
        and path
        and hidden_states.device.type == "xpu"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and 0 < hidden_states.shape[0] <= 64
        and indexer.index_n_heads == 4
        and indexer.index_kv_heads == 1
        and indexer.index_head_dim == 128
        and indexer.compress_ratio == 4
        and indexer.token_topk == 2048
        and metadata.is_cuda_graph
        and metadata.token_to_kv_pool.qsa_compressed_dtype == torch.bfloat16
        and torch.xpu.get_device_properties(hidden_states.device).device_id == 0xE223
    ):
        return False
    rows = hidden_states.shape[0]
    rotary = indexer.rotary_emb
    cache = getattr(rotary, "cos_sin_cache", None)
    if not (
        rotary.rotary_dim == 64
        and rotary.is_neox_style
        and not getattr(rotary, "mrope_interleaved_glm", False)
        and cache is not None
        and cache.device == hidden_states.device == positions.device
        and cache.dtype in (torch.bfloat16, torch.float32)
        and cache.ndim == 2
        and cache.shape[1] == 64
        and cache.stride(1) == 1
        and positions.dtype == torch.int64
        and positions.ndim in (1, 2)
        and positions.shape[-1] == rows
        and (positions.ndim == 1 or positions.shape[0] == 3)
    ):
        return False
    if positions.ndim == 2:
        axes = getattr(rotary, "axis_map", None)
        if not (
            axes is not None
            and axes.device == hidden_states.device
            and axes.dtype == torch.int64
            and axes.shape == (32,)
            and axes.is_contiguous()
        ):
            return False
    for name in (
        "sequence_lengths",
        "token_to_batch_idx",
        "decode_logical_positions",
        "pending_ring_slots",
        "graph_write_locs",
        "graph_ring_group_locs",
        "graph_compressed_page_table",
        "graph_compressed_lengths",
    ):
        value = getattr(metadata, name, None)
        if (
            value is None
            or value.device != hidden_states.device
            or value.shape[0] != rows
        ):
            return False
    columns = (
        metadata.graph_compressed_page_table.shape[1]
        * metadata.token_to_kv_pool.qsa_compressed_page_size
    )
    library = _library(path)
    if not 0 < columns <= _decode_topk_column_limit():
        return False
    if columns > 4096 and not (
        os.environ.get("SGLANG_XPU_QSA_INDEX_SCORE") == "1"
        and hasattr(library, "xpu_qsa_index_score")
    ):
        # A long capture must score the paged keys directly. The portable
        # gathered-key reference would materialize a large expanded cache.
        return False
    return all(
        hasattr(library, name)
        for name in ("xpu_qsa_rope", "xpu_qsa_store_rows", "xpu_qsa_expand")
    )


def can_capture_mtp_indexer(backend, forward_batch):
    """The fixed, one-request MTP seed/lookup uses persistent GPU buffers.

    The full indexer/metadata guard must also pass. Sparse sharing is bound
    before graph capture; its decode lookup and accepted-row draft-extend
    writeback consume replay-updated device indices and lengths only.
    """
    requests = getattr(forward_batch, "req_pool_indices", None)
    if (
        os.environ.get("SGLANG_XPU_QSA_MTP_GRAPH") != "1"
        or not 1 <= forward_batch.batch_size <= 8
        or not isinstance(requests, torch.Tensor)
        or requests.numel() != forward_batch.batch_size
    ):
        return False
    mode = forward_batch.forward_mode
    if mode.is_decode():
        method = getattr(backend, "should_reuse_mtp_sparse_indices", None)
    elif mode.is_draft_extend_v2():
        method = getattr(backend, "should_capture_mtp_sparse_indices", None)
    else:
        return False
    return method is not None and method(forward_batch)


def try_refresh_graph_metadata(
    backend,
    metadata,
    bs,
    req_pool_indices,
    seq_lens,
    forward_mode,
    spec_info,
    num_padding,
):
    """Refresh the existing XPU graph buffers without host row-layout staging."""
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    table = backend.req_to_token
    indexer = metadata.indexer_metadata
    pool = indexer.token_to_kv_pool
    if not (
        os.environ.get("SGLANG_XPU_QSA_METADATA_SYCL") == "1"
        and path
        and seq_lens.device.type == "xpu"
        and 1 <= bs <= 8
        and table is not None
        and table.device == seq_lens.device == req_pool_indices.device
        and table.ndim == 2
        and table.stride(1) == 1
        and table.dtype in (torch.int32, torch.int64)
        and seq_lens.dtype in (torch.int32, torch.int64)
        and req_pool_indices.dtype in (torch.int32, torch.int64)
        and seq_lens.ndim == req_pool_indices.ndim == 1
        and seq_lens.numel() >= bs
        and req_pool_indices.numel() >= bs
        and seq_lens.stride(0) == req_pool_indices.stride(0) == 1
        and pool.qsa_compress_ratio == 4
        and torch.xpu.get_device_properties(seq_lens.device).device_id == 0xE223
    ):
        return False
    library = _library(path)
    if not hasattr(library, "xpu_qsa_graph_metadata"):
        return False
    ring_size = getattr(pool, "qsa_pending_ring_size", pool.qsa_compress_ratio)
    refresh = library.xpu_qsa_graph_metadata
    extra_arguments = ()
    if ring_size != pool.qsa_compress_ratio:
        if not hasattr(library, "xpu_qsa_graph_metadata_causal"):
            return False
        refresh = library.xpu_qsa_graph_metadata_causal
        extra_arguments = (ring_size,)
    if forward_mode.is_target_verify():
        mode, extend = 1, int(spec_info.draft_token_num)
    elif forward_mode.is_draft_extend_v2():
        mode, extend = 2, 0
    elif forward_mode.is_decode():
        mode, extend = 0, 0
    else:
        return False
    rows = metadata.sequence_lengths.numel()
    if (
        not 1 <= rows <= 64
        or (mode == 0 and rows != bs)
        or (mode == 1 and rows != bs * extend)
    ):
        return False
    output = (
        metadata.sequence_lengths,
        metadata.row_req_pool_indices,
        indexer.graph_prefix_lengths,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
        indexer.graph_compressed_page_table,
    )
    for i, value in enumerate(output):
        if not (
            value is not None
            and value.device == seq_lens.device
            and value.is_contiguous()
            and value.shape[0] == rows
            and value.dtype == (torch.int64 if i == 6 else torch.int32)
        ):
            return False
    if metadata.sequence_lengths.data_ptr() != indexer.sequence_lengths.data_ptr():
        return False
    if metadata.row_req_pool_indices.data_ptr() != indexer.req_pool_indices.data_ptr():
        return False
    extend_lens = backend._stage_extend_lens(spec_info, bs, rows) if mode == 2 else None
    if extend_lens is not None and not (
        extend_lens.device == seq_lens.device
        and extend_lens.dtype in (torch.int32, torch.int64)
        and extend_lens.ndim == 1
        and extend_lens.numel() >= bs
        and extend_lens.stride(0) == 1
    ):
        return False
    stream = torch.xpu.current_stream(seq_lens.device)
    status = refresh(
        stream.sycl_queue,
        seq_lens.data_ptr(),
        req_pool_indices.data_ptr(),
        extend_lens.data_ptr() if extend_lens is not None else None,
        table.data_ptr(),
        *[v.data_ptr() for v in output],
        mode,
        bs,
        rows,
        max(0, min(int(num_padding or 0), bs)),
        extend,
        0,
        seq_lens.element_size(),
        req_pool_indices.element_size(),
        extend_lens.element_size() if extend_lens is not None else 4,
        table.element_size(),
        table.shape[0],
        table.stride(0),
        indexer.graph_compressed_page_table.shape[1],
        indexer.compress_ratio,
        pool.qsa_compressed_page_size * indexer.compress_ratio,
        *extra_arguments,
    )
    if status:
        raise RuntimeError(
            f"Native QSA graph metadata submission failed: status={status}"
        )
    for value in (seq_lens, req_pool_indices, extend_lens, table, *output):
        if value is not None:
            value.record_stream(stream)
    return True


def try_store_rows(buffer, locations, source):
    """Store valid pool rows, keeping the last source for duplicate locations.

    Unsupported layouts and aliased inputs use the ordinary Torch assignment.
    The native bounds guard skips invalid out-of-pool locations; pool producers
    own the valid-index contract, as for the native attention slot reader.
    """
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    if not path or not (
        buffer.device.type == "xpu"
        and buffer.device == source.device == locations.device
        and buffer.dtype == source.dtype
        and buffer.dtype in (torch.bfloat16, torch.int64)
        and buffer.ndim >= 2
        and source.ndim >= 2
        and buffer.shape[0] > 0
        and source.shape[1:] == buffer.shape[1:]
        and buffer.numel() > 0
        and buffer.is_contiguous()
        and locations.ndim == 1
        and locations.dtype in (torch.int32, torch.int64)
        and 0 <= locations.numel() == source.shape[0] <= 8192
        and torch.xpu.get_device_properties(buffer.device).device_id == 0xE223
        and not torch._C._overlaps(buffer, source)
        and not torch._C._overlaps(buffer, locations)
    ):
        return False
    if locations.numel() == 0:
        return True
    library = _library(path)
    if not hasattr(library, "xpu_qsa_store_rows"):
        return False
    source = source.reshape(source.shape[0], -1)
    locations = locations.contiguous()
    stream = torch.xpu.current_stream(buffer.device)
    status = library.xpu_qsa_store_rows(
        stream.sycl_queue,
        buffer.data_ptr(),
        source.data_ptr(),
        locations.data_ptr(),
        source.shape[0],
        source.shape[1],
        buffer.shape[0],
        buffer.element_size(),
        locations.element_size(),
        source.stride(0),
        source.stride(1),
    )
    if status:
        raise RuntimeError(f"Native QSA store submission failed: status={status}")
    for tensor in (buffer, source, locations):
        tensor.record_stream(stream)
    return True


def can_expand_blocks(blocks, positions, lengths, ratio, tokens):
    return bool(os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")) and (
        blocks.device.type == "xpu"
        and blocks.dtype == torch.int32
        and blocks.ndim == 2
        and 2 <= ratio <= 128
        and 0 < tokens <= 8192
        and blocks.shape[1] == (tokens + ratio - 1) // ratio
        and positions.numel() == lengths.numel() == blocks.shape[0]
        and positions.ndim == lengths.ndim == 1
        and positions.device == lengths.device == blocks.device
        and positions.dtype in (torch.int32, torch.int64)
        and lengths.dtype in (torch.int32, torch.int64)
        and torch.xpu.get_device_properties(blocks.device).device_id == 0xE223
        and hasattr(_library(os.environ["SGLANG_XPU_QSA_SYCL_LIB"]), "xpu_qsa_expand")
    )


def expand_blocks(blocks, positions, lengths, ratio, tokens):
    """Preserve the Torch fallback's exact integer order without sorting."""
    if not can_expand_blocks(blocks, positions, lengths, ratio, tokens):
        raise ValueError("Unsupported B70 QSA expansion geometry")
    output = torch.empty(
        (blocks.shape[0], tokens + ratio - 1), device=blocks.device, dtype=torch.int32
    )
    if blocks.shape[0] == 0:
        return output
    blocks = blocks.contiguous()
    positions = positions.to(torch.int64).contiguous()
    lengths = lengths.to(torch.int64).contiguous()
    stream = torch.xpu.current_stream(blocks.device)
    library = _library(os.environ["SGLANG_XPU_QSA_SYCL_LIB"])
    tensors = (blocks, positions, lengths, output)
    status = library.xpu_qsa_expand(
        stream.sycl_queue,
        *[tensor.data_ptr() for tensor in tensors],
        blocks.shape[0],
        blocks.shape[1],
        ratio,
        tokens,
    )
    if status:
        raise RuntimeError(f"Native QSA expansion submission failed: status={status}")
    for tensor in tensors:
        tensor.record_stream(stream)
    return output


def _head_partition(q, k):
    if q.ndim != 3 or k.ndim != 3:
        return 0
    geometry = (q.shape[1], k.shape[1])
    if geometry == (24, 2):
        return 1
    if os.environ.get("SGLANG_XPU_GGUF_TP_EP") == "1":
        return {(12, 1): 2, (6, 1): 4}.get(geometry, 0)
    return 0


def available(q, k, v, slots):
    tp = _head_partition(q, k)
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    geometry_ok = bool(path) and tp != 0 and (
        q.device.type == "xpu"
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.ndim == k.ndim == v.ndim == 3
        and q.shape[2] == 256
        and k.shape == v.shape
        and k.shape[2] == 256
        and k.shape[0] > 0
        and q.shape[0] > 0
        and k.device == v.device == slots.device == q.device
        and k.stride(1) == v.stride(1) == 256
        and k.stride(2) == v.stride(2) == 1
        and k.stride(0) >= k.shape[1] * 256
        and v.stride(0) >= v.shape[1] * 256
        and slots.ndim == 2
        and slots.shape[0] == q.shape[0]
        and slots.shape[1] > 0
        and slots.dtype == torch.int32
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    )
    if not geometry_ok:
        return False
    # Retain the fallback when an older library has only the full-head entry.
    return tp == 1 or hasattr(_library(path), f"xpu_qsa_bf16_tp{tp}")


def trim_compacted_prefill_indices(indices, sequence_length):
    """Drop whole padding chunks after QSAIndexer's stable compaction.

    The caller must supply fresh full-prefix indices with at most
    sequence_length valid entries per row. Keeping complete 64-cell chunks
    preserves both the original chunk results and the merge order.
    """
    if sequence_length < 1:
        raise ValueError("QSA prefill sequence length must be positive")
    width = (sequence_length + 63) // 64 * 64
    return indices[:, :width] if width < indices.shape[1] else indices


def sparse_attention(q, k, v, slots, scale=None, *, int8=False):
    """Read selected BF16 KV rows without expanding them to query-head count.

    Negative slots are masked. Producers must supply valid positive physical
    slots; the native bounds guard also masks out-of-pool slots. The two-stage
    FP32 softmax/reduction changes rounding relative to Torch's dense reference.
    """
    if int8:
        from sglang.kernels.ops.attention.xpu_qsa_int8 import available as check
    else:
        check = available
    if not check(q, k, v, slots):
        raise ValueError("Unsupported native B70 QSA geometry")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("QSA scale must be finite and positive")
    q = q.contiguous()
    slots = slots.contiguous()
    chunks = (slots.shape[1] + 63) // 64
    batch = int(os.environ.get("SGLANG_XPU_QSA_PREFILL_QUERY_BATCH", "0"))
    if batch not in (0, 8, 16, 32, 64, 128, 256):
        raise ValueError("QSA prefill query batch must be 0, 8, 16, 32, 64, 128 or 256")
    # Strata's prompt path bounds attention scratch by its query batch. Each
    # query keeps the same 64-cell chunks and merge order; batches share only
    # temporary storage on the current in-order queue. Decode stays unchanged.
    rows = min(batch, q.shape[0]) if batch and q.shape[0] > 4 else q.shape[0]
    shape = (rows, k.shape[1], chunks, q.shape[1] // k.shape[1])
    acc = torch.empty((*shape, 256), device=q.device, dtype=torch.float32)
    maximum = torch.empty(shape, device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = torch.empty_like(q)
    stream = torch.xpu.current_stream(q.device)
    library = _library(os.environ["SGLANG_XPU_QSA_SYCL_LIB"])
    tensors = (q, k, v, slots, acc, maximum, denominator, output)
    tp = _head_partition(q, k)
    name = "xpu_qsa_int8" if int8 else "xpu_qsa_bf16"
    function = getattr(library, name if tp == 1 else f"{name}_tp{tp}")
    for first in range(0, q.shape[0], rows):
        count = min(rows, q.shape[0] - first)
        status = function(
            stream.sycl_queue,
            q.data_ptr() + first * q.stride(0) * q.element_size(),
            k.data_ptr(), v.data_ptr(),
            slots.data_ptr() + first * slots.stride(0) * slots.element_size(),
            acc.data_ptr(), maximum.data_ptr(), denominator.data_ptr(),
            output.data_ptr() + first * output.stride(0) * output.element_size(),
            count, slots.shape[1], k.shape[0], k.stride(0), v.stride(0), scale,
        )
        if status:
            raise RuntimeError(f"Native BF16 QSA submission failed: status={status}")
    for tensor in tensors:
        tensor.record_stream(stream)
    return output
