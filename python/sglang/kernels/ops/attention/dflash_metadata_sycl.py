"""Opt-in single-request DFlash verify metadata in one SYCL launch.

Consume the translator's current source on every call. Host-fed plans are not
cached or captured here. Other layouts retain the attention backend's path.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.speculative.dflash_native import (
    native_window,
    require_native_window,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind

logger = logging.getLogger(__name__)
OUTPUT_NAMES = (
    "qo_indptr",
    "kv_indptr",
    "cuda_graph_kv_indices",
    "mask_indptr",
    "window_kv_indptr",
    "cuda_graph_window_kv_indices",
    "cuda_graph_window_kv_offsets",
)


@lru_cache(maxsize=1)
def load_metadata_sycl():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("SYCL metadata requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        library.dflash_metadata_abi.argtypes = []
        library.dflash_metadata_abi.restype = ctypes.c_int
        if library.dflash_metadata_abi() != 1:
            raise RuntimeError("Unsupported SYCL metadata ABI")
        library.dflash_metadata_fill.argtypes = [ctypes.c_void_p] * 12 + [
            ctypes.c_int
        ] * 6
        library.dflash_metadata_fill.restype = ctypes.c_int
        library.dflash_metadata_error.argtypes = []
        library.dflash_metadata_error.restype = ctypes.c_char_p
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL metadata symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    logger.info(
        "Using SYCL single-request DFlash M=%d verify metadata (window=2047)",
        native_window(library),
    )
    return library


def try_update_metadata(backend, plan, bs, seq_lens, spec_info, req_pool_indices):
    """Return the existing metadata tuple, or None without changing buffers."""
    if not (
        backend.is_draft_runner
        and bs == 1
        and backend.dcp_size == 1
        and backend.sliding_window_size == 2047
        and backend._target_verify_num_tokens_per_req(spec_info) in (4, 8)
        and (spec_info is None or spec_info.custom_mask is None)
        and not isinstance(backend.token_to_kv_pool, BaseSWAKVPool)
        and backend.kv_index_translator.window_kv_kind == IdSpaceKind.FULL
        and seq_lens.device.type == "xpu"
        and seq_lens.numel() == 1
        and seq_lens.is_contiguous()
        and seq_lens.dtype in (torch.int32, torch.int64)
    ):
        return None
    source = backend.kv_index_translator.read_source(
        plan, req_pool_indices=req_pool_indices, bs=1
    )
    outputs = [getattr(backend, name) for name in OUTPUT_NAMES]
    qo, kv, full, mask, wp, window, offset = outputs
    dtypes = (
        qo.dtype,
        torch.int32,
        torch.int64,
        mask.dtype,
        torch.int32,
        torch.int64,
        torch.int32,
    )
    if not (
        qo.dtype in (torch.int32, torch.int64)
        and mask.dtype in (torch.int32, torch.int64)
        and source.entry_page_size in (1, 128)
        and backend.kv_index_translator.page_size == 128
        and source.ids.ndim == 2
        and source.ids.stride(1) == 1
        and source.ids.dtype in (torch.int32, torch.int64)
        and source.row_ids.ndim == 1
        and source.row_ids.numel() >= 1
        and source.row_ids.is_contiguous()
        and source.row_ids.dtype in (torch.int32, torch.int64)
        and all(
            t.device == seq_lens.device and t.is_contiguous() and t.dtype == dtype
            for t, dtype in zip(outputs, dtypes)
        )
        and full.numel() >= backend.max_context_len
        and window.numel() >= 2047
        and min(qo.numel(), kv.numel(), mask.numel(), wp.numel()) >= 2
        and offset.numel() >= 1
        and source.ids.device == source.row_ids.device == seq_lens.device
        and (
            source.v2p is None
            or (
                source.entry_page_size == 1
                and source.v2p.device == seq_lens.device
                and source.v2p.ndim == 1
                and source.v2p.is_contiguous()
                and source.v2p.dtype in (torch.int32, torch.int64)
            )
        )
        and torch.xpu.get_device_properties(seq_lens.device).device_id == 0xE223
    ):
        return None
    bits = (
        int(source.ids.dtype == torch.int64)
        | (2 * int(source.row_ids.dtype == torch.int64))
        | (4 * int(source.v2p is not None and source.v2p.dtype == torch.int64))
        | (8 * int(qo.dtype == torch.int64))
        | (16 * int(mask.dtype == torch.int64))
    )
    library = load_metadata_sycl()
    require_native_window(library, backend._target_verify_num_tokens_per_req(spec_info))
    tensors = (seq_lens, source.row_ids, source.ids, source.v2p, *outputs)
    code = library.dflash_metadata_fill(
        torch.xpu.current_stream().sycl_queue,
        *[t.data_ptr() if t is not None else None for t in tensors],
        source.row_stride,
        source.entry_page_size,
        128,
        bits,
        int(seq_lens.dtype == torch.int64),
        full.numel(),
    )
    if code:
        raise RuntimeError(library.dflash_metadata_error().decode())
    return (
        qo[:2],
        kv[:2],
        None,
        mask[:2],
        wp[:2],
        window,
        backend.cuda_graph_window_num_kv_splits,
        offset,
    )
