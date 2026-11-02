"""Opt-in B70 M4/M8 bidirectional draft attention using pinned SYCL-TLA decode.

The caller preserves the existing no-mask, GQA4, scale and window checks.
Backend-provided CSR values must describe 0..2047 valid prefix slots. Every
call packs live K/V and produces metadata on device; no weight cache or host
readback is used. Restart and recapture after flag or library changes.
"""

import ctypes
import logging
from functools import lru_cache

import torch
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_attention_tla():
    path = envs.SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB.get()
    if not path:
        raise RuntimeError(
            "SYCL-TLA attention requires SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB"
        )
    library = ctypes.CDLL(path)
    try:
        library.dflash_attention_tla_abi.argtypes = []
        library.dflash_attention_tla_abi.restype = ctypes.c_int
        if library.dflash_attention_tla_abi() != 1:
            raise RuntimeError("Unsupported SYCL-TLA attention ABI")
        library.dflash_attention_tla_error.argtypes = []
        library.dflash_attention_tla_error.restype = ctypes.c_char_p
        library.dflash_attention_tla_run.argtypes = [ctypes.c_void_p] * 14 + [
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.dflash_attention_tla_run.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL-TLA attention symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_attention_tla.py"
        ) from exc
    return library


@lru_cache(maxsize=1)
def _announce():
    logger.info(
        "Using SYCL-TLA B70 M4 bidirectional draft attention: "
        "window2047, dynamic KV packing, no weight cache"
    )


def can_use_attention_tla(q, k, v, kb, vb, ptr, ids, out):
    return _can_use_attention_layout(q, k, v, kb, vb, ptr, ids, out, windows=(4,))


def _can_use_attention_layout(q, k, v, kb, vb, ptr, ids, out, *, windows):
    if not (
        not torch.compiler.is_compiling()
        and q.device.type == "xpu"
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[0] in windows
        and q.shape == out.shape == (q.shape[0], 32, 128)
        and q.stride() in ((4096, 128, 1), (6144, 128, 1))
        and k.shape == v.shape == (q.shape[0], 8, 128)
        and kb.ndim == vb.ndim == 3
        and kb.shape == vb.shape
        and kb.shape[1:] == (8, 128)
        and kb.shape[0] > 0
        and ptr.shape == (2,)
        and ptr.dtype == torch.int32
        and ids.ndim == 1
        and ids.numel() >= 2047
        and ids.dtype == torch.int64
        and all(t.device == q.device for t in (k, v, kb, vb, ptr, ids, out))
        and all(t.dtype == q.dtype for t in (k, v, kb, vb, out))
        and all(t.is_contiguous() for t in (k, v, kb, vb, ptr, ids, out))
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, kb, vb)):
        return False
    if any(torch._C._overlaps(out, t) for t in (q, k, v, kb, vb, ptr, ids)):
        return False
    return torch.xpu.get_device_properties(q.device).device_id == 0xE223


def try_attention_tla(q, k, v, kb, vb, ptr, ids, out):
    """Return False without writes or loading a library for unsupported layouts."""
    if not can_use_attention_tla(q, k, v, kb, vb, ptr, ids, out):
        return False
    library = load_attention_tla()
    pk = torch.empty((2176, 8, 128), device=q.device, dtype=q.dtype)
    pv = torch.empty_like(pk)
    lengths = torch.empty(4, device=q.device, dtype=torch.int32)
    cuq = torch.empty(5, device=q.device, dtype=torch.int32)
    table = torch.empty((4, 17), device=q.device, dtype=torch.int32)
    stream = torch.xpu.current_stream(q.device)
    tensors = (q, k, v, kb, vb, ptr, ids, out, pk, pv, lengths, cuq, table)
    for tensor in tensors:
        tensor.record_stream(stream)
    rc = library.dflash_attention_tla_run(
        stream.sycl_queue,
        *[tensor.data_ptr() for tensor in tensors],
        q.stride(0),
        q.device.index,
    )
    if rc:
        message = library.dflash_attention_tla_error().decode()
        raise RuntimeError(message or f"SYCL-TLA attention failed with status {rc}")
    _announce()
    return True


@lru_cache(maxsize=1)
def load_attention_paged_tla():
    path = envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.get()
    if not path:
        raise RuntimeError(
            "Paged SYCL-TLA attention requires SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB"
        )
    library = ctypes.CDLL(path)
    try:
        library.dflash_attention_paged_tla_abi.argtypes = []
        library.dflash_attention_paged_tla_abi.restype = ctypes.c_int
        if library.dflash_attention_paged_tla_abi() != 1:
            raise RuntimeError("Unsupported paged SYCL-TLA attention ABI")
        library.dflash_attention_paged_tla_error.argtypes = []
        library.dflash_attention_paged_tla_error.restype = ctypes.c_char_p
        library.dflash_attention_paged_tla_run.argtypes = [ctypes.c_void_p] * 11 + [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.dflash_attention_paged_tla_run.restype = ctypes.c_int
        window_query = getattr(library, "dflash_attention_paged_tla_window", None)
        library.window_size = 4  # Legacy ABI1 libraries only implement M4.
        if window_query is not None:
            window_query.argtypes = []
            window_query.restype = ctypes.c_int
            library.window_size = window_query()
        if library.window_size not in (4, 8):
            raise RuntimeError("Unsupported paged SYCL-TLA attention window")
    except AttributeError as exc:
        raise RuntimeError(
            "Paged SYCL-TLA symbols missing; rebuild with --paged-kv"
        ) from exc
    return library


def can_use_attention_paged_tla(
    q, k, v, kb, vb, ptr, ids, out, saved_paged_kv, *, page_size
):
    """The backend vouches for saved KV, logical page alignment and ownership.

    Full-prefix CSR and the M4/M8 current write locations must be valid, with
    every logical 128-token page internally contiguous. Current tokens continue
    that logical sequence. The last page belongs exclusively to this request;
    only its invalid suffix is cleared. These are allocator/value contracts,
    not tensor metadata properties; no device-to-host validation is performed.
    """
    if not _can_use_attention_layout(q, k, v, kb, vb, ptr, ids, out, windows=(4, 8)):
        return False
    if page_size != 128 or kb.shape[0] % 128 or kb.shape[0] < 128:
        return False
    if not isinstance(saved_paged_kv, tuple) or len(saved_paged_kv) != 3:
        return False
    fullptr, fullids, loc = saved_paged_kv
    if not all(isinstance(t, torch.Tensor) for t in saved_paged_kv):
        return False
    if not (
        fullptr.shape == (2,)
        and fullptr.dtype == torch.int32
        and fullids.ndim == 1
        and fullids.numel() >= 2047
        and fullids.dtype == torch.int64
        and loc.shape == (q.shape[0],)
        and loc.dtype == torch.int64
        and all(t.device == q.device and t.is_contiguous() for t in saved_paged_kv)
    ):
        return False
    return not any(torch._C._overlaps(out, t) for t in saved_paged_kv)


@lru_cache(maxsize=2)
def _announce_paged(window):
    logger.info(
        "Using SYCL-TLA B70 M%d saved paged KV attention: "
        "page128, device-aligned window2047, final-page padding clear",
        window,
    )


def try_attention_paged_tla(
    q, k, v, kb, vb, ptr, ids, out, saved_paged_kv, *, page_size
):
    if not can_use_attention_paged_tla(
        q,
        k,
        v,
        kb,
        vb,
        ptr,
        ids,
        out,
        saved_paged_kv,
        page_size=page_size,
    ):
        return False
    library = load_attention_paged_tla()
    window = q.shape[0]
    if window != library.window_size:
        raise RuntimeError(
            f"Paged SYCL-TLA window mismatch: M{window} input, "
            f"M{library.window_size} library; rebuild with --paged-kv "
            f"--window-size {window} and recapture"
        )
    fullptr, fullids, loc = saved_paged_kv
    lengths = torch.empty(window, device=q.device, dtype=torch.int32)
    cuq = torch.empty(window + 1, device=q.device, dtype=torch.int32)
    table = torch.empty((window, 18), device=q.device, dtype=torch.int32)
    stream = torch.xpu.current_stream(q.device)
    tensors = (q, kb, vb, fullptr, fullids, loc, out, lengths, cuq, table)
    for tensor in tensors:
        tensor.record_stream(stream)
    rc = library.dflash_attention_paged_tla_run(
        stream.sycl_queue,
        *[tensor.data_ptr() for tensor in tensors],
        q.stride(0),
        q.device.index,
        kb.shape[0],
    )
    if rc:
        raise RuntimeError(library.dflash_attention_paged_tla_error().decode())
    _announce_paged(window)
    return True
