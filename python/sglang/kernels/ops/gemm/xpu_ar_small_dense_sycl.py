"""Opt-in B70 AR small GEMV with live FP8 weights and retained rounding.

Build with test/manual/kernels/build_xpu_ar_small_dense.py. Warm before graph
capture; restart and recapture after changing the flag or library. The shared
gate/up keeps FP32 dequantization and its explicit FMA/reduction tree. The
shared down projection retains FP16 scales and scaled-operand rounding.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_ar_small_dense(q, w, xs, ws):
    if q.ndim != 2 or w.ndim != 2:
        return False
    n, k = w.shape
    tensors = (q, w, xs, ws)
    if not (
        (n, k) in ((1024, 2048), (2048, 512))
        and q.shape == (1, k)
        and q.dtype == w.dtype == torch.float8_e4m3fn
        and xs.dtype == ws.dtype == torch.float32
        and xs.numel() == k // 128
        and ws.shape == (n // 128, k // 128)
        and q.is_contiguous()
        and w.is_contiguous()
        and xs.is_contiguous()
        and q.data_ptr() % 4 == w.data_ptr() % 4 == 0
        and all(0 < s < 2**31 for s in ws.stride())
        and q.device.type == "xpu"
        and all(t.device == q.device for t in tensors)
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    return torch.xpu.get_device_properties(q.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _load_library(path):
    if not path:
        raise RuntimeError(
            "Set SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB to the built library"
        )
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the SYCL AR small dense before XPU graph capture")
    library = ctypes.CDLL(path)
    try:
        library.ar_small_abi.argtypes = []
        library.ar_small_abi.restype = ctypes.c_int
        if library.ar_small_abi() != 1:
            raise RuntimeError("Unsupported SYCL AR small dense ABI; rebuild it")
        library.ar_small_error.argtypes = []
        library.ar_small_error.restype = ctypes.c_char_p
        library.ar_small_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5
        library.ar_small_run.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("Library lacks SYCL AR small dense ABI; rebuild it") from exc
    logger.info(
        "Using B70 AR SYCL small dense: gate1024x2048 row4 FP32/FMA, "
        "down2048x512 row2 FP16, SG32/GRF256"
    )
    return library


@lru_cache(maxsize=2)
def _log_shape(n, k):
    logger.info("B70 AR SYCL small dense dispatch: N=%d K=%d", n, k)


def xpu_ar_small_dense(q, w, xs, ws):
    """Return BF16 output without changing activation quantization or weights."""
    if not can_use_ar_small_dense(q, w, xs, ws):
        raise ValueError("Unsupported B70 AR small dense layout")
    library = _load_library(envs.SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB.get())
    n, k = w.shape
    _log_shape(n, k)
    out = torch.empty((1, n), dtype=torch.bfloat16, device=q.device)
    stream = torch.xpu.current_stream(q.device)
    status = library.ar_small_run(
        stream.sycl_queue,
        *[t.data_ptr() for t in (q, w, xs, ws, out)],
        n,
        k,
        *ws.stride(),
        4 if k == 2048 else 2,
    )
    if status:
        raise RuntimeError(library.ar_small_error().decode())
    for tensor in (q, w, xs, ws, out):
        tensor.record_stream(stream)
    return out
