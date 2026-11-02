"""Opt-in SYCL block-FP8 projections for B70 TP2/TP4 pure AR decode.

Retains activation quantization, FP16 dequantization and BF16 output.
The FP32 tree reduction can change rounded outputs relative to scaled_mm.
"""
import ctypes
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_MODES = {6144: 1, 4608: 0, 3072: 2, 2560: 2}


def can_use_tp_projection(q, w, xs, ws):
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    if q.ndim != 2 or w.ndim != 2:
        return False
    n, k = w.shape
    if not (
        n in _MODES and k == 2048 and q.shape == (1, k)
        and q.dtype == w.dtype == torch.float8_e4m3fn
        and xs.dtype == ws.dtype == torch.float32
        and xs.shape == (1, 16) and ws.shape == (n // 128, 16)
        and q.is_contiguous() and w.is_contiguous() and xs.is_contiguous()
        and all(t.device == q.device for t in (w, xs, ws))
        and q.device.type == "xpu"
        and q.data_ptr() % 16 == w.data_ptr() % 16 == 0
        and xs.data_ptr() % 4 == ws.data_ptr() % 4 == 0
        and all(0 < s < 2**31 for s in ws.stride())
    ):
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec

    tp = get_parallel().tp_size
    if tp not in (2, 4) or get_parallel().pp_size != 1:
        return False
    if get_spec().speculative_algorithm is not None:
        return False
    if n not in ((6144, 4608) if tp == 2 else (3072, 2560)):
        return False
    return torch.xpu.get_device_properties(q.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_TP_BLOCK_FP8_SYCL_LIB to the built library")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the TP projection before XPU graph capture")
    lib = ctypes.CDLL(path)
    lib.tp_dense01_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
    lib.tp_dense01_run.restype = ctypes.c_int
    lib.tp_dense01_error.argtypes = []
    lib.tp_dense01_error.restype = ctypes.c_char_p
    logger.info("B70 TP SYCL projection library: path=%s sha256=%s", path,
                hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return lib


@lru_cache(maxsize=4)
def _log_shape(n, mode):
    logger.info("B70 TP SYCL projection enabled: N=%d K=2048 row_tile=%d; FP32 tree/BF16 output", n, (4, 8, 16)[mode])


def tp_projection(q, w, xs, ws):
    if not can_use_tp_projection(q, w, xs, ws):
        raise ValueError("Unsupported B70 TP block-FP8 projection layout")
    lib = _library(envs.SGLANG_XPU_TP_BLOCK_FP8_SYCL_LIB.get())
    n = w.shape[0]
    mode = _MODES[n]
    _log_shape(n, mode)
    out = torch.empty((1, n), device=q.device, dtype=torch.bfloat16)
    stream = torch.xpu.current_stream(q.device)
    rc = lib.tp_dense01_run(stream.sycl_queue, q.data_ptr(), w.data_ptr(),
                           xs.data_ptr(), ws.data_ptr(), out.data_ptr(),
                           n, *ws.stride(), mode)
    if rc:
        raise RuntimeError((rc, lib.tp_dense01_error().decode()))
    for tensor in (q, w, xs, ws, out):
        tensor.record_stream(stream)
    return out
