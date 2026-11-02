"""Opt-in B70 TP shared/output projections using the single-GPU SYCL method.

Native scaled_mm uses different dequantization precision on these shapes.
Keep the calibrated FP16/FP32 boundary per shape, FP32 accumulation and BF16
output. Different reduction trees can change individual rounded outputs.
"""
import ctypes
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_MODES = {
    2: {(2048, 2048): 3, (512, 2048): 6, (2048, 256): 2},
    4: {(2048, 1024): 2, (256, 2048): 5},
}


def can_use_tp_aux_projection(q, w, xs, ws):
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    if q.ndim != 2 or q.shape[0] != 1 or w.ndim != 2:
        return False
    n, k = w.shape
    if not (
        q.device.type == "xpu"
        and all(t.device == q.device for t in (w, xs, ws))
        and q.dtype == w.dtype == torch.float8_e4m3fn
        and xs.dtype == ws.dtype == torch.float32
        and q.shape[1] == k
        and xs.shape == (1, k // 128)
        and ws.shape == (n // 128, k // 128)
        and q.is_contiguous() and w.is_contiguous() and xs.is_contiguous()
        and q.data_ptr() % 16 == w.data_ptr() % 16 == 0
        and xs.data_ptr() % 4 == ws.data_ptr() % 4 == 0
        and all(0 < stride < 2**31 for stride in ws.stride())
    ):
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    return (
        parallel.tp_size in _MODES
        and (n, k) in _MODES[parallel.tp_size]
        and parallel.pp_size == 1
        # Shared-expert and attention output projections retain these TP
        # shards under EP2; only the routed expert matrices change ownership.
        and (
            parallel.moe_ep_size == 1
            or (parallel.tp_size == 2 and parallel.moe_ep_size == 2)
        )
        and parallel.attn_cp_size == 1
        and not parallel.attn_dp_enabled
        and get_spec().speculative_algorithm is None
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    )


@lru_cache(maxsize=1)
def _library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_TP_AUX_FP8_SYCL_LIB to the built library")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm TP auxiliary projections before graph capture")
    library = ctypes.CDLL(path)
    library.tp_aux_fp8_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5
    library.tp_aux_fp8_run.restype = ctypes.c_int
    library.tp_aux_fp8_error.restype = ctypes.c_char_p
    logger.info(
        "B70 TP auxiliary SYCL library: path=%s sha256=%s", path,
        hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
    return library


@lru_cache(maxsize=5)
def _log_shape(n, k, mode):
    logger.info(
        "B70 TP auxiliary SYCL enabled: N=%d K=%d row_tile=%d dequant=%s",
        n, k, (1, 2, 4, 8)[mode % 4], "FP16" if mode < 4 else "FP32",
    )


def tp_aux_projection(q, w, xs, ws):
    if not can_use_tp_aux_projection(q, w, xs, ws):
        raise ValueError("Unsupported B70 TP auxiliary FP8 layout")
    from sglang.srt.runtime_context import get_parallel

    n, k = w.shape
    mode = _MODES[get_parallel().tp_size][(n, k)]
    library = _library(envs.SGLANG_XPU_TP_AUX_FP8_SYCL_LIB.get())
    _log_shape(n, k, mode)
    output = torch.empty((1, n), device=q.device, dtype=torch.bfloat16)
    stream = torch.xpu.current_stream(q.device)
    tensors = (q, w, xs, ws, output)
    code = library.tp_aux_fp8_run(
        stream.sycl_queue, *[tensor.data_ptr() for tensor in tensors],
        n, k, *ws.stride(), mode,
    )
    if code:
        raise RuntimeError((code, library.tp_aux_fp8_error().decode()))
    for tensor in tensors:
        tensor.record_stream(stream)
    return output
