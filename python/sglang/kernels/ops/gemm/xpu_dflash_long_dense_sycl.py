"""Opt-in M4/N2048/K4096 FP16-operand SYCL projection with split-K=8.

Scales and operands retain FP16 rounding; eight FP32 partials change reduction
order before BF16 output. Warm packed weights before capture; restart and
recapture after changing weights, flags or libraries. No derived FP16 weights.
"""

import ctypes
import logging
from functools import lru_cache
import torch
from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import packed_weight
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_long_dense():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("SYCL long dense requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        library.dflash_long_dense_abi.argtypes = []
        library.dflash_long_dense_abi.restype = ctypes.c_int
        if library.dflash_long_dense_abi() != 1:
            raise RuntimeError("Unsupported SYCL long dense ABI")
        library.dflash_long_dense_gemm.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int
        ] * 2
        library.dflash_long_dense_gemm.restype = ctypes.c_int
        library.dflash_long_dense_error.argtypes = []
        library.dflash_long_dense_error.restype = ctypes.c_char_p
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL long dense symbols missing; rebuild with test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    logger.info("Using SYCL M=4 long dense: 2048x4096 (split-K=8, SG=4, GRF=128)")
    return library


def can_use_long_dense(q, w, xs, ws, block_size, bias):
    return (
        q.ndim == w.ndim == xs.ndim == ws.ndim == 2
        and q.device.type == "xpu"
        and q.shape == (4, 4096)
        and w.shape == (2048, 4096)
        and q.dtype == w.dtype == torch.float8_e4m3fn
        and xs.dtype == ws.dtype == torch.float32
        and q.is_contiguous()
        and w.is_contiguous()
        and xs.is_contiguous()
        and xs.shape == (4, 32)
        and ws.shape == (16, 32)
        and q.device == w.device == xs.device == ws.device
        and tuple(block_size) == (128, 128)
        and bias is None
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    )


def long_dense_sycl(q, w, xs, ws):
    """Caller checks the B70 M4 block128 contract before entering this path."""
    library = load_long_dense()
    packed = packed_weight(w)
    prepared = torch.empty((4096, 4), device=q.device, dtype=torch.float16)
    partial = torch.empty((8, 4, 2048), device=q.device, dtype=torch.float32)
    out = torch.empty((4, 2048), device=q.device, dtype=torch.bfloat16)
    rc = library.dflash_long_dense_gemm(
        torch.xpu.current_stream().sycl_queue,
        q.data_ptr(),
        packed.data_ptr(),
        xs.data_ptr(),
        ws.data_ptr(),
        prepared.data_ptr(),
        out.data_ptr(),
        partial.data_ptr(),
        *ws.stride(),
    )
    if rc:
        raise RuntimeError(library.dflash_long_dense_error().decode())
    return out
