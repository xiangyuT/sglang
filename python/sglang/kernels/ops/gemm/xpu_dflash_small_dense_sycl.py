"""Opt-in M=4 K512 SYCL projection with live FP8 weights and block scales.

Scales and operands retain the reference FP16 rounding; reduction is FP32.
The vector reduction can round differently from the retained DPAS path.
There is no derived weight cache. Restart and recapture after flag changes.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_small_dense():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("SYCL small dense requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        library.dflash_small_dense_abi.argtypes = []
        library.dflash_small_dense_abi.restype = ctypes.c_int
        if library.dflash_small_dense_abi() != 1:
            raise RuntimeError("Unsupported small SYCL dense ABI")
        library.dflash_small_dense_error.argtypes = []
        library.dflash_small_dense_error.restype = ctypes.c_char_p
        library.dflash_small_dense_gemm.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_int
        ] * 2
        library.dflash_small_dense_gemm.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError(
            "Small SYCL dense symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    logger.info("Using SYCL M=4 small dense: 2048x512 (ESIMD VL=128, live FP8)")
    return library


def small_dense_sycl(q, w, xs, ws):
    """Caller enforces the B70 contiguous M=4 block-128 projection contract."""
    library = load_small_dense()
    prepared = torch.empty((4, 512), device=q.device, dtype=torch.float16)
    out = torch.empty((4, 2048), device=q.device, dtype=torch.bfloat16)
    rc = library.dflash_small_dense_gemm(
        torch.xpu.current_stream().sycl_queue,
        q.data_ptr(),
        xs.data_ptr(),
        w.data_ptr(),
        ws.data_ptr(),
        prepared.data_ptr(),
        out.data_ptr(),
        *ws.stride(),
    )
    if rc:
        raise RuntimeError(library.dflash_small_dense_error().decode())
    return out
