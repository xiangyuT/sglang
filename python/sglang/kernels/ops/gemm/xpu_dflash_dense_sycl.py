"""Opt-in sequential-K SYCL projections with byte-preserving FP8 packing.

Warm before capture; restart and recapture after changing weights or flags.
Activation and weight scales remain dynamic. Source FP8 weights stay available
for prefill. The subgroup matrix intrinsic and FP16 operand rounding match
those of the retained M=4 projection or the M=8 native projection; there are no split partial sums.
A 16/32 KiB lane-major FP16 activation buffer is prepared per projection.
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

logger = logging.getLogger(__name__)
SHAPES = {(12288, 2048): 0, (9216, 2048): 1}
_CACHE_ATTRIBUTE = "_sglang_dflash_dense_sycl_packed"


@lru_cache(maxsize=1)
def load_dense_sycl():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("SYCL dense requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        abi = library.dflash_seq_dense_abi
        abi.argtypes = []
        abi.restype = ctypes.c_int
        if abi() != 2:
            raise RuntimeError(
                "Unsupported sequential SYCL dense ABI; rebuild the ABI 2 native library"
            )
        library.dflash_seq_dense_error.argtypes = []
        library.dflash_seq_dense_error.restype = ctypes.c_char_p
        library.dflash_seq_dense_pack.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int
        ] * 2
        library.dflash_seq_dense_pack.restype = ctypes.c_int
        library.dflash_seq_dense_gemm.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_int
        ] * 5
        library.dflash_seq_dense_gemm.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError(
            "Sequential SYCL dense symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    logger.info(
        "Using sequential SYCL M=%d dense: 12288x2048 and 9216x2048 "
        "(SG=4, GRF=128, lane-major FP16 A, vector FP8 B)",
        native_window(library),
    )
    return library


def packed_weight(weight):
    """Keep layout bytes on the source tensor, without a global model cache."""
    version = weight._version
    entry = getattr(weight, _CACHE_ATTRIBUTE, None)
    if entry is not None and entry[0] == version:
        return entry[1]
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError(
            "Warm SYCL dense packed weights before XPU graph capture; "
            "restart and recapture after source weight changes"
        )
    library = load_dense_sycl()
    packed = torch.empty(weight.numel() // 4, device=weight.device, dtype=torch.int32)
    rc = library.dflash_seq_dense_pack(
        torch.xpu.current_stream().sycl_queue,
        weight.data_ptr(),
        packed.data_ptr(),
        *weight.shape,
    )
    if rc:
        raise RuntimeError(library.dflash_seq_dense_error().decode())
    setattr(weight, _CACHE_ATTRIBUTE, (version, packed))
    return packed


def dense_sycl(q, w, xs, ws):
    """Caller enforces the B70 contiguous M4/M8 block-128 tensor contract."""
    mode = SHAPES[tuple(w.shape)]
    library = load_dense_sycl()
    rows = q.shape[0]
    require_native_window(library, rows)
    packed = packed_weight(w)
    n, k = w.shape
    prepared = torch.empty((k, rows), device=q.device, dtype=torch.float16)
    out = torch.empty((rows, n), device=q.device, dtype=torch.bfloat16)
    rc = library.dflash_seq_dense_gemm(
        torch.xpu.current_stream().sycl_queue,
        q.data_ptr(),
        packed.data_ptr(),
        xs.data_ptr(),
        ws.data_ptr(),
        prepared.data_ptr(),
        out.data_ptr(),
        n,
        k,
        ws.stride(0),
        ws.stride(1),
        mode,
    )
    if rc:
        raise RuntimeError(library.dflash_seq_dense_error().decode())
    return out
