"""Opt-in B70 AR BF16 E256/top8 routing, with native FP32 rounding.

Build using test/manual/kernels/build_xpu_ar_topk.py. The BMG correctly-rounded
divide option is required: generic approximate reciprocals can change tokens.
Warm the library before capture and restart after changing its path or contents.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_ar_topk(logits, weights, ids, renormalize):
    if not (
        isinstance(renormalize, bool)
        and logits.shape == (1, 256)
        and weights.shape == ids.shape == (1, 8)
        and logits.dtype == torch.bfloat16
        and weights.dtype == torch.float32
        and ids.dtype == torch.int32
        and logits.device.type == "xpu"
        and logits.device == weights.device == ids.device
        and all(t.is_contiguous() for t in (logits, weights, ids))
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (logits, weights)):
        return False
    return torch.xpu.get_device_properties(logits.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _load_library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_AR_TOPK_SYCL_LIB to the built AR library")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the SYCL AR top-k library before XPU graph capture")
    library = ctypes.CDLL(path)
    try:
        library.ar_topk_abi.argtypes = []
        library.ar_topk_abi.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("Library lacks AR top-k ABI; rebuild the library") from exc
    if library.ar_topk_abi() != 1:
        raise RuntimeError("Unsupported SYCL AR top-k ABI; rebuild the library")
    library.ar_topk_error.argtypes = []
    library.ar_topk_error.restype = ctypes.c_char_p
    library.ar_topk_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    library.ar_topk_run.restype = ctypes.c_int
    logger.info("Using B70 AR SYCL E256/top8, SG32/GRF128, FP32 rounded divide")
    return library


def xpu_ar_topk(logits, weights, ids, renormalize=False):
    """Write FP32 top8 weights and int32 IDs, breaking score ties by low ID."""
    if not can_use_ar_topk(logits, weights, ids, renormalize):
        raise ValueError("Unsupported B70 AR BF16 [1,256] -> FP32/int32 [1,8]")
    library = _load_library(envs.SGLANG_XPU_AR_TOPK_SYCL_LIB.get())
    stream = torch.xpu.current_stream(logits.device)
    status = library.ar_topk_run(
        stream.sycl_queue,
        logits.data_ptr(),
        weights.data_ptr(),
        ids.data_ptr(),
        int(renormalize),
    )
    if status:
        raise RuntimeError(library.ar_topk_error().decode())
    for tensor in (logits, weights, ids):
        tensor.record_stream(stream)
