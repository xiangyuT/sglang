"""Opt-in asynchronous SYCL copies for B70 M4/M8 graph-input geometry.

Five contiguous INT64 inputs contain M/M/M/1/1 elements. Unsupported layouts
and alias hazards return False for the original foreach path. Inputs and
outputs are recorded on the caller's stream for allocator lifetime safety.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.attention.dflash_gdn_sycl import load_gdn
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_small_copy():
    library = load_gdn()
    try:
        library.dflash_small_copy_abi.argtypes = []
        library.dflash_small_copy_abi.restype = ctypes.c_int
        if library.dflash_small_copy_abi() != 1:
            raise RuntimeError("Unsupported SYCL small-copy ABI")
        library.dflash_small_copy.argtypes = [ctypes.c_void_p] * 11
        library.dflash_small_copy.restype = ctypes.c_int
        if hasattr(library, "dflash_small_copy_m8"):
            library.dflash_small_copy_m8.argtypes = [ctypes.c_void_p] * 11
            library.dflash_small_copy_m8.restype = ctypes.c_int
        library.dflash_small_copy_error.argtypes = []
        library.dflash_small_copy_error.restype = ctypes.c_char_p
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL small-copy symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_gdn.py"
        ) from exc
    return library


@lru_cache(maxsize=2)
def _announce(window):
    logger.info(
        "Using SYCL B70 M%d graph-input copies: 5 INT64 buffers (%d/%d/%d/1/1)",
        window,
        window,
        window,
        window,
    )


def eligible(dsts, srcs):
    if len(dsts) != 5 or len(srcs) != 5:
        return False
    device = dsts[0].device
    if device.type != "xpu":
        return False
    window = dsts[0].numel()
    if window not in (4, 8):
        return False
    for n, dst, src in zip((window, window, window, 1, 1), dsts, srcs):
        if not (
            dst.dtype == src.dtype == torch.int64
            and dst.device == src.device == device
            and dst.is_contiguous()
            and src.is_contiguous()
            and dst.ndim == src.ndim == 1
            and dst.shape == src.shape
            and dst.numel() == n
        ):
            return False
    if torch.xpu.get_device_properties(device).device_id != 0xE223:
        return False
    dst_ranges = [(t.data_ptr(), t.data_ptr() + t.numel() * 8) for t in dsts]
    src_ranges = [(t.data_ptr(), t.data_ptr() + t.numel() * 8) for t in srcs]
    for i, a in enumerate(dst_ranges):
        for j, b in enumerate(dst_ranges):
            if i < j and max(a[0], b[0]) < min(a[1], b[1]):
                return False
        for j, b in enumerate(src_ranges):
            if max(a[0], b[0]) < min(a[1], b[1]) and not (i == j and a == b):
                return False
    return True


def submit(dsts, srcs):
    library = load_small_copy()
    window = dsts[0].numel()
    function = (
        library.dflash_small_copy
        if window == 4
        else getattr(library, "dflash_small_copy_m8", None)
    )
    if function is None:
        raise RuntimeError(
            "SYCL small-copy library lacks M8 support; rebuild and recapture"
        )
    stream = torch.xpu.current_stream()
    for tensor in (*dsts, *srcs):
        tensor.record_stream(stream)
    rc = function(stream.sycl_queue, *[t.data_ptr() for t in (*dsts, *srcs)])
    if rc:
        raise RuntimeError(library.dflash_small_copy_error().decode())
    _announce(window)


def try_small_copy(dsts, srcs):
    if (
        not envs.SGLANG_XPU_PAGED_DFLASH.get()
        or not envs.SGLANG_XPU_DFLASH_SMALL_COPY_SYCL.get()
        or torch.compiler.is_compiling()
        or not eligible(dsts, srcs)
    ):
        return False
    submit(dsts, srcs)
    return True
