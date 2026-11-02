# SPDX-License-Identifier: Apache-2.0
"""Bounded accepted-prefix side-state copies, without a host mask read."""

import ctypes
import logging
import math
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


def _entry_contiguous(tensor, first):
    stride = 1
    for size, actual in reversed(
        list(zip(tensor.shape[first:], tensor.stride()[first:]))
    ):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True


def _range(tensor):
    end = sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1
    return tensor.data_ptr(), tensor.data_ptr() + end * tensor.element_size()


def eligible(dst, src, destination, step, source_indices):
    if not (
        source_indices is None
        and dst.device.type == "xpu"
        and src.device == dst.device
        and dst.dtype == src.dtype
        and dst.dtype in (torch.bfloat16, torch.float32, torch.int64)
        and dst.ndim >= 3
        and src.ndim == dst.ndim + 1
        and 1 <= dst.shape[0] == src.shape[0] <= 48
        and dst.shape[1] >= 1
        and src.shape[1] >= destination.numel()
        and 1 <= src.shape[2] <= 4
        and dst.shape[2:] == src.shape[3:]
        and _entry_contiguous(dst, 2)
        and _entry_contiguous(src, 3)
        and destination.ndim == 1
        and 1 <= destination.numel() <= 8
        and all(
            t.device == dst.device
            and t.shape == destination.shape
            and t.is_contiguous()
            and t.dtype in (torch.int32, torch.int64)
            for t in (destination, step)
        )
        and all(s >= 0 for t in (dst, src) for s in t.stride())
    ):
        return False
    row_bytes = math.prod(dst.shape[2:]) * dst.element_size()
    if not (
        16 <= row_bytes <= 16 * 1024 * 1024
        and row_bytes % 16 == 0
        and dst.stride(1) * dst.element_size() >= row_bytes
        and src.stride(2) * src.element_size() >= row_bytes
        and (
            destination.numel() == 1
            or (
                src.stride(1) * src.element_size() >= row_bytes
                and src.stride(1) * src.element_size() % 16 == 0
            )
        )
        and all(t.data_ptr() % 16 == 0 for t in (dst, src))
        and all(
            s * t.element_size() % 16 == 0
            for t, dims in ((dst, (0, 1)), (src, (0, 2)))
            for s in (t.stride(i) for i in dims)
        )
    ):
        return False
    a, b = _range(dst), _range(src)
    return (
        max(a[0], b[0]) >= min(a[1], b[1])
        and torch.xpu.get_device_properties(dst.device).device_id == 0xE223
    )


@lru_cache(maxsize=1)
def _library():
    path = os.environ.get("SGLANG_XPU_GGUF_STATE_COPY_LIB")
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Set and warm SGLANG_XPU_GGUF_STATE_COPY_LIB before capture")
    lib = ctypes.CDLL(path)
    lib.xpu_gguf_state_copy.argtypes = (
        [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_int64] * 4
    )
    lib.xpu_gguf_state_copy.restype = ctypes.c_int
    lib.xpu_gguf_state_copy_error.restype = ctypes.c_char_p
    if hasattr(lib, "xpu_gguf_state_copy_rows"):
        lib.xpu_gguf_state_copy_rows.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8 + [ctypes.c_int64] * 5
        )
        lib.xpu_gguf_state_copy_rows.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def _announce():
    logger.info(
        "GGUF SYCL PLE state commit enabled; device mask, exact bytes, no CPU nonzero"
    )


def try_scatter(dst, src, destination, step, source_indices=None):
    if os.environ.get("SGLANG_XPU_GGUF_STATE_COPY") != "1" or not eligible(
        dst, src, destination, step, source_indices
    ):
        return False
    lib = _library()
    batch = destination.numel()
    if batch > 1 and not hasattr(lib, "xpu_gguf_state_copy_rows"):
        return False
    stream = torch.xpu.current_stream(dst.device)
    item = dst.element_size()
    arguments = (
        stream.sycl_queue,
        dst.data_ptr(),
        src.data_ptr(),
        destination.data_ptr(),
        step.data_ptr(),
    )
    if batch > 1:
        rc = lib.xpu_gguf_state_copy_rows(
            *arguments,
            dst.shape[0],
            dst.shape[1],
            src.shape[1],
            src.shape[2],
            batch,
            math.prod(dst.shape[2:]) * item,
            int(destination.dtype == torch.int64),
            int(step.dtype == torch.int64),
            dst.stride(0) * item,
            dst.stride(1) * item,
            src.stride(0) * item,
            src.stride(1) * item,
            src.stride(2) * item,
        )
    else:
        rc = lib.xpu_gguf_state_copy(
            *arguments,
            dst.shape[0],
            dst.shape[1],
            src.shape[2],
            math.prod(dst.shape[2:]) * item,
            int(destination.dtype == torch.int64),
            int(step.dtype == torch.int64),
            dst.stride(0) * item,
            dst.stride(1) * item,
            src.stride(0) * item,
            src.stride(2) * item,
        )
    if rc:
        raise RuntimeError((rc, lib.xpu_gguf_state_copy_error().decode()))
    for tensor in (dst, src, destination, step):
        tensor.record_stream(stream)
    _announce()
    return True
