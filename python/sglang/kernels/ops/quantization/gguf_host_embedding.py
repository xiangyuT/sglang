# SPDX-License-Identifier: Apache-2.0
"""Keep original Q8 token rows in pinned host storage; gather selected rows.

The model owns the pinned weight for the lifetime of all graph replays, just
as it owns device parameters. No host readback of token IDs is performed.
"""
import ctypes
from functools import lru_cache
import logging
import os

import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library as _gguf_library

logger = logging.getLogger(__name__)


@lru_cache(None)
def _library(path):
    library = ctypes.CDLL(path)
    library.xpu_gguf_host_embedding.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    library.xpu_gguf_host_embedding.restype = ctypes.c_int
    library.xpu_gguf_host_embedding_error.restype = ctypes.c_char_p
    return library


def _check_host_pointer(path, queue, pointer):
    library = _gguf_library(path)
    if library.xpu_gguf_gpu_cache_pointer_kind(queue, pointer) != 1:
        raise ValueError("Token embedding must be USM host storage in this XPU context")


def prepare_host_embedding(layer):
    weight = layer.weight
    if not (weight.device.type == "xpu" and weight.dtype == torch.uint8
            and weight.ndim == 2 and weight.shape[1] == 2720
            and weight.is_contiguous()
            and torch.xpu.get_device_properties(weight.device).device_id == 0xE223):
        raise ValueError("Pinned-host token embedding requires original B70 Q8 rows of width 2560")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Move token embedding before graph capture")
    host = torch.empty(weight.shape, dtype=weight.dtype, device="cpu", pin_memory=True)
    host.copy_(weight.detach())
    queue = torch.xpu.current_stream(weight.device).sycl_queue
    _check_host_pointer(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"), queue, host.data_ptr())
    replacement = torch.nn.Parameter(host, requires_grad=False)
    replacement.__dict__.update(weight.__dict__)
    layer.weight = replacement
    logger.info("GGUF Q8 token embedding on pinned host: %d bytes, original rows; GPU gather preserves graph replay", host.numel())


def gather_host_embedding(weight, ids):
    if not (weight.device.type == "cpu" and weight.dtype == torch.uint8
            and weight.ndim == 2 and 1 <= weight.shape[0] <= 262144
            and weight.shape[1] == 2720 and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0 and ids.device.type == "xpu"
            and ids.dtype in (torch.int32, torch.int64) and ids.is_contiguous()
            and ids.numel() <= 4096
            and torch.xpu.get_device_properties(ids.device).device_id == 0xE223):
        raise ValueError("Unsupported B70 pinned-host embedding geometry")
    output = torch.empty((ids.numel(), 2720), dtype=torch.uint8, device=ids.device)
    if ids.numel() == 0:
        return output
    stream = torch.xpu.current_stream(ids.device)
    _check_host_pointer(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"), stream.sycl_queue, weight.data_ptr())
    ids = ids.reshape(-1).long()
    library = _library(os.environ["SGLANG_XPU_GGUF_HOST_EMBEDDING_LIB"])
    status = library.xpu_gguf_host_embedding(
        stream.sycl_queue, weight.data_ptr(), ids.data_ptr(), output.data_ptr(),
        ids.numel(), weight.shape[0], 2720,
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_host_embedding_error().decode()))
    for value in (ids, output):
        value.record_stream(stream)
    return output
