# SPDX-License-Identifier: Apache-2.0
"""Fuse route scatter, weighting and the Omix Torch FP32 sum (vt0=4)."""
import ctypes
from functools import lru_cache
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


@lru_cache(None)
def _library(path):
    library = ctypes.CDLL(path)
    library.xpu_gguf_route_reduce.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
    library.xpu_gguf_route_reduce.restype = ctypes.c_int
    library.xpu_gguf_route_reduce_error.restype = ctypes.c_char_p
    logger.info("GGUF fused prefill route reduction: scatter+FP32 weights+Torch vt0=4 sum; BF16/FP32 output")
    return library


def route_metadata(order, route_count, device):
    """Upload the unique sorted-to-route map and its inverse in one transfer."""
    order = np.asarray(order, dtype=np.int32)
    if (order.ndim != 1 or route_count < 1 or len(order) > route_count
            or (len(order) and (order.min() < 0 or order.max() >= route_count))
            or len(np.unique(order)) != len(order)):
        raise ValueError("Prefill route order must contain unique in-range indices")
    inverse = np.full(route_count, -1, dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    both = torch.as_tensor(np.concatenate((order, inverse)), device=device)
    return both[:len(order)], both[len(order):]


def reduce_routes(sorted_output, inverse, weights, dtype):
    if not (sorted_output.device.type == "xpu" and sorted_output.dtype == torch.float32
            and sorted_output.ndim == 2 and sorted_output.shape[1] == 2560
            and sorted_output.is_contiguous() and sorted_output.shape[0] > 0
            and weights.ndim == 2 and 5 <= weights.shape[0] <= 4096
            and 1 <= weights.shape[1] <= 10
            and sorted_output.shape[0] <= weights.numel()
            and weights.device == sorted_output.device
            and inverse.device == sorted_output.device and inverse.dtype == torch.int32
            and inverse.shape == (weights.numel(),) and inverse.is_contiguous()
            and dtype in (torch.bfloat16, torch.float32)
            and torch.xpu.get_device_properties(sorted_output.device).device_id == 0xE223):
        raise ValueError("Unsupported B70 routed-prefill reduction geometry")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Fused prefill route reduction requires eager execution")
    weights = weights.float().contiguous()
    output = torch.empty((weights.shape[0], 2560), dtype=dtype, device=weights.device)
    stream = torch.xpu.current_stream(weights.device)
    library = _library(os.environ["SGLANG_XPU_GGUF_PREFILL_REDUCE_LIB"])
    status = library.xpu_gguf_route_reduce(
        stream.sycl_queue, sorted_output.data_ptr(), inverse.data_ptr(),
        weights.data_ptr(), output.data_ptr(), weights.shape[0], weights.shape[1],
        2560, int(dtype == torch.float32), sorted_output.shape[0],
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_route_reduce_error().decode()))
    for value in (sorted_output, inverse, weights, output):
        value.record_stream(stream)
    return output
