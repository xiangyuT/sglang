# SPDX-License-Identifier: Apache-2.0
"""Reconstruct exactly the existing BF16 matrix without an FP32 temporary."""
import ctypes
from functools import lru_cache
import logging
import os

import torch

from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

logger = logging.getLogger(__name__)
SUPPORTED = frozenset((8, 14, 20, 21, 23, 42))


@lru_cache(None)
def _library(path):
    library = ctypes.CDLL(path)
    library.xpu_gguf_reconstruct_bf16.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
    library.xpu_gguf_reconstruct_bf16.restype = ctypes.c_int
    library.xpu_gguf_reconstruct_bf16_error.restype = ctypes.c_char_p
    logger.info("GGUF fused BF16 dense reconstruction: original decoder arithmetic and one BF16 rounding; Q8 SoA16 decoded directly")
    return library


def reconstruct_bf16(packed, kind, width, *, q8_soa=False):
    geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
    if not (geometry and kind in SUPPORTED and width in (640, 2560, 6144)
            and width % geometry[0] == 0 and packed.device.type == "xpu"
            and packed.dtype == torch.uint8 and packed.ndim == 2
            and packed.is_contiguous() and packed.data_ptr() % 2 == 0
            and 1 <= packed.shape[0] <= 248320
            and packed.shape[1] == width // geometry[0] * geometry[1]
            and type(q8_soa) is bool
            and (not q8_soa or (kind == 8 and packed.shape[0] % 16 == 0))
            and torch.xpu.get_device_properties(packed.device).device_id == 0xE223):
        raise ValueError("Unsupported B70 dense BF16 reconstruction geometry")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Fused dense BF16 reconstruction requires eager execution")
    output = torch.empty((packed.shape[0], width), dtype=torch.bfloat16, device=packed.device)
    stream = torch.xpu.current_stream(packed.device)
    library = _library(os.environ["SGLANG_XPU_GGUF_RECONSTRUCT_BF16_LIB"])
    status = library.xpu_gguf_reconstruct_bf16(
        stream.sycl_queue, packed.data_ptr(), output.data_ptr(),
        kind, width, packed.shape[0], int(q8_soa),
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_reconstruct_bf16_error().decode()))
    for value in (packed, output):
        value.record_stream(stream)
    return output
