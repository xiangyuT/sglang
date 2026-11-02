# SPDX-License-Identifier: Apache-2.0
"""Opt-in wider Q6_K head with unchanged per-token arithmetic.

The caller owns the verified SoA16 layout marker; incomplete tiles retain the
original row layout. Quantization keeps the original four-row launches, and
raw tails keep the existing four-row DP4A kernel. No scratch is shared across
concurrent calls or graphs.
"""

import ctypes
import os
from functools import lru_cache

import torch
from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library as _iq4_library


@lru_cache(maxsize=1)
def library(path):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the wider Q6 head library before graph capture")
    lib = ctypes.CDLL(path)
    lib.xpu_gguf_dense_q6_wide.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
    lib.xpu_gguf_dense_q6_wide.restype = ctypes.c_int
    lib.xpu_gguf_dense_q6_wide_error.restype = ctypes.c_char_p
    return lib


def warm_from_env():
    if os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS") in ("8", "16"):
        path = os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_LIB")
        if not path:
            raise RuntimeError("Set SGLANG_XPU_GGUF_Q6_HEAD_LIB for wider Q6 heads")
        library(path)


def try_project(x, weight):
    wave = os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS")
    if not (
        wave in ("8", "16")
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 5 <= x.shape[0] <= 64
        and x.shape[1] == 2560
        and weight.device == x.device
        and weight.dtype == torch.uint8
        and weight.ndim == 2
        and 16 <= weight.shape[0] <= 248320
        and weight.shape[1] == 2100
        and weight.is_contiguous()
        and weight.data_ptr() % 64 == 0
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    ):
        return None
    wave_rows = int(wave)
    path = os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_LIB")
    if not path:
        raise RuntimeError("Set SGLANG_XPU_GGUF_Q6_HEAD_LIB for wider Q6 heads")
    wide = library(path)
    base = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    quantizer = _iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    activation = x.contiguous()
    tokens, width = activation.shape
    rows = weight.shape[0]
    block_bytes = (width // 32) * 36
    scratch = torch.empty(tokens * block_bytes, dtype=torch.uint8, device=x.device)
    output = torch.empty((tokens, rows), dtype=torch.float32, device=x.device)
    stream = torch.xpu.current_stream(x.device)
    for first in range(0, tokens, 4):
        rc = quantizer.xpu_iq4_quantize_layout(
            stream.sycl_queue,
            activation.data_ptr() + first * width * 2,
            scratch.data_ptr() + first * block_bytes,
            width,
            min(4, tokens - first),
            1,
            0,
        )
        if rc:
            raise RuntimeError(quantizer.xpu_iq4_error().decode())
    full = rows // 16 * 16
    for first in range(0, tokens, wave_rows):
        count = min(wave_rows, tokens - first)
        function = (
            wide.xpu_gguf_dense_q6_wide if count > 4 else base.xpu_gguf_dense_q6_soa
        )
        rc = function(
            stream.sycl_queue,
            weight.data_ptr(),
            scratch.data_ptr() + first * block_bytes,
            output.data_ptr() + first * rows * 4,
            14,
            width,
            full,
            count,
            2100,
            rows,
        )
        if rc:
            raise RuntimeError(f"Q6 head submission failed: {rc}")
        if full != rows:
            for tail_first in range(0, count, 4):
                row = first + tail_first
                rc = base.xpu_dense_dp4a(
                    stream.sycl_queue,
                    weight.data_ptr() + full * 2100,
                    scratch.data_ptr() + row * block_bytes,
                    output.data_ptr() + (row * rows + full) * 4,
                    14,
                    width,
                    rows - full,
                    min(4, count - tail_first),
                    2100,
                    rows,
                )
                if rc:
                    raise RuntimeError(base.xpu_dense_dp4a_error().decode())
    for value in (activation, weight, scratch, output):
        value.record_stream(stream)
    return output
