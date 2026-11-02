# SPDX-License-Identifier: Apache-2.0
"""Opt-in HC mix/injection for the measured faster 8..24-row geometries."""

import ctypes
import os
from functools import lru_cache

import torch


def eligible(x):
    return (
        os.environ.get("SGLANG_XPU_GGUF_HC_WIDE_MIX") == "1"
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 8 <= x.shape[0] <= 24
        and x.shape[0] % 4 == 0
        and x.shape[1] == 10240
        and x.is_contiguous()
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


@lru_cache(maxsize=1)
def library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SGLANG_XPU_GGUF_HC_WIDE_LIB before graph capture")
    lib = ctypes.CDLL(path)
    lib.xpu_hc_wide_down_low.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int]
    lib.xpu_hc_wide_up_mix.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int]
    for name in ("xpu_hc_wide_down_low", "xpu_hc_wide_up_mix"):
        getattr(lib, name).restype = ctypes.c_int
        getattr(lib, name + "_error").restype = ctypes.c_char_p
    return lib


def warm_from_env():
    if os.environ.get("SGLANG_XPU_GGUF_HC_WIDE_MIX") == "1":
        library(os.environ.get("SGLANG_XPU_GGUF_HC_WIDE_LIB"))


def mix(normed, down_gate, up):
    if not (
        eligible(normed)
        and down_gate.shape == (324, 10240)
        and up.shape == (10240, 320)
        and down_gate.device == up.device == normed.device
        and down_gate.dtype == up.dtype == torch.bfloat16
        and down_gate.is_contiguous()
        and up.is_contiguous()
    ):
        raise ValueError("Unsupported native HC wide mix geometry")
    lib = library(os.environ.get("SGLANG_XPU_GGUF_HC_WIDE_LIB"))
    count = normed.shape[0]
    scores = torch.empty((count, 324), dtype=torch.bfloat16, device=normed.device)
    low = torch.empty((count, 320), dtype=torch.bfloat16, device=normed.device)
    partial = torch.empty((10, count, 324), dtype=torch.float32, device=normed.device)
    output = torch.empty((count, 2560), dtype=torch.bfloat16, device=normed.device)
    stream = torch.xpu.current_stream(normed.device)
    status = lib.xpu_hc_wide_down_low(
        stream.sycl_queue, normed.data_ptr(), down_gate.data_ptr(),
        scores.data_ptr(), partial.data_ptr(), low.data_ptr(), count,
    )
    if status:
        raise RuntimeError(lib.xpu_hc_wide_down_low_error().decode())
    status = lib.xpu_hc_wide_up_mix(
        stream.sycl_queue, low.data_ptr(), up.data_ptr(), normed.data_ptr(),
        output.data_ptr(), count,
    )
    if status:
        raise RuntimeError(lib.xpu_hc_wide_up_mix_error().decode())
    for value in (normed, down_gate, up, scores, low, partial, output):
        value.record_stream(stream)
    return output, scores[:, 320:324]
