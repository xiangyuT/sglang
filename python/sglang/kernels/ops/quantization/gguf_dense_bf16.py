# SPDX-License-Identifier: Apache-2.0
"""Bounded compressed projections with one final BF16 store per output row."""

import os

import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library as _iq4_library
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES


def supports(x, parts, packed_parts, q8_soa_parts=None):
    if q8_soa_parts is None:
        q8_soa_parts = (False,) * len(parts)
    return (
        os.environ.get("SGLANG_XPU_GGUF_DENSE_BF16") == "1"
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= 8
        and (x.shape[0] <= 4 or (
            os.environ.get("SGLANG_XPU_GGUF_DENSE_Q8_WAVE_ROWS") == "8"
            and bool(parts)
            and all(kind == 8 and q8_soa and not packed
                    for (kind, _), q8_soa, packed in zip(parts, q8_soa_parts, packed_parts))
        ))
        and x.shape[1] in (640, 2560, 6144)
        and len(parts) == len(packed_parts)
        and len(parts) == len(q8_soa_parts)
        and all(type(v) is bool for v in q8_soa_parts)
        and bool(parts)
        and all(
            (kind in (8, 14, 20, 23, 42) or (kind == 21 and packed))
            and weight.dtype == torch.uint8
            and weight.device == x.device
            and weight.ndim == 2
            and weight.is_contiguous()
            and weight.shape[0] >= 1
            and x.shape[1] % BLOCK_GEOMETRY[GGML_TYPES[kind]][0] == 0
            and weight.shape[1] == x.shape[1] // BLOCK_GEOMETRY[GGML_TYPES[kind]][0] * BLOCK_GEOMETRY[GGML_TYPES[kind]][1]
            and (kind not in (21, 23) or x.shape[1] in (2560, 6144))
            and (kind not in (8, 14, 20, 21) or weight.shape[0] <= 16384)
            and (kind != 21 or (weight.shape[0] >= 16 and weight.shape[0] % 16 == 0))
            and (not q8_soa or (kind == 8 and not packed and weight.shape[0] % 16 == 0))
            for (kind, weight), packed, q8_soa in zip(parts, packed_parts, q8_soa_parts)
        )
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


def quantize_batch(x, scratch, *, input_permutation=None):
    """Encode independent token rows once, before bounded Q8 dot waves."""
    if not (x.device.type == "xpu" and x.dtype == torch.bfloat16
            and x.ndim == 2 and 5 <= x.shape[0] <= 32
            and x.shape[1] in (640, 2560, 6144)
            and input_permutation in (None, "gdn_16x3x128")
            and (input_permutation is None or x.shape[1] == 6144)
            and scratch.device == x.device and scratch.dtype == torch.uint8
            and scratch.is_contiguous()
            and scratch.numel() >= x.shape[0] * (x.shape[1] // 32) * 36
            and not torch._C._is_alias_of(x, scratch)):
        raise ValueError("Invalid batch Q8 activation geometry or scratch")
    library = _iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    if not hasattr(library, "xpu_iq4_quantize_layout_wide"):
        raise RuntimeError("Rebuild the IQ4 library for batched Q8 activation encoding")
    activations = x.contiguous()
    stream = torch.xpu.current_stream(x.device)
    rc = library.xpu_iq4_quantize_layout_wide(
        stream.sycl_queue, activations.data_ptr(), scratch.data_ptr(),
        x.shape[1], x.shape[0], 1, int(input_permutation is not None),
    )
    if rc:
        raise RuntimeError(library.xpu_iq4_error().decode())
    activations.record_stream(stream)
    scratch.record_stream(stream)


def project(x, parts, *, packed_parts, input_permutation=None, scratch=None,
            q8_soa_parts=None, quantized_input=None):
    if q8_soa_parts is None:
        q8_soa_parts = (False,) * len(parts)
    if not supports(x, parts, packed_parts, q8_soa_parts):
        raise ValueError("Unsupported B70 direct-BF16 projection")
    if input_permutation not in (None, "gdn_16x3x128"):
        raise ValueError("Unsupported compressed activation permutation")
    tokens, width = x.shape
    if quantized_input is not None:
        if not (all(kind == 8 and soa for (kind, _), soa in zip(parts, q8_soa_parts))
                and quantized_input.device == x.device
                and quantized_input.dtype == torch.uint8
                and quantized_input.is_contiguous()
                and quantized_input.numel() == tokens * (width // 32) * 36):
            raise ValueError("Invalid pre-encoded Q8 activation wave")
        scratch = quantized_input
    columns = sum(weight.shape[0] for _, weight in parts)
    output = torch.empty((tokens, columns), dtype=torch.bfloat16, device=x.device)
    activations = x.contiguous()
    if scratch is None:
        scratch = torch.empty(tokens * (width // 32) * 36, dtype=torch.uint8, device=x.device)
    elif not (scratch.device == x.device and scratch.dtype == torch.uint8
              and scratch.is_contiguous() and scratch.numel() >= tokens * (width // 32) * 36):
        raise ValueError("Insufficient or invalid compressed projection scratch")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    quantizer = _iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    stream = torch.xpu.current_stream(x.device)
    # Keep each existing quantizer launch and its 32-value subgroup arithmetic.
    # The wider dot consumes both consecutive activation images afterward.
    for row in range(0, tokens if quantized_input is None else 0, 4):
        rc = quantizer.xpu_iq4_quantize_layout(
            stream.sycl_queue, activations.data_ptr() + row * width * 2,
            scratch.data_ptr() + row * (width // 32) * 36,
            width, min(4, tokens - row), 1, int(input_permutation is not None),
        )
        if rc:
            raise RuntimeError(quantizer.xpu_iq4_error().decode())
    offset = 0
    for (kind, weight), packed, q8_soa in zip(parts, packed_parts, q8_soa_parts):
        arguments = (
            stream.sycl_queue, weight.data_ptr(), scratch.data_ptr(),
            output.data_ptr() + 2 * offset,
        )
        if q8_soa:
            if not hasattr(library, "xpu_gguf_dense_q8_soa"):
                raise RuntimeError("Rebuild the GGUF library for Q8 SoA16 projections")
            name = "xpu_gguf_dense_q8_soa_wide" if tokens > 4 else "xpu_gguf_dense_q8_soa"
            if not hasattr(library, name):
                raise RuntimeError("Rebuild the GGUF library for eight-row Q8 projections")
            function = getattr(library, name)
            rc = function(*arguments, width, weight.shape[0], tokens, columns, 1)
        elif kind == 23:
            if os.environ.get("SGLANG_XPU_GGUF_IQ4_WG8") == "1":
                function = library.xpu_iq4_s4_wg_bf16
                rc = function(*arguments, width, weight.shape[0], tokens, weight.shape[1], columns, 8)
            else:
                function = library.xpu_iq4_s4_bf16
                rc = function(*arguments, width, weight.shape[0], tokens, weight.shape[1], columns, 1)
        elif kind == 21 and packed:
            function = library.xpu_iq3_packed_bf16
            rc = function(*arguments, width, weight.shape[0], tokens, weight.shape[1], columns)
        elif kind == 42:
            function = library.xpu_gguf_dense_q8_bf16
            rc = function(*arguments, kind, width, weight.shape[0], tokens, weight.shape[1], columns)
        else:
            function = library.xpu_dense_dp4a_bf16
            rc = function(*arguments, kind, width, weight.shape[0], tokens, weight.shape[1], columns)
        if rc:
            raise RuntimeError((kind, rc))
        offset += weight.shape[0]
        weight.record_stream(stream)
    for tensor in (activations, scratch, output):
        tensor.record_stream(stream)
    return output
