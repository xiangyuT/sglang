# SPDX-License-Identifier: Apache-2.0
"""Original mixed GGUF projections sharing one Q8_1 activation image."""

import os

import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library as _iq4_library
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

DENSE_TYPES = frozenset((8, 14, 20, 21, 23, 42))


def permute_q6_fields(packed, *, inverse=False):
    """Losslessly interleave complete Q6_K 16-row tiles; retain the raw tail."""
    if not (
        packed.device.type == "xpu" and packed.dtype == torch.uint8
        and packed.ndim == 2 and 16 <= packed.shape[0] <= 248320
        and packed.shape[1] == 2100 and packed.is_contiguous()
        and packed.data_ptr() % 64 == 0
        and torch.xpu.get_device_properties(packed.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 Q6 field permutation")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_q6_field_permute"):
        raise RuntimeError("Rebuild the GGUF library for Q6 SoA16 projections")
    output = torch.empty_like(packed)
    stream = torch.xpu.current_stream(packed.device)
    status = library.xpu_gguf_q6_field_permute(
        stream.sycl_queue, packed.data_ptr(), output.data_ptr(), packed.shape[0], int(inverse),
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_dense_q6_soa_error().decode()))
    for value in (packed, output):
        value.record_stream(stream)
    return output


def permute_q8_fields(packed, width, *, inverse=False):
    """Use the existing MTP SoA16 byte layout without a CPU prefill transfer."""
    if not (
        packed.device.type == "xpu" and packed.dtype == torch.uint8
        and packed.ndim == 2 and width in (640, 2560, 6144)
        and 16 <= packed.shape[0] <= 248320 and packed.shape[0] % 16 == 0
        and packed.shape[1] == width // 32 * 34 and packed.is_contiguous()
        and packed.data_ptr() % 64 == 0
        and torch.xpu.get_device_properties(packed.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 Q8 field permutation")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_q8_field_permute"):
        raise RuntimeError("Rebuild the GGUF library for Q8 SoA16 projections")
    output = torch.empty_like(packed)
    stream = torch.xpu.current_stream(packed.device)
    status = library.xpu_gguf_q8_field_permute(
        stream.sycl_queue, packed.data_ptr(), output.data_ptr(),
        width, packed.shape[0], int(inverse),
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_dense_q8_soa_error().decode()))
    for value in (packed, output):
        value.record_stream(stream)
    return output


def permute_iq3_fields(packed, width, *, inverse=False):
    """Return a lossless 16-row field permutation, with unchanged byte count."""
    if not (
        packed.device.type == "xpu"
        and packed.dtype == torch.uint8
        and packed.ndim == 2
        and width in (2560, 6144)
        and 16 <= packed.shape[0] <= 16384
        and packed.shape[0] % 16 == 0
        and packed.shape[1] == width // 256 * 110
        and packed.is_contiguous()
        and packed.data_ptr() % 64 == 0
        and torch.xpu.get_device_properties(packed.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 IQ3 field permutation")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_iq3_field_permute"):
        raise RuntimeError("Rebuild the GGUF library for packed IQ3 projections")
    output = torch.empty_like(packed)
    stream = torch.xpu.current_stream(packed.device)
    status = library.xpu_iq3_field_permute(
        stream.sycl_queue, packed.data_ptr(), output.data_ptr(),
        width, packed.shape[0], int(inverse),
    )
    if status:
        raise RuntimeError((status, library.xpu_iq3_packed_error().decode()))
    for value in (packed, output):
        value.record_stream(stream)
    return output


def dense_q8_projection(
    x, parts, *, input_permutation=None, scratch=None, iq3_packed_parts=None,
    q8_soa_parts=None, q6_soa_parts=None,
):
    """Return fresh FP32 rows; each part is ``(GGML type, packed row tensor)``.

    Parts default to original GGUF rows. Explicit ``iq3_packed_parts`` metadata
    selects the lossless IQ3 field layout independently of environment flags.
    This uses Q8_1 activations rather than expanded-BF16 GEMM arithmetic. An
    optional scratch image may be reused on the same ordered stream.
    """
    if not (
        x.device.type == "xpu"
        and x.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.shape[1] in (640, 2560, 6144)
        and x.shape[0] <= 4
        and parts
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
        and (
            input_permutation is None
            or (input_permutation == "gdn_16x3x128" and x.shape[1] == 6144)
        )
    ):
        raise ValueError("Unsupported B70 compressed dense geometry")
    if iq3_packed_parts is None:
        iq3_packed_parts = (False,) * len(parts)
    if q8_soa_parts is None:
        q8_soa_parts = (False,) * len(parts)
    if q6_soa_parts is None:
        q6_soa_parts = (False,) * len(parts)
    if len(q6_soa_parts) != len(parts) or any(type(v) is not bool for v in q6_soa_parts):
        raise ValueError("Invalid Q6 SoA16 layout metadata")
    if len(q8_soa_parts) != len(parts) or any(type(v) is not bool for v in q8_soa_parts):
        raise ValueError("Invalid Q8 SoA16 layout metadata")
    if len(iq3_packed_parts) != len(parts) or any(
        type(value) is not bool for value in iq3_packed_parts
    ):
        raise ValueError("Invalid IQ3 layout metadata")
    width = x.shape[1]
    for (kind, packed), field_packed, q8_soa, q6_soa in zip(
        parts, iq3_packed_parts, q8_soa_parts, q6_soa_parts
    ):
        if kind not in DENSE_TYPES:
            raise ValueError("Unsupported original dense GGUF format")
        qk, size = BLOCK_GEOMETRY[GGML_TYPES[kind]]
        if not (
            width % qk == 0
            and packed.device == x.device
            and packed.dtype == torch.uint8
            and packed.ndim == 2
            and packed.shape[0] > 0
            and packed.shape[1] == width // qk * size
            and packed.is_contiguous()
            and packed.data_ptr() % 64 == 0
        ):
            raise ValueError("Invalid original compressed dense rows")
        if field_packed and not (
            kind == 21 and width in (2560, 6144) and packed.shape[0] % 16 == 0
        ):
            raise ValueError("Packed IQ3 layout requires complete 16-row tiles")
        if q6_soa and not (kind == 14 and width == 2560 and packed.shape[0] >= 16
                              and not field_packed and not q8_soa):
            raise ValueError("Packed Q6 layout requires Q6_K width2560 rows")
        if q8_soa and not (kind == 8 and not field_packed and packed.shape[0] % 16 == 0):
            raise ValueError("Packed Q8 layout requires original Q8_0 whole 16-row tiles")
    rows = sum(packed.shape[0] for _, packed in parts)
    output = torch.empty((x.shape[0], rows), dtype=torch.float32, device=x.device)
    if not x.shape[0]:
        return output
    iq4 = _iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not (
        hasattr(iq4, "xpu_iq4_mmvq_output_stride")
        and hasattr(library, "xpu_gguf_dense_q8")
    ):
        raise RuntimeError("Rebuild the GGUF and IQ4 libraries for mixed projections")
    activation = x.contiguous()
    required = x.shape[0] * (width // 32) * 36
    if scratch is None:
        scratch = torch.empty(required, dtype=torch.uint8, device=x.device)
    elif not (
        scratch.device == x.device
        and scratch.dtype == torch.uint8
        and scratch.is_contiguous()
        and scratch.numel() >= required
        and scratch.data_ptr() % 64 == 0
    ):
        raise ValueError("Invalid Q8_1 dense scratch image")
    stream = torch.xpu.current_stream(x.device)
    status = iq4.xpu_iq4_quantize_layout(
        stream.sycl_queue,
        activation.data_ptr(),
        scratch.data_ptr(),
        width,
        x.shape[0],
        {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[x.dtype],
        int(input_permutation is not None),
    )
    if status:
        raise RuntimeError((status, iq4.xpu_iq4_error().decode()))
    offset = 0
    for (kind, packed), field_packed, q8_soa, q6_soa in zip(
        parts, iq3_packed_parts, q8_soa_parts, q6_soa_parts
    ):
        arguments = (
            stream.sycl_queue,
            packed.data_ptr(),
            scratch.data_ptr(),
            output.data_ptr() + offset * output.element_size(),
        )
        if q6_soa:
            if not hasattr(library, "xpu_gguf_dense_q6_soa"):
                raise RuntimeError("Rebuild the GGUF library for Q6 SoA16 projections")
            full = packed.shape[0] // 16 * 16
            status = library.xpu_gguf_dense_q6_soa(
                *arguments, kind, width, full, x.shape[0], packed.shape[1], rows,
            )
            error = library.xpu_gguf_dense_q6_soa_error
            if status:
                raise RuntimeError((status, error().decode()))
            tail = packed.shape[0] - full
            if tail:
                status = library.xpu_dense_dp4a(
                    stream.sycl_queue, packed.data_ptr() + full * packed.shape[1],
                    scratch.data_ptr(), output.data_ptr() + (offset + full) * 4,
                    kind, width, tail, x.shape[0], packed.shape[1], rows,
                )
                error = library.xpu_dense_dp4a_error
        elif q8_soa:
            if not hasattr(library, "xpu_gguf_dense_q8_soa"):
                raise RuntimeError("Rebuild the GGUF library for Q8 SoA16 projections")
            status = library.xpu_gguf_dense_q8_soa(
                *arguments, width, packed.shape[0], x.shape[0], rows, 0,
            )
            error = library.xpu_gguf_dense_q8_soa_error
        elif kind == 23:
            if os.environ.get("SGLANG_XPU_GGUF_IQ4_RAW_S4") == "1":
                if not hasattr(library, "xpu_iq4_s4"):
                    raise RuntimeError("Rebuild the GGUF library for IQ4 S4 projections")
                status = library.xpu_iq4_s4(
                    *arguments, width, packed.shape[0], x.shape[0],
                    packed.shape[1], rows, 1,
                )
                error = library.xpu_iq4_s4_error
            else:
                status = iq4.xpu_iq4_mmvq_output_stride(
                    *arguments, width, packed.shape[0], x.shape[0], rows
                )
                error = iq4.xpu_iq4_error
        elif field_packed:
            if not hasattr(library, "xpu_iq3_packed"):
                raise RuntimeError("Rebuild the GGUF library for packed IQ3 projections")
            status = library.xpu_iq3_packed(
                *arguments, width, packed.shape[0], x.shape[0], packed.shape[1], rows
            )
            error = library.xpu_iq3_packed_error
        elif kind == 21 and os.environ.get("SGLANG_XPU_GGUF_IQ3_XMX") == "1":
            if not hasattr(library, "xpu_iq3_xmx"):
                raise RuntimeError("Rebuild the GGUF library for IQ3 XMX projections")
            status = library.xpu_iq3_xmx(
                *arguments, width, packed.shape[0], x.shape[0], packed.shape[1], rows
            )
            error = library.xpu_iq3_xmx_error
        elif kind in (8, 14, 20) and os.environ.get("SGLANG_XPU_GGUF_DENSE_DP4A") == "1":
            if not hasattr(library, "xpu_dense_dp4a"):
                raise RuntimeError("Rebuild the GGUF library for dense DP4A projections")
            status = library.xpu_dense_dp4a(
                *arguments, kind, width, packed.shape[0], x.shape[0],
                packed.shape[1], rows,
            )
            error = library.xpu_dense_dp4a_error
        else:
            status = library.xpu_gguf_dense_q8(
                *arguments,
                kind,
                width,
                packed.shape[0],
                x.shape[0],
                packed.shape[1],
                rows,
            )
            error = library.xpu_gguf_dense_q8_error
        if status:
            raise RuntimeError((status, error().decode()))
        offset += packed.shape[0]
        packed.record_stream(stream)
    for tensor in (activation, scratch, output):
        tensor.record_stream(stream)
    return output
