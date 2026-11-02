# SPDX-License-Identifier: Apache-2.0
"""Strata's compressed GGUF decoders on the current Torch XPU stream."""

import ctypes
import os
from functools import lru_cache

import torch

from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

SUPPORTED_TYPES = frozenset((8, 14, 16, 20, 21, 22, 23, 29, 42))


@lru_cache(maxsize=1)
def _library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SGLANG_XPU_GGUF_SYCL_LIB before graph capture")
    library = ctypes.CDLL(path)
    library.xpu_gguf_dequant.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    library.xpu_gguf_dequant.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_mixed_max_rows"):
        library.xpu_gguf_mixed_max_rows.argtypes = []
        library.xpu_gguf_mixed_max_rows.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_iq4_down_permute"):
        library.xpu_gguf_iq4_down_permute.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
        library.xpu_gguf_iq4_down_permute.restype = ctypes.c_int
        library.xpu_gguf_iq4_cache_dequant.argtypes = library.xpu_gguf_dequant.argtypes
        library.xpu_gguf_iq4_cache_dequant.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_iq3_gu_permute"):
        library.xpu_gguf_iq3_gu_permute.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
        library.xpu_gguf_iq3_gu_permute.restype = ctypes.c_int
        library.xpu_gguf_iq3_cache_dequant.argtypes = library.xpu_gguf_dequant.argtypes
        library.xpu_gguf_iq3_cache_dequant.restype = ctypes.c_int
    library.xpu_gguf_error.restype = ctypes.c_char_p
    for name, integers in (
        ("xpu_dense_dp4a_bf16", 6),
        ("xpu_iq4_s4_bf16", 6),
        ("xpu_iq4_s4_wg_bf16", 6),
        ("xpu_iq3_packed_bf16", 5),
        ("xpu_gguf_dense_q8_bf16", 6),
    ):
        if hasattr(library, name):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * integers
            function.restype = ctypes.c_int
            getattr(library, name + "_error").restype = ctypes.c_char_p
    if hasattr(library, "xpu_bf16_raw_2da_gemm"):
        library.xpu_bf16_raw_2da_gemm.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
        )
        library.xpu_bf16_raw_2da_gemm.restype = ctypes.c_int
        library.xpu_bf16_raw_2da_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_hc_combine_norm"):
        library.xpu_gguf_hc_combine_norm.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int]
        )
        library.xpu_gguf_hc_combine_norm.restype = ctypes.c_int
        library.xpu_gguf_hc_combine_norm_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_top10_norm"):
        library.xpu_gguf_top10_norm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
        library.xpu_gguf_top10_norm.restype = ctypes.c_int
        library.xpu_gguf_top10_norm_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_mtp_top1_probability"):
        library.xpu_mtp_top1_probability.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
        )
        library.xpu_mtp_top1_probability.restype = ctypes.c_int
        library.xpu_mtp_top1_probability_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_hc_norm"):
        library.xpu_gguf_hc_norm.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int,
            ctypes.c_float,
        ]
        library.xpu_gguf_hc_norm.restype = ctypes.c_int
        library.xpu_gguf_hc_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_gpu_cache_prepare"):
        library.xpu_gguf_gpu_cache_prepare.argtypes = [ctypes.c_void_p] * 11 + [
            ctypes.c_int
        ] * 6
        library.xpu_gguf_gpu_cache_prepare.restype = ctypes.c_int
        if hasattr(library, "xpu_gguf_gpu_cache_prepare_parallel"):
            library.xpu_gguf_gpu_cache_prepare_parallel.argtypes = [
                ctypes.c_void_p
            ] * 11 + [ctypes.c_int] * 7
            library.xpu_gguf_gpu_cache_prepare_parallel.restype = ctypes.c_int
        library.xpu_gguf_gpu_cache_error.restype = ctypes.c_char_p
        library.xpu_gguf_gpu_cache_pointer_kind.argtypes = [ctypes.c_void_p] * 2
        library.xpu_gguf_gpu_cache_pointer_kind.restype = ctypes.c_int
        if hasattr(library, "xpu_gguf_gpu_cache_stage"):
            library.xpu_gguf_gpu_cache_stage.argtypes = [ctypes.c_void_p] * 8 + [
                ctypes.c_int
            ] * 3
            library.xpu_gguf_gpu_cache_stage.restype = ctypes.c_int
            if hasattr(library, "xpu_gguf_gpu_cache_stage_lowbit"):
                library.xpu_gguf_gpu_cache_stage_lowbit.argtypes = (
                    library.xpu_gguf_gpu_cache_stage.argtypes
                )
                library.xpu_gguf_gpu_cache_stage_lowbit.restype = ctypes.c_int
            if hasattr(library, "xpu_gguf_gpu_cache_stage_encoded"):
                library.xpu_gguf_gpu_cache_stage_encoded.argtypes = (
                    library.xpu_gguf_gpu_cache_stage.argtypes
                )
                library.xpu_gguf_gpu_cache_stage_encoded.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_gpu_cache_prepare_mixed"):
        library.xpu_gguf_gpu_cache_prepare_mixed.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 8
        library.xpu_gguf_gpu_cache_prepare_mixed.restype = ctypes.c_int
        library.xpu_gguf_gpu_cache_stage_mixed.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4
        library.xpu_gguf_gpu_cache_stage_mixed.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_dense_q8"):
        library.xpu_gguf_dense_q8.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
        library.xpu_gguf_dense_q8.restype = ctypes.c_int
        library.xpu_gguf_dense_q8_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_dense_q8_soa"):
        library.xpu_gguf_dense_q8_soa.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        library.xpu_gguf_dense_q8_soa.restype = ctypes.c_int
        library.xpu_gguf_dense_q8_soa_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_dense_q8_soa_wide"):
        library.xpu_gguf_dense_q8_soa_wide.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        library.xpu_gguf_dense_q8_soa_wide.restype = ctypes.c_int
        library.xpu_gguf_q8_field_permute.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        library.xpu_gguf_q8_field_permute.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_dense_q6_soa"):
        library.xpu_gguf_dense_q6_soa.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
        library.xpu_gguf_dense_q6_soa.restype = ctypes.c_int
        library.xpu_gguf_dense_q6_soa_error.restype = ctypes.c_char_p
        library.xpu_gguf_q6_field_permute.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
        library.xpu_gguf_q6_field_permute.restype = ctypes.c_int
    if hasattr(library, "xpu_iq3_xmx"):
        library.xpu_iq3_xmx.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        library.xpu_iq3_xmx.restype = ctypes.c_int
        library.xpu_iq3_xmx_error.restype = ctypes.c_char_p
    for name in ("xpu_dense_dp4a", "xpu_iq4_s4"):
        if hasattr(library, name):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
            function.restype = ctypes.c_int
            getattr(library, name + "_error").restype = ctypes.c_char_p
    if hasattr(library, "xpu_hc_esimd"):
        library.xpu_hc_esimd.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int, ctypes.c_float, ctypes.c_int,
        ]
        library.xpu_hc_esimd.restype = ctypes.c_int
        library.xpu_hc_esimd_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_moe_input_quantize"):
        library.xpu_moe_input_quantize.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
        )
        library.xpu_moe_input_quantize.restype = ctypes.c_int
        library.xpu_moe_activate_quantize.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int]
        )
        library.xpu_moe_activate_quantize.restype = ctypes.c_int
        library.xpu_moe_quant_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_top10"):
        library.xpu_gguf_top10.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
        library.xpu_gguf_top10.restype = ctypes.c_int
        library.xpu_gguf_top10_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_iq3_packed"):
        library.xpu_iq3_packed.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        library.xpu_iq3_packed.restype = ctypes.c_int
        library.xpu_iq3_field_permute.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        )
        library.xpu_iq3_field_permute.restype = ctypes.c_int
        library.xpu_iq3_packed_error.restype = ctypes.c_char_p
    library.xpu_gguf_matvec.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
    library.xpu_gguf_matvec.restype = ctypes.c_int
    library.xpu_gguf_expert_rows.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int] * 4
        + [ctypes.c_int64]
        + [ctypes.c_int] * 3
    )
    library.xpu_gguf_expert_rows.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_expert_rows_general"):
        library.xpu_gguf_expert_rows_general.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int] * 4
            + [ctypes.c_int64]
            + [ctypes.c_int] * 4
        )
        library.xpu_gguf_expert_rows_general.restype = ctypes.c_int
        library.xpu_gguf_scatter_routes.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int
        ] * 3
        library.xpu_gguf_scatter_routes.restype = ctypes.c_int
    if hasattr(library, "xpu_expert_q8_rows"):
        library.xpu_expert_quantize.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int
        ] * 2
        library.xpu_expert_quantize.restype = ctypes.c_int
        library.xpu_expert_quantize_error.restype = ctypes.c_char_p
        library.xpu_expert_q8_rows.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int] * 4
            + [ctypes.c_int64]
            + [ctypes.c_int] * 4
        )
        library.xpu_expert_q8_rows.restype = ctypes.c_int
        library.xpu_expert_q8_error.restype = ctypes.c_char_p
        if hasattr(library, "xpu_expert_q8_float_rows"):
            library.xpu_expert_q8_float_rows.argtypes = (
                [ctypes.c_void_p] * 9
                + [ctypes.c_int] * 4
                + [ctypes.c_int64]
                + [ctypes.c_int] * 4
            )
            library.xpu_expert_q8_float_rows.restype = ctypes.c_int
    if hasattr(library, "xpu_raw_q2_down"):
        library.xpu_raw_q2_down.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
        library.xpu_raw_q2_down.restype = ctypes.c_int
        library.xpu_raw_q2_down_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_gguf_moe_q8"):
        library.xpu_gguf_moe_q8.argtypes = (
            [ctypes.c_void_p] * 15
            + [ctypes.c_int] * 6
            + [ctypes.c_int64, ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        )
        library.xpu_gguf_moe_q8.restype = ctypes.c_int
        for name in ("xpu_gguf_moe_q8_addressed", "xpu_gguf_moe_q8_addressed_prepared"):
            if not hasattr(library, name):
                continue
            function = getattr(library, name)
            function.argtypes = (
                library.xpu_gguf_moe_q8.argtypes
                + [ctypes.c_void_p] * 4
                + [ctypes.c_int] * 2
                + [ctypes.c_void_p]
            )
            function.restype = ctypes.c_int
        if hasattr(library, "xpu_gguf_moe_q8_pvr"):
            library.xpu_gguf_moe_q8_pvr.argtypes = (
                library.xpu_gguf_moe_q8.argtypes
                + [ctypes.c_void_p] * 4
                + [ctypes.c_int] * 2
            )
            library.xpu_gguf_moe_q8_pvr.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_moe_q8_mixed_addressed"):
        library.xpu_gguf_moe_q8_mixed_addressed.argtypes = (
            library.xpu_gguf_moe_q8_addressed.argtypes + [ctypes.c_int] * 2
        )
        library.xpu_gguf_moe_q8_mixed_addressed.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_iq2s_pack"):
        library.xpu_gguf_iq2s_pack.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
        library.xpu_gguf_iq2s_pack.restype = ctypes.c_int
        library.xpu_gguf_iq2s_gu.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5
        library.xpu_gguf_iq2s_gu.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_iq2s_pack_expert"):
        library.xpu_gguf_iq2s_pack_expert.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int
        ] * 3
        library.xpu_gguf_iq2s_pack_expert.restype = ctypes.c_int
        library.xpu_gguf_q2_pack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        library.xpu_gguf_q2_pack.restype = ctypes.c_int
        library.xpu_gguf_q2_down.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
        library.xpu_gguf_q2_down.restype = ctypes.c_int
    if hasattr(library, "xpu_gguf_copy_create"):
        library.xpu_gguf_copy_create.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        library.xpu_gguf_copy_create.restype = ctypes.c_void_p
        library.xpu_gguf_copy_expert.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_uint64
        ] * 3
        library.xpu_gguf_copy_expert.restype = ctypes.c_int
        library.xpu_gguf_copy_destroy.argtypes = [ctypes.c_void_p]
        library.xpu_gguf_copy_destroy.restype = ctypes.c_int
        library.xpu_gguf_copy_error.restype = ctypes.c_char_p
        if hasattr(library, "xpu_gguf_copy_group"):
            library.xpu_gguf_copy_group.argtypes = [ctypes.c_void_p] * 5 + [
                ctypes.c_int
            ] * 4
            library.xpu_gguf_copy_group.restype = ctypes.c_int
    return library


def dequantize_gguf(packed, kind, *, dtype=torch.float16, out=None):
    """Decode integer GGUF type IDs without requantizing their compressed bytes.

    Returns a flat FP16/FP32 tensor. Callers supply the logical matrix shape;
    this component does not yet install a whole-model GGUF execution backend.
    """
    geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
    if not (
        geometry
        and kind in SUPPORTED_TYPES
        and dtype in (torch.float16, torch.float32)
        and packed.device.type == "xpu"
        and packed.dtype == torch.uint8
        and packed.is_contiguous()
        and packed.data_ptr() % 2 == 0
        and packed.numel() > 0
        and packed.numel() % geometry[1] == 0
        and torch.xpu.get_device_properties(packed.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 GGUF compressed tensor or output dtype")
    count = packed.numel() // geometry[1] * geometry[0]
    if out is None:
        out = torch.empty(count, dtype=dtype, device=packed.device)
    elif not (
        out.numel() == count
        and out.dtype == dtype
        and out.device == packed.device
        and out.is_contiguous()
    ):
        raise ValueError("GGUF output shape, dtype or device mismatch")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    stream = torch.xpu.current_stream(packed.device)
    status = library.xpu_gguf_dequant(
        stream.sycl_queue,
        packed.data_ptr(),
        out.data_ptr(),
        kind,
        count,
        int(dtype == torch.float32),
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_error().decode()))
    for tensor in (packed, out):
        tensor.record_stream(stream)
    return out


def matmul_gguf(x, packed, kind, rows):
    """FP32 fused decode/GEMV with a fixed per-row reduction order.

    Original compressed bytes are preserved. This initial execution path does
    not quantize activations to Q8_1 and is not Strata's tuned INT8 MMVQ.
    """
    geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
    if not (
        geometry
        and kind in SUPPORTED_TYPES
        and x.device.type == "xpu"
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim == 2
        and x.shape[1] > 0
        and rows > 0
        and x.shape[1] % geometry[0] == 0
        and packed.device == x.device
        and packed.dtype == torch.uint8
        and packed.is_contiguous()
        and packed.data_ptr() % 2 == 0
        and packed.numel() == rows * (x.shape[1] // geometry[0]) * geometry[1]
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 GGUF matrix or activation shape")
    if x.shape[0] == 0:
        return torch.empty((0, rows), device=x.device, dtype=x.dtype)
    # The native FP32 GEMM fallback has nondeterministic last-bit differences
    # on B70, which can cross BF16 rounding boundaries and alter greedy output.
    # Keep the same subgroup reduction for prefill and short decode batches.
    activation = x.float().contiguous()
    output = torch.empty((x.shape[0], rows), dtype=torch.float32, device=x.device)
    stream = torch.xpu.current_stream(x.device)
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    status = library.xpu_gguf_matvec(
        stream.sycl_queue,
        packed.data_ptr(),
        activation.data_ptr(),
        output.data_ptr(),
        kind,
        x.shape[1],
        rows,
        x.shape[0],
        packed.numel() // rows,
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_error().decode()))
    for tensor in (activation, packed, output):
        tensor.record_stream(stream)
    return output.to(x.dtype)


def expert_matvec_gguf(
    x,
    packed,
    kind,
    slots,
    rows,
    *,
    up=None,
    tokens=None,
    q8_activations=False,
    q8_workspace=None,
):
    """Evaluate routed expert rows in one launch, pairing gate/up when provided.

    ``slots`` is a device int32 vector into the compressed cache. The caller
    validates logical routes before constructing it; invalid device slot values
    produce NaNs without reading outside the cache. Gate/up share one activation
    row, or select rows through ``tokens``; down receives one row per route.
    FP32 arithmetic and
    each row's subgroup reduction order match ``matmul_gguf`` by default.
    ``q8_activations`` selects Strata's finite Q8_1/lane8 arithmetic and changes
    the numerical contract; it requires the optional expert-Q8 library symbols.
    An optional disjoint ``q8_workspace`` belongs to the caller and must remain
    owned until the current stream's work completes.
    """
    geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
    if not (
        geometry
        and kind in SUPPORTED_TYPES
        and x.device.type == "xpu"
        and x.dtype == torch.float32
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] > 0
        and x.shape[1] % geometry[0] == 0
        and rows > 0
        and packed.device == x.device
        and packed.dtype == torch.uint8
        and packed.ndim == 2
        and packed.shape[0] > 0
        and packed.shape[1] == rows * (x.shape[1] // geometry[0]) * geometry[1]
        and packed.is_contiguous()
        and packed.data_ptr() % 2 == 0
        and slots.device == x.device
        and slots.dtype == torch.int32
        and slots.ndim == 1
        and slots.numel() > 0
        and slots.is_contiguous()
        and x.shape[0] > 0
        and (
            (up is None and tokens is None and x.shape[0] == slots.numel())
            or (up is not None and (tokens is not None or x.shape[0] == 1))
        )
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 GGUF routed matrix geometry")
    if up is not None and not (
        up.device == packed.device
        and up.dtype == packed.dtype
        and up.shape == packed.shape
        and up.is_contiguous()
        and up.data_ptr() % 2 == 0
    ):
        raise ValueError("GGUF gate/up cache geometry mismatch")
    if tokens is not None and not (
        tokens.device == x.device
        and tokens.dtype == torch.int32
        and tokens.shape == slots.shape
        and tokens.is_contiguous()
    ):
        raise ValueError("GGUF route-to-token map geometry mismatch")
    output = torch.empty((slots.numel(), rows), dtype=torch.float32, device=x.device)
    output_up = torch.empty_like(output) if up is not None else None
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    stream = torch.xpu.current_stream(x.device)
    quantized = None
    if q8_activations:
        if not (
            kind in (16, 22, 29, 42)
            and x.shape[1] in (640, 2560, 6144)
            and x.shape[0] <= 2048
            and hasattr(library, "xpu_expert_q8_rows")
        ):
            raise ValueError("Unsupported Strata Q8_1 expert geometry or library")
        shape = (x.shape[0], x.shape[1] // 32 * 36)
        if q8_workspace is None:
            quantized = torch.empty(shape, dtype=torch.uint8, device=x.device)
        elif (
            q8_workspace.device == x.device
            and q8_workspace.dtype == torch.uint8
            and q8_workspace.shape == shape
            and q8_workspace.is_contiguous()
        ):
            quantized = q8_workspace
        else:
            raise ValueError("Expert Q8 workspace shape, dtype or device mismatch")
        if not hasattr(library, "xpu_expert_q8_float_rows"):
            status = library.xpu_expert_quantize(
                stream.sycl_queue,
                x.data_ptr(),
                quantized.data_ptr(),
                x.shape[1],
                x.shape[0],
            )
            if status:
                raise RuntimeError(
                    (status, library.xpu_expert_quantize_error().decode())
                )
    arguments = [
        stream.sycl_queue,
        packed.data_ptr(),
        up.data_ptr() if up is not None else None,
        x.data_ptr(),
        slots.data_ptr(),
        output.data_ptr(),
        output_up.data_ptr() if output_up is not None else None,
        kind,
        x.shape[1],
        rows,
        slots.numel(),
        packed.shape[1],
        packed.shape[1] // rows,
        int(up is not None),
        packed.shape[0],
    ]
    if q8_activations:
        arguments.insert(5, tokens.data_ptr() if tokens is not None else None)
        arguments.append(x.shape[0])
        if hasattr(library, "xpu_expert_q8_float_rows"):
            arguments.insert(8, quantized.data_ptr())
            status = library.xpu_expert_q8_float_rows(*arguments)
        else:
            arguments[3] = quantized.data_ptr()
            status = library.xpu_expert_q8_rows(*arguments)
    elif tokens is None:
        status = library.xpu_gguf_expert_rows(*arguments)
    else:
        if not hasattr(library, "xpu_gguf_expert_rows_general"):
            raise RuntimeError("Rebuild the GGUF SYCL library for grouped experts")
        arguments.insert(5, tokens.data_ptr())
        arguments.append(x.shape[0])
        status = library.xpu_gguf_expert_rows_general(*arguments)
    if status:
        error = (
            library.xpu_expert_q8_error if q8_activations else library.xpu_gguf_error
        )
        raise RuntimeError((status, error().decode()))
    for tensor in (x, packed, slots, tokens, output, up, output_up, quantized):
        if tensor is not None:
            tensor.record_stream(stream)
    return (output, output_up) if up is not None else output


def scatter_expert_routes(source, indices, out):
    """Place unique route rows without changing their FP32 values.

    The residency planner supplies unique, in-range device indices. Native
    bounds checks skip invalid destinations without writing outside ``out``.
    """
    if not (
        source.device.type == "xpu"
        and source.dtype == torch.float32
        and source.ndim == 2
        and source.shape[0] > 0
        and source.shape[1] > 0
        and source.is_contiguous()
        and indices.device == source.device
        and indices.dtype == torch.int32
        and indices.shape == (source.shape[0],)
        and indices.is_contiguous()
        and out.device == source.device
        and out.dtype == torch.float32
        and out.ndim == 2
        and out.shape[0] > 0
        and out.shape[1] == source.shape[1]
        and out.is_contiguous()
        and source.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
        and torch.xpu.get_device_properties(source.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 GGUF route scatter geometry")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_scatter_routes"):
        raise RuntimeError("Rebuild the GGUF SYCL library for route scatter")
    stream = torch.xpu.current_stream(source.device)
    status = library.xpu_gguf_scatter_routes(
        stream.sycl_queue,
        source.data_ptr(),
        indices.data_ptr(),
        out.data_ptr(),
        source.shape[0],
        out.shape[0],
        source.shape[1],
    )
    if status:
        raise RuntimeError((status, library.xpu_gguf_error().decode()))
    for tensor in (source, indices, out):
        tensor.record_stream(stream)
    return out
