# SPDX-License-Identifier: Apache-2.0
"""Strata's B70 IQ4_XS/Q8_1 pair-reuse MMVQ on the current Torch stream."""

import ctypes
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SGLANG_XPU_GGUF_IQ4_SYCL_LIB before capture")
    library = ctypes.CDLL(path)
    library.xpu_iq4_quantize.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    library.xpu_iq4_quantize.restype = ctypes.c_int
    library.xpu_iq4_mmvq.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    library.xpu_iq4_mmvq.restype = ctypes.c_int
    library.xpu_iq4_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_iq4_mmvq_output_stride"):
        library.xpu_iq4_mmvq_output_stride.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int
        ] * 4
        library.xpu_iq4_mmvq_output_stride.restype = ctypes.c_int
    for name in ("xpu_iq4_quantize_layout", "xpu_iq4_quantize_layout_wide"):
        if hasattr(library, name):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
            function.restype = ctypes.c_int
    return library


def iq4_xs_mmvq(x, packed, *, input_permutation=None):
    """FP32 logits from original IQ4_XS weights and finite Q8_1 activations.

    Inputs are finite FP16/BF16/FP32 rows. Q8_1 uses Strata's 32-value blocks,
    half scale, round-away integer conversion, and finite-half saturation.
    Outputs intentionally preserve FP32 rather than rounding logits to BF16.
    Longer token batches use consecutive groups of at most four rows.
    ``gdn_16x3x128`` restores the original GGUF value-head input order inside
    quantization; it is supported only by the layout-aware native ABI.
    """
    if not (
        x.device.type == "xpu"
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 2
        and x.shape[1] in (2560, 6144)
        and packed.device == x.device
        and packed.dtype == torch.uint8
        and packed.ndim == 2
        and packed.shape[0] > 0
        and packed.shape[1] == x.shape[1] // 256 * 136
        and packed.is_contiguous()
        and packed.data_ptr() % 64 == 0
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
        and (
            input_permutation is None
            or (input_permutation == "gdn_16x3x128" and x.shape[1] == 6144)
        )
    ):
        raise ValueError("Unsupported B70 IQ4_XS matrix or activation geometry")
    output = torch.empty(
        (x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32
    )
    if x.shape[0] == 0:
        return output
    library = _library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    s4_library = None
    if os.environ.get("SGLANG_XPU_GGUF_IQ4_RAW_S4") == "1":
        from sglang.kernels.ops.quantization.gguf_sycl import _library as _gguf_library

        s4_library = _gguf_library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
        if not hasattr(s4_library, "xpu_iq4_s4"):
            raise RuntimeError("Rebuild the GGUF library for IQ4 S4 projections")
    layout_aware = hasattr(library, "xpu_iq4_quantize_layout")
    if input_permutation is not None and not layout_aware:
        raise ValueError("Rebuild the IQ4 library for the GDN input layout")
    activation = x.contiguous() if layout_aware else x.float().contiguous()
    q8 = torch.empty(
        (x.shape[0], x.shape[1] // 32 * 36), dtype=torch.uint8, device=x.device
    )
    stream = torch.xpu.current_stream(x.device)
    for start in range(0, x.shape[0], 4):
        tokens = min(4, x.shape[0] - start)
        arguments = [
            stream.sycl_queue,
            activation[start:].data_ptr(),
            q8[start:].data_ptr(),
            x.shape[1],
            tokens,
        ]
        if layout_aware:
            arguments.extend(
                (
                    {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[x.dtype],
                    int(input_permutation is not None),
                )
            )
            status = library.xpu_iq4_quantize_layout(*arguments)
        else:
            status = library.xpu_iq4_quantize(*arguments)
        if status:
            raise RuntimeError((status, library.xpu_iq4_error().decode()))
        arguments = (
            stream.sycl_queue, packed.data_ptr(), q8[start:].data_ptr(),
            output[start:].data_ptr(), x.shape[1], packed.shape[0], tokens,
        )
        if s4_library is not None:
            status = s4_library.xpu_iq4_s4(
                *arguments, packed.shape[1], packed.shape[0], 1
            )
        else:
            status = library.xpu_iq4_mmvq(*arguments)
        if status:
            error = (s4_library.xpu_iq4_s4_error if s4_library is not None
                     else library.xpu_iq4_error)
            raise RuntimeError((status, error().decode()))
    for tensor in (activation, q8, packed, output):
        tensor.record_stream(stream)
    return output
