"""Opt-in exact block128 quantization for the measured B70 DFlash shapes."""

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


def can_use_dflash_quant(x):
    if not (
        not torch.compiler.is_compiling()
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and tuple(x.shape)
        in ((4, 2048), (4, 4096), (3, 2048), (8, 2048), (8, 4096), (7, 2048))
        and x.is_contiguous()
    ):
        return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    return torch.xpu.get_device_properties(x.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _announce():
    logger.info(
        "Using SYCL B70 DFlash block128 quant: BF16 M4/M8 K2048,K4096 or M3/M7 K2048; "
        "FP8/FP32 bits, shared SMALL_TLA_LIB, no extra weight cache"
    )


def dflash_quant(x):
    if not can_use_dflash_quant(x):
        raise ValueError("Unsupported B70 DFlash block128 quantization contract")
    from sglang.kernels.ops.gemm.xpu_dflash_small_tla import quantize_bf16_block128

    result = quantize_bf16_block128(x)
    _announce()
    return result
