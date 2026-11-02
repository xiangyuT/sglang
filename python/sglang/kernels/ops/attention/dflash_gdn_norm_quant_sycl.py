"""Opt-in B70 M8 gated RMSNorm128 and block128 FP8 quantization.

The BF16 boundary and clustered reduction match the retained norm/quant path.
There is no persistent tensor state or weight cache. Warm the library before
capture; restart and recapture after flags or library changes.
"""

import ctypes
import logging
import math
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_gdn_norm_quant():
    path = envs.SGLANG_XPU_DFLASH_GDN_NORM_QUANT_LIB.get()
    if not path:
        raise RuntimeError(
            "SYCL norm/quant requires SGLANG_XPU_DFLASH_GDN_NORM_QUANT_LIB"
        )
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SYCL GDN norm/quant library before graph capture")
    library = ctypes.CDLL(path)
    try:
        library.dflash_gdn_norm_quant_abi.argtypes = []
        library.dflash_gdn_norm_quant_abi.restype = ctypes.c_int
        if library.dflash_gdn_norm_quant_abi() != 1:
            raise RuntimeError("Unsupported SYCL GDN norm/quant ABI")
        library.dflash_gdn_norm_quant_error.argtypes = []
        library.dflash_gdn_norm_quant_error.restype = ctypes.c_char_p
        library.dflash_gdn_norm_quant_run.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
        ]
        library.dflash_gdn_norm_quant_run.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("Rebuild with build_xpu_dflash_gdn_norm_quant.py") from exc
    logger.info(
        "Using SYCL B70 DFlash M8 GDN gated RMSNorm128/block128 quant (SG=2, GRF=256)"
    )
    return library


def can_use_gdn_norm_quant(x, z, weight, eps):
    if not (
        not torch.compiler.is_compiling()
        and x.device.type == "xpu"
        and x.dtype == z.dtype == weight.dtype == torch.bfloat16
        and x.shape == z.shape == (256, 128)
        and weight.shape == (128,)
        and x.is_contiguous()
        and z.is_contiguous()
        and weight.is_contiguous()
        and x.device == z.device == weight.device
        and isinstance(eps, (int, float))
        and math.isfinite(eps)
        and eps > 0
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, z, weight)):
        return False
    return torch.xpu.get_device_properties(x.device).device_id == 0xE223


def can_fuse_gdn_norm_quant(x, z, norm, projection):
    """Only the measured gated-norm and block-FP8 output projection contract."""
    if not (
        envs.SGLANG_XPU_PAGED_DFLASH.get()
        and envs.SGLANG_XPU_DFLASH_QUANT_SYCL.get()
        and envs.SGLANG_XPU_DFLASH_GDN_NORM_QUANT_SYCL.get()
    ):
        return False
    method = getattr(projection, "quant_method", None)
    weight = getattr(projection, "weight", None)
    if not (
        norm.norm_before_gate
        and norm.activation in ("swish", "silu")
        and getattr(norm, "group_size", None) in (None, 128)
        and getattr(norm, "bias", None) is None
        and type(method).__name__ == "Fp8LinearMethod"
        and method.block_quant
        and tuple(method.weight_block_size) == (128, 128)
        and not getattr(method, "use_marlin", False)
        and not getattr(method, "use_mxfp8", False)
        and isinstance(weight, torch.Tensor)
        and weight.shape == (2048, 4096)
        and weight.dtype == torch.float8_e4m3fn
        and weight.device == x.device
        and getattr(projection, "bias", None) is None
    ):
        return False
    return can_use_gdn_norm_quant(x, z, norm.weight, norm.eps)


def gdn_norm_quant(x, z, weight, eps):
    if not can_use_gdn_norm_quant(x, z, weight, eps):
        raise ValueError("Unsupported B70 M8 GDN norm/block128 quant layout")
    library = load_gdn_norm_quant()
    stream = torch.xpu.current_stream(x.device)
    q = torch.empty((8, 4096), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((8, 32), device=x.device, dtype=torch.float32)
    rc = library.dflash_gdn_norm_quant_run(
        stream.sycl_queue,
        x.data_ptr(),
        z.data_ptr(),
        weight.data_ptr(),
        q.data_ptr(),
        scale.data_ptr(),
        None,
        None,
        256,
        eps,
        0,
        448.0,
    )
    if rc:
        raise RuntimeError(library.dflash_gdn_norm_quant_error().decode())
    for tensor in (x, z, weight, q, scale):
        tensor.record_stream(stream)
    return q, scale
