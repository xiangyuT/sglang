"""Opt-in B70 TP2 decode fusions with the native BF16 boundaries."""

import ctypes
import hashlib
import logging
import math
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _runtime(device):
    if (torch.is_grad_enabled() or torch.compiler.is_compiling()
            or device.type != "xpu"):
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec

    p = get_parallel()
    return (
        p.tp_size == 2 and p.pp_size == 1 and p.moe_ep_size in (1, 2)
        and p.attn_cp_size == 1 and not p.attn_dp_enabled
        and get_spec().speculative_algorithm is None
        and torch.xpu.get_device_properties(device).device_id == 0xE223
    )


@lru_cache(maxsize=2)
def _library(kind, path):
    if not path:
        raise RuntimeError(f"Set the TP {kind} SYCL library path")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError(f"Warm TP {kind} before graph capture")
    lib = ctypes.CDLL(path)
    if kind == "qk-rope":
        lib.tp_qk_rope_run.argtypes = [ctypes.c_void_p] * 10 + [
            ctypes.c_int, ctypes.c_float,
        ]
        lib.tp_qk_rope_run.restype = ctypes.c_int
        lib.tp_qk_rope_error.restype = ctypes.c_char_p
    else:
        lib.tp_gdn_norm_quant_abi.restype = ctypes.c_int
        if lib.tp_gdn_norm_quant_abi() != 1:
            raise RuntimeError("Unsupported TP gated norm/quant ABI")
        lib.tp_gdn_norm_quant_run.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float,
        ]
        lib.tp_gdn_norm_quant_run.restype = ctypes.c_int
        lib.tp_gdn_norm_quant_error.restype = ctypes.c_char_p
    logger.info("B70 TP2 SYCL %s enabled: library=%s sha256=%s", kind, path,
                hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return lib


def try_qk_rope(qg, k, qw, kw, cache, positions, eps, nq, nk, dim, rotary, neox):
    if not envs.SGLANG_XPU_TP_QK_ROPE_SYCL.get():
        return None
    tensors = (qg, k, qw, kw, cache, positions)
    if not (
        (nq, nk, dim, rotary) == (8, 1, 256, 64) and neox
        and qg.shape == (1, 4096) and k.shape == (1, 256)
        and qw.shape == kw.shape == (256,)
        and cache.ndim == 2 and cache.shape[1] == 64 and cache.shape[0] > 0
        and positions.shape == (1,) and positions.dtype == torch.int64
        and all(t.dtype == torch.bfloat16 for t in tensors[:-1])
        and all(t.device == qg.device and t.is_contiguous() for t in tensors)
        and isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0
        and _runtime(qg.device)
    ):
        return None
    lib = _library("qk-rope", envs.SGLANG_XPU_TP_QK_ROPE_SYCL_LIB.get())
    qo = torch.empty((1, 2048), device=qg.device, dtype=qg.dtype)
    ko = torch.empty_like(k)
    gate = torch.empty_like(qo)
    stream = torch.xpu.current_stream(qg.device)
    rc = lib.tp_qk_rope_run(
        stream.sycl_queue, *[t.data_ptr() for t in tensors],
        qo.data_ptr(), ko.data_ptr(), gate.data_ptr(), cache.shape[0], eps,
    )
    if rc:
        raise RuntimeError(lib.tp_qk_rope_error().decode())
    for t in tensors + (qo, ko, gate):
        t.record_stream(stream)
    return qo, ko, gate


def try_gdn_norm_quant(x, z, norm, projection):
    if not envs.SGLANG_XPU_TP_GDN_NORM_QUANT_SYCL.get():
        return None
    w = getattr(norm, "weight", None)
    pw = getattr(projection, "weight", None)
    method = getattr(projection, "quant_method", None)
    if not (
        x.shape == z.shape == (16, 128)
        and isinstance(w, torch.Tensor) and w.shape == (128,)
        and all(t.dtype == torch.bfloat16 and t.device == x.device
                and t.is_contiguous() for t in (x, z, w))
        and norm.norm_before_gate and norm.activation in ("swish", "silu")
        and getattr(norm, "group_size", None) in (None, 128)
        and getattr(norm, "bias", None) is None
        and not norm._forward_hooks and not norm._forward_pre_hooks
        and isinstance(norm.eps, (int, float)) and math.isfinite(norm.eps)
        and norm.eps > 0
        and type(method).__name__ == "Fp8LinearMethod"
        and method.block_quant and tuple(method.weight_block_size) == (128, 128)
        and not any(getattr(method, name, False) for name in
                    ("use_marlin", "use_mxfp8", "block_fp8_as_mxfp8"))
        and isinstance(pw, torch.Tensor) and pw.shape == (2048, 2048)
        and pw.dtype == torch.float8_e4m3fn and pw.device == x.device
        and getattr(projection, "bias", None) is None
        and not projection._forward_hooks and not projection._forward_pre_hooks
        and _runtime(x.device)
    ):
        return None
    lib = _library("gdn-norm", envs.SGLANG_XPU_TP_GDN_NORM_QUANT_SYCL_LIB.get())
    quant = torch.empty((1, 2048), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 16), device=x.device, dtype=torch.float32)
    stream = torch.xpu.current_stream(x.device)
    rc = lib.tp_gdn_norm_quant_run(
        stream.sycl_queue, x.data_ptr(), z.data_ptr(), w.data_ptr(),
        quant.data_ptr(), scale.data_ptr(), None, None, 16, norm.eps, 0, 448.,
    )
    if rc:
        raise RuntimeError(lib.tp_gdn_norm_quant_error().decode())
    for t in (x, z, w, quant, scale):
        t.record_stream(stream)
    return quant, scale
