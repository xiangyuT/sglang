"""Opt-in B70 M=4/M=8 block-FP8 projections for DFlash verification.

The tested native path rounds scales to FP16, then dequantized operands to
FP16, and accumulates in FP32. Keep both rounding steps before DPAS. Other
shapes, including the attention output projection, retain native scaled_mm.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_decode_xpu import fp8_e4m3_div_256
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_GEOMETRY = {
    (12288, 2048): (64, 32, 4),
    (9216, 2048): (64, 32, 4),
    (2048, 512): (64, 64, 8),
}


@lru_cache(maxsize=1)
def _announce():
    logger.info("Using B70 DFlash M=4 dense DPAS with FP16 scale/operand rounding")


@triton.jit
def _half_operand(raw, scale):
    # FP8/256 is exactly representable in FP16. After rounding the scale to
    # FP16, multiplying its FP32 expansion by 256 cannot overflow FP32.
    return (fp8_e4m3_div_256(raw) * (scale.to(tl.float16).to(tl.float32) * 256.0)).to(
        tl.float16
    )


@triton.jit
def _dense_dpas(
    X,
    W,
    XS,
    WS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
):
    mi = tl.arange(0, 4)
    ni = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((4, BN), 0, tl.float32)
    for block in range(K // BK):
        ki = block * BK + kk
        a = tl.load(X + mi[:, None] * K + ki[None, :])
        b = tl.load(W + ni[:, None] * K + ki[None, :])
        sa = tl.load(XS + mi[:, None] * (K // 128) + ki[None, :] // 128)
        sb = tl.load(WS + (ni[:, None] // 128) * S0 + (ki[None, :] // 128) * S1)
        acc = tl.dot(
            _half_operand(a, sa),
            tl.trans(_half_operand(b, sb)),
            acc,
            input_precision="ieee",
        )
    tl.store(Y + mi[:, None] * N + ni[None, :], acc.to(tl.bfloat16))


def can_use_dense_dpas(q, w, xs, ws, block_size, bias):
    supported = (
        q.ndim == w.ndim == xs.ndim == ws.ndim == 2
        and q.device.type == "xpu"
        and (
            q.shape[0] == 4
            or (
                q.shape[0] == 8
                and envs.SGLANG_XPU_DFLASH_DENSE_SYCL.get()
                and tuple(w.shape) in ((12288, 2048), (9216, 2048))
            )
        )
        and tuple(w.shape) in _GEOMETRY
        and q.shape[1] == w.shape[1]
        and q.dtype == w.dtype == torch.float8_e4m3fn
        and xs.dtype == ws.dtype == torch.float32
        and q.is_contiguous()
        and w.is_contiguous()
        and xs.is_contiguous()
        and xs.shape == (q.shape[0], q.shape[1] // 128)
        and ws.shape == (w.shape[0] // 128, w.shape[1] // 128)
        and q.device == w.device == xs.device == ws.device
        and tuple(block_size) == (128, 128)
        and bias is None
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    )
    if (
        supported
        and q.shape[0] == 4
        and envs.SGLANG_XPU_DFLASH_DENSE_SYCL.get()
        and tuple(w.shape) in ((12288, 2048), (9216, 2048))
    ):
        # An M8 server can still prefill four prompt tokens. Its fixed-width
        # M8 library must not consume the retained M4 activation geometry.
        from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import load_dense_sycl
        from sglang.kernels.ops.speculative.dflash_native import native_window

        if native_window(load_dense_sycl()) == 8:
            return False
    return supported


def dense_dpas(q, w, xs, ws):
    """Run an eligible projection; caller checks can_use_dense_dpas first."""
    if envs.SGLANG_XPU_DFLASH_SMALL_DENSE_SYCL.get() and tuple(w.shape) == (
        2048,
        512,
    ):
        from sglang.kernels.ops.gemm.xpu_dflash_small_dense_sycl import (
            small_dense_sycl,
        )

        return small_dense_sycl(q, w, xs, ws)
    if envs.SGLANG_XPU_DFLASH_DENSE_SYCL.get():
        from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import SHAPES, dense_sycl

        if tuple(w.shape) in SHAPES:
            return dense_sycl(q, w, xs, ws)
    _announce()
    n, k = w.shape
    bn, bk, warps = _GEOMETRY[(n, k)]
    out = torch.empty((4, n), device=q.device, dtype=torch.bfloat16)
    _dense_dpas[(n // bn,)](
        q,
        w,
        xs,
        ws,
        out,
        N=n,
        K=k,
        BN=bn,
        BK=bk,
        S0=ws.stride(0),
        S1=ws.stride(1),
        num_warps=warps,
        num_stages=2,
        grf_mode="256",
    )
    return out
