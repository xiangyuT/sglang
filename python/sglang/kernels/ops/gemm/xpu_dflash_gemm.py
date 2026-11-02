"""Small-M shared block-FP8 projection for the experimental B70 DFlash path.

Retains FP32 dequantization instead of selecting an FP16 DPAS path. FP32
reduction order can differ; FP8 activation quantization and scales are unchanged.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_decode_xpu import fp8_e4m3_div_256

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _announce(path):
    logger.info("Using B70 DFlash small-M block-FP8 projection: %s", path)


@triton.jit
def _decode_fp8(value):
    # Restore the exponent in the decoded FP32 bits. Compensating in an
    # arbitrary FP32 scale could overflow even when the final product is finite.
    small = fp8_e4m3_div_256(value)
    bits = small.to(tl.uint32, bitcast=True)
    magnitude = value.to(tl.uint8, bitcast=True) & 127
    restored = tl.where((magnitude == 0) | (magnitude == 127), bits, bits + 0x04000000)
    return restored.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _shared_gemv(X, W, XS, WS, Y, M: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, 2048)
    raw = tl.load(W + rows[:, None] * 2048 + cols[None, :])
    scale = tl.load(WS + (rows[:, None] // 128) * S0 + (cols[None, :] // 128) * S1)
    weight = _decode_fp8(raw) * scale
    for token in tl.static_range(M):
        raw_x = tl.load(X + token * 2048 + cols)
        sx = tl.load(XS + token * 16 + cols // 128)
        x = _decode_fp8(raw_x) * sx
        tl.store(Y + token * 1024 + rows, tl.sum(weight * x[None, :], 1))


def shared_gemv(q, w, xs, ws):
    """Guarded [M,2048] @ [1024,2048].T, M=2..8, BF16 output."""
    _announce("shared FP32 vector decode, M=2..8")
    out = torch.empty((q.shape[0], 1024), device=q.device, dtype=torch.bfloat16)
    _shared_gemv[(256,)](
        q,
        w,
        xs,
        ws,
        out,
        M=q.shape[0],
        S0=ws.stride(0),
        S1=ws.stride(1),
        num_warps=4,
        grf_mode="256",
    )
    return out
