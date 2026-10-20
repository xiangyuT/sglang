"""Single-row block-FP8 W8A8 GEMV for XPU decode projections.

Consumes the existing per-128 activation quantization and 128x128 weight
scales. The FP8 checkpoint and activation quantization are unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    X,
    W,
    XS,
    WS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BN: tl.constexpr,
    FP16_DEQUANT: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, K)
    groups = cols // 128
    # Match oneDNN's Xe2 block-FP8 GEMM: scales and the scaled operands
    # round to FP16 before FP32 accumulation. Scaling an FP32 group dot
    # afterward is mathematically valid but changes the native result.
    xs = tl.load(XS + groups)
    x = tl.load(X + cols).to(tl.float32)
    # The guarded shapes are multiples of the row tile. Avoid a masked FP8
    # load: Intel Triton cannot materialize its implicit integer zero in FP8.
    w = tl.load(W + rows[:, None] * K + cols[None, :])
    ws = tl.load(
        WS + (rows[:, None] // 128) * WS0 + groups[None, :] * WS1,
    )
    if FP16_DEQUANT:
        xs = xs.to(tl.float16).to(tl.float32)
        ws = ws.to(tl.float16).to(tl.float32)
        x = (x * xs).to(tl.float16).to(tl.float32)
        w = (w.to(tl.float32) * ws).to(tl.float16).to(tl.float32)
    else:
        # oneDNN chooses an FP32 SIMD path for the small shared gate/up.
        x = x * xs
        w = w.to(tl.float32) * ws
    out = tl.sum(w * x[None, :], 1)
    tl.store(Y + rows, out, rows < N)


def xpu_block_fp8_gemv(q_input, weight, activation_scale, weight_scale, *, rows=4):
    """Return BF16 ``q_input[1,K] @ weight[N,K].T`` with block scales."""
    n, k = weight.shape
    assert rows in (1, 2, 4, 8) and n % rows == 0
    out = torch.empty((1, n), dtype=torch.bfloat16, device=q_input.device)
    _gemv_kernel[(triton.cdiv(n, rows),)](
        q_input,
        weight,
        activation_scale,
        weight_scale,
        out,
        N=n,
        K=k,
        WS0=weight_scale.stride(0),
        WS1=weight_scale.stride(1),
        BN=rows,
        FP16_DEQUANT=(n, k) != (1024, 2048),
        num_warps=4,
    )
    return out
