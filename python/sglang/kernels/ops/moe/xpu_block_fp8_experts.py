"""B70 single-token, top-8 block-FP8 experts with BF16 activations.

Keep the native W8A16 boundaries: FP32 scales, BF16 weight truncation,
BF16 gate/up outputs, BF16 SiLU before the multiply, and BF16 down outputs
before the FP32 router-weighted sum. No checkpoint repacking is needed.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_decode_xpu import fp8_e4m3_div_256


@triton.jit
def _dequant_weight(value, scale, SCALED_FP8: tl.constexpr = False):
    if SCALED_FP8:
        scaled = fp8_e4m3_div_256(value) * (scale * 256)
    else:
        scaled = value.to(tl.float32) * scale
    # The native Xe2 BF16-destination multiply truncates towards zero.
    return (scaled.to(tl.int32, bitcast=True) & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _gate_up_kernel(
    X,
    W,
    S,
    Ids,
    A,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    SCALED_FP8: tl.constexpr = False,
):
    route = tl.program_id(1)
    expert = tl.load(Ids + route)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, H)
    x = tl.load(X + cols).to(tl.float32)
    gate = tl.load(W + expert * (2 * I * H) + rows[:, None] * H + cols[None, :])
    up = tl.load(W + expert * (2 * I * H) + (rows[:, None] + I) * H + cols[None, :])
    scale_base = expert * (2 * I // 128) * (H // 128)
    gate_scale = tl.load(
        S + scale_base + (rows[:, None] // 128) * (H // 128) + cols[None, :] // 128
    )
    up_scale = tl.load(
        S
        + scale_base
        + ((rows[:, None] + I) // 128) * (H // 128)
        + cols[None, :] // 128
    )
    gate = tl.sum(_dequant_weight(gate, gate_scale, SCALED_FP8) * x[None, :], 1)
    up = tl.sum(_dequant_weight(up, up_scale, SCALED_FP8) * x[None, :], 1)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(A + route * I + rows, silu * up)


@triton.jit
def _down_combine_kernel(
    A,
    W,
    S,
    Ids,
    Weights,
    Y,
    H: tl.constexpr,
    I: tl.constexpr,
    T: tl.constexpr,
    BN: tl.constexpr,
    SCALED_FP8: tl.constexpr = False,
):
    routes = tl.arange(0, T)
    experts = tl.load(Ids + routes)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, I)
    x = tl.load(A + routes[:, None] * I + cols[None, :]).to(tl.float32)
    w = tl.load(
        W
        + experts[:, None, None] * (H * I)
        + rows[None, :, None] * I
        + cols[None, None, :]
    )
    s = tl.load(
        S
        + experts[:, None, None] * (H // 128) * (I // 128)
        + (rows[None, :, None] // 128) * (I // 128)
        + cols[None, None, :] // 128
    )
    partial = tl.sum(_dequant_weight(w, s, SCALED_FP8) * x[:, None, :], 2)
    partial = partial.to(tl.bfloat16).to(tl.float32)
    weights = tl.load(Weights + routes)
    tl.store(Y + rows, tl.sum(partial * weights[:, None], 0))


def xpu_block_fp8_experts(
    hidden_states, w13, w2, s13, s2, topk_ids, topk_weights, *, scaled_fp8=False
):
    """Compute the guarded [1,2048], top-8, I=512 paged-expert step.

    ``scaled_fp8`` needs finite scales that are zero or within [2**-118, 2**118]
    in magnitude, checked over the full host store before selecting this path.
    IDs are valid resident slots, including zero-weight padding remapped by
    the device planner. Inputs/scales are contiguous BF16/FP32, weights are
    contiguous e4m3 with [128,128] scales. The caller applies the layer scale.
    """
    intermediate = torch.empty(
        (8, 512), dtype=hidden_states.dtype, device=hidden_states.device
    )
    output = torch.empty_like(hidden_states)
    _gate_up_kernel[(512, 8)](
        hidden_states,
        w13,
        s13,
        topk_ids,
        intermediate,
        H=2048,
        I=512,
        BN=1,
        SCALED_FP8=scaled_fp8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _down_combine_kernel[(2048,)](
        intermediate,
        w2,
        s2,
        topk_ids,
        topk_weights,
        output,
        H=2048,
        I=512,
        T=8,
        BN=1,
        SCALED_FP8=scaled_fp8,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output
