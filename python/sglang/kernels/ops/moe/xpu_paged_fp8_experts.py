"""B70 single-token experts with pinned-host fetch during compute and cache fill.

Scale rows are copied first; resident weights keep the original arithmetic.
All cache writes have one route owner even for duplicate/padded slots.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.paged_experts_xpu import _gather_kernel
from sglang.kernels.ops.moe.xpu_block_fp8_experts import _dequant_weight
from sglang.srt.environ import envs


@triton.jit
def _lookup_missing_slot(Ids, Dst, Src, Count, route, T: tl.constexpr):
    slot = tl.load(Ids + route)
    n = tl.load(Count)
    expert = 0
    missing = False
    owner = False
    if n > 0:
        plans = tl.arange(0, 8)
        dst = tl.load(Dst + plans, plans < n, other=-1)
        src = tl.load(Src + plans, plans < n, other=0)
        match = (dst == slot) & (plans < n)
        missing = tl.sum(match.to(tl.int32), 0) > 0
        expert = tl.sum(tl.where(match, src, 0), 0)
        if missing:
            routes = tl.arange(0, T)
            slots = tl.load(Ids + routes)
            first = tl.min(tl.where(slots == slot, routes, T), 0)
            owner = route == first
    return slot, expert, missing, owner


@triton.jit
def _paged_gate_up_kernel(
    X,
    W,
    S,
    Ids,
    A,
    Hosts,
    Dst,
    Src,
    Count,
    H: tl.constexpr,
    INTER: tl.constexpr,
    T: tl.constexpr,
    BN: tl.constexpr,
):
    route = tl.program_id(1)
    slot, expert, missing, owner = _lookup_missing_slot(Ids, Dst, Src, Count, route, T)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, H)
    x = tl.load(X + cols).to(tl.float32)
    if missing:
        host = tl.load(Hosts).to(tl.pointer_type(tl.uint32))
        gpu = W.to(tl.pointer_type(tl.uint32))
        words = tl.arange(0, H // 4)
        goff = rows[:, None] * (H // 4) + words[None, :]
        uoff = (rows[:, None] + INTER) * (H // 4) + words[None, :]
        gw = tl.load(host + expert * (2 * INTER * H // 4) + goff)
        uw = tl.load(host + expert * (2 * INTER * H // 4) + uoff)
        tl.store(gpu + slot * (2 * INTER * H // 4) + goff, gw, owner)
        tl.store(gpu + slot * (2 * INTER * H // 4) + uoff, uw, owner)
        shift = tl.arange(0, 4) * 8
        gate = (
            ((gw[:, :, None] >> shift[None, None, :]) & 255)
            .to(tl.uint8)
            .reshape(BN, H)
            .to(tl.float8e4nv, bitcast=True)
        )
        up = (
            ((uw[:, :, None] >> shift[None, None, :]) & 255)
            .to(tl.uint8)
            .reshape(BN, H)
            .to(tl.float8e4nv, bitcast=True)
        )
    else:
        gate = tl.load(W + slot * (2 * INTER * H) + rows[:, None] * H + cols[None, :])
        up = tl.load(
            W + slot * (2 * INTER * H) + (rows[:, None] + INTER) * H + cols[None, :]
        )
    scale_base = slot * (2 * INTER // 128) * (H // 128)
    gs = tl.load(
        S + scale_base + (rows[:, None] // 128) * (H // 128) + cols[None, :] // 128
    )
    us = tl.load(
        S
        + scale_base
        + ((rows[:, None] + INTER) // 128) * (H // 128)
        + cols[None, :] // 128
    )
    gate = (
        tl.sum(_dequant_weight(gate, gs, True) * x[None, :], 1)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    up = (
        tl.sum(_dequant_weight(up, us, True) * x[None, :], 1)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(A + route * INTER + rows, silu * up)


@triton.jit
def _paged_down_combine_kernel(
    A,
    W,
    S,
    Ids,
    Weights,
    Y,
    Hosts,
    Dst,
    Src,
    Count,
    H: tl.constexpr,
    INTER: tl.constexpr,
    T: tl.constexpr,
    BN: tl.constexpr,
):
    routes = tl.arange(0, T)
    slots = tl.load(Ids + routes)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, INTER)
    x = tl.load(A + routes[:, None] * INTER + cols[None, :]).to(tl.float32)
    n = tl.load(Count)
    if n > 0:
        plans = tl.arange(0, 8)
        dst = tl.load(Dst + plans, plans < n, other=-1)
        src = tl.load(Src + plans, plans < n, other=0)
        match = (slots[:, None] == dst[None, :]) & (plans[None, :] < n)
        missing = tl.sum(match.to(tl.int32), 1) > 0
        experts = tl.sum(tl.where(match, src[None, :], 0), 1)
        earlier = (slots[:, None] == slots[None, :]) & (
            routes[:, None] > routes[None, :]
        )
        owner = tl.sum(earlier.to(tl.int32), 1) == 0
        host = tl.load(Hosts + 1).to(tl.pointer_type(tl.uint32))
        gpu = W.to(tl.pointer_type(tl.uint32))
        base = tl.where(
            missing, host + experts * (H * INTER // 4), gpu + slots * (H * INTER // 4)
        )
        words = tl.arange(0, INTER // 4)
        off = rows[None, :, None] * (INTER // 4) + words[None, None, :]
        value = tl.load(base[:, None, None] + off)
        tl.store(
            gpu + slots[:, None, None] * (H * INTER // 4) + off,
            value,
            (missing & owner)[:, None, None],
        )
        shift = tl.arange(0, 4) * 8
        w = (
            ((value[:, :, :, None] >> shift[None, None, None, :]) & 255)
            .to(tl.uint8)
            .reshape(T, BN, INTER)
            .to(tl.float8e4nv, bitcast=True)
        )
    else:
        w = tl.load(
            W
            + slots[:, None, None] * (H * INTER)
            + rows[None, :, None] * INTER
            + cols[None, None, :]
        )
    s = tl.load(
        S
        + slots[:, None, None] * (H // 128) * (INTER // 128)
        + (rows[None, :, None] // 128) * (INTER // 128)
        + cols[None, None, :] // 128
    )
    partial = (
        tl.sum(_dequant_weight(w, s, True) * x[:, None, :], 2)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    weights = tl.load(Weights + routes)
    tl.store(Y + rows, tl.sum(partial * weights[:, None], 0))


def xpu_paged_block_fp8_experts(x, w1, w2, s1, s2, ids, weights, hp, gp, state, ar_joint=None, scales_prepared=False):
    """Compute and fill all missed weights for the guarded M=1/top-8 layout.

    Pointer tables have explicit W13,W2,S13,S2 order; pinned hosts outlive graph
    replay. Planner buffers contain at most eight distinct missing slots. The
    caller checks the block-FP8 layout and full host-store scaled-decode bounds.
    Each raw weight word has one writing route owner. The two ordered kernels
    cover both weight planes before the next decode/prefill can use the cache.
    """
    # Pointer order is explicit W13,W2,S13,S2. Only FP32 scale rows copied here.
    if not scales_prepared:
        _gather_kernel[(1,)](
            hp[2:],
            gp[2:],
            state.src,
            state.dst,
            state.count,
            ROWS=(s1[0].numel(), s2[0].numel()),
            GRID=1,
            BLOCK=256,
            num_warps=4,
        )
    a = torch.empty((8, 512), dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)
    use_sycl = False
    if ar_joint is not None:
        from sglang.srt.layers.ar101_joint import run as run_joint
        run_joint(ar_joint, x, w1, s1, ids, a, hp, state)
        use_sycl = True
    if not use_sycl and envs.SGLANG_XPU_AR_EXPERT_GATE_SYCL.get() and not torch.compiler.is_compiling():
        from sglang.srt.runtime_context import get_spec

        if get_spec().speculative_algorithm is None:
            from sglang.kernels.ops.moe.xpu_ar_expert_gate_sycl import (
                can_use_ar_expert_gate,
                xpu_ar_expert_gate,
            )

            use_sycl = can_use_ar_expert_gate(x, w1, s1, ids, a, hp, state)
            if use_sycl:
                xpu_ar_expert_gate(x, w1, s1, ids, a, hp, state)
    if not use_sycl:
        _paged_gate_up_kernel[(512, 8)](
            x,
            w1,
            s1,
            ids,
            a,
            hp,
            state.dst,
            state.src,
            state.count,
            H=2048,
            INTER=512,
            T=8,
            BN=1,
            num_warps=4,
            enable_fp_fusion=False,
        )
    _paged_down_combine_kernel[(2048,)](
        a,
        w2,
        s2,
        ids,
        weights,
        y,
        hp,
        state.dst,
        state.src,
        state.count,
        H=2048,
        INTER=512,
        T=8,
        BN=1,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y
