"""B70 single-token GDN with the installed native BF16 rounding boundaries.

Conv products and beta stay FP32; a+dt_bias rounds to BF16 before softplus.
Prefill, other devices and unsupported layouts remain on the native backend.
"""
# Recurrent kernel adapted from FLA's fused_recurrent.py through SGLang:
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp


@triton.jit
def _conv_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    stride_qkvz_batch: tl.constexpr,
    stride_qkvz_dim: tl.constexpr,
    stride_ba_batch: tl.constexpr,
    stride_ba_dim: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_pos: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_indices: tl.constexpr,
    QKV_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    qkv_mask = dim_idx < QKV_DIM

    x = tl.load(
        mixed_qkvz + batch_idx * stride_qkvz_batch + dim_idx * stride_qkvz_dim,
        mask=qkv_mask,
        other=0.0,
    )

    state_slot = tl.load(conv_state_indices + batch_idx * stride_state_indices).to(
        tl.int64
    )
    # Treat every out-of-range index as padding so stale replay metadata cannot
    # turn an indexed state update into an OOB access.
    valid_slot = (
        (state_slot != PAD_SLOT_ID) & (state_slot >= 0) & (state_slot < NUM_STATE_SLOTS)
    )
    state_base = (
        conv_state + state_slot * stride_state_batch + dim_idx * stride_state_dim
    )

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(conv_bias + dim_idx, mask=qkv_mask, other=0.0).to(tl.float32)

    # Match the deployed direct-Triton update exactly. Its effective decode
    # state length is width-1 even when the physical cache tensor is wider.
    for pos in tl.static_range(KERNEL_WIDTH - 1):
        state_value = tl.load(
            state_base + pos * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        weight_value = tl.load(
            conv_weight + dim_idx * stride_weight_dim + pos * stride_weight_width,
            mask=qkv_mask,
            other=0.0,
        )
        # Do not force an FP32 multiply here. This expression deliberately
        # retains the operand types/order of causal_conv1d_triton.py.
        acc += state_value.to(tl.float32) * weight_value.to(tl.float32)

    last_weight = tl.load(
        conv_weight
        + dim_idx * stride_weight_dim
        + (KERNEL_WIDTH - 1) * stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    acc += x.to(tl.float32) * last_weight.to(tl.float32)
    if SILU_ACTIVATION:
        conv_out = acc / (1.0 + tl.exp(-acc))
    else:
        conv_out = acc

    # The legacy kernel leaves padded rows' input unchanged.
    conv_out = tl.where(valid_slot, conv_out, x)
    tl.store(
        mixed_qkv + batch_idx * QKV_DIM + dim_idx,
        conv_out,
        mask=qkv_mask,
    )

    # The direct-Triton wrapper sets effective state_len=width-1 for decode.
    for pos in tl.static_range(KERNEL_WIDTH - 2):
        next_value = tl.load(
            state_base + (pos + 1) * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        tl.store(
            state_base + pos * stride_state_pos,
            next_value,
            mask=qkv_mask & valid_slot,
        )
    tl.store(
        state_base + (KERNEL_WIDTH - 2) * stride_state_pos,
        x,
        mask=qkv_mask & valid_slot,
    )

    # The first feature lanes also materialize the smaller downstream tensors.
    z_mask = dim_idx < V_DIM
    z_value = tl.load(
        mixed_qkvz
        + batch_idx * stride_qkvz_batch
        + (QKV_DIM + dim_idx) * stride_qkvz_dim,
        mask=z_mask,
        other=0.0,
    )
    tl.store(z + batch_idx * V_DIM + dim_idx, z_value, mask=z_mask)

    gate_mask = dim_idx < NUM_V_HEADS
    b_value = tl.load(
        mixed_ba + batch_idx * stride_ba_batch + dim_idx * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    a_value = tl.load(
        mixed_ba
        + batch_idx * stride_ba_batch
        + (NUM_V_HEADS + dim_idx) * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    tl.store(b + batch_idx * NUM_V_HEADS + dim_idx, b_value, mask=gate_mask)
    tl.store(a + batch_idx * NUM_V_HEADS + dim_idx, a_value, mask=gate_mask)


@triton.jit
def _recurrent_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if (state_idx < 0) | (state_idx >= NUM_STATE_SLOTS):
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = (a_val + dt_bias_val).to(tl.bfloat16).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def can_use_xpu_gdn_decode(
    qkvz, ba, conv_state, ssm_state, conv_weights, conv_bias, A_log, dt_bias
):
    return (
        qkvz.device.type == "xpu"
        and tuple(qkvz.shape) == (1, 12288)
        and tuple(ba.shape) == (1, 64)
        and conv_state.ndim == 3
        and tuple(conv_state.shape[1:]) == (8192, 3)
        and ssm_state.ndim == 4
        and tuple(ssm_state.shape[1:]) == (32, 128, 128)
        and conv_state.shape[0] == ssm_state.shape[0]
        and tuple(conv_weights.shape) == (8192, 4)
        and tuple(A_log.shape) == tuple(dt_bias.shape) == (32,)
        and conv_bias is None
        and qkvz.dtype
        == ba.dtype
        == conv_state.dtype
        == conv_weights.dtype
        == dt_bias.dtype
        == torch.bfloat16
        and ssm_state.dtype == A_log.dtype == torch.float32
        and all(
            t.device == qkvz.device and t.is_contiguous()
            for t in (qkvz, ba, conv_state, ssm_state, conv_weights, A_log, dt_bias)
        )
        and torch.xpu.get_device_properties(qkvz.device).device_id == 0xE223
    )


def xpu_gdn_decode(
    qkvz, ba, conv_state, ssm_state, conv_weights, A_log, dt_bias, indices
):
    """Update one valid/indexed cache row; negative/out-of-range rows are padding.

    The caller checks the fixed B70 BF16 layout with can_use_xpu_gdn_decode.
    State indices are one contiguous int32/int64 value, resident on this XPU.
    Both kernels use the current stream and are safe for changing-input replay.
    """
    mixed = torch.empty((1, 8192), device=qkvz.device, dtype=qkvz.dtype)
    z = torch.empty((1, 32, 128), device=qkvz.device, dtype=qkvz.dtype)
    b = torch.empty((1, 32), device=qkvz.device, dtype=qkvz.dtype)
    a = torch.empty_like(b)
    _conv_kernel[(1, 32)](
        mixed,
        z,
        b,
        a,
        qkvz,
        ba,
        conv_state,
        conv_weights,
        None,
        indices,
        qkvz.stride(0),
        qkvz.stride(1),
        ba.stride(0),
        ba.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weights.stride(0),
        conv_weights.stride(1),
        indices.stride(0),
        QKV_DIM=8192,
        V_DIM=4096,
        NUM_V_HEADS=32,
        NUM_STATE_SLOTS=conv_state.shape[0],
        STATE_LEN=conv_state.shape[2],
        KERNEL_WIDTH=4,
        HAS_BIAS=False,
        SILU_ACTIVATION=True,
        PAD_SLOT_ID=-1,
        BLOCK_SIZE=256,
        num_warps=8,
        num_stages=2,
    )
    return recurrent_only(mixed, a, b, A_log, dt_bias, ssm_state, indices, z)


def recurrent_only(mixed, a, b, A_log, dt_bias, ssm_state, indices, z):
    """Original B1 recurrence shared by the two convolution producers."""
    out = torch.empty_like(z)
    _recurrent_kernel[(32, 32)](
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=ssm_state,
        ht=ssm_state,
        ssm_state_indices=indices,
        scale=128**-0.5,
        stride_mixed_qkv_tok=mixed.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=ssm_state.stride(0),
        stride_final_state_token=ssm_state.stride(0),
        stride_indices_seq=indices.stride(0),
        H=16,
        HV=32,
        K=128,
        V=128,
        BK=128,
        BV=4,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        NUM_STATE_SLOTS=ssm_state.shape[0],
        num_warps=1,
        num_stages=3,
    )
    return out, z
