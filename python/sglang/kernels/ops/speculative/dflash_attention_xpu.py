"""B70 split-KV attention for a single bidirectional DFlash draft block.

Prefix slots come from the existing window rail. Current tokens are dense.
This changes only the draft proposal path, not target verification.
"""

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs


@triton.jit
def first(
    Q,
    K,
    V,
    KB,
    VB,
    INDPTR,
    IDS,
    P,
    MAX,
    SUM,
    M: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    KB0: tl.constexpr,
    KB1: tl.constexpr,
    VB0: tl.constexpr,
    VB1: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    split = tl.program_id(0)
    head = tl.program_id(1)
    r = tl.arange(0, BM)
    t = r // (H // HK)
    qh = head * (H // HK) + r % (H // HK)
    d = tl.arange(0, D)
    n = split * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(INDPTR)
    length = tl.load(INDPTR + 1) - start
    q = tl.load(
        Q + t[:, None] * Q0 + qh[:, None] * Q1 + d[None, :], t[:, None] < M, other=0
    )
    ids = tl.load(IDS + start + n, n < length, other=0)
    k1 = tl.load(
        KB + ids[None, :] * KB0 + head * KB1 + d[:, None], n[None, :] < length, other=0
    )
    k2 = tl.load(
        K + (n[None, :] - length) * K0 + head * K1 + d[:, None],
        (n[None, :] >= length) & (n[None, :] < length + M),
        other=0,
    )
    k = tl.where(n[None, :] < length, k1, k2)
    s = tl.dot(q, k) * SCALE
    valid = (t[:, None] < M) & (n[None, :] < length + M)
    valid &= (n[None, :] >= length) | ((length + t[:, None]) <= n[None, :] + WINDOW)
    s = tl.where(valid, s, -float("inf"))
    maximum = tl.max(s, 1)
    safe = tl.where(maximum == -float("inf"), 0, maximum)
    p = tl.exp(s - safe[:, None])
    denom = tl.sum(p, 1)
    v1 = tl.load(
        VB + ids[:, None] * VB0 + head * VB1 + d[None, :], n[:, None] < length, other=0
    )
    v2 = tl.load(
        V + (n[:, None] - length) * V0 + head * V1 + d[None, :],
        (n[:, None] >= length) & (n[:, None] < length + M),
        other=0,
    )
    v = tl.where(n[:, None] < length, v1, v2)
    a = tl.dot(p.to(tl.bfloat16), v)
    offset = split * M * H + t * H + qh
    tl.store(P + offset[:, None] * D + d[None, :], a, t[:, None] < M)
    tl.store(MAX + offset, maximum, t < M)
    tl.store(SUM + offset, denom, t < M)


@triton.jit
def merge(
    P,
    MAX,
    SUM,
    OUT,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    off = s * M * H + t * H + h
    ma = tl.load(MAX + off, s < SPLITS, other=-float("inf"))
    su = tl.load(SUM + off, s < SPLITS, other=0)
    weight = tl.exp(ma - tl.max(ma, 0))
    denom = tl.sum(weight * su, 0)
    a = tl.load(P + off[:, None] * D + d[None, :], s[:, None] < SPLITS, other=0)
    y = tl.sum(a * weight[:, None], 0) / denom
    tl.store(OUT + t * (H * D) + h * D + d, y)


def _run(q, k, v, kb, vb, ptr, ids, out, window=2047):
    block = 64
    m, h, d = q.shape
    hk = k.shape[1]
    splits = triton.cdiv(window + m, block)
    p = torch.empty((splits, m, h, d), device=q.device, dtype=torch.float32)
    ma = torch.empty((splits, m, h), device=q.device, dtype=torch.float32)
    su = torch.empty_like(ma)
    o = out
    first[(splits, hk)](
        q,
        k,
        v,
        kb,
        vb,
        ptr,
        ids,
        p,
        ma,
        su,
        M=m,
        H=h,
        HK=hk,
        D=d,
        SPLITS=splits,
        BLOCK=block,
        BM=max(16, triton.next_power_of_2(m * (h // hk))),
        Q0=q.stride(0),
        Q1=q.stride(1),
        K0=k.stride(0),
        K1=k.stride(1),
        V0=v.stride(0),
        V1=v.stride(1),
        KB0=kb.stride(0),
        KB1=kb.stride(1),
        VB0=vb.stride(0),
        VB1=vb.stride(1),
        SCALE=d**-0.5,
        WINDOW=window,
        num_warps=4,
        grf_mode="256",
    )
    merge[(m, h)](
        p,
        ma,
        su,
        o,
        M=m,
        H=h,
        D=d,
        SPLITS=splits,
        BS=triton.next_power_of_2(splits),
        num_warps=4,
    )
    return o


def try_xpu_dflash_attention(
    q,
    k,
    v,
    out,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale,
    *,
    sliding_window_size=-1,
    logit_cap=0.0,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    page_size=1,
    score_mod=None,
    aux_tensors=None,
    extend_seq_lens_cpu=None,
    saved_paged_kv=None,
):
    """Return False for an unsupported contract; otherwise fill out and return True.

    The Triton backend provides one zero-based uniform query block and a prefix
    window of at most sliding_window_size tokens. All lengths stay on device.
    For the supported flat 3-D KV buffers, page128 addressing simplifies to
    (slot//128)*(128*stride0)+(slot%128)*stride0 == slot*stride0.
    """
    if not (
        q.device.type == "xpu"
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and 2 <= q.shape[0] <= 8
        and tuple(q.shape[1:]) == (32, 128)
        and tuple(k.shape) == tuple(v.shape) == (q.shape[0], 8, 128)
        and k_buffer.ndim == v_buffer.ndim == 3
        and tuple(k_buffer.shape) == tuple(v_buffer.shape)
        and tuple(k_buffer.shape[1:]) == (8, 128)
        and out.shape == q.shape
        and out.is_contiguous()
        and all(
            t.dtype == q.dtype and t.device == q.device and t.stride(-1) == 1
            for t in (k, v, out, k_buffer, v_buffer)
        )
        and q.stride(-1) == 1
        and qo_indptr.shape == kv_indptr.shape == (2,)
        and qo_indptr.is_contiguous()
        and kv_indptr.is_contiguous()
        and kv_indices.ndim == 1
        and kv_indices.is_contiguous()
        and all(
            t.device == q.device and t.dtype in (torch.int32, torch.int64)
            for t in (qo_indptr, kv_indptr, kv_indices)
        )
        and max_len_extend == q.shape[0]
        and not is_causal
        and custom_mask is None
        and sinks is None
        and score_mod is None
        and aux_tensors is None
        and page_size in (1, 128)
        and sliding_window_size == 2047
        and logit_cap == 0.0
        and xai_temperature_len == -1
        and isinstance(k_scale, (int, float))
        and isinstance(v_scale, (int, float))
        and k_scale == v_scale == 1.0
        and sm_scale == 128**-0.5
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    ):
        return False
    if (
        envs.SGLANG_XPU_PAGED_DFLASH.get()
        and envs.SGLANG_XPU_DFLASH_ATTENTION.get()
        and envs.SGLANG_XPU_DFLASH_ATTENTION_TLA.get()
    ):
        from sglang.kernels.ops.speculative.dflash_attention_tla import (
            try_attention_paged_tla,
            try_attention_tla,
        )

        if (
            envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA.get()
            and saved_paged_kv is not None
        ):
            if try_attention_paged_tla(
                q,
                k,
                v,
                k_buffer,
                v_buffer,
                kv_indptr,
                kv_indices,
                out,
                saved_paged_kv,
                page_size=page_size,
            ):
                return True
        if try_attention_tla(q, k, v, k_buffer, v_buffer, kv_indptr, kv_indices, out):
            return True
    _run(q, k, v, k_buffer, v_buffer, kv_indptr, kv_indices, out, sliding_window_size)
    return True
