"""BF16 exact top-k using 32-bit score/local-index keys; smallest-ID ties."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered(x):
    x = tl.where(x == 0, 0.0, x).to(tl.bfloat16)
    bits = x.to(tl.uint16, bitcast=True).to(tl.uint32)
    key = tl.where((bits & 32768) != 0, (~bits) & 65535, bits ^ 32768)
    return tl.where(x != x, 65535, key)


@triton.jit
def _value(o):
    bits = tl.where((o & 32768) != 0, o ^ 32768, (~o) & 65535).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def first(
    X,
    PV,
    PI,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
    TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tl.arange(0, BLOCK)
    i = tile * BLOCK + j
    x = tl.load(X + row * V + i, i < V, other=-float("inf"))
    key = (_ordered(x) << SHIFT) | (BLOCK - 1 - j).to(tl.uint32)
    key = tl.where(i < V, key, 0)
    best = tl.topk(key, K)
    vals = _value(best >> SHIFT)
    idx = tl.where(best != 0, tile * BLOCK + BLOCK - 1 - (best & (BLOCK - 1)), V)
    vals = tl.where(best != 0, vals, -float("inf"))
    tl.store(PV + (row * TILES + tile) * K + tl.arange(0, K), vals)
    tl.store(PI + (row * TILES + tile) * K + tl.arange(0, K), idx)


@triton.jit
def merge(
    PV,
    PI,
    VAL,
    IDX,
    V: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(PV + row * TILES * K + j, j < TILES * K, other=-float("inf"))
    key = (_ordered(x) << SHIFT) | (BLOCK - 1 - j).to(tl.uint32)
    original_id = tl.load(PI + row * TILES * K + j, j < TILES * K, other=V)
    key = tl.where((j < TILES * K) & (original_id < V), key, 0)
    best = tl.topk(key, K)
    pos = BLOCK - 1 - (best & (BLOCK - 1))
    ids = tl.load(PI + row * TILES * K + pos)
    tl.store(VAL + row * K + tl.arange(0, K), _value(best >> SHIFT))
    tl.store(IDX + row * K + tl.arange(0, K), ids)


def can_use_xpu_dflash_topk(x, k):
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and k in (16, 64)
        and 1 <= x.shape[0] <= 8
        and k <= x.shape[1] <= 262144
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


def xpu_dflash_topk(x, k=16):
    """Return sorted BF16 values/int64 IDs; equal scores choose smaller IDs.

    The caller checks can_use_xpu_dflash_topk. Each tile emits sorted candidates,
    with increasing IDs among ties; concatenating tiles preserves that tie order.
    Thus both stages can pack BF16 scores and local positions into 32-bit keys.
    """
    block = 4096
    n, v = x.shape
    t = triton.cdiv(v, block)
    mergeblock = triton.next_power_of_2(t * k)
    pv = torch.empty((n, t, k), device=x.device, dtype=x.dtype)
    pi = torch.empty((n, t, k), device=x.device, dtype=torch.int32)
    vals = torch.empty((n, k), device=x.device, dtype=x.dtype)
    ids = torch.empty((n, k), device=x.device, dtype=torch.int64)
    first[(n, t)](
        x,
        pv,
        pi,
        V=v,
        K=k,
        BLOCK=block,
        SHIFT=block.bit_length() - 1,
        TILES=t,
        num_warps=4,
        grf_mode="256",
    )
    merge[(n,)](
        pv,
        pi,
        vals,
        ids,
        V=v,
        K=k,
        TILES=t,
        BLOCK=mergeblock,
        SHIFT=mergeblock.bit_length() - 1,
        num_warps=4,
        grf_mode="256",
    )
    return vals, ids
