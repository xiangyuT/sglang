"""Compact active-expert scheduling for the experimental B70 DFlash path.

The external SYCL library uses the pinned native W8A16 DPAS mainloop.
Weights retain their resident-slot layout; only routing metadata is compacted.
"""

import ctypes
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@lru_cache(maxsize=1)
def load_compact_moe():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError(
            "SGLANG_XPU_DFLASH_COMPACT_MOE requires SGLANG_XPU_DFLASH_NATIVE_LIB; build the library with test/manual/kernels/build_xpu_dflash_native.py"
        )
    library = ctypes.CDLL(path)
    library.dflash_compact_abi.argtypes = []
    library.dflash_compact_abi.restype = ctypes.c_int
    if library.dflash_compact_abi() != 1:
        raise RuntimeError("Unsupported compact MoE library ABI")
    library.dflash_compact_gemm.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
    library.dflash_compact_gemm.restype = ctypes.c_int
    library.dflash_compact_error.argtypes = []
    library.dflash_compact_error.restype = ctypes.c_char_p
    return library


@triton.jit
def max_op(a, b):
    return tl.maximum(a, b)


@triton.jit
def sort_routes(
    Top, Sorted, Slots, Cmap, R: tl.constexpr, BR: tl.constexpr, E: tl.constexpr
):
    r = tl.arange(0, BR)
    # Sentinel slots follow all valid resident IDs. Only the sort is padded;
    # the native GEMMs and output permutation retain the real route count.
    ids = tl.load(Top + r, r < R, other=E).to(tl.int32)
    key = tl.sort(ids * BR + r, descending=False)
    route = key % BR
    tl.store(Sorted + r, route)
    tl.store(Slots + r, key // BR)
    tl.store(Cmap + route, r, route < R)


@triton.jit
def prepare_routes(Slots, Ids, Counts, Offsets, R: tl.constexpr, BR: tl.constexpr):
    r = tl.arange(0, BR)
    slot = tl.load(Slots + r)
    previous = tl.load(Slots + tl.maximum(r - 1, 0))
    following = tl.load(Slots + tl.minimum(r + 1, BR - 1))
    first = (r < R) & ((r == 0) | (slot != previous))
    last = (r < R) & ((r == R - 1) | (slot != following))
    rank = tl.cumsum(first.to(tl.int32), 0) - 1
    start = tl.associative_scan(tl.where(first, r, 0), 0, max_op)
    active = tl.sum(first.to(tl.int32), 0)
    padding = (r < R) & (r >= active)
    tl.store(Ids + r, 0, padding)
    tl.store(Counts + r, 0, padding)
    tl.store(Offsets + r, 0, padding)
    tl.store(Ids + rank, slot, first)
    tl.store(Offsets + rank, r, first)
    tl.store(Counts + rank, r - start + 1, last)


@triton.jit
def shuffle_input(X, Sorted, A, H: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = off // H
    k = off % H
    route = tl.load(Sorted + row, row < R, other=0)
    value = tl.load(X + (route // 8) * H + k, off < R * H, other=0)
    tl.store(A + off, value, off < R * H)


def mm(x, w, s, counts, ids, offsets, out):
    code = load_compact_moe().dflash_compact_gemm(
        torch.xpu.current_stream().sycl_queue,
        x.data_ptr(),
        w.data_ptr(),
        s.data_ptr(),
        out.data_ptr(),
        counts.data_ptr(),
        ids.data_ptr(),
        offsets.data_ptr(),
        counts.numel(),
        w.shape[1],
        w.shape[2],
    )

    if code:
        raise RuntimeError(load_compact_moe().dflash_compact_error().decode())


def xpu_compact_moe(x, w1, w2, s1, s2, topids, weights, *, paging=None):
    """Top-8 W8A16 experts with native Silu/combine and scaling factor 1.0.

    Resident slot IDs must be in [0, E); the pager maps padded routes to slot
    zero with zero weight. The surrounding contract applies routed scaling.
    This entry point does not support torch.compile or arbitrary MoE layouts.
    """
    m, h = x.shape
    e = w1.shape[0]
    if not (
        x.device.type == "xpu"
        and 2 <= m <= 8
        and h == 2048
        and x.dtype == torch.bfloat16
        and w1.shape == (e, 1024, 2048)
        and w2.shape == (e, 2048, 512)
        and w1.dtype == w2.dtype == torch.float8_e4m3fn
        and s1.shape == (e, 8, 16)
        and s2.shape == (e, 16, 4)
        and s1.dtype == s2.dtype == weights.dtype == torch.float32
        and topids.dtype == torch.int32
        and topids.shape == weights.shape == (m, 8)
        and all(
            t.device == x.device and t.is_contiguous()
            for t in (x, w1, w2, s1, s2, topids, weights)
        )
    ):
        raise ValueError("Unsupported B70 compact W8A16 MoE tensor contract")
    if paging is not None:
        from sglang.kernels.ops.moe.xpu_compact_paged import prepare_paged

        paged_mm = prepare_paged(paging, w1, w2, s1, s2, x.device)
    r = m * 8
    br = triton.next_power_of_2(r)
    cmap = torch.empty((r,), device=x.device, dtype=torch.int32)
    sorted_routes = torch.empty((br,), device=x.device, dtype=torch.int32)
    shuffle = torch.empty((r, h), device=x.device, dtype=x.dtype)
    ids = torch.empty((r,), device=x.device, dtype=torch.int32)
    counts = torch.empty_like(ids)
    offsets = torch.empty_like(ids)
    fused_routes = (
        paging is not None
        and (m == 4 or (m == 8 and envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.get()))
        and envs.SGLANG_XPU_PAGED_DFLASH.get()
        and (
            envs.SGLANG_XPU_DFLASH_FUSED_ROUTES.get()
            or envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.get()
        )
    )
    if fused_routes:
        if envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.get():
            paged_mm.bind_sycl_routes(
                topids, sorted_routes, cmap, ids, counts, offsets, e
            )
        else:
            paged_mm.bind_fused_routes(
                topids, sorted_routes, cmap, ids, counts, offsets, e
            )
    else:
        slots = torch.empty_like(sorted_routes)
        sort_routes[(1,)](
            topids, sorted_routes, slots, cmap, R=r, BR=br, E=e, num_warps=4
        )
        prepare_routes[(1,)](slots, ids, counts, offsets, R=r, BR=br, num_warps=4)
    shuffle_input[(triton.cdiv(r * h, 512),)](
        x, sorted_routes, shuffle, H=h, R=r, BLOCK=512, num_warps=4
    )
    gate_up = torch.empty((r, w1.shape[1]), device=x.device, dtype=x.dtype)
    intermediate = torch.empty((r, w1.shape[1] // 2), device=x.device, dtype=x.dtype)
    partial = torch.empty((r, h), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    if paging is not None:
        if not fused_routes:
            paged_mm.bind_routes(ids)
        paged_mm(shuffle, w1, s1, counts, ids, offsets, gate_up, 0)
    else:
        mm(shuffle, w1, s1, counts, ids, offsets, gate_up)
    torch.ops.sgl_kernel.silu_and_mul(intermediate, gate_up)
    if paging is not None:
        paged_mm(intermediate, w2, s2, counts, ids, offsets, partial, 1)
    else:
        mm(intermediate, w2, s2, counts, ids, offsets, partial)
    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(partial, out, cmap, 1.0, weights)
    return out
