"""Opt-in native host fetch/cache fill for compact B70 W8A16 experts.

Pinned host planes and pointer tables must remain alive through graph replay.
The planner owns unique (host expert, resident slot) pairs and every missed
slot must occur in the active routes. No tensor values are read by Python.
"""

import ctypes
import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.paged_experts_xpu import _gather_kernel
from sglang.kernels.ops.speculative.dflash_native import (
    native_window,
    require_native_window,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_sycl_routes():
    library = load_paged_moe()
    try:
        abi = library.dflash_routes_abi
        prepare = library.dflash_routes_prepare
        error = library.dflash_routes_error
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL route symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if abi() != 1:
        raise RuntimeError("Unsupported SYCL route library ABI")
    prepare.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int]
    prepare.restype = ctypes.c_int
    error.argtypes = []
    error.restype = ctypes.c_char_p
    logger.info(
        "Using SYCL M=%d DFlash route metadata and scale copies (WG=128)",
        native_window(library),
    )
    return library


@lru_cache(maxsize=1)
def load_paged_moe():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("Native paged MoE requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        abi = library.dflash_paged_abi
        gemm = library.dflash_paged_gemm
        error = library.dflash_paged_error
    except AttributeError as exc:
        raise RuntimeError(
            "Native paged MoE symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if abi() != 1:
        raise RuntimeError("Unsupported native paged MoE library ABI")
    gemm.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4
    gemm.restype = ctypes.c_int
    error.argtypes = []
    error.restype = ctypes.c_char_p
    return library


@triton.jit
def classify_routes(Ids, Dst, Src, Count, HostIds, R: tl.constexpr, BR: tl.constexpr):
    r = tl.arange(0, BR)
    p = tl.arange(0, BR)
    count = tl.load(Count)
    ids = tl.load(Ids + r, r < R, other=-2)
    dst = tl.load(Dst + p, p < count, other=-1)
    src = tl.load(Src + p, p < count, other=-1)
    match = (ids[:, None] == dst[None, :]) & (p[None, :] < count)
    host = tl.max(tl.where(match, src[None, :], -1), 1)
    tl.store(HostIds + r, host, r < R)


@triton.jit
def _prepare_fused_routes(
    Top,
    Sorted,
    Cmap,
    Ids,
    Counts,
    Offsets,
    HostIds,
    Dst,
    Src,
    Count,
    Hp,
    Gp,
    R: tl.constexpr,
    BR: tl.constexpr,
    E: tl.constexpr,
):
    r = tl.arange(0, BR)
    old = tl.load(Top + r, r < R, other=E).to(tl.int32)
    key = tl.sort(old * BR + r, descending=False)
    routes = key % BR
    slots = key // BR
    tl.store(Sorted + r, routes)
    tl.store(Cmap + routes, r, routes < R)

    # R is only 32 on the enabled path. Pairwise equality avoids gathering a
    # sorted predecessor and also handles all routes sharing one expert.
    same = (slots[:, None] == slots[None, :]) & (r[None, :] < R)
    first = (r < R) & (tl.sum((same & (r[None, :] < r[:, None])).to(tl.int32), 1) == 0)
    lengths = tl.sum(same.to(tl.int32), 1)
    rank = tl.cumsum(first.to(tl.int32), 0) - 1
    active = tl.sum(first.to(tl.int32), 0)
    padding = (r < R) & (r >= active)
    tl.store(Ids + r, 0, padding)
    tl.store(Counts + r, 0, padding)
    tl.store(Offsets + r, 0, padding)
    tl.store(Ids + rank, slots, first)
    tl.store(Offsets + rank, r, first)
    tl.store(Counts + rank, lengths, first)

    n = tl.load(Count)
    plans = tl.arange(0, BR)
    dst = tl.load(Dst + plans, plans < n, other=-1)
    src = tl.load(Src + plans, plans < n, other=-1)
    match = (slots[:, None] == dst[None, :]) & (plans[None, :] < n)
    host_ids = tl.max(tl.where(match, src[None, :], -1), 1)
    tl.store(HostIds + rank, host_ids, first)
    # Match classify_routes even for zero-count padding (whose slot ID is 0).
    zero_host = tl.max(tl.where((plans < n) & (dst == 0), src, -1), 0)
    tl.store(HostIds + r, zero_host, padding)

    # Copy the two small scale planes before native DPAS consumes them. The
    # unchanged native mainloop fetches and fills raw weight words itself.
    if n > 0:
        off = tl.arange(0, 256)
        for plane in tl.static_range(2):
            row = 128 if plane == 0 else 64
            host = tl.load(Hp + 2 + plane).to(tl.pointer_type(tl.uint32))
            gpu = tl.load(Gp + 2 + plane).to(tl.pointer_type(tl.uint32))
            for j in range(tl.cdiv(n * row, 256)):
                position = off + j * 256
                plan = position // row
                inside = position % row
                valid = position < n * row
                source = tl.load(Src + plan, valid, other=0)
                dest = tl.load(Dst + plan, valid, other=0)
                value = tl.load(host + source * row + inside, valid, other=0)
                tl.store(gpu + dest * row + inside, value, valid)


class _PagedMatmul:
    def __init__(self, paging):
        self.hp, self.gp, self.state, self.hosts = paging
        self.library = load_paged_moe()

    def _allocate_host_ids(self, ids):
        r = ids.numel()
        state = self.state
        if state.src.numel() < r or state.dst.numel() < r:
            raise ValueError("Native paged MoE plan capacity is smaller than routes")
        return torch.empty_like(ids)

    def bind_fused_routes(self, topids, sorted_routes, cmap, ids, counts, offsets, e):
        self.host_ids = self._allocate_host_ids(ids)
        r = ids.numel()
        _prepare_fused_routes[(1,)](
            topids,
            sorted_routes,
            cmap,
            ids,
            counts,
            offsets,
            self.host_ids,
            self.state.dst,
            self.state.src,
            self.state.count,
            self.hp,
            self.gp,
            R=r,
            BR=triton.next_power_of_2(r),
            E=e,
            num_warps=4,
        )

    def bind_sycl_routes(self, topids, sorted_routes, cmap, ids, counts, offsets, e):
        if topids.shape not in ((4, 8), (8, 8)) or ids.numel() != topids.numel():
            raise ValueError(
                "SYCL route preparation requires M4/M8 with eight routes each"
            )
        library = load_sycl_routes()
        if topids.shape[0] == 4 and native_window(library) == 8:
            # Four-row prompt tails keep the retained M4 preparation in an
            # M8 process; the paged expert arithmetic supports both shapes.
            return self.bind_fused_routes(
                topids, sorted_routes, cmap, ids, counts, offsets, e
            )
        require_native_window(library, topids.shape[0])
        self.host_ids = self._allocate_host_ids(ids)
        code = library.dflash_routes_prepare(
            torch.xpu.current_stream().sycl_queue,
            topids.data_ptr(),
            sorted_routes.data_ptr(),
            cmap.data_ptr(),
            ids.data_ptr(),
            counts.data_ptr(),
            offsets.data_ptr(),
            self.host_ids.data_ptr(),
            self.state.dst.data_ptr(),
            self.state.src.data_ptr(),
            self.state.count.data_ptr(),
            self.hp.data_ptr(),
            self.gp.data_ptr(),
            128,
        )
        if code:
            raise RuntimeError(library.dflash_routes_error().decode())

    def bind_routes(self, ids):
        r = ids.numel()
        state = self.state
        self.host_ids = self._allocate_host_ids(ids)
        classify_routes[(1,)](
            ids,
            state.dst,
            state.src,
            state.count,
            self.host_ids,
            R=r,
            BR=triton.next_power_of_2(r),
            num_warps=4,
        )
        # W13,W2,S13,S2 pointer order. Scales precede both GEMMs; raw
        # weight words are filled by the first M tile inside native DPAS.
        _gather_kernel[(1,)](
            self.hp[2:],
            self.gp[2:],
            state.src,
            state.dst,
            state.count,
            ROWS=(128, 64),
            GRID=1,
            BLOCK=256,
            num_warps=4,
        )

    def __call__(self, x, w, s, counts, ids, offsets, out, plane):
        code = self.library.dflash_paged_gemm(
            torch.xpu.current_stream().sycl_queue,
            x.data_ptr(),
            w.data_ptr(),
            s.data_ptr(),
            out.data_ptr(),
            counts.data_ptr(),
            ids.data_ptr(),
            offsets.data_ptr(),
            self.hosts[plane].data_ptr(),
            self.host_ids.data_ptr(),
            counts.numel(),
            w.shape[1],
            w.shape[2],
            0,  # Expert-major scheduling; tile-major was slower on B70.
        )
        if code:
            raise RuntimeError(self.library.dflash_paged_error().decode())


def prepare_paged(paging, w1, w2, s1, s2, device):
    hp, gp, state, hosts = paging
    if len(hosts) != 4:
        raise ValueError("Native paged MoE requires W13,W2,S13,S2 host planes")
    e = hosts[0].shape[0]
    shapes = ((e, 1024, 2048), (e, 2048, 512), (e, 8, 16), (e, 16, 4))
    if not all(
        h.device.type == "cpu"
        and h.shape == shape
        and h.dtype == g.dtype
        and h.is_pinned()
        and h.is_contiguous()
        and h.data_ptr() % 64 == 0
        for h, g, shape in zip(hosts, (w1, w2, s1, s2), shapes)
    ):
        raise ValueError("Unsupported native paged MoE pinned host tensor contract")
    if not (
        all(
            t.device == device
            and t.dtype == torch.uint64
            and t.shape == (4,)
            and t.is_contiguous()
            for t in (hp, gp)
        )
        and state.count.numel() == 1
        and all(
            t.device == device and t.dtype == torch.int32 and t.is_contiguous()
            for t in (state.src, state.dst, state.count)
        )
    ):
        raise ValueError("Unsupported native paged MoE pointer/plan contract")
    return _PagedMatmul(paging)
