# SPDX-License-Identifier: Apache-2.0
"""Serialized GPU expert LRU with byte-preserving pinned-mirror exchange.

This is a preparation primitive. The model owns the state for the lifetime of
all graph replays and serializes its prefill/decode phases. Expert arithmetic
and prefill integration remain separate from this module.

SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW=1 permits up to 32 rows on partial
257-expert EP2 partitions. GPU table rows 256..511 alias cold pinned mirrors
only when a protected union exceeds the resident capacity. Logical expert
ownership and the E-K host mirror table keep their original bounds.
"""

import atexit
import os
from dataclasses import dataclass

import numpy as np
import torch

from sglang.kernels.ops.quantization.gguf_iq2s_pvr import IQ2SGroupedRoutes
from sglang.kernels.ops.quantization.gguf_expert_geometry import expert_field_sizes
from sglang.kernels.ops.quantization.gguf_sycl import _library

_live_states = {}
_GU_BYTES = {16: 422400, 21: 704000, 22: 524800, 23: 870400, 29: 358400, 42: 460800}


def empty_expert_aos(rows, sizes, device):
    """Allocate one encoded expert arena with three non-overlapping field views."""
    if rows < 1 or len(sizes) != 3 or any(size < 1 or size % 16 for size in sizes):
        raise ValueError("Invalid expert AoS allocation geometry")
    packed = torch.empty((rows, sum(sizes)), dtype=torch.uint8, device=device)
    views, offset = [], 0
    for size in sizes:
        views.append(packed[:, offset:offset + size])
        offset += size
    return tuple(views)


def make_host_expert_aos(fields):
    """One primary pinned allocation, with gate/up/down contiguous per expert.

    Field views retain their row geometry; only their outer byte stride changes.
    The returned views own the allocation. No secondary host weight copy is kept.
    """
    if len(fields) != 3 or any(
        x.device.type != "cpu" or x.dtype != torch.uint8 or x.ndim != 2
        or not x.is_contiguous() or x.shape[0] != fields[0].shape[0]
        or x.shape[1] < 1 for x in fields
    ):
        raise ValueError("Expert AoS conversion requires three contiguous CPU byte fields")
    rows = fields[0].shape[0]
    if rows == 0:
        return tuple(fields)
    sizes = tuple(x.shape[1] for x in fields)
    packed = torch.empty((rows, sum(sizes)), dtype=torch.uint8, pin_memory=True)
    views, offset = [], 0
    for source, size in zip(fields, sizes):
        view = packed[:, offset:offset + size]
        view.copy_(source)
        views.append(view)
        offset += size
    return tuple(views)


def expert_aos_views_valid(fields, rows, sizes, device):
    if len(fields) != 3 or len(sizes) != 3 or any(
        value.device != device or value.dtype != torch.uint8
        or value.shape != (rows, size)
        for value, size in zip(fields, sizes)
    ):
        return False
    if rows == 0:
        return True
    stride = sum(sizes)
    base, offset = fields[0].untyped_storage().data_ptr(), 0
    for value, size in zip(fields, sizes):
        if (value.stride() != (stride, 1)
                or value.untyped_storage().data_ptr() != base
                or value.untyped_storage().nbytes() < rows * stride
                or value.data_ptr() != base + offset or value.data_ptr() % 16):
            return False
        offset += size
    return True


def encode_host_experts(raw, kind, field, *, packed_gu=False, down_kind=42, down_soa=False, gu_soa=False):
    """Preserve encoded bytes with optional lossless PVR/IQ3/IQ4 field order."""
    sizes = expert_field_sizes(kind, down_kind)
    size = sizes[field] if field in (0, 1, 2) else None
    if not (
        kind in _GU_BYTES
        and field in (0, 1, 2)
        and raw.device.type == "cpu"
        and raw.dtype == torch.uint8
        and raw.ndim == 2
        and raw.shape[1] == size
        and raw.is_contiguous()
    ):
        raise ValueError("Invalid host expert encoding geometry")
    if packed_gu and kind not in (16, 29):
        raise ValueError("Packed gate/up requires IQ2_XXS or IQ1_M")
    if down_soa and (kind not in (21, 23) or down_kind != 20):
        raise ValueError("IQ4 down field order requires mixed IQ3/IQ4 expert caches")
    if gu_soa and (kind != 21 or down_kind not in (8, 20)):
        raise ValueError("IQ3 gate/up field order requires a mixed IQ3 expert cache")
    if gu_soa and field < 2 and raw.shape[0]:
        experts = raw.shape[0]
        source = raw.numpy().reshape(experts, 160, 4, 10, 110)
        output = torch.empty_like(raw, pin_memory=raw.is_pinned())
        target = output.numpy().reshape(experts, 160, 4400)
        for begin, end, dest in [(0, 2, 0), (106, 110, 80), (66, 74, 240), (74, 106, 560)]:
            target[..., dest:dest+40*(end-begin)] = (
                source[..., begin:end].transpose(0, 1, 3, 2, 4)
                .reshape(experts, 160, 40*(end-begin))
            )
        target[..., 1840:] = (
            source[..., 2:66].reshape(experts, 160, 4, 10, 8, 2, 4)
            .transpose(0, 1, 3, 5, 2, 4, 6).reshape(experts, 160, 2560)
        )
        return output
    if down_soa and field == 2 and raw.shape[0]:
        experts = raw.shape[0]
        source = raw.numpy().reshape(experts, 640, 4, 5, 4, 18)
        output = torch.empty_like(raw, pin_memory=raw.is_pinned())
        target = output.numpy().reshape(experts, 640, 5, 288)
        target[..., :32] = source[..., :2].transpose(0, 1, 3, 2, 4, 5).reshape(experts, 640, 5, 32)
        target[..., 32:] = (
            source[..., 2:].reshape(experts, 640, 4, 5, 4, 2, 2, 4)
            .transpose(0, 1, 3, 6, 2, 4, 5, 7).reshape(experts, 640, 5, 256)
        )
        return output
    if packed_gu and field < 2 and raw.shape[0]:
        experts, block = raw.shape[0], 66 if kind == 16 else 56
        source = raw.numpy().reshape(experts, 80, 8, 10, block)
        output = torch.empty(raw.shape, dtype=torch.uint8, pin_memory=raw.is_pinned())
        target = output.numpy().reshape(experts, 80, 10, 8 * block)
        if kind == 16:
            fields = source[..., 2:].reshape(experts, 80, 8, 10, 8, 8)
            target[..., :256] = (
                fields[..., :4]
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(experts, 80, 10, 256)
            )
            target[..., 256:512] = (
                fields[..., 4:]
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(experts, 80, 10, 256)
            )
            target[..., 512:] = (
                source[..., :2].transpose(0, 1, 3, 2, 4).reshape(experts, 80, 10, 16)
            )
        else:
            for begin, end, dest in ((0, 32, 0), (32, 48, 256), (48, 56, 384)):
                target[..., dest : dest + 8 * (end - begin)] = (
                    source[..., begin:end]
                    .transpose(0, 1, 3, 2, 4)
                    .reshape(experts, 80, 10, 8 * (end - begin))
                )
        return output
    if kind != 22 or raw.shape[0] == 0:
        return raw
    experts, down = raw.shape[0], field == 2
    tiles, block, packet = (160, 18, 288) if down else (40, 82, 1312)
    source = raw.numpy().reshape(experts, tiles, 16, 10, block)
    output = torch.empty(raw.shape, dtype=torch.uint8, pin_memory=raw.is_pinned())
    target = output.numpy().reshape(experts, tiles, 10, packet)
    header = 256 if down else 1280
    target[..., header:] = (
        source[..., :2].transpose(0, 1, 3, 2, 4).reshape(experts, tiles, 10, 32)
    )
    for p in range(2 if down else 8):
        start, stride = (2 + 8 * p, 128) if down else (2 + 4 * p, 160)
        high = start + 4 if down else start + 32
        target[..., p * stride : p * stride + 64] = (
            source[..., start : start + 4]
            .transpose(0, 1, 3, 2, 4)
            .reshape(experts, tiles, 10, 64)
        )
        target[..., p * stride + 64 : p * stride + 128] = (
            source[..., high : high + 4]
            .transpose(0, 1, 3, 2, 4)
            .reshape(experts, tiles, 10, 64)
        )
        if not down:
            qh = source[..., 66 + p].transpose(0, 1, 3, 2)
            scale = source[..., 74 + p].transpose(0, 1, 3, 2)
            target[..., p * 160 + 128 : p * 160 + 160] = np.stack(
                (qh, scale), axis=-1
            ).reshape(experts, tiles, 10, 32)
    return output


@dataclass
class GPUCacheRoutes:
    grouping: IQ2SGroupedRoutes
    error: torch.Tensor
    actual_groups: torch.Tensor
    owner: "GPUExpertCache"

    @property
    def slots(self):
        return self.grouping.route_slots


class GPUExpertCache:
    """Own encoded resident slots, pinned mirrors and stable GPU address tables.

    ``resident_experts`` names each row of the three device buffers, and
    ``mirror_experts`` names each row of the three host buffers. Together they
    cover every logical expert exactly once. IQ2_S uses the already qualified
    IQ2S/Q2 PVR byte permutations in both tiers. Mixed caches may also use
    opt-in IQ3 gate/up and IQ4 down field orders in both tiers.
    No weights are requantized. The caller must destroy replayable graphs
    before closing a state.
    """

    def __init__(
        self,
        kind,
        resident_experts,
        resident_buffers,
        mirror_experts,
        mirrors,
        *,
        layout,
        lru_order=None,
        down_kind=42,
        host_layout="fields",
        resident_layout="fields",
        down_soa=False,
        gu_soa=False,
    ):
        self._closed = True
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Create GPU expert cache state before capture")
        expected_layout = "iq2s_q2_pvr" if kind == 22 else "original"
        self.packed_gu = kind in (16, 29) and layout == "lowbit_gu_packed"
        if kind not in _GU_BYTES or (layout != expected_layout and not self.packed_gu):
            raise ValueError("Unsupported expert format or encoded cache layout")
        self.kind = kind
        self.down_kind = down_kind
        self.gu_soa = bool(gu_soa)
        if self.gu_soa and (kind != 21 or down_kind not in (8, 20)):
            raise ValueError("IQ3 gate/up field order requires a mixed IQ3 expert cache")
        self.down_soa = bool(down_soa)
        if self.down_soa and (kind not in (21, 23) or down_kind != 20):
            raise ValueError("IQ4 down field order requires a mixed expert cache")
        self.sizes = expert_field_sizes(kind, down_kind)
        resident_experts = tuple(int(e) for e in resident_experts)
        mirror_experts = tuple(int(e) for e in mirror_experts)
        self.experts = len(resident_experts) + len(mirror_experts)
        all_experts = resident_experts + mirror_experts
        if not (
            1 <= len(resident_experts) <= self.experts <= 512
            and len(set(all_experts)) == self.experts
            and set(all_experts) == set(range(self.experts))
            and len(resident_buffers) == len(mirrors) == 3
        ):
            raise ValueError("Resident and mirror rows must partition all experts")
        self.union_overflow = (
            os.environ.get("SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW") == "1"
            and kind in (21, 23) and down_kind in (8, 20)
        )
        if self.union_overflow and (
            self.experts != 257 or len(resident_experts) > 256
        ):
            raise ValueError("Expert-union overflow requires a partial 257-expert EP2 cache")
        self.device = resident_buffers[0].device
        if (
            self.device.type != "xpu"
            or torch.xpu.get_device_properties(self.device).device_id != 0xE223
        ):
            raise ValueError("GPU expert-cache primitive currently requires B70")
        if resident_layout not in ("fields", "expert_aos"):
            raise ValueError("Unsupported resident expert layout")
        self.resident_layout = resident_layout
        if resident_layout == "expert_aos":
            self._check_expert_aos(resident_buffers, len(resident_experts), self.device)
        else:
            self._check_buffers(resident_buffers, len(resident_experts), self.device)
        if host_layout not in ("fields", "expert_aos"):
            raise ValueError("Unsupported host expert layout")
        self.host_layout = host_layout
        if host_layout == "expert_aos":
            self._check_expert_aos(mirrors, len(mirror_experts), torch.device("cpu"))
        else:
            self._check_buffers(mirrors, len(mirror_experts), torch.device("cpu"))
        self.library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
        self.max_rows = 4
        if kind in (21, 23) and down_kind in (8, 20):
            requested = int(os.environ.get("SGLANG_XPU_GGUF_EXPERT_WAVE_ROWS", "4"))
            maximum = 32 if self.union_overflow else 16
            if not 4 <= requested <= maximum:
                raise ValueError(f"GGUF expert wave rows must be between 4 and {maximum}")
            self.max_rows = requested
            if requested > 4 and (
                not hasattr(self.library, "xpu_gguf_mixed_max_rows")
                or self.library.xpu_gguf_mixed_max_rows() < requested
            ):
                raise ValueError("Rebuild the GGUF library for wider mixed expert waves")
        if self.gu_soa and any(not hasattr(self.library, name) for name in (
            "xpu_gguf_mixed_packets_gu_soa", "xpu_gguf_iq3_gu_permute", "xpu_gguf_iq3_cache_dequant",
        )):
            raise ValueError("Rebuild the GGUF library for IQ3 gate/up field order")
        if self.down_soa and any(not hasattr(self.library, name) for name in (
            "xpu_gguf_mixed_packets_soa", "xpu_gguf_iq4_down_permute", "xpu_gguf_iq4_cache_dequant",
        )):
            raise ValueError("Rebuild the GGUF library for IQ4 down field order")
        if not hasattr(self.library, "xpu_gguf_gpu_cache_prepare"):
            raise ValueError("Rebuild the GGUF library for GPU expert caching")
        queue = torch.xpu.current_stream(self.device).sycl_queue
        for value in mirrors:
            if (
                value.numel()
                and self.library.xpu_gguf_gpu_cache_pointer_kind(
                    queue, value.data_ptr()
                )
                != 1
            ):
                raise ValueError(
                    "Mirror buffers must be USM host allocations in this context"
                )
        self.resident_chunks = [tuple(resident_buffers)]
        self.mirrors = tuple(mirrors)
        self._workspaces = {}
        self._coalesced_prefill = True
        order = (
            resident_experts if lru_order is None else tuple(int(e) for e in lru_order)
        )
        if len(order) != len(resident_experts) or set(order) != set(resident_experts):
            raise ValueError("Initial LRU order must name every resident exactly once")
        expert_slots = [-1] * self.experts
        slot_experts = [-1] * self.experts
        addresses = [[0, 0, 0] for _ in range(512 if self.union_overflow else self.experts)]
        mirror_addresses = [[0, 0, 0] for _ in range(self.experts)]
        used = [0] * self.experts
        for slot, expert in enumerate(resident_experts):
            expert_slots[expert] = slot
            slot_experts[slot] = expert
            addresses[slot] = [
                value.data_ptr() + slot * value.stride(0)
                for value in resident_buffers
            ]
        for row, expert in enumerate(mirror_experts):
            mirror_addresses[expert] = [
                value.data_ptr() + row * value.stride(0)
                for value in mirrors
            ]
        for stamp, expert in enumerate(order, 1):
            used[expert] = stamp

        def tensor(value, dtype):
            return torch.tensor(value, dtype=dtype, device=self.device)

        self.expert_slots = tensor(expert_slots, torch.int32)
        self.slot_experts = tensor(slot_experts, torch.int32)
        self.last_used = tensor(used, torch.uint64)
        self.counters = tensor([len(resident_experts), 0], torch.uint64)
        self.capacity = tensor([len(resident_experts)], torch.int32)
        self.slot_addresses = tensor(addresses, torch.uint64)
        self.mirror_addresses = tensor(mirror_addresses, torch.uint64)
        self._closed = False
        # CPU pinned allocation ownership cannot be inferred by Torch from the
        # native USM accesses. Retain it explicitly until close synchronizes.
        _live_states[id(self)] = self

    def _check_buffers(self, buffers, rows, device):
        if len(buffers) != 3:
            raise ValueError("An expert cache requires gate, up and down buffers")
        if any(
            value.device != device
            or value.dtype != torch.uint8
            or value.shape != (rows, size)
            or not value.is_contiguous()
            or (value.numel() and value.data_ptr() % 16)
            for value, size in zip(buffers, self.sizes)
        ):
            raise ValueError("Invalid encoded expert buffer geometry")
        ranges = sorted(
            (value.data_ptr(), value.data_ptr() + value.numel())
            for value in buffers
            if value.numel()
        )
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("Encoded expert buffers must not overlap")

    def _check_expert_aos(self, fields, rows, device):
        if not expert_aos_views_valid(fields, rows, self.sizes, device):
            raise ValueError("Invalid expert AoS views: expected one arena and exact disjoint field rows")

    def residency_snapshot(self, *, zero_expert=None):
        """Read live placement and the last GPU prepare outside graph execution.

        Byte counts describe encoded payload exchanged in each direction, not
        hardware PCIe transactions. The native miss counter is per prepare,
        so this readback deliberately does not report cumulative request work.
        """
        if self._closed or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Read an open expert cache outside graph capture")
        if zero_expert is not None and not 0 <= zero_expert < self.experts:
            raise ValueError("Invalid zero-expert identity")
        torch.xpu.synchronize(self.device)
        locations = self.expert_slots.cpu()
        mirror = self.mirror_addresses.cpu().ne(0).all(dim=1)
        capacity = int(self.capacity.item())
        live = locations >= 0
        if not (
            int(live.sum()) == capacity
            and torch.equal(mirror, ~live)
            and torch.equal(
                locations[live].sort().values,
                torch.arange(capacity, dtype=locations.dtype),
            )
        ):
            raise RuntimeError("GPU slots and host mirrors no longer partition experts")
        misses = int(self.counters.cpu()[1])
        encoded_bytes = sum(self.sizes)
        zero_resident = zero_expert is not None and bool(live[zero_expert])
        return {
            "logical_experts_including_dummy": self.experts,
            "allocated_slots": capacity,
            "resident_real_experts": capacity - int(zero_resident),
            "mirrored_experts_including_dummy": int(mirror.sum()),
            "zero_expert_resident": zero_resident,
            "last_gpu_prepare_misses": misses,
            "encoded_bytes_per_expert": encoded_bytes,
            "last_gpu_prepare_h2d_payload_bytes": misses * encoded_bytes,
            "last_gpu_prepare_d2h_payload_bytes": misses * encoded_bytes,
            "counter_scope": "last GPU prepare, not cumulative request work",
        }

    def prepare(self, ids):
        """Queue GPU grouping, LRU planning and reversible mirror exchanges.

        The returned grouping uses the route count as a fixed launch bound;
        inactive groups have empty ranges. Invalid device IDs set ``error``
        and invalid slot metadata; callers must not accept such outputs.
        """
        if self._closed:
            raise RuntimeError("GPU expert cache was closed")
        if not (
            ids.device == self.device
            and ids.dtype in (torch.int32, torch.int64)
            and ids.ndim == 2
            and 1 <= ids.shape[0] <= self.max_rows
            and 1 <= ids.shape[1] <= 10
            and ids.is_contiguous()
        ):
            raise ValueError(f"GPU cache routes require contiguous rows<={self.max_rows}/topk<=10")
        stream = torch.xpu.current_stream(self.device)
        routes = ids.numel()
        key = (stream.sycl_queue, ids.shape[0], ids.shape[1])
        if key not in self._workspaces:
            self._workspaces[key] = (
                torch.empty(6 * routes + 3, dtype=torch.int32, device=self.device),
                torch.empty((routes, 4), dtype=torch.uint64, device=self.device),
            )
        meta, swaps = self._workspaces[key]
        fields = (
            self.mirror_addresses,
            self.expert_slots,
            self.slot_experts,
            self.last_used,
            self.counters,
            self.capacity,
            self.slot_addresses,
        )
        planner = os.environ.get("SGLANG_XPU_GGUF_PARALLEL_PLAN", "0")
        function = self.library.xpu_gguf_gpu_cache_prepare
        extra = ()
        if planner != "0":
            workgroup = 128 if planner == "1" else int(planner)
            if workgroup not in (64, 128, 256) or not hasattr(
                self.library, "xpu_gguf_gpu_cache_prepare_parallel"
            ):
                raise ValueError(
                    "Parallel GPU planner requires a rebuilt library and WG64/128/256"
                )
            function = self.library.xpu_gguf_gpu_cache_prepare_parallel
            extra = (workgroup,)
        mixed = self.kind in (21, 23) or self.down_kind != 42
        if mixed:
            function = getattr(self.library, "xpu_gguf_gpu_cache_prepare_mixed", None)
            if function is None:
                raise ValueError("Rebuild the GGUF library for mixed expert cache formats")
            extra = (0 if planner == "0" else workgroup,)
        status = function(
            stream.sycl_queue,
            ids.data_ptr(),
            *(value.data_ptr() for value in fields),
            meta.data_ptr(),
            swaps.data_ptr(),
            self.kind,
            *((self.down_kind,) if mixed else ()),
            ids.shape[0],
            ids.shape[1],
            self.experts,
            self.experts,
            int(ids.dtype == torch.int64),
            *extra,
        )
        if status:
            raise RuntimeError(
                (status, self.library.xpu_gguf_gpu_cache_error().decode())
            )
        for value in (ids, meta, swaps, *fields):
            value.record_stream(stream)
        for chunk in self.resident_chunks:
            for value in chunk:
                value.record_stream(stream)
        grouping = IQ2SGroupedRoutes(
            data=meta,
            route_slots=meta[:routes],
            group_slots=meta[routes : 2 * routes],
            starts=meta[2 * routes : 3 * routes + 1],
            token_rows=meta[3 * routes + 1 : 4 * routes + 1],
            destinations=meta[4 * routes + 1 : 5 * routes + 1],
            groups=routes,
        )
        return GPUCacheRoutes(
            grouping,
            meta[6 * routes + 2 : 6 * routes + 3],
            meta[6 * routes + 1 : 6 * routes + 2],
            self,
        )

    def owned_resident_storage_bytes(self):
        """Count primary and post-capture chunks once, including AoS aliases."""
        return sum(
            {
                value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                for chunk in self.resident_chunks
                for value in chunk
            }.values()
        )

    def grow(self, extra_buffers, experts):
        """Initialization-only growth; graph-facing table addresses stay fixed."""
        if self._closed or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Grow an open GPU cache outside graph capture")
        experts = tuple(int(e) for e in experts)
        if not experts or len(set(experts)) != len(experts):
            raise ValueError("Growth must name distinct mirrored experts")
        if self.union_overflow and int(self.capacity.item()) + len(experts) > 256:
            raise ValueError("Disable expert-union overflow before full EP2 residency")
        chunks = (*self.resident_chunks, tuple(extra_buffers))
        if self.resident_layout == "expert_aos":
            self._check_expert_aos(extra_buffers, len(experts), self.device)
            ranges = sorted(
                (chunk[0].data_ptr(), chunk[0].data_ptr() + chunk[0].shape[0] * sum(self.sizes))
                for chunk in chunks
            )
        else:
            self._check_buffers(extra_buffers, len(experts), self.device)
            ranges = sorted(
                (value.data_ptr(), value.data_ptr() + value.numel())
                for chunk in chunks for value in chunk
            )
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("Extra slots must not overlap existing resident storage")
        torch.xpu.synchronize(self.device)
        slots = self.expert_slots.cpu()
        inverse = self.slot_experts.cpu()
        addresses = self.slot_addresses.cpu()
        mirror = self.mirror_addresses.cpu()
        used = self.last_used.cpu()
        counters = self.counters.cpu()
        capacity = int(self.capacity.item())
        if capacity + len(experts) > self.experts or any(
            not 0 <= expert < self.experts or slots[expert].item() >= 0
            for expert in experts
        ):
            raise ValueError("Growth exceeds available mirrored experts")
        positions = []
        for field, size in enumerate(self.sizes):
            base = self.mirrors[field].data_ptr()
            stride = self.mirrors[field].stride(0)
            rows = []
            for expert in experts:
                delta = int(mirror[expert, field].item()) - base
                if (
                    delta < 0
                    or delta % stride
                    or delta // stride >= self.mirrors[field].shape[0]
                ):
                    raise ValueError("Mirror address is outside its owned storage")
                rows.append(delta // stride)
            positions.append(rows)
        for field, rows in enumerate(positions):
            extra_buffers[field].copy_(
                self.mirrors[field].index_select(0, torch.tensor(rows))
            )
        torch.xpu.synchronize(self.device)
        stamp = int(counters[0].item())
        for row, expert in enumerate(experts):
            slot = capacity + row
            slots[expert], inverse[slot] = slot, expert
            mirror[expert] = 0
            addresses[slot] = torch.tensor(
                [
                    value.data_ptr() + row * value.stride(0)
                    for value in extra_buffers
                ],
                dtype=torch.uint64,
            )
            stamp += 1
            used[expert] = stamp
        counters[0] = stamp
        self.resident_chunks.append(tuple(extra_buffers))
        for target, source in (
            (self.expert_slots, slots),
            (self.slot_experts, inverse),
            (self.slot_addresses, addresses),
            (self.mirror_addresses, mirror),
            (self.last_used, used),
            (self.counters, counters),
        ):
            target.copy_(source)
        self.capacity.fill_(capacity + len(experts))
        torch.xpu.synchronize(self.device)
        self.compact_mirrors()

    def compact_mirrors(self):
        receipt = self._compact_mirror_storage()
        if receipt["released_storage_bytes"]:
            # The helper's frame has released all old field views. Dropping
            # Tensor ownership alone leaves pinned blocks cached by Torch.
            torch.accelerator.memory.empty_host_cache()
        return receipt

    def _compact_mirror_storage(self):
        """Reclaim dead host rows after growth, preserving replay table pointers.

        The serialized owner must be idle, as for grow(). Logical placement and
        encoded bytes are unchanged. Read the LIVE address table: exchanges may
        have permuted rows independently of the original expert numbering.
        """
        if self._closed or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Compact an open GPU cache outside graph capture")
        torch.xpu.synchronize(self.device)
        slots = self.expert_slots.cpu().tolist()
        table = self.mirror_addresses.cpu()
        cold = [expert for expert, slot in enumerate(slots) if slot < 0]
        if (len(cold) != self.experts - int(self.capacity.item())
                or any(bool(table[e].ne(0).any()) for e, slot in enumerate(slots) if slot >= 0)
                or any(not bool(table[e].ne(0).all()) for e in cold)):
            raise RuntimeError("Resident slots and mirrors must partition all experts")

        def storage_bytes(fields):
            return sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                        for v in fields}.values())

        before = storage_bytes(self.mirrors)
        compact_bytes = len(cold) * sum(self.sizes)
        if before == compact_bytes:
            return dict(cold_experts=len(cold), before_storage_bytes=before,
                        after_storage_bytes=before, released_storage_bytes=0)

        positions = []
        for field, source in enumerate(self.mirrors):
            rows = []
            for expert in cold:
                delta = int(table[expert, field]) - source.data_ptr()
                if delta < 0 or delta % source.stride(0) or delta // source.stride(0) >= source.shape[0]:
                    raise RuntimeError("Live mirror address is outside owned storage")
                rows.append(delta // source.stride(0))
            if len(set(rows)) != len(rows):
                raise RuntimeError("Distinct cold experts cannot alias a mirror row")
            positions.append(rows)

        if self.host_layout == "expert_aos" and cold:
            arena = torch.empty((len(cold), sum(self.sizes)), dtype=torch.uint8, pin_memory=True)
            offsets = (0, self.sizes[0], self.sizes[0] + self.sizes[1])
            compact = tuple(arena[:, offset:offset + size] for offset, size in zip(offsets, self.sizes))
        else:
            compact = tuple(torch.empty((len(cold), size), dtype=torch.uint8, pin_memory=True)
                            for size in self.sizes)
        updated = torch.zeros_like(table)
        for field, (source, target, rows) in enumerate(zip(self.mirrors, compact, positions)):
            for row, old_row in enumerate(rows):
                target[row].copy_(source[old_row])
                updated[cold[row], field] = target.data_ptr() + row * target.stride(0)
        queue = torch.xpu.current_stream(self.device).sycl_queue
        if any(t.numel() and self.library.xpu_gguf_gpu_cache_pointer_kind(queue, t.data_ptr()) != 1
               for t in compact):
            raise RuntimeError("Compacted mirrors must remain USM host allocations")
        # Captured kernels retain this table's address, never the old host base.
        # Publish only after all copies/validation succeed; drain the update
        # before releasing the only owner of the previous pinned allocations.
        self.mirror_addresses.copy_(updated)
        torch.xpu.synchronize(self.device)
        self.mirrors = compact
        receipt = dict(cold_experts=len(cold), before_storage_bytes=before,
                       after_storage_bytes=storage_bytes(compact),
                       released_storage_bytes=before - storage_bytes(compact))
        self.mirror_compaction_receipt = receipt
        return receipt

    def stage_original(self, experts, buffers):
        """Stage bounded FP32-prefill weights without disturbing decode LRU.

        Sources are the current GPU slots or their pinned mirrors; IQ2S/Q2
        tiles and mixed-field layouts are restored to original byte order for FP32 dots.
        This eager-only operation leaves the decode address table unchanged.
        """
        if self._closed or torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Stage prefill weights on an open eager GPU cache")
        experts = tuple(int(e) for e in experts)
        if (
            not experts
            or len(set(experts)) != len(experts)
            or min(experts) < 0
            or max(experts) >= self.experts
        ):
            raise ValueError("Prefill stage requires distinct valid logical experts")
        self._check_buffers(buffers, len(experts), self.device)
        if not hasattr(self.library, "xpu_gguf_gpu_cache_stage"):
            raise ValueError("Rebuild the GGUF library for GPU-cache prefill staging")
        ids = torch.tensor(experts, dtype=torch.int32, device=self.device)
        stream = torch.xpu.current_stream(self.device)
        inputs = (
            ids,
            self.expert_slots,
            self.slot_addresses,
            self.mirror_addresses,
            *buffers,
        )
        coalesced = self._coalesced_prefill and self.kind == 22
        if coalesced and not hasattr(self.library, "xpu_gguf_gpu_cache_stage_encoded"):
            raise ValueError("Rebuild the GGUF library for encoded prefill staging")
        if self.packed_gu:
            if not hasattr(self.library, "xpu_gguf_gpu_cache_stage_lowbit"):
                raise ValueError("Rebuild the GGUF library for packed gate/up prefill")
            function = self.library.xpu_gguf_gpu_cache_stage_lowbit
        else:
            function = (
                self.library.xpu_gguf_gpu_cache_stage_encoded
                if coalesced
                else self.library.xpu_gguf_gpu_cache_stage
            )
        mixed = self.kind in (21, 23) or self.down_kind != 42
        if mixed:
            function = getattr(self.library, "xpu_gguf_gpu_cache_stage_mixed", None)
            if function is None:
                raise ValueError("Rebuild the GGUF library for mixed expert staging")
        status = function(
            stream.sycl_queue,
            *(value.data_ptr() for value in inputs),
            self.kind,
            *((self.down_kind,) if mixed else ()),
            len(experts),
            self.experts,
        )
        if status:
            raise RuntimeError(
                (status, self.library.xpu_gguf_gpu_cache_error().decode())
            )
        if self.gu_soa:
            for buffer in buffers[:2]:
                temporary = torch.empty_like(buffer)
                status = self.library.xpu_gguf_iq3_gu_permute(
                    stream.sycl_queue, buffer.data_ptr(), temporary.data_ptr(), len(experts), 1,
                )
                if status:
                    raise RuntimeError((status, "Restore IQ3 gate/up row order"))
                buffer.copy_(temporary)
                temporary.record_stream(stream)
        if self.down_soa:
            # Stage both tiers contiguously, then restore original row order
            # entirely in VRAM. Decode slots and host mirrors retain one copy.
            temporary = torch.empty_like(buffers[2])
            status = self.library.xpu_gguf_iq4_down_permute(
                stream.sycl_queue, buffers[2].data_ptr(), temporary.data_ptr(), len(experts), 1,
            )
            if status:
                raise RuntimeError((status, "Restore IQ4 down row order"))
            buffers[2].copy_(temporary)
            temporary.record_stream(stream)
        if coalesced:
            from sglang.kernels.ops.quantization.gguf_iq2s_pvr import repack_iq2s_cache

            # PCIe reads stay contiguous in encoded order. The inverse byte
            # permutation runs entirely in VRAM after staging, without changing
            # resident or mirror ownership or the FP32 prefill arithmetic.
            repack_iq2s_cache(
                buffers[0],
                buffers[1],
                range(len(experts)),
                to_packed=False,
                down=buffers[2],
            )
        for value in inputs:
            value.record_stream(stream)
        for chunk in self.resident_chunks:
            for value in chunk:
                value.record_stream(stream)

    def close(self):
        if not self._closed:
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError("Close GPU cache after destroying replayable graphs")
            torch.xpu.synchronize(self.device)
            self._closed = True
            _live_states.pop(id(self), None)


def _close_states():
    for state in list(_live_states.values()):
        state.close()


atexit.register(_close_states)
