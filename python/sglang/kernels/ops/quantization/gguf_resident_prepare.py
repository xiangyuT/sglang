# SPDX-License-Identifier: Apache-2.0
"""Expert workspace preparation on the actual consumer stream.

The native preparation must run after the expert workspace is selected: an
eager stream and a graph capture stream can own different scratch allocations.
The resident path uses fixed expert slots. The paged path consumes the existing
GPU cache plan and clips its logical dummy group. Neither preparation changes
encoded bytes, plans evictions or replaces the pager.
"""

import ctypes
from dataclasses import dataclass, replace
from functools import lru_cache
import logging
import os

import torch

from sglang.kernels.ops.quantization.gguf_iq2s_pvr import IQ2SGroupedRoutes

logger = logging.getLogger(__name__)
_seen = set()


@lru_cache(maxsize=8)
def _library(path):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the resident preparation library before capture")
    lib = ctypes.CDLL(path)
    lib.resident_prepare.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
    lib.resident_prepare.restype = ctypes.c_int
    if hasattr(lib, "resident_prepare_mapped"):
        lib.resident_prepare_mapped.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5
        lib.resident_prepare_mapped.restype = ctypes.c_int
    if hasattr(lib, "resident_prepare_q8"):
        lib.resident_prepare_q8.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5
        lib.resident_prepare_q8.restype = ctypes.c_int
    lib.resident_error.restype = ctypes.c_char_p
    return lib


@dataclass(frozen=True)
class ResidentRoutes(IQ2SGroupedRoutes):
    ids: torch.Tensor
    zero_slot: int
    library: object
    local_mapping: torch.Tensor | None = None

    def prepare_resident(self, workspace, stream, zero_slot, *, x=None):
        """Prepare the selected consumer workspace, optionally including Q8 input."""
        rows, topk = self.ids.shape
        count = rows * topk
        if zero_slot != self.zero_slot:
            raise ValueError("Resident zero slot changed after route preparation")
        for name, width in (("gate", 640), ("up", 640), ("down", 2560)):
            value = workspace[name]
            if not (
                value.device == self.ids.device
                and value.dtype == torch.float32
                and value.shape == (count, width)
                and value.is_contiguous()
            ):
                raise ValueError(
                    "Resident preparation requires actual-stream FP32 scratch"
                )
        function = self.library.resident_prepare
        arguments = (stream.sycl_queue, self.ids.data_ptr())
        if self.local_mapping is not None:
            mapping = self.local_mapping
            if not (
                mapping.device == self.ids.device
                and mapping.dtype == torch.int32
                and mapping.shape == (512,)
                and mapping.is_contiguous()
                and hasattr(self.library, "resident_prepare_mapped")
            ):
                raise ValueError(
                    "Resident preparation requires a native E512 expert map"
                )
            function = self.library.resident_prepare_mapped
            arguments += (mapping.data_ptr(),)
        arguments += (
            self.data.data_ptr(),
            workspace["gate"].data_ptr(),
            workspace["up"].data_ptr(),
            workspace["down"].data_ptr(),
            rows,
            topk,
            zero_slot,
            int(self.ids.dtype == torch.int64),
        )
        if self.local_mapping is not None:
            arguments += (512,)
        if x is not None:
            if not hasattr(self.library, "resident_prepare_q8"):
                raise ValueError(
                    "Rebuild resident preparation for the Q8 input producer"
                )
            value, scratch = workspace["x"], workspace["scratch"]
            if not (
                x.device == value.device == scratch.device == self.ids.device
                and x.dtype == torch.bfloat16
                and x.shape == (rows, 2560)
                and x.is_contiguous()
                and value.dtype == torch.float32
                and value.shape == x.shape
                and value.is_contiguous()
                and scratch.dtype == torch.uint8
                and scratch.ndim == 1
                and scratch.is_contiguous()
                and scratch.numel() >= rows * 80 * 36
            ):
                raise ValueError(
                    "Resident input producer requires actual-stream BF16/FP32/Q8 scratch"
                )
            function = self.library.resident_prepare_q8
            arguments = (
                stream.sycl_queue,
                self.ids.data_ptr(),
                self.local_mapping.data_ptr()
                if self.local_mapping is not None
                else None,
                self.data.data_ptr(),
                workspace["gate"].data_ptr(),
                workspace["up"].data_ptr(),
                workspace["down"].data_ptr(),
                x.data_ptr(),
                value.data_ptr(),
                scratch.data_ptr(),
                rows,
                topk,
                zero_slot,
                int(self.ids.dtype == torch.int64),
                512,
            )
        status = function(*arguments)
        if status:
            raise RuntimeError((status, self.library.resident_error().decode()))
        self.ids.record_stream(stream)
        if self.local_mapping is not None:
            self.local_mapping.record_stream(stream)
        if x is not None:
            for value in (x, workspace["x"], workspace["scratch"]):
                value.record_stream(stream)
        phase = "capture" if torch.xpu.is_current_stream_capturing() else "eager"
        key = (self.ids.device, phase, x is not None)
        if key not in _seen:
            _seen.add(key)
            logger.info(
                "GGUF resident EP preparation invoked: device=%s phase=%s rows=%d topk=%d zero=%d input_q8=%s; identity mapping, no LRU/exchange",
                self.ids.device,
                phase,
                rows,
                topk,
                zero_slot,
                x is not None,
            )
        return x is not None


class ResidentPreparation:
    def __init__(self, cache, zero_slot, library):
        self.cache, self.zero_slot, self.library = cache, zero_slot, library
        self.workspaces = {}

    def supports_deferred_mapping(self, x, ids, mapping):
        return (
            hasattr(self.library, "resident_prepare_mapped")
            and not self.cache._closed
            and x.device == ids.device == mapping.device == self.cache.device
            and x.device.type == "xpu" and x.dtype == torch.bfloat16
            and x.ndim == 2 and x.shape[1] == 2560 and x.is_contiguous()
            and ids.dtype in (torch.int32, torch.int64) and ids.ndim == 2
            and ids.shape[0] == x.shape[0] and 1 <= ids.shape[0] <= 4
            and 1 <= ids.shape[1] <= 10 and ids.is_contiguous()
            and mapping.dtype == torch.int32 and mapping.shape == (512,)
            and mapping.is_contiguous()
        )

    def routes_dispatch(self, output):
        deferred = getattr(output, "xpu_gguf_resident_routing", None)
        mapping = None
        if deferred is not None:
            owner, mapping = deferred
            if owner is not self or not self.supports_deferred_mapping(
                output.hidden_states, output.topk_output.topk_ids, mapping
            ):
                raise ValueError("Deferred GGUF expert map does not belong to this consumer")
        return self.routes(output.topk_output.topk_ids, local_mapping=mapping)

    def routes(self, ids, *, local_mapping=None):
        from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUCacheRoutes

        if self.cache._closed:
            raise RuntimeError("Resident expert cache was closed")
        if not (ids.device == self.cache.device
                and ids.dtype in (torch.int32, torch.int64) and ids.ndim == 2
                and 1 <= ids.shape[0] <= 4 and 1 <= ids.shape[1] <= 10
                and ids.is_contiguous()):
            raise ValueError("Resident routes require contiguous rows1..4/topk1..10 IDs")
        stream = torch.xpu.current_stream(ids.device)
        key = (stream.sycl_queue, *ids.shape)
        count = ids.numel()
        if key not in self.workspaces:
            self.workspaces[key] = torch.empty(6 * count + 3, dtype=torch.int32,
                                               device=ids.device)
        data = self.workspaces[key]
        # Metadata is deliberately deferred until fused_q8_experts selects its
        # own workspace. No caller may consume these ranges before that step.
        grouping = ResidentRoutes(
            data, data[:count], data[count:2*count], data[2*count:3*count+1],
            data[3*count+1:4*count+1], data[4*count+1:5*count+1], count,
            ids, self.zero_slot, self.library, local_mapping,
        )
        return GPUCacheRoutes(grouping, data[6*count+2:6*count+3],
                              data[6*count+1:6*count+2], self.cache)


def configure(cache, zero_slot):
    """Opt in only for immutable, completely resident EP2/EP4 partitions."""
    path = os.environ.get("SGLANG_XPU_GGUF_RESIDENT_PREP_LIB")
    if not path:
        return None
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Configure resident preparation before capture")
    if not (zero_slot in (128, 256) and cache.experts == zero_slot + 1
            and cache.kind in (16, 21, 22, 23, 29)
            and (cache.kind == 22 or cache.packed_gu)
            and cache.device.type == "xpu"
            and torch.xpu.get_device_properties(cache.device).device_id == 0xE223):
        raise ValueError("Resident preparation requires native full B70 EP2/EP4 with zero clipping")
    if not (torch.equal(cache.expert_slots.cpu(), torch.arange(cache.experts, dtype=torch.int32))
            and bool((cache.mirror_addresses == 0).all().item())
            and int(cache.capacity.item()) == cache.experts):
        raise ValueError("Resident preparation requires fixed identity slots and no mirrors")
    return ResidentPreparation(cache, zero_slot, _library(path))


@lru_cache(maxsize=8)
def _paged_library(path):
    if not path:
        raise ValueError("Paged expert preparation requires its native library")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the paged preparation library before capture")
    lib = ctypes.CDLL(path)
    lib.paged_prepare.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3
    lib.paged_prepare.restype = ctypes.c_int
    if hasattr(lib, "paged_prepare_arena_supported"):
        lib.paged_prepare_arena_supported.argtypes = []
        lib.paged_prepare_arena_supported.restype = ctypes.c_int
    if hasattr(lib, "paged_prepare_max_rows"):
        lib.paged_prepare_max_rows.argtypes = []
        lib.paged_prepare_max_rows.restype = ctypes.c_int
    lib.paged_error.restype = ctypes.c_char_p
    return lib


def prepare_paged(grouping, workspace, stream, zero_expert, *, x=None):
    """Clip the logical dummy after paging, preserving the original cache plan.

    Physical slots can move on every replay. Native grouping retains logical
    expert identities, so clipping never assumes a fixed slot for the dummy.
    Scratch and the optional Q8 producer belong to the actual consumer stream.
    """
    value = workspace["x"]
    rows = value.shape[0]
    count = grouping.groups
    if not (
        isinstance(zero_expert, int) and 1 <= zero_expert <= 511
        and 1 <= rows <= (32 if os.environ.get("SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW") == "1" else 16) and count % rows == 0 and 1 <= count // rows <= 10
        and value.device.type == "xpu" and value.dtype == torch.float32
        and value.shape == (rows, 2560) and value.is_contiguous()
        and torch.xpu.get_device_properties(value.device).device_id == 0xE223
        and grouping.data.dtype == grouping.starts.dtype == torch.int32
        and grouping.data.device == grouping.starts.device == value.device
        and grouping.data.is_contiguous() and grouping.starts.is_contiguous()
        and grouping.data.numel() == 6 * count + 3
        and grouping.starts.numel() == count + 1
    ):
        raise ValueError("Paged preparation requires native B70 grouped cache metadata")
    for name, width in (("gate", 640), ("up", 640), ("down", 2560)):
        tensor = workspace[name]
        if not (
            tensor.device == value.device and tensor.dtype == torch.float32
            and tensor.shape == (count, width) and tensor.is_contiguous()
        ):
            raise ValueError("Paged preparation requires actual-stream FP32 scratch")
    if "paged_starts" not in workspace:
        workspace["paged_starts"] = torch.empty_like(grouping.starts)
    starts = workspace["paged_starts"]
    if not (
        starts.device == value.device and starts.dtype == torch.int32
        and starts.shape == grouping.starts.shape and starts.is_contiguous()
    ):
        raise ValueError("Paged clipped ranges must match the consumer workspace")
    scratch = workspace["scratch"]
    if x is not None and not (
        x.device == scratch.device == value.device
        and x.dtype == torch.bfloat16 and x.shape == value.shape
        and x.is_contiguous() and scratch.dtype == torch.uint8
        and scratch.ndim == 1 and scratch.is_contiguous()
        and scratch.numel() >= rows * 80 * 36
    ):
        raise ValueError("Paged input producer requires BF16/FP32/Q8 scratch")
    lib = _paged_library(os.environ.get("SGLANG_XPU_GGUF_PAGED_PREP_LIB"))
    if rows > 4 and (not hasattr(lib, "paged_prepare_max_rows")
                     or lib.paged_prepare_max_rows() < rows):
        raise ValueError("Rebuild paged preparation for wider expert waves")
    if workspace["gate"].data_ptr() == workspace["down"].data_ptr():
        count_bytes = count * 640 * 4
        if (workspace["up"].data_ptr() != workspace["down"].data_ptr() + count_bytes
                or workspace["activation"].data_ptr()
                != workspace["down"].data_ptr() + 2 * count_bytes):
            raise ValueError("Invalid paged expert scratch arena offsets")
        if (not hasattr(lib, "paged_prepare_arena_supported")
                or lib.paged_prepare_arena_supported() != 1):
            raise ValueError("Rebuild paged preparation for non-racing arena clears")
    status = lib.paged_prepare(
        stream.sycl_queue, grouping.data.data_ptr(), starts.data_ptr(),
        workspace["gate"].data_ptr(), workspace["up"].data_ptr(),
        workspace["down"].data_ptr(), x.data_ptr() if x is not None else None,
        value.data_ptr() if x is not None else None,
        scratch.data_ptr() if x is not None else None,
        rows, count // rows, zero_expert,
    )
    if status:
        raise RuntimeError((status, lib.paged_error().decode()))
    phase = "capture" if torch.xpu.is_current_stream_capturing() else "eager"
    key = (value.device, phase, "paged", x is not None)
    if key not in _seen:
        _seen.add(key)
        logger.info(
            "GGUF paged expert preparation: device=%s phase=%s logical_zero=%d "
            "input_q8=%s; existing LRU/exchange retained, no fixed zero slot",
            value.device, phase, zero_expert, x is not None,
        )
    return replace(grouping, starts=starts), x is not None
