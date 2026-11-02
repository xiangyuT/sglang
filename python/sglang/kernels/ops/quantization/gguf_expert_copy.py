# SPDX-License-Identifier: Apache-2.0
"""Bounded pinned expert staging on the current Torch XPU stream.

The cache policy and compressed layouts remain owned by the MoE method. This
adapter replaces only allocating/copying three CPU buffers and submitting
three Torch H2D copies for one miss. It retains one 11.54 MiB host ring per
stream/library, with no additional device staging allocation or scatter.
"""

import atexit
import logging
import os
from functools import lru_cache

import numpy as np
import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.srt.model_loader.gguf_rows import (
    CompressedGGUFExpertSlice,
    CompressedGGUFRows,
)

logger = logging.getLogger(__name__)
_RING_SLOTS = 8
_RING_STRIDE = 1511424  # Round the 1,510,400-byte IQ2S expert up to 4096.
_pools = {}


class _CopyRing:
    def __init__(self, library, stream, slots=_RING_SLOTS):
        self.library = library
        self.stream = stream
        self.storage = torch.empty(
            (slots, _RING_STRIDE), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.handle = library.xpu_gguf_copy_create(
            stream.sycl_queue, self.storage.data_ptr(), _RING_STRIDE, slots
        )
        if not self.handle:
            raise RuntimeError(library.xpu_gguf_copy_error().decode())

    def close(self):
        if self.handle:
            status = self.library.xpu_gguf_copy_destroy(self.handle)
            if status:
                # Keep both the handle and borrowed pinned memory on failure.
                raise RuntimeError(self.library.xpu_gguf_copy_error().decode())
            self.handle = None


def _close_pools():
    for pool in _pools.values():
        pool.close()
    _pools.clear()


atexit.register(_close_pools)


@lru_cache(maxsize=1)
def _log_submission():
    logger.info("Native GGUF expert H2D submitted; bounded pinned ring, original bytes")


def _source_description(source, experts):
    if type(source) is CompressedGGUFExpertSlice:
        mapping = source.source._mapping
        offset = source.start * source.row_bytes
        stride = source.source_rows_per_expert * source.row_bytes
    elif type(source) is CompressedGGUFRows:
        mapping = source._mapping
        offset = 0
        stride = source.rows // experts * source.row_bytes
    else:
        return None
    size = source.rows // experts * source.row_bytes
    if not (
        isinstance(mapping, np.ndarray)
        and mapping.dtype == np.uint8
        and mapping.flags.c_contiguous
        and source.rows % experts == 0
        and 0 < size <= _RING_STRIDE
        and offset + (experts - 1) * stride + size <= mapping.nbytes
    ):
        return None
    return (mapping.ctypes.data + offset, stride, size)


class ExpertCopies:
    def __init__(self, sources, descriptions, library, path, device, experts):
        self.sources = sources  # Keep the read-only mappings alive.
        self.descriptions = descriptions
        self.library = library
        self.path = path
        self.device = device
        self.experts = experts
        self.expert_bytes = sum(row[2] for row in descriptions)
        self.submissions = 0

    def copy(self, layer, expert, slot):
        if not 0 <= expert < self.experts:
            raise ValueError("Expert source index outside the mapped table")
        buffers = (layer.gguf_gate, layer.gguf_up, layer.gguf_down)
        if not 0 <= slot < buffers[0].shape[0]:
            raise ValueError("Expert destination outside the cache")
        stream = torch.xpu.current_stream(self.device)
        key = (self.path, self.device, stream.sycl_queue)
        if key not in _pools:
            _pools[key] = _CopyRing(self.library, stream)
        pool = _pools[key]
        destinations = [
            buffer.data_ptr() + slot * row[2]
            for buffer, row in zip(buffers, self.descriptions)
        ]
        sources = [row[0] + expert * row[1] for row in self.descriptions]
        sizes = [row[2] for row in self.descriptions]
        status = self.library.xpu_gguf_copy_expert(
            pool.handle, *destinations, *sources, *sizes
        )
        if status:
            raise RuntimeError(self.library.xpu_gguf_copy_error().decode())
        for buffer in buffers:
            buffer.record_stream(stream)
        self.submissions += 1
        _log_submission()

    def group_routes(self, ids_cpu, resident):
        """Prepare the same packed route image without a Torch transfer sync."""
        from sglang.kernels.ops.quantization.gguf_iq2s_pvr import IQ2SGroupedRoutes

        if not hasattr(self.library, "xpu_gguf_copy_group"):
            return None
        ids = np.asarray(ids_cpu)
        if (
            ids.dtype.kind not in "iu"
            or ids.ndim != 2
            or not 1 <= ids.shape[0] <= 4
            or not 1 <= ids.shape[1] <= 10
        ):
            raise ValueError("Native IQ2S route grouping requires rows<=4/topk<=10")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        groups = len(resident)
        experts = np.fromiter(resident.keys(), dtype=np.int32, count=groups)
        slots = np.fromiter(resident.values(), dtype=np.int32, count=groups)
        routes = ids.size
        data = torch.empty(
            3 * routes + 2 * groups + 1, device=self.device, dtype=torch.int32
        )
        stream = torch.xpu.current_stream(self.device)
        key = (self.path, self.device, stream.sycl_queue)
        if key not in _pools:
            _pools[key] = _CopyRing(self.library, stream)
        status = self.library.xpu_gguf_copy_group(
            _pools[key].handle,
            data.data_ptr(),
            ids.ctypes.data,
            experts.ctypes.data,
            slots.ctypes.data,
            ids.shape[0],
            ids.shape[1],
            groups,
            self.experts,
        )
        if status:
            if status == 2:
                raise ValueError("Invalid native IQ2S resident route mapping")
            raise RuntimeError(self.library.xpu_gguf_copy_error().decode())
        data.record_stream(stream)
        end_slots = routes + groups
        end_starts = end_slots + groups + 1
        end_rows = end_starts + routes
        return IQ2SGroupedRoutes(
            data,
            data[:routes],
            data[routes:end_slots],
            data[end_slots:end_starts],
            data[end_starts:end_rows],
            data[end_rows:],
            groups,
        )


def make_expert_copies(sources, layer, experts):
    """Resolve source geometry once at bind time; unsupported paths stay Torch."""
    path = os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")
    if not (
        os.environ.get("SGLANG_XPU_GGUF_NATIVE_STAGE") == "1"
        and path
        and len(sources) == 3
        and experts > 0
        and sources[0].kind in (16, 22, 29, 42)
        and sources[0].kind == sources[1].kind
        and sources[2].kind == 42
    ):
        return None
    descriptions = [_source_description(source, experts) for source in sources]
    if any(row is None for row in descriptions):
        return None
    if sum(row[2] for row in descriptions) > _RING_STRIDE:
        return None
    buffers = (layer.gguf_gate, layer.gguf_up, layer.gguf_down)
    device = buffers[0].device
    if not (
        device.type == "xpu"
        and all(
            buffer.device == device
            and buffer.dtype == torch.uint8
            and buffer.ndim == 2
            and buffer.shape[1] == row[2]
            and buffer.is_contiguous()
            for buffer, row in zip(buffers, descriptions)
        )
        and torch.xpu.get_device_properties(device).device_id == 0xE223
    ):
        return None
    library = _library(path)
    if not hasattr(library, "xpu_gguf_copy_create"):
        return None
    return ExpertCopies(sources, descriptions, library, path, device, experts)
