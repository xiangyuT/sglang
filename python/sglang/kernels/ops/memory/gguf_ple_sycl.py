# SPDX-License-Identifier: Apache-2.0
"""Opt-in IQ4_NL PLE lookup from a persistent pinned compressed host table."""
import ctypes
import logging
import os
from functools import lru_cache

import numpy as np
import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _library():
    path = os.environ.get("SGLANG_XPU_GGUF_PLE_LIB")
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Build and warm the PLE library before graph capture")
    lib = ctypes.CDLL(path)
    lib.xpu_gguf_ple_host_pointer.argtypes = [ctypes.c_void_p] * 2
    lib.xpu_gguf_ple_host_pointer.restype = ctypes.c_int
    lib.xpu_gguf_ple_gather.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_int] * 2
    lib.xpu_gguf_ple_gather.restype = ctypes.c_int
    lib.xpu_gguf_ple_error.restype = ctypes.c_char_p
    return lib


class PinnedPLE:
    def __init__(self, table):
        self.tables = (table,) if isinstance(table, torch.Tensor) else tuple(table)
        if not self.tables:
            raise ValueError("PLE requires at least one compressed host chunk")
        self.chunk_rows = self.tables[0].shape[0]
        if not (self.chunk_rows > 0
                and all(t.device.type == "cpu" and t.dtype == torch.uint8
                        and t.ndim == 2 and t.shape[1] == 90
                        and t.is_contiguous() and t.is_pinned()
                        and 0 < t.shape[0] <= self.chunk_rows for t in self.tables)
                and all(t.shape[0] == self.chunk_rows for t in self.tables[:-1])
                and torch.xpu.get_device_properties().device_id == 0xE223):
            raise ValueError("PLE requires a pinned contiguous IQ4_NL width160 table on B70")
        self.rows = sum(t.shape[0] for t in self.tables)
        self.device = torch.device("xpu", torch.xpu.current_device())
        self.lib = _library()
        if any(self.lib.xpu_gguf_ple_host_pointer(torch.xpu.current_stream().sycl_queue, t.data_ptr()) != 1 for t in self.tables):
            raise ValueError("PLE table must be a USM host allocation in this context")
        self.addresses = torch.tensor([t.data_ptr() for t in self.tables], dtype=torch.uint64, device=self.device)

    def gather(self, indices, out=None):
        if not (indices.device == self.device and indices.dtype in (torch.int32, torch.int64)
                and 0 < indices.numel() <= 16384 and indices.is_contiguous()):
            raise ValueError("PLE indices must be contiguous bounded XPU integer rows")
        shape = (*indices.shape, 160)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=indices.device)
        elif out.shape != shape or out.dtype != torch.bfloat16 or out.device != indices.device or not out.is_contiguous():
            raise ValueError("PLE output must be contiguous BF16 with the requested shape")
        stream = torch.xpu.current_stream(indices.device)
        rc = self.lib.xpu_gguf_ple_gather(stream.sycl_queue, self.addresses.data_ptr(), indices.data_ptr(),
                                           out.data_ptr(), self.rows, self.chunk_rows, indices.numel(),
                                           int(indices.dtype == torch.int64))
        if rc:
            raise RuntimeError((rc, self.lib.xpu_gguf_ple_error().decode()))
        for tensor in (indices, out, self.addresses):
            tensor.record_stream(stream)
        return out


def create_pinned_ple(source):
    if os.environ.get("SGLANG_XPU_GGUF_PLE_SYCL") != "1":
        return None
    if source.kind != 20 or source.width != 160 or source.row_bytes != 90:
        raise ValueError("Pinned PLE requires original IQ4_NL width160 rows")
    # Runtime host storage only: no model file or expanded weight artifact is written.
    # Bounded allocations avoid the pinned allocator's 32-GiB rounded request.
    chunk_rows = (1 << 30) // 90
    tables = []
    for start in range(0, source.rows, chunk_rows):
        stop = min(start + chunk_rows, source.rows)
        table = torch.empty((stop - start, 90), dtype=torch.uint8, device="cpu", pin_memory=True)
        np.copyto(table.numpy(), source._mapping[start:stop])
        tables.append(table)
    result = PinnedPLE(tables)
    logger.info("GGUF PLE SYCL: %d original compressed bytes in %d USM host chunks; no CPU index readback", source.rows * 90, len(tables))
    return result
