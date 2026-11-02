# SPDX-License-Identifier: Apache-2.0
"""Bounded row access to original compressed GGUF tensors.

The file stays compressed and read-only. Advanced indexing copies only selected
rows, so a PLE table or an expert source need not be expanded or pinned in full.
"""

from __future__ import annotations

import numpy as np
import torch
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY


class CompressedGGUFRows:
    def __init__(self, shard, tensor):
        if len(tensor.shape) < 2 or not any(t is tensor for t in shard.tensors):
            raise ValueError("Expected a matrix belonging to the GGUF shard")
        self.width = tensor.shape[0]
        self.rows = tensor.elements // self.width
        self.kind = tensor.type_id
        block, size = BLOCK_GEOMETRY[tensor.type_name]
        if self.width % block:
            raise ValueError("GGUF rows must contain whole quantization blocks")
        self.row_bytes = self.width // block * size
        self.shard, self.tensor = shard, tensor
        self._mapping = np.memmap(
            shard.path,
            mode="r",
            dtype=np.uint8,
            offset=shard.data_start + tensor.offset,
            shape=(self.rows, self.row_bytes),
        )

    def packed_rows(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.rows):
            raise ValueError("GGUF row IDs must be integer indices within the table")
        packed = self._mapping[ids]
        # Advanced indexing already copies these rows. Keep a private writable
        # buffer for scalar/basic selections as well, without copying a whole
        # expert's already-owned advanced-index result a second time.
        if not packed.flags.writeable or np.shares_memory(packed, self._mapping):
            packed = np.array(packed, copy=True)
        return torch.from_numpy(packed)

    def packed_row_range(self, start, stop, *, pin_memory=False):
        """Copy one contiguous row range into an owned CPU transfer buffer."""
        if (
            not isinstance(start, (int, np.integer))
            or not isinstance(stop, (int, np.integer))
            or start < 0
            or stop < start
            or stop > self.rows
        ):
            raise ValueError("GGUF row range must stay within the table")
        packed = torch.empty(
            (stop - start, self.row_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        np.copyto(packed.numpy(), self._mapping[start:stop])
        return packed

    def decode_rows(self, ids, *, device, dtype=torch.bfloat16):
        packed = self.packed_rows(ids)
        shape = (*packed.shape[:-1], self.width)
        if packed.numel() == 0:
            return torch.empty(shape, device=device, dtype=dtype)
        if torch.device(device).type == "xpu":
            from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf

            # Decode in FP32 before converting to the model activation dtype.
            values = dequantize_gguf(
                packed.to(device).reshape(-1), self.kind, dtype=torch.float32
            ).view(shape)
        else:
            import gguf

            raw = packed.numpy().reshape(-1)
            if self.kind == 42:
                blocks = raw.reshape(-1, 18)
                scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
                lane = np.arange(64)
                code = (blocks[:, 2 + lane // 4] >> (lane % 4 * 2)) & 3
                decoded = (code.astype(np.float32) - 1) * scale
            else:
                decoded = gguf.dequantize(raw, gguf.GGMLQuantizationType(self.kind))
            values = torch.from_numpy(decoded.reshape(shape)).to(device)
        return values.to(dtype)


class CompressedGGUFExpertSlice(CompressedGGUFRows):
    """A row section inside each expert of a fused compressed tensor.

    The MTP checkpoint stores gate and up consecutively inside every expert.
    Keep their original file mapping shared; only requested rows are copied.
    """

    def __init__(self, source, start, stop):
        if (
            len(source.tensor.shape) != 3
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= source.tensor.shape[1]
        ):
            raise ValueError("Expert slice must select rows within each expert")
        self.source = source
        self.start = start
        self.rows_per_expert = stop - start
        self.source_rows_per_expert = source.tensor.shape[1]
        self.rows = self.rows_per_expert * source.tensor.shape[2]
        self.width, self.row_bytes, self.kind = (
            source.width,
            source.row_bytes,
            source.kind,
        )

    def packed_rows(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.rows):
            raise ValueError("GGUF row IDs must be integer indices within the table")
        physical = (
            ids // self.rows_per_expert * self.source_rows_per_expert
            + self.start
            + ids % self.rows_per_expert
        )
        return self.source.packed_rows(physical)

    def packed_row_range(self, start, stop, *, pin_memory=False):
        if (
            not isinstance(start, (int, np.integer))
            or not isinstance(stop, (int, np.integer))
            or start < 0
            or stop < start
            or stop > self.rows
        ):
            raise ValueError("GGUF row range must stay within the expert slice")
        if (
            start < stop
            and start // self.rows_per_expert == (stop - 1) // self.rows_per_expert
        ):
            physical = (
                start // self.rows_per_expert * self.source_rows_per_expert
                + self.start
                + start % self.rows_per_expert
            )
            return self.source.packed_row_range(
                physical, physical + stop - start, pin_memory=pin_memory
            )
        packed = torch.empty(
            (stop - start, self.row_bytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        cursor = start
        while cursor < stop:
            end = min(stop, (cursor // self.rows_per_expert + 1) * self.rows_per_expert)
            packed[cursor - start : end - start].copy_(
                self.packed_row_range(cursor, end)
            )
            cursor = end
        return packed
