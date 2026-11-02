# SPDX-License-Identifier: Apache-2.0
"""Reversible IQ2_S cache tiles and grouped route metadata for native DPAS."""

import os
from dataclasses import dataclass

import numpy as np
import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library


@dataclass(frozen=True)
class IQ2SGroupedRoutes:
    data: torch.Tensor
    route_slots: torch.Tensor
    group_slots: torch.Tensor
    starts: torch.Tensor
    token_rows: torch.Tensor
    destinations: torch.Tensor
    groups: int


def group_iq2s_routes(ids_cpu, resident, device):
    ids = np.asarray(ids_cpu)
    if not (
        ids.ndim == 2
        and ids.dtype.kind in "iu"
        and 1 <= ids.shape[0] <= 4
        and 1 <= ids.shape[1] <= 10
    ):
        raise ValueError("IQ2S route grouping requires one to four rows, top-k <= 10")
    flat = ids.reshape(-1)
    order = np.argsort(flat, kind="stable")
    experts, counts = np.unique(flat, return_counts=True)
    try:
        grouped_slots = np.array([resident[int(e)] for e in experts], dtype=np.int32)
        route_slots = np.array([resident[int(e)] for e in flat], dtype=np.int32)
    except KeyError as error:
        raise ValueError("An IQ2S route has no resident expert") from error
    if np.any(grouped_slots < 0) or np.unique(grouped_slots).size != experts.size:
        raise ValueError("Distinct selected experts require distinct resident slots")
    starts = np.concatenate(([0], counts.cumsum())).astype(np.int32)
    packed = np.concatenate(
        (route_slots, grouped_slots, starts, order // ids.shape[1], order)
    ).astype(np.int32)
    data = torch.from_numpy(packed).to(device)
    routes, groups = flat.size, experts.size
    offset = routes
    gs = data[offset : offset + groups]
    offset += groups
    st = data[offset : offset + groups + 1]
    offset += groups + 1
    tr = data[offset : offset + routes]
    offset += routes
    return IQ2SGroupedRoutes(
        data, data[:routes], gs, st, tr, data[offset : offset + routes], groups
    )


def repack_iq2s_cache(gate, up, slot_ids, *, to_packed, down=None):
    """Permute only selected slots, preserving their byte count and identity.

    Callers own a single active model stream and track each slot's current
    format. Selected slots must be distinct: overlapping in-place transforms
    would otherwise race between workgroups.
    """
    ids = tuple(int(i) for i in slot_ids)
    if not ids:
        return
    if not (
        gate.device.type == "xpu"
        and gate.dtype == up.dtype == torch.uint8
        and gate.ndim == up.ndim == 2
        and gate.shape == up.shape
        and gate.shape[1] == 524800
        and gate.device == up.device
        and gate.is_contiguous()
        and up.is_contiguous()
        and len(set(ids)) == len(ids)
        and min(ids) >= 0
        and max(ids) < gate.shape[0]
        and not torch._C._overlaps(gate, up)
        and torch.xpu.get_device_properties(gate.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 IQ2S in-place cache conversion")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_iq2s_pack"):
        raise ValueError("Rebuild the GGUF library for the IQ2S packed cache")
    indices = torch.tensor(ids, dtype=torch.int32, device=gate.device)
    stream = torch.xpu.current_stream(gate.device)
    if down is not None:
        if not (
            down.dtype == torch.uint8
            and down.device == gate.device
            and down.shape == (gate.shape[0], 460800)
            and down.is_contiguous()
            and not torch._C._overlaps(down, gate)
            and not torch._C._overlaps(down, up)
        ):
            raise ValueError("Unsupported packed Q2_0 expert-down cache")
        if not hasattr(library, "xpu_gguf_iq2s_pack_expert"):
            raise ValueError(
                "Rebuild the GGUF library for paired expert cache conversion"
            )
        status = library.xpu_gguf_iq2s_pack_expert(
            stream.sycl_queue,
            gate.data_ptr(),
            up.data_ptr(),
            down.data_ptr(),
            indices.data_ptr(),
            len(ids),
            gate.shape[0],
            int(not to_packed),
        )
        down.record_stream(stream)
    else:
        status = library.xpu_gguf_iq2s_pack(
            stream.sycl_queue,
            gate.data_ptr(),
            up.data_ptr(),
            indices.data_ptr(),
            len(ids),
            gate.shape[0],
            int(not to_packed),
        )
    if status:
        raise RuntimeError((status, library.xpu_gguf_error().decode()))
    for tensor in (gate, up, indices):
        tensor.record_stream(stream)
