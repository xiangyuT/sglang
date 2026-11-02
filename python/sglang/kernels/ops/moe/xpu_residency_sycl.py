"""Opt-in single-wave B70 expert LRU with the retained integer policy."""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.speculative.dflash_native import (
    native_window,
    require_native_window,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_sycl_planner(ids, weights, state, mapped_ids, mapped_weights):
    if not (
        ids.device.type == "xpu"
        and ids.ndim == 2
        and ids.shape == weights.shape
        and ids.shape[0] in (4, 8)
        and ids.shape[1] == 8
        and ids.dtype == torch.int32
        and weights.dtype == mapped_weights.dtype == torch.float32
        and state.slot_expert.numel() == 192
        and state.expert_slot.numel() == 256
        and state.slot_lastuse.numel() == 192
        and state.step.numel() == state.count.numel() == 1
        and min(
            state.src.numel(),
            state.dst.numel(),
            mapped_ids.numel(),
            mapped_weights.numel(),
        )
        >= ids.numel()
    ):
        return False
    integers = (
        state.step,
        state.slot_expert,
        state.expert_slot,
        state.slot_lastuse,
        state.src,
        state.dst,
        state.count,
        mapped_ids,
    )
    supported = (
        all(t.dtype == torch.int32 for t in integers)
        and all(
            t.is_contiguous() and t.device == ids.device
            for t in (ids, weights, mapped_weights, *integers)
        )
        and torch.xpu.get_device_properties(ids.device).device_id == 0xE223
    )
    if supported and ids.shape[0] == 4 and native_window(load_sycl_planner()) == 8:
        return False
    return supported


@lru_cache(maxsize=1)
def load_sycl_planner():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError(
            "SYCL residency planner requires SGLANG_XPU_DFLASH_NATIVE_LIB"
        )
    library = ctypes.CDLL(path)
    try:
        abi = library.dflash_planner_abi
        decide = library.dflash_planner_decide
        error = library.dflash_planner_error
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL planner symbols missing; rebuild with test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if abi() != 1:
        raise RuntimeError("Unsupported SYCL residency planner ABI")
    decide.argtypes = [ctypes.c_void_p] * 12
    decide.restype = ctypes.c_int
    error.argtypes = []
    error.restype = ctypes.c_char_p
    logger.info(
        "Using SYCL E=256/K=192/R=%d expert residency (SG=32)",
        native_window(library) * 8,
    )
    return library


def decide_sycl(ids, weights, state, mapped_ids, mapped_weights):
    library = load_sycl_planner()
    require_native_window(library, ids.shape[0])
    tensors = (
        ids,
        weights,
        state.step,
        state.slot_expert,
        state.expert_slot,
        state.slot_lastuse,
        state.src,
        state.dst,
        state.count,
        mapped_ids,
        mapped_weights,
    )
    code = library.dflash_planner_decide(
        torch.xpu.current_stream().sycl_queue, *[t.data_ptr() for t in tensors]
    )
    if code:
        raise RuntimeError(library.dflash_planner_error().decode())
