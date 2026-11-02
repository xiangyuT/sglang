"""Opt-in B70 AR K192 gate/up with device-selected hit/miss row tiles.

Build with test/manual/kernels/build_xpu_ar_expert_gate.py. SiLU explicitly
uses the retained native reciprocal; BF16 conversion boundaries are preserved.
Warm the library before capture; restart and recapture after changing it.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_ar_expert_gate(x, w, s, ids, a, hosts, state):
    tensors = (x, w, s, ids, a, hosts, state.dst, state.src, state.count)
    if not (
        x.shape == (1, 2048)
        and w.shape == (192, 1024, 2048)
        and s.shape == (192, 8, 16)
        and ids.shape in ((8,), (1, 8))
        and a.shape == (8, 512)
        and hosts.shape == (4,)
        and state.dst.shape == state.src.shape
        and state.dst.shape in ((8,), (192,))
        and state.count.shape == (1,)
        and x.dtype == a.dtype == torch.bfloat16
        and w.dtype == torch.float8_e4m3fn
        and s.dtype == torch.float32
        and hosts.dtype == torch.uint64
        and ids.dtype
        == state.dst.dtype
        == state.src.dtype
        == state.count.dtype
        == torch.int32
        and x.device.type == "xpu"
        and all(t.device == x.device and t.is_contiguous() for t in tensors)
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    return torch.xpu.get_device_properties(x.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _load_library(path):
    if not path:
        raise RuntimeError(
            "Set SGLANG_XPU_AR_EXPERT_GATE_SYCL_LIB to the built library"
        )
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the SYCL AR expert gate before XPU graph capture")
    library = ctypes.CDLL(path)
    try:
        library.ar_expert_gate_abi.argtypes = []
        library.ar_expert_gate_abi.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("Library lacks AR expert gate ABI; rebuild it") from exc
    if library.ar_expert_gate_abi() != 1:
        raise RuntimeError("Unsupported SYCL AR expert gate ABI; rebuild it")
    library.ar_expert_gate_error.argtypes = []
    library.ar_expert_gate_error.restype = ctypes.c_char_p
    library.ar_expert_gate_run.argtypes = [ctypes.c_void_p] * 10
    library.ar_expert_gate_run.restype = ctypes.c_int
    logger.info("Using B70 AR K192 SYCL expert gate/up, rows2/4 SG32/GRF256")
    return library


def xpu_ar_expert_gate(x, w, s, ids, a, hosts, state):
    """Write gate/up and missed raw weights; caller gathers FP32 scales first.

    Hosts are pinned USM stores, with explicit W13,W2,S13,S2 pointer order.
    Device plans contain at most eight distinct slots and remain live through
    replay. Duplicate routes have one cache writer; the next down kernel is
    ordered on the same stream. Input values are never read by Python.
    """
    if not can_use_ar_expert_gate(x, w, s, ids, a, hosts, state):
        raise ValueError("Unsupported B70 AR K192 FP8 expert gate/up layout")
    library = _load_library(envs.SGLANG_XPU_AR_EXPERT_GATE_SYCL_LIB.get())
    stream = torch.xpu.current_stream(x.device)
    tensors = (x, w, s, ids, a, hosts, state.dst, state.src, state.count)
    status = library.ar_expert_gate_run(
        stream.sycl_queue, *[t.data_ptr() for t in tensors]
    )
    if status:
        raise RuntimeError(library.ar_expert_gate_error().decode())
    for tensor in tensors:
        tensor.record_stream(stream)
