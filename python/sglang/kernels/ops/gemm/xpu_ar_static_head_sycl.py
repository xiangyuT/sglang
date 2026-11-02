"""Opt-in B70 single-row BF16 head with a lossless static weight layout.

Build test/manual/kernels/build_xpu_ar_static_head.py and warm before capture.
Low bytes remain verbatim; a 256-value block stores either signed 3-bit high
byte deltas or full high bytes. The original weight is retained for fallback:
this reduces reads, adds GPU storage, and is not a VRAM-saving feature.

The runtime covers the unquantized 248320x2048 head in ordinary AR. Its FP32
reduction can differ from native GEMV at a BF16 rounding boundary. Online
weight updates are rejected by the engine while the startup flag is enabled.
Direct external weight writes, device moves, and library/flag changes require
restart and graph recapture; Python checks do not run during graph replay.
"""

import ctypes
import logging
import os
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_ar_static_head(hidden, weight):
    if not (
        hidden.shape == (1, 2048)
        and weight.shape == (248320, 2048)
        and hidden.dtype == weight.dtype == torch.bfloat16
        and hidden.device.type == "xpu"
        and hidden.device == weight.device
        and hidden.is_contiguous()
        and weight.is_contiguous()
        and hidden.data_ptr() % 4 == weight.data_ptr() % 4 == 0
    ):
        return False
    if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
        return False
    return torch.xpu.get_device_properties(hidden.device).device_id == 0xE223


@lru_cache(maxsize=1)
def _load_library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL_LIB")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the static packed head before XPU graph capture")
    library = ctypes.CDLL(path)
    try:
        library.ar_static_head_abi.argtypes = []
        library.ar_static_head_abi.restype = ctypes.c_int
        if library.ar_static_head_abi() != 1:
            raise RuntimeError("Unsupported static packed head ABI; rebuild it")
        library.ar_static_head_error.argtypes = []
        library.ar_static_head_error.restype = ctypes.c_char_p
        for name, pointers, integers in (
            ("metadata", 4, 1),
            ("pack", 6, 1),
            ("verify", 7, 1),
            ("gemv", 8, 4),
        ):
            function = getattr(library, "ar_static_head_" + name)
            function.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int] * integers
            function.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("Library lacks static packed head ABI; rebuild it") from exc
    return library


def _version(weight):
    try:
        return weight._version
    except RuntimeError:
        # Inference tensors have no version counter. The engine rejects online
        # updates independently of this diagnostic check, including before warmup.
        return None


class PackedBf16Head:
    """Prepared static layout; Nx2048, N divisible by eight, B70 only."""

    def __init__(self, weight):
        if not (
            weight.device.type == "xpu"
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.shape[1] == 2048
            and 0 < weight.shape[0] < 2**21
            and weight.shape[0] % 8 == 0
            and weight.is_contiguous()
            and weight.data_ptr() % 4 == 0
            and torch.xpu.get_device_properties(weight.device).device_id == 0xE223
        ):
            raise ValueError("Unsupported static BF16 head layout")
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Warm the static packed head before XPU graph capture")
        self.weight = weight
        self.pointer = weight.data_ptr()
        self.version = _version(weight)
        self.library = _load_library(
            envs.SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL_LIB.get()
        )
        self.blocks = weight.numel() // 256
        self.headers = torch.empty(self.blocks, dtype=torch.int32, device=weight.device)
        counter = torch.zeros(1, dtype=torch.int32, device=weight.device)
        self._call("metadata", weight, self.headers, counter, integers=(self.blocks,))
        # One preparation-time synchronization sizes the exact fallback storage.
        self.bad_blocks = int(counter[0])
        self.low = torch.empty(weight.numel(), dtype=torch.uint8, device=weight.device)
        self.codes = torch.empty(
            weight.numel() // 2, dtype=torch.uint8, device=weight.device
        )
        self.fallback = torch.empty(
            max(1, self.bad_blocks * 256), dtype=torch.uint8, device=weight.device
        )
        self._call("pack", weight, *self.buffers, integers=(self.blocks,))
        if self.verify() != 0:
            raise RuntimeError("Static packed head did not reconstruct weight bits")

    @property
    def buffers(self):
        return self.low, self.codes, self.headers, self.fallback

    def _call(self, name, *tensors, integers):
        stream = torch.xpu.current_stream(self.weight.device)
        status = getattr(self.library, "ar_static_head_" + name)(
            stream.sycl_queue, *[t.data_ptr() for t in tensors], *integers
        )
        if status:
            raise RuntimeError(self.library.ar_static_head_error().decode())
        for tensor in tensors:
            tensor.record_stream(stream)

    def verify(self):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Verify static packed head outside graph capture")
        errors = torch.zeros(1, dtype=torch.int32, device=self.weight.device)
        self._call(
            "verify", self.weight, *self.buffers, errors, integers=(self.blocks,)
        )
        return int(errors[0])

    def gemv(self, hidden, *, packed=True):
        if not (
            hidden.shape == (1, 2048)
            and hidden.dtype == torch.bfloat16
            and hidden.device == self.weight.device
            and hidden.is_contiguous()
            and hidden.data_ptr() % 4 == 0
        ):
            raise ValueError("Unsupported static BF16 head activation")
        if (
            self.weight.data_ptr() != self.pointer
            or _version(self.weight) != self.version
        ):
            raise RuntimeError(
                "Static packed head weight changed; restart and recapture"
            )
        out = torch.empty(
            (1, self.weight.shape[0]), dtype=torch.bfloat16, device=hidden.device
        )
        self._call(
            "gemv",
            hidden,
            self.weight,
            *self.buffers,
            out,
            integers=(self.weight.shape[0], 8, int(packed), 0),
        )
        return out

    def storage(self):
        return dict(
            raw_bytes=self.weight.numel() * 2,
            encoded_allocated_bytes=sum(
                t.numel() * t.element_size() for t in self.buffers
            ),
            logical_streamed_bytes=(
                self.low.numel()
                + (self.blocks - self.bad_blocks) * 128
                + self.headers.numel() * 4
                + self.bad_blocks * 256
            ),
            bad_blocks=self.bad_blocks,
            total_blocks=self.blocks,
            raw_weight_retained=True,
        )


def xpu_ar_static_head(hidden, layer, *, ar246_decode=False, ar449_decode=False, ar449_output=None):
    weight = layer.weight
    if not can_use_ar_static_head(hidden, weight):
        raise ValueError("Unsupported B70 AR static head layout")
    prepared = getattr(layer, "_xpu_ar_static_packed_head", None)
    if prepared is None:
        prepared = PackedBf16Head(weight)
        layer._xpu_ar_static_packed_head = prepared
        logger.info(
            "Using B70 AR static lossless BF16 packed head: N=248320 K=2048 "
            "row8/plain FP32, SG32/GRF256; storage=%s",
            prepared.storage(),
        )
    if prepared.weight is not weight:
        raise RuntimeError("Static packed head weight replaced; restart and recapture")
    if ar246_decode and ar449_decode and os.environ.get('CODEX_AR449_HEAD')=='1':
        if weight.data_ptr()!=prepared.pointer or _version(weight)!=prepared.version:
            raise RuntimeError('Static packed head weight changed; restart and recapture')
        from sglang.srt.layers.ar449_head import try_run as q5_run
        output=q5_run(prepared,hidden,output=ar449_output)
        if output is not None:return output
    if ar246_decode and os.environ.get('CODEX_AR246_HEAD')=='1':
        if weight.data_ptr()!=prepared.pointer or _version(weight)!=prepared.version:
            raise RuntimeError('Static packed head weight changed; restart and recapture')
        from sglang.srt.layers.ar246_head import try_run
        output=try_run(prepared,hidden)
        if output is not None:return output
    return prepared.gemv(hidden)
