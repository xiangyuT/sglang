"""Opt-in B70 M4/M8 projection: SYCL FP8 quantization and sycl-tla FP16 DPAS.

Weights/scales are static and retain a 2 MiB FP16 operand cache per projection.
Warm on the capture stream first. Restart and recapture after weight, scale,
flag or library changes. Graph replay itself does not run Python version checks.
"""

import ctypes
import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_CACHE_ATTRIBUTE = "_sglang_dflash_small_tla_cache"


@lru_cache(maxsize=1)
def load_small_tla():
    path = envs.SGLANG_XPU_DFLASH_SMALL_TLA_LIB.get()
    if not path:
        raise RuntimeError("SYCL small TLA requires SGLANG_XPU_DFLASH_SMALL_TLA_LIB")
    library = ctypes.CDLL(path)
    try:
        library.dflash_small_tla_abi.argtypes = []
        library.dflash_small_tla_abi.restype = ctypes.c_int
        if library.dflash_small_tla_abi() != 1:
            raise RuntimeError("Unsupported SYCL small TLA ABI")
        for name in ("dflash_small_tla_error", "dflash_small_tla_quant_error"):
            fn = getattr(library, name)
            fn.argtypes = []
            fn.restype = ctypes.c_char_p
        for name, pointers, integers in (
            ("dflash_small_tla_pack", 4, 4),
            ("dflash_small_tla_gemm", 6, 2),
        ):
            fn = getattr(library, name)
            fn.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int] * integers
            fn.restype = ctypes.c_int
        library.dflash_small_tla_quantize.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ]
        library.dflash_small_tla_quantize.restype = ctypes.c_int
        if hasattr(library, "dflash_small_tla_gemm_m8"):
            library.dflash_small_tla_gemm_m8.argtypes = [ctypes.c_void_p] * 6 + [
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.dflash_small_tla_gemm_m8.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL small TLA symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_small_tla.py"
        ) from exc
    return library


@lru_cache(maxsize=2)
def _announce_projection(rows):
    logger.info(
        "Using SYCL quant + sycl-tla B70 M%d small projection: "
        "FP16 static cache, 2 MiB/weight (SG=4, GRF=256)",
        rows,
    )


def can_use_small_tla(x, weight, scales, block_size, input_scale=None, bias=None):
    if not (
        not torch.compiler.is_compiling()
        and x.device.type == "xpu"
        and input_scale is None
        and bias is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] in (4, 8)
        and x.shape[1] == 512
        and weight.shape == (2048, 512)
        and weight.dtype == torch.float8_e4m3fn
        and scales.shape == (16, 4)
        and scales.dtype == torch.float32
        and tuple(block_size) == (128, 128)
        and x.is_contiguous()
        and weight.is_contiguous()
        and all(s > 0 for s in scales.stride())
        and x.device == weight.device == scales.device
    ):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, scales)):
        return False
    return torch.xpu.get_device_properties(x.device).device_id == 0xE223


@dataclass
class _WeightCache:
    identity: tuple
    packed: torch.Tensor
    scale_owner: torch.Tensor
    library: object
    ready: object
    stream_queue: int
    captured: bool = False

    def use_stream(self, stream):
        queue = stream.sycl_queue
        if self.stream_queue != queue:
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm SYCL small TLA weights on the capture stream first"
                )
            stream.wait_event(self.ready)
            self.packed.record_stream(stream)
            self.stream_queue = queue


def _weight_cache(weight, scales):
    library = load_small_tla()
    identity = (
        weight.data_ptr(),
        weight._version,
        scales.data_ptr(),
        scales._version,
        tuple(scales.stride()),
        id(library),
    )
    entry = getattr(weight, _CACHE_ATTRIBUTE, None)
    stream = torch.xpu.current_stream(weight.device)
    if entry is not None and entry.identity == identity:
        entry.use_stream(stream)
        return entry
    if entry is not None and entry.captured:
        raise RuntimeError(
            "SYCL small TLA weights, scales or library changed after graph capture; "
            "restart and recapture"
        )
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SYCL small TLA weights before XPU graph capture")
    packed = torch.empty_like(weight, dtype=torch.float16)
    rc = library.dflash_small_tla_pack(
        stream.sycl_queue,
        weight.data_ptr(),
        scales.data_ptr(),
        packed.data_ptr(),
        2048,
        512,
        *scales.stride(),
    )
    if rc:
        raise RuntimeError(library.dflash_small_tla_error().decode())
    ready = torch.xpu.Event()
    ready.record(stream)
    entry = _WeightCache(identity, packed, scales, library, ready, stream.sycl_queue)
    setattr(weight, _CACHE_ATTRIBUTE, entry)
    return entry


def _quantize(x, library=None):
    """Block128 BF16 quantization; also exposed to focused bit-pattern tests."""
    if not (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[1] % 128 == 0
    ):
        raise ValueError("SYCL small TLA quantization requires contiguous BF16 rows")
    library = library if library is not None else load_small_tla()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (x.shape[0], x.shape[1] // 128), device=x.device, dtype=torch.float32
    )
    rc = library.dflash_small_tla_quantize(
        torch.xpu.current_stream(x.device).sycl_queue,
        x.data_ptr(),
        q.data_ptr(),
        scales.data_ptr(),
        x.numel() // 128,
        1e-10,
        448.0,
        2,
    )
    if rc:
        raise RuntimeError(library.dflash_small_tla_quant_error().decode())
    return q, scales


def quantize_bf16_block128(x):
    """Use the bundled quantizer without creating a projection weight cache."""
    q, scales = _quantize(x)
    stream = torch.xpu.current_stream(x.device)
    for tensor in (x, q, scales):
        tensor.record_stream(stream)
    return q, scales


def small_tla(x, weight, scales):
    if not can_use_small_tla(x, weight, scales, (128, 128)):
        raise ValueError("Unsupported B70 M4/M8 small TLA projection contract")
    rows = x.shape[0]
    library = load_small_tla()
    function = (
        library.dflash_small_tla_gemm
        if rows == 4
        else getattr(library, "dflash_small_tla_gemm_m8", None)
    )
    if function is None:
        raise RuntimeError(
            "SYCL small TLA library lacks M8 support; rebuild and recapture"
        )
    _announce_projection(rows)
    entry = _weight_cache(weight, scales)
    q, xs = _quantize(x, entry.library)
    prepared = torch.empty_like(x, dtype=torch.float16)
    out = torch.empty((rows, 2048), device=x.device, dtype=torch.bfloat16)
    stream = torch.xpu.current_stream(x.device)
    for tensor in (x, entry.packed, q, xs, prepared, out):
        tensor.record_stream(stream)
    rc = function(
        stream.sycl_queue,
        q.data_ptr(),
        entry.packed.data_ptr(),
        xs.data_ptr(),
        prepared.data_ptr(),
        out.data_ptr(),
        2048,
        512,
    )
    if rc:
        raise RuntimeError(entry.library.dflash_small_tla_error().decode())
    entry.captured |= torch.xpu.is_current_stream_capturing()
    return out
