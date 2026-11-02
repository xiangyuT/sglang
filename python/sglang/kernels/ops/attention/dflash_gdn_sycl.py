"""Opt-in B70 M4/M8 GDN verify/commit with pool-owned FP32 delta records.

The separate native library uses precise math. Records are one in-flight
linear window per pool, so concurrent requests, PP and radix tracking are
rejected. The original conv rollback and non-verify paths remain with their
existing backends. Restart and recapture after changing flags or libraries.
"""

import ctypes
import logging
from dataclasses import dataclass
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
STATE_ELEMENTS = 524288
RECORD_ELEMENTS = 66048


@dataclass(frozen=True)
class GdnScratch:
    """A verify window tied to one immutable source layout, or one layer view."""

    source: torch.Tensor
    records: torch.Tensor
    source_index: torch.Tensor
    output: torch.Tensor
    source_layout: tuple

    @staticmethod
    def layout(source):
        return (
            source.data_ptr(),
            tuple(source.shape),
            tuple(source.stride()),
            source.dtype,
            source.device,
        )

    @classmethod
    def create(cls, source, records, source_index, output):
        return cls(source, records, source_index, output, cls.layout(source))

    @property
    def buffers(self):
        # Source is charged by the checkpoint pool, not charged a second time.
        return [self.records, self.source_index, self.output]

    @property
    def steps(self):
        return self.output.shape[-3]

    def at_layer_idx(self, layer):
        return type(self).create(
            self.source[layer],
            self.records[layer],
            self.source_index[layer],
            self.output[layer],
        )

    def matches_source(self, source):
        return self.source_layout == self.layout(source)


@lru_cache(maxsize=1)
def load_gdn():
    path = envs.SGLANG_XPU_DFLASH_GDN_LIB.get()
    if not path:
        raise RuntimeError(
            "SYCL GDN requires SGLANG_XPU_DFLASH_GDN_LIB; build with "
            "test/manual/kernels/build_xpu_dflash_gdn.py"
        )
    library = ctypes.CDLL(path)
    try:
        library.dflash_gdn_abi.argtypes = []
        library.dflash_gdn_abi.restype = ctypes.c_int
        if library.dflash_gdn_abi() != 1:
            raise RuntimeError("Unsupported SYCL GDN ABI")
        library.dflash_gdn_error.argtypes = []
        library.dflash_gdn_error.restype = ctypes.c_char_p
        library.dflash_gdn_verify.argtypes = (
            [ctypes.c_void_p] * 15
            + [ctypes.c_int] * 3
            + [ctypes.c_int64]
            + [ctypes.c_int] * 2
        )
        library.dflash_gdn_verify.restype = ctypes.c_int
        library.dflash_gdn_commit.argtypes = (
            [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_int64] * 5
        )
        library.dflash_gdn_commit.restype = ctypes.c_int
        if hasattr(library, "dflash_gdn_supports_steps"):
            library.dflash_gdn_supports_steps.argtypes = [ctypes.c_int]
            library.dflash_gdn_supports_steps.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError("SYCL GDN symbols missing; rebuild the GDN library") from exc
    return library


def _checked(rc):
    if rc:
        raise RuntimeError(load_gdn().dflash_gdn_error().decode())


def validate_runtime():
    from sglang.srt.runtime_context import (
        get_memory,
        get_parallel,
        get_schedule,
        get_spec,
    )

    spec = get_spec()
    algorithm = getattr(spec.speculative_algorithm, "name", spec.speculative_algorithm)
    if not (
        envs.SGLANG_XPU_PAGED_DFLASH.get()
        and algorithm == "DFLASH"
        and spec.speculative_num_draft_tokens in (4, 8)
        and get_parallel().tp_size == get_parallel().pp_size == 1
        and get_schedule().max_running_requests == 1
        and get_memory().disable_radix_cache
    ):
        raise ValueError(
            "SYCL GDN requires DFLASH M4/M8, TP1/PP1, one request, radix off"
        )


def allocate_scratch(temporal, steps, *, is_kda, replayssm, envelope):
    """Allocate once inside the owning pool's memory region, before capture."""
    validate_runtime()
    if not (
        temporal.device.type == "xpu"
        and torch.xpu.get_device_properties(temporal.device).device_id == 0xE223
        and temporal.dtype == torch.float32
        and temporal.ndim == 5
        and 1 <= temporal.shape[0] <= 30
        and tuple(temporal.shape[2:]) == (32, 128, 128)
        and tuple(temporal.stride()[2:]) == (16384, 128, 1)
        and steps in (4, 8)
        and not (is_kda or replayssm or envelope)
        and not torch.xpu.is_current_stream_capturing()
    ):
        raise ValueError(
            "SYCL GDN requires B70 FP32 GDN state and the M4/M8 snapshot pool"
        )
    library = load_gdn()
    if steps == 8 and not (
        hasattr(library, "dflash_gdn_supports_steps")
        and library.dflash_gdn_supports_steps(steps) == 1
    ):
        raise RuntimeError("SYCL GDN library lacks M8 support; rebuild and recapture")
    layers, device = temporal.shape[0], temporal.device
    records = torch.empty((layers, RECORD_ELEMENTS), device=device, dtype=torch.float32)
    indices = torch.empty((layers, 1), device=device, dtype=torch.int32)
    output = torch.empty(
        (layers, 1, steps, 32, 128), device=device, dtype=torch.bfloat16
    )
    logger.info(
        "Using SYCL B70 DFlash GDN M=%d/BV=2; pool-owned delta replay, "
        "FP32 state, original conv rollback; checkpoint allocation retained",
        steps,
    )
    return GdnScratch.create(temporal, records, indices, output)


def verify_gdn(
    layer_cache,
    *,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    indices,
    cu_seqlens,
    steps,
    native_bf16_gdn,
    retrieve_parent_token,
    stable_rows,
):
    """Selected raw verify cannot fall back to stale snapshot contents."""
    source = layer_cache.temporal
    scratch = layer_cache.dflash_sycl
    if not scratch.matches_source(source):
        raise ValueError(
            "SYCL GDN layer/source mapping changed; rebuild the pool and graphs"
        )
    records, saved, output = scratch.records, scratch.source_index, scratch.output
    if not (
        native_bf16_gdn
        and steps in (4, 8)
        and steps == scratch.steps
        and not stable_rows
        and retrieve_parent_token is None
        and tuple(q.shape) == tuple(k.shape) == (1, steps, 16, 128)
        and tuple(v.shape) == (1, steps, 32, 128)
        and all(tuple(t.stride()) == (8192 * steps, 8192, 128, 1) for t in (q, k, v))
        and all(
            t.shape == (steps, 32) and t.stride(0) in (32, 64) and t.stride(1) == 1
            for t in (a, b)
        )
        and all(t.dtype == torch.bfloat16 for t in (q, k, v, a, b, dt_bias, output))
        and A_log.dtype == source.dtype == records.dtype == torch.float32
        and all(t.shape == (32,) and t.is_contiguous() for t in (A_log, dt_bias))
        and source.ndim == 4
        and tuple(source.shape[1:]) == (32, 128, 128)
        and tuple(source.stride()[1:]) == (16384, 128, 1)
        and source.stride(0) >= STATE_ELEMENTS
        and records.shape == (RECORD_ELEMENTS,)
        and records.is_contiguous()
        and saved.shape == indices.shape == (1,)
        and saved.dtype == indices.dtype == torch.int32
        and saved.is_contiguous()
        and indices.is_contiguous()
        and cu_seqlens.shape == (2,)
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.is_contiguous()
        and output.shape == (1, steps, 32, 128)
        and output.is_contiguous()
        and all(
            t.device == q.device
            for t in (
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                source,
                records,
                saved,
                indices,
                cu_seqlens,
                output,
            )
        )
    ):
        raise ValueError(
            "Unsupported selected SYCL GDN verify layout; snapshots are stale"
        )
    tensors = [
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        source,
        indices,
        None,
        None,
        output,
        records,
        saved,
    ]
    _checked(
        load_gdn().dflash_gdn_verify(
            torch.xpu.current_stream().sycl_queue,
            *[t.data_ptr() if t is not None else None for t in tensors],
            steps,
            1,
            0,
            source.stride(0),
            a.stride(0),
            b.stride(0),
        )
    )
    return output


def commit_gdn(cache, destination, last_correct, track, track_steps, src_raw):
    """Commit the accepted state; caller retains conv rollback and PLE handling."""
    source = cache.temporal
    scratch = cache.dflash_sycl
    if not scratch.matches_source(source):
        raise ValueError(
            "SYCL GDN pool/source mapping changed; rebuild the pool and graphs"
        )
    records, saved = scratch.records, scratch.source_index
    if not (
        track is None
        and track_steps is None
        and src_raw is None
        and source.dtype == records.dtype == torch.float32
        and source.ndim == 5
        and 1 <= source.shape[0] <= 30
        and tuple(source.shape[2:]) == (32, 128, 128)
        and tuple(source.stride()[2:]) == (16384, 128, 1)
        and source.stride(1) >= STATE_ELEMENTS
        and scratch.steps in (4, 8)
        and scratch.output.shape == (source.shape[0], 1, scratch.steps, 32, 128)
        and records.shape == (source.shape[0], RECORD_ELEMENTS)
        and saved.shape == (source.shape[0], 1)
        and saved.dtype == torch.int32
        and records.is_contiguous()
        and saved.is_contiguous()
        and destination.shape == last_correct.shape == (1,)
        and destination.dtype in (torch.int32, torch.int64)
        and last_correct.dtype in (torch.int32, torch.int64)
        and all(
            t.device == source.device
            for t in (records, saved, destination, last_correct)
        )
    ):
        raise ValueError("Unsupported selected SYCL GDN commit/tracking layout")
    destination = destination.to(torch.int32).contiguous()
    last_correct = last_correct.to(torch.int32).contiguous()
    tensors = [source, saved, records, source, destination, last_correct]
    _checked(
        load_gdn().dflash_gdn_commit(
            torch.xpu.current_stream().sycl_queue,
            *[t.data_ptr() for t in tensors],
            scratch.steps,
            1,
            source.shape[0],
            1,
            source.stride(1),
            source.stride(1),
            source.stride(0),
            records.stride(0),
            source.stride(0),
        )
    )
