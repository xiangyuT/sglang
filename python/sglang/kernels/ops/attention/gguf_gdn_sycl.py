# SPDX-License-Identifier: Apache-2.0
"""Pool-owned NEXTN verification for full and TP2/TP4 GDN head partitions."""

import ctypes
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)
STATE_ELEMENTS = 48 * 128 * 128
RECORD_ELEMENTS = 82688


def scratch_bytes(heads, batch, layers=36):
    """Persistent delta records, live source IDs and BF16 verify outputs."""
    if heads not in (48, 24, 12) or not 1 <= batch <= 8 or layers != 36:
        raise ValueError("Unsupported GGUF GDN scratch geometry")
    return layers * batch * (RECORD_ELEMENTS * heads // 48 * 4 + 4 + 4 * heads * 128 * 2)


@dataclass(frozen=True)
class GGUFGdnScratch:
    source: torch.Tensor
    records: torch.Tensor
    source_index: torch.Tensor
    output: torch.Tensor
    source_layout: tuple

    @staticmethod
    def layout(source):
        return source.data_ptr(), tuple(source.shape), tuple(source.stride()), source.dtype, source.device

    @classmethod
    def create(cls, source, records, source_index, output):
        return cls(source, records, source_index, output, cls.layout(source))

    @property
    def buffers(self):
        return [self.records, self.source_index, self.output]

    def matches_source(self, source):
        return self.source_layout == self.layout(source)

    def at_layer_idx(self, layer):
        return type(self).create(self.source[layer], self.records[layer], self.source_index[layer], self.output[layer])


@lru_cache(maxsize=1)
def load_library():
    path = os.environ.get("SGLANG_XPU_GGUF_GDN_SYCL_LIB")
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm SGLANG_XPU_GGUF_GDN_SYCL_LIB before graph capture")
    library = ctypes.CDLL(path)
    for heads in (48, 24, 12):
        if not hasattr(library, f"nextn_gdn{heads}_verify"):
            continue
        verify = getattr(library, f"nextn_gdn{heads}_verify")
        verify.argtypes = ([ctypes.c_void_p] * 15 + [ctypes.c_int] * 3
                           + [ctypes.c_int64] * 2 + [ctypes.c_int] * 2)
        verify.restype = ctypes.c_int
        if hasattr(library, f"nextn_gdn{heads}_verify_batch"):
            batch_verify = getattr(library, f"nextn_gdn{heads}_verify_batch")
            batch_verify.argtypes = verify.argtypes + [ctypes.c_int]
            batch_verify.restype = ctypes.c_int
        commit = getattr(library, f"nextn_gdn{heads}_commit")
        commit.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
                           + [ctypes.c_int64] * 5)
        commit.restype = ctypes.c_int
        getattr(library, f"nextn_gdn{heads}_error").restype = ctypes.c_char_p
    return library


def checked(rc, heads=48):
    if rc:
        raise RuntimeError(getattr(load_library(), f"nextn_gdn{heads}_error")().decode())


def allocate_scratch(source, steps, *, is_kda, replayssm, envelope):
    from sglang.srt.runtime_context import get_exec, get_memory, get_model, get_parallel, get_schedule, get_spec

    spec = get_spec()
    parallel = get_parallel()
    tp = parallel.tp_size
    max_batch = get_schedule().max_running_requests
    supported_tp = tp == 1 or (
        os.environ.get("SGLANG_XPU_GGUF_TP_EP") == "1"
        and tp in (2, 4) and parallel.moe_ep_size == tp
    )
    heads = 48 // tp if supported_tp else 0
    algorithm = getattr(spec.speculative_algorithm, "name", spec.speculative_algorithm)
    if not (
        algorithm == "EAGLE"
        and not spec.speculative_adaptive
        and spec.speculative_eagle_topk == 1
        and get_model().quantization == "gguf"
        and supported_tp and parallel.pp_size == 1
        and isinstance(max_batch, int) and 1 <= max_batch <= 8
        and (get_memory().disable_radix_cache or get_exec().mamba.enable_mamba_extra_buffer)
        and not (is_kda or replayssm or envelope)
        and steps == 4
        and source.device.type == "xpu"
        and torch.xpu.get_device_properties(source.device).device_id == 0xE223
        and source.dtype == torch.float32
        and source.ndim == 5
        and source.shape[0] == 36
        and tuple(source.shape[2:]) == (heads, 128, 128)
        and tuple(source.stride()[2:]) == (16384, 128, 1)
        and not torch.xpu.is_current_stream_capturing()
    ):
        raise ValueError("GGUF SYCL GDN requires B70, TP1/2/4 with matching heads, PP1, BS1..8, fixed NEXTN3/window4, FP32 state and radix off or extra-buffer tracking")
    library = load_library()
    if not hasattr(library, f"nextn_gdn{heads}_verify"):
        raise ValueError("Rebuild GGUF GDN library for the selected TP head partition")
    if os.environ.get("SGLANG_XPU_GGUF_GDN_BATCH") == "1":
        if heads != 24 or not hasattr(library, "nextn_gdn24_verify_batch"):
            raise ValueError("GGUF batched GDN requires TP2 and a rebuilt GDN library")
    records = torch.empty((36, max_batch, RECORD_ELEMENTS * heads // 48), dtype=torch.float32, device=source.device)
    indices = torch.empty((36, max_batch), dtype=torch.int32, device=source.device)
    output = torch.empty((36, max_batch, 4, heads, 128), dtype=torch.bfloat16, device=source.device)
    logger.info("GGUF H%d/HV%d SYCL GDN enabled: 36 pool-owned delta records, batch capacity=%d, BV4; original conv/PLE rollback retained", heads // 3, heads, max_batch)
    return GGUFGdnScratch.create(source, records, indices, output)


def verify_gdn(layer_cache, *, q, k, v, a, b, A_log, dt_bias, indices, cu_seqlens,
               steps, native_bf16_gdn, retrieve_parent_token, stable_rows):
    source = layer_cache.temporal
    scratch = layer_cache.gguf_gdn_sycl
    heads = source.shape[1]
    batch = indices.numel()
    key_heads, mixed_width = heads // 3, heads // 3 * 5 * 128
    if not (
        scratch.matches_source(source)
        and native_bf16_gdn
        and steps == 4
        and retrieve_parent_token is None
        and not stable_rows
        and heads in (48, 24, 12)
        and 1 <= batch <= scratch.output.shape[0]
        and q.shape == k.shape == (1, batch * 4, key_heads, 128)
        and v.shape == (1, batch * 4, heads, 128)
        and all(t.stride(1) == mixed_width and t.stride(-1) == 1 for t in (q, k, v))
        and all(t.shape == (batch * 4, heads) and t.stride(0) in (heads, 2 * heads) and t.stride(1) == 1 for t in (a, b))
        and all(t.dtype == torch.bfloat16 for t in (q, k, v, a, b, dt_bias))
        and A_log.dtype == source.dtype == torch.float32
        and A_log.shape == dt_bias.shape == (heads,)
        and A_log.is_contiguous() and dt_bias.is_contiguous()
        and source.ndim == 4 and tuple(source.shape[1:]) == (heads, 128, 128)
        and tuple(source.stride()[1:]) == (16384, 128, 1)
        and indices.shape == (batch,)
        and indices.dtype in (torch.int32, torch.int64)
        and cu_seqlens.shape == (batch + 1,)
    ):
        raise ValueError("Unsupported selected GGUF GDN verify layout; restart without the flag")
    indices = indices.to(torch.int32).contiguous()
    stream = torch.xpu.current_stream(q.device)
    verify = getattr(load_library(), f"nextn_gdn{heads}_verify")
    # Batch only independent request verification. Accepted-prefix commit keeps
    # its existing checkpoint-before-destination ordering across all requests.
    if os.environ.get("SGLANG_XPU_GGUF_GDN_BATCH") == "1" and heads == 24 and batch > 1:
        batch_verify = getattr(load_library(), "nextn_gdn24_verify_batch", None)
        if batch_verify is None:
            raise ValueError("Rebuild GGUF GDN library for batched verify")
        tensors = [q, k, v, a, b, A_log, dt_bias, source, indices, None, None,
                   scratch.output, scratch.records, scratch.source_index]
        checked(batch_verify(
            stream.sycl_queue, *[t.data_ptr() if t is not None else None for t in tensors],
            4, 2, 0, source.stride(0), 0, a.stride(0), b.stride(0), batch,
        ), heads)
    else:
        for request in range(batch):
            first = request * 4
            tensors = [q[:, first:first + 4], k[:, first:first + 4], v[:, first:first + 4],
                       a[first:first + 4], b[first:first + 4], A_log, dt_bias, source,
                       indices[request:request + 1], None, None,
                       scratch.output[request], scratch.records[request],
                       scratch.source_index[request:request + 1]]
            checked(verify(
                stream.sycl_queue, *[t.data_ptr() if t is not None else None for t in tensors],
                4, 2, 0, source.stride(0), 0, a.stride(0), b.stride(0),
            ), heads)
    tensors = [q, k, v, a, b, A_log, dt_bias, source, indices,
               scratch.output, scratch.records, scratch.source_index]
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)
    return scratch.output[:batch].view(1, batch * 4, heads, 128)


def commit_gdn(cache, destination, last_correct, track, track_steps, src_raw):
    scratch, source = cache.gguf_gdn_sycl, cache.temporal
    batch = destination.numel()
    tracking = track is not None and track_steps is not None
    if not (scratch.matches_source(source) and src_raw is None
            and 1 <= batch <= scratch.output.shape[1]
            and destination.shape == last_correct.shape == (batch,)
            and ((track is None and track_steps is None) or (
                tracking and track.shape == track_steps.shape == (batch,)
                and all(t.device == source.device and t.dtype in (torch.int32, torch.int64)
                        for t in (track, track_steps))))):
        raise ValueError("Unsupported selected GGUF GDN commit layout")
    destination = destination.to(torch.int32).contiguous()
    last_correct = last_correct.to(torch.int32).contiguous()
    stream = torch.xpu.current_stream(source.device)
    heads = source.shape[2]
    commit = getattr(load_library(), f"nextn_gdn{heads}_commit")
    # Extra-buffer tracking owns distinct checkpoint slots. Both folds must
    # read the original live state: checkpoint every row BEFORE mutating any
    # live destination, otherwise the deltas would be applied twice. Negative
    # track steps are masked by the native kernel, including graph padding.
    targets = []
    if tracking:
        track = track.to(torch.int32).contiguous()
        track_steps = track_steps.to(torch.int32).contiguous()
        targets.append((track, track_steps))
    targets.append((destination, last_correct))
    for target, steps in targets:
        for request in range(batch):
            tensors = [source, scratch.source_index[:, request], scratch.records[:, request],
                       source, target[request:request + 1], steps[request:request + 1]]
            checked(commit(
                stream.sycl_queue, *[t.data_ptr() for t in tensors],
                4, 2, source.shape[0], scratch.source_index.stride(0),
                source.stride(1), source.stride(1), source.stride(0),
                scratch.records.stride(0), source.stride(0),
            ), heads)
    tensors = [source, scratch.source_index, scratch.records, destination, last_correct]
    for tensor in tensors + ([track, track_steps] if tracking else []):
        tensor.record_stream(stream)
