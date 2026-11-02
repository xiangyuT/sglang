# SPDX-License-Identifier: Apache-2.0
"""Bounded expert-grouped FP16 prefill over original compressed GGUF slots.

Expert bytes and residency stay unchanged. Unlike the FP32 route-wise reference,
this opt-in path rounds decoded weights and SwiGLU inputs to FP16 and accumulates
XMX products in FP32. Decode never enters this module.
"""
import ctypes
from functools import lru_cache
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf, scatter_expert_routes

_workspaces = {}
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _library(path):
    if not path:
        raise RuntimeError('Set SGLANG_XPU_GGUF_PREFILL_SYCL_LIB to the grouped prefill library')
    lib = ctypes.CDLL(path)
    lib.xpu_gguf_prefill_f16.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    lib.xpu_gguf_prefill_f16.restype = ctypes.c_int
    lib.xpu_gguf_prefill_error.restype = ctypes.c_char_p
    return lib


def grouped_f16(x, weight, counts, *, out=None):
    counts = tuple(int(n) for n in counts)
    if not (x.device.type == 'xpu' and x.dtype == torch.float16 and x.ndim == 2
            and x.is_contiguous() and weight.device == x.device
            and weight.dtype == torch.float16 and weight.ndim == 3
            and weight.is_contiguous() and weight.shape[0] == len(counts)
            and 1 <= len(counts) <= 32 and min(counts) >= 0 and max(counts) <= 32768
            and sum(counts) == x.shape[0] and weight.shape[2] == x.shape[1]
            and x.shape[1] > 0 and x.shape[1] % 32 == 0
            and weight.shape[1] > 0 and weight.shape[1] % 64 == 0
            and torch.xpu.get_device_properties(x.device).device_id == 0xE223):
        raise ValueError('Grouped FP16 prefill requires bounded B70 contiguous expert groups')
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('Grouped FP16 prefill requires eager execution')
    shape = (x.shape[0], weight.shape[1])
    if out is None:
        out = torch.empty(shape, device=x.device, dtype=torch.float32)
    elif not (out.shape == shape and out.dtype == torch.float32 and out.device == x.device
              and out.is_contiguous()):
        raise ValueError('Grouped FP16 prefill output does not match')
    if not x.shape[0]:
        return out
    lib = _library(os.environ.get('SGLANG_XPU_GGUF_PREFILL_SYCL_LIB'))
    cpu_counts = (ctypes.c_int32 * len(counts))(*counts)
    stream = torch.xpu.current_stream(x.device)
    status = lib.xpu_gguf_prefill_f16(stream.sycl_queue, x.data_ptr(), weight.data_ptr(),
                                   out.data_ptr(), ctypes.addressof(cpu_counts),
                                   len(counts), weight.shape[1], x.shape[1])
    if status:
        raise RuntimeError((status, lib.xpu_gguf_prefill_error().decode()))
    for value in (x, weight, out):
        value.record_stream(stream)
    return out


def apply_prefill(method, x, weights, ids_cpu, experts, *, output_dtype=None):
    short_f16_format = (
        method.sources[0].kind in (21, 23) and method.sources[2].kind == 8
    ) or (method.sources[0].kind == method.sources[2].kind == 42)
    if not (x.device.type == 'xpu' and x.dtype == torch.bfloat16 and x.ndim == 2
            and x.shape[0] > 0 and (x.shape[0] > 4 or short_f16_format)
            and method.hidden_size == 2560 and method.intermediate == 640
            and method.sources[0].kind in (16, 21, 22, 23, 29, 42)
            and method.sources[2].kind in (8, 20, 42)
            and method._gpu_cache is not None and ids_cpu.shape == tuple(weights.shape)
            and ids_cpu.shape[0] == x.shape[0]):
        raise ValueError('Unsupported grouped IQ4 prefill geometry')
    stream = torch.xpu.current_stream(x.device)
    use_i8 = (os.environ.get('SGLANG_XPU_GGUF_PREFILL_I8') == '1'
              and method.sources[0].kind == 21 and method.sources[2].kind == 20)
    if use_i8 and os.environ.get('SGLANG_XPU_GGUF_PREFILL_NATIVE_LAYER') == '1':
        from sglang.kernels.ops.quantization.gguf_prefill_layer import apply_layer
        return apply_layer(method, x, weights, ids_cpu, experts, output_dtype=output_dtype)
    native_f16_format = (
        method.sources[0].kind in (21, 23) and method.sources[2].kind == 8
    ) or (
        os.environ.get('SGLANG_XPU_GGUF_PREFILL_NATIVE_MTP') == '1'
        and method.sources[0].kind == 42 and method.sources[2].kind == 42
    )
    if (not use_i8 and os.environ.get('SGLANG_XPU_GGUF_PREFILL_NATIVE_F16_LAYER') == '1'
            and native_f16_format):
        from sglang.kernels.ops.quantization.gguf_prefill_f16_layer import apply_layer
        return apply_layer(method, x, weights, ids_cpu, experts, output_dtype=output_dtype)
    f16_limit = int(os.environ.get('SGLANG_XPU_GGUF_PREFILL_F16_GROUP', '32'))
    if not 1 <= f16_limit <= 32:
        raise ValueError('FP16 prefill expert group must be between1 and32')
    H, I, limit = method.hidden_size, method.intermediate, 16 if use_i8 else f16_limit
    key = (x.device, stream.sycl_queue, f16_limit)
    if key not in _workspaces:
        _workspaces[key] = (
            torch.empty((f16_limit, 2 * I, H), dtype=torch.float16, device=x.device),
            torch.empty((f16_limit, H, I), dtype=torch.float16, device=x.device),
            {},
        )
        logger.info('GGUF grouped FP16 XMX prefill: bounded %d-expert arena; original compressed weights and decode path retained', f16_limit)
    gu_weight, down_weight, encoded_cache = _workspaces[key]
    sizes = method._gpu_cache.sizes
    if (sizes, limit) not in encoded_cache:
        encoded_cache[(sizes, limit)] = tuple(torch.empty((limit, size), dtype=torch.uint8, device=x.device)
                                    for size in sizes)
    flat = ids_cpu.reshape(-1)
    # Sorting routes by expert makes every expert a single matrix product.
    order = np.argsort(flat, kind='stable')
    ordered = flat[order]
    counts = np.bincount(ordered, minlength=method.num_experts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    parts = torch.empty((flat.size, H), dtype=torch.float32, device=x.device)
    for first in range(0, len(experts), limit):
        group = experts[first:first + limit]
        count = len(group)
        packed = tuple(buffer[:count] for buffer in encoded_cache[(sizes, limit)])
        method._gpu_cache.stage_original(group, packed)
        selections = np.concatenate([order[offsets[e]:offsets[e + 1]] for e in group])
        destinations = torch.as_tensor(selections.astype(np.int32), device=x.device)
        token_ids = torch.as_tensor((selections // ids_cpu.shape[1]).astype(np.int64), device=x.device)
        activation = x.index_select(0, token_ids).to(torch.float16)
        local_counts = counts[group].tolist()
        if use_i8:
            from sglang.kernels.ops.quantization.gguf_prefill_i8 import grouped_i8

            result = grouped_i8(activation, torch.cat(packed[:2], dim=1),
                                packed[2], local_counts)
            scatter_expert_routes(result, destinations, parts)
            continue
        gate = dequantize_gguf(packed[0].view(-1), method.sources[0].kind,
                              dtype=torch.float16).view(count, I, H)
        up = dequantize_gguf(packed[1].view(-1), method.sources[1].kind,
                            dtype=torch.float16).view(count, I, H)
        # A single contiguous group lets the native launch evaluate all experts.
        gu_weight[:count].copy_(torch.cat((gate, up), dim=1))
        gu = grouped_f16(activation, gu_weight[:count], local_counts)
        hidden = (F.silu(gu[:, :I]) * gu[:, I:]).to(torch.float16)
        dequantize_gguf(packed[2].view(-1), method.sources[2].kind, dtype=torch.float16,
                        out=down_weight[:count].view(-1))
        result = grouped_f16(hidden, down_weight[:count], local_counts)
        scatter_expert_routes(result, destinations, parts)
    result = (parts.view(*ids_cpu.shape, H) * weights.float().unsqueeze(-1)).sum(dim=1)
    return result.to(x.dtype if output_dtype is None else output_dtype)
