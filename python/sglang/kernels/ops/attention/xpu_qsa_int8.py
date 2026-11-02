# SPDX-License-Identifier: Apache-2.0
"""B70 QSA block-64 INT8 KV with inline FP16 scales; no BF16 cache shadow."""

import os

import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import _head_partition, _library

HEAD_DIM = 256
BLOCK_SIZE = 64
RECORD_BYTES = HEAD_DIM + HEAD_DIM // BLOCK_SIZE * 2


def _record_ok(tensor):
    if not (
        tensor.dtype == torch.int8
        and tensor.ndim == 3
        and tensor.shape[0] > 0
        and tensor.shape[1] in (1, 2)
        and tensor.shape[2] == HEAD_DIM
        and tensor.stride() == (tensor.shape[1] * RECORD_BYTES, RECORD_BYTES, 1)
    ):
        return False
    # The logical payload view must also own the trailing inline scale bytes.
    end = tensor.storage_offset() + tensor.shape[0] * tensor.stride(0)
    return tensor.untyped_storage().nbytes() >= end


def available(q, k, v, slots):
    tp = _head_partition(q, k)
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    if not (
        path
        and tp
        and _record_ok(k)
        and _record_ok(v)
        and q.device.type == "xpu"
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[0] > 0
        and q.shape[2] == HEAD_DIM
        and k.shape == v.shape
        and q.device == k.device == v.device == slots.device
        and slots.dtype == torch.int32
        and slots.ndim == 2
        and slots.shape[0] == q.shape[0]
        and slots.shape[1] > 0
        and torch.xpu.get_device_properties(q.device).device_id == 0xE223
    ):
        return False
    return hasattr(
        _library(path), "xpu_qsa_int8" if tp == 1 else f"xpu_qsa_int8_tp{tp}"
    )


def quantize_and_store(k_buffer, v_buffer, locations, key, value, owners, error):
    """Store last source per valid slot. Invalid (including negative) slots skip.

    ``error`` is a sticky device flag for nonfinite/overflowing scales. Inspect
    it at validation/health boundaries, never by synchronizing every decode.
    """
    if locations.numel() == 0:
        return
    path = os.environ.get("SGLANG_XPU_QSA_SYCL_LIB")
    if not (
        path
        and _record_ok(k_buffer)
        and _record_ok(v_buffer)
        and k_buffer.shape == v_buffer.shape
        and key.dtype == value.dtype == torch.bfloat16
        and key.ndim == value.ndim == 3
        and key.shape == value.shape
        and key.shape[1:] == k_buffer.shape[1:]
        and key.stride(2) == value.stride(2) == 1
        and locations.ndim == 1
        and locations.dtype in (torch.int32, torch.int64)
        and locations.numel() == key.shape[0]
        and owners.dtype == torch.int32
        and owners.shape == (k_buffer.shape[0],)
        and owners.is_contiguous()
        and error.dtype == torch.int32
        and error.numel() == 1
        and error.is_contiguous()
        and key.device.type == "xpu"
        and all(
            t.device == key.device
            for t in (value, k_buffer, v_buffer, locations, owners, error)
        )
        and torch.xpu.get_device_properties(key.device).device_id == 0xE223
        and not torch._C._overlaps(key, k_buffer)
        and not torch._C._overlaps(value, v_buffer)
    ):
        raise ValueError(
            "INT8 KV requires B70 BF16 inputs and block-64 inline QSA records"
        )
    function = getattr(_library(path), "xpu_qsa_int8_store", None)
    if function is None:
        raise RuntimeError("Rebuild SGLANG_XPU_QSA_SYCL_LIB with INT8 KV support")
    locations = locations.contiguous()
    stream = torch.xpu.current_stream(key.device)
    # Keep launches bounded without losing last-source semantics across chunks.
    for first in range(0, key.shape[0], 8192):
        count = min(8192, key.shape[0] - first)
        status = function(
            stream.sycl_queue,
            key.data_ptr() + first * key.stride(0) * key.element_size(),
            value.data_ptr() + first * value.stride(0) * value.element_size(),
            locations.data_ptr() + first * locations.element_size(),
            k_buffer.data_ptr(),
            v_buffer.data_ptr(),
            owners.data_ptr(),
            error.data_ptr(),
            count,
            key.shape[1],
            k_buffer.shape[0],
            locations.element_size(),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
        )
        if status:
            raise RuntimeError(f"INT8 KV store submission failed: status={status}")
    for tensor in (k_buffer, v_buffer, locations, key, value, owners, error):
        tensor.record_stream(stream)
