"""Bounded GGUF verify projections consumed directly by strided GDN kernels."""
import os

import torch


def try_verify_views(qkvz, ba, *, is_verify, key_heads, value_heads, head_dim):
    if not (
        os.environ.get("SGLANG_XPU_GGUF_GDN_VERIFY_VIEWS") == "1"
        and os.environ.get("SGLANG_XPU_GGUF_GDN_SYCL") == "1"
        and is_verify
        and key_heads in (16, 8, 4) and value_heads == 3 * key_heads and head_dim == 128
        and isinstance(qkvz, torch.Tensor) and isinstance(ba, torch.Tensor)
        and qkvz.device.type == "xpu"
        and qkvz.device == ba.device
        and qkvz.dtype == ba.dtype == torch.bfloat16
        and qkvz.shape == (4, key_heads * 8 * 128)
        and ba.shape == (4, 2 * value_heads)
        and qkvz.is_contiguous() and ba.is_contiguous()
        and torch.xpu.get_device_properties(qkvz.device).device_id == 0xE223
    ):
        return None
    # Conv1D honors the projected row stride. Native GDN consumes B/A with
    # stride 96 and retains the original four intermediate rollback windows.
    width = key_heads * 5 * 128
    return (qkvz[:, :width], qkvz[:, width:].view(4, value_heads, 128),
            ba[:, :value_heads], ba[:, value_heads:])
