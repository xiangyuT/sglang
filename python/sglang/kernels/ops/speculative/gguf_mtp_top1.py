# SPDX-License-Identifier: Apache-2.0
"""B70 greedy NEXTN argmax and actual confidence for the pinned GGUF head."""

import os

import torch


def try_top1(logits):
    if not (
        os.environ.get("SGLANG_XPU_GGUF_MTP_TOP1") == "1"
        and logits.device.type == "xpu"
        and logits.dtype == torch.float32
        and logits.shape == (1, 106299)
        and logits.is_contiguous()
        and torch.xpu.get_device_properties(logits.device).device_id == 0xE223
    ):
        return None
    from sglang.kernels.ops.quantization.gguf_sycl import _library

    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    probability = torch.empty((1, 1), dtype=torch.float32, device=logits.device)
    indices = torch.empty((1, 1), dtype=torch.int64, device=logits.device)
    keys = torch.empty((1, 64), dtype=torch.uint64, device=logits.device)
    sums = torch.empty((1, 64), dtype=torch.float32, device=logits.device)
    stream = torch.xpu.current_stream(logits.device)
    rc = library.xpu_mtp_top1_probability(
        stream.sycl_queue, logits.data_ptr(), probability.data_ptr(),
        indices.data_ptr(), keys.data_ptr(), sums.data_ptr(), 1, 106299, 64,
    )
    if rc:
        raise RuntimeError(library.xpu_mtp_top1_probability_error().decode())
    for tensor in (logits, probability, indices, keys, sums):
        tensor.record_stream(stream)
    return probability, indices
