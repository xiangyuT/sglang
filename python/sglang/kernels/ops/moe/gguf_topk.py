# SPDX-License-Identifier: Apache-2.0
"""Optional SYCL selection for the small B70 GGUF E512/top-10 router.

Torch computes softmax. Optional selected-weight normalization uses the same
one-row FP32 reduction tree for batches1..8, retaining the pinned XPU radix
ordering, including tied scores.
"""

import os

import torch


def supports(scores: torch.Tensor) -> bool:
    return (
        scores.device.type == "xpu"
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 8
        and scores.shape[1] == 512
        and scores.is_contiguous()
        and not (torch.is_grad_enabled() and scores.requires_grad)
        and torch.xpu.get_device_properties(scores.device).device_id == 0xE223
    )


def select(scores: torch.Tensor, *, normalize=False) -> tuple[torch.Tensor, torch.Tensor]:
    if not supports(scores):
        raise ValueError("Unsupported B70 E512/top-10 score layout")
    from sglang.kernels.ops.quantization.gguf_sycl import _library

    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_top10"):
        raise RuntimeError("Rebuild the GGUF library for E512/top-10 routing")
    weights = torch.empty((scores.shape[0], 10), dtype=torch.float32, device=scores.device)
    ids = torch.empty((scores.shape[0], 10), dtype=torch.int64, device=scores.device)
    stream = torch.xpu.current_stream(scores.device)
    function = library.xpu_gguf_top10_norm if normalize else library.xpu_gguf_top10
    error = library.xpu_gguf_top10_norm_error if normalize else library.xpu_gguf_top10_error
    status = function(
        stream.sycl_queue, scores.data_ptr(), weights.data_ptr(), ids.data_ptr(),
        scores.shape[0],
    )
    if status:
        raise RuntimeError((status, error().decode()))
    for tensor in (scores, weights, ids):
        tensor.record_stream(stream)
    return weights, ids
