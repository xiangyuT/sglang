# SPDX-License-Identifier: Apache-2.0
"""Discard unused prefix MoE outputs in the single-layer GGUF NEXTN draft.

Attention still processes every prefix position and writes its complete KV.
Only LAST hidden state is consumed after the final, token-independent MoE.
Eight identical tail rows retain the existing FP16 XMX arithmetic contract.
"""
from contextlib import contextmanager
import logging
import os

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


@contextmanager
def mtp_prefill_tail(model, batch):
    enabled = (
        os.environ.get("SGLANG_XPU_GGUF_MTP_PREFILL_LAST_ONLY") == "1"
        and os.environ.get("SGLANG_XPU_GGUF_PREFILL_F16") == "1"
        and batch.input_ids is not None
        and batch.input_ids.device.type == "xpu"
        and batch.batch_size == 1
        and batch.capture_hidden_mode == CaptureHiddenMode.LAST
        and not batch.return_logprob
        and batch.forward_mode.is_extend()
        and not batch.forward_mode.is_draft_extend_v2()
        and not batch.forward_mode.is_target_verify()
        and model.config.num_hidden_layers == 1
        and tuple(model.config.layer_types) == ("full_attention",)
        and not model.model.has_ple
    )
    if not enabled:
        yield
        return
    parallel = get_parallel()
    if (parallel.pp_size != 1 or parallel.attn_cp_size != 1
            or torch.xpu.is_current_stream_capturing()):
        yield
        return
    from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
    methods = [module.quant_method for module in model.model.modules()
               if isinstance(getattr(module, "quant_method", None), CompressedGGUFMoEMethod)]
    if len(methods) != 1 or any(
        method.cache_slots != method.num_experts
        or method._gpu_cache is None
        or method._gpu_cache._closed
        or method._gpu_cache.device.type != "xpu"
        for method in methods
    ):
        yield
        return
    previous = [getattr(method, "_mtp_prefill_last_only", False) for method in methods]
    try:
        for method in methods:
            method._mtp_prefill_last_only = True
        yield
    finally:
        for method, value in zip(methods, previous):
            method._mtp_prefill_last_only = value


def apply_tail(method, x, weights, ids_cpu, *, output_dtype=None):
    from sglang.kernels.ops.quantization.gguf_prefill import apply_prefill
    tail_ids = np.broadcast_to(ids_cpu[-1:], (8, ids_cpu.shape[1])).copy()
    tail = apply_prefill(
        method, x[-1:].expand(8, -1).contiguous(),
        weights[-1:].expand(8, -1).contiguous(),
        tail_ids, np.unique(tail_ids).tolist(), output_dtype=output_dtype,
    )
    result = tail.new_zeros((x.shape[0], method.hidden_size))
    result[-1].copy_(tail[0])
    if not getattr(method, "_mtp_prefill_tail_logged", False):
        logger.info("GGUF single-layer MTP prefill: routed MoE %d -> 8 identical tail rows; full prefix KV retained", x.shape[0])
        method._mtp_prefill_tail_logged = True
    return result
