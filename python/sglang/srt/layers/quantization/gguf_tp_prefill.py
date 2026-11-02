# SPDX-License-Identifier: Apache-2.0
"""Opt-in FP32 long-row partials for the fully resident GGUF TP/EP layout."""
import logging
import os
import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill

logger = logging.getLogger(__name__)


def reduce_prefill_partial(value, group):
    """Sum FP32 branch partials and cast once at the model boundary.

    This is an explicit model dtype transition, not a change to the general
    all-reduce API. Native byte all-gather preserves the FP32 partial bits.
    """
    if not (value.device.type == "xpu" and value.dtype == torch.float32
            and value.ndim == 2 and value.shape[0] > 0
            and value.shape[1] == 2560 and value.is_contiguous()
            and group.world_size in (2, 4)):
        raise ValueError("GGUF FP32 prefill partial requires H2560 rows and TP2/TP4")
    communicator = group.xpu_communicator
    peer = None if communicator is None else communicator.peer_allreduce
    if peer is None:
        raise RuntimeError("GGUF FP32 prefill partials require native peer all-gather")
    flat = value.view(-1)
    result = torch.empty_like(value, dtype=torch.bfloat16)
    out = result.view(-1)
    capacity = peer.max_gather_bytes // value.element_size()
    # The native transport, fixed rank-order sum and static slice copies are
    # graph safe. The same long-row partial occurs in batched verification;
    # it must not be classified as eager-only merely by token-row count.
    for start in range(0, flat.numel(), capacity):
        local = flat[start:start + capacity]
        gathered = torch.empty((group.world_size, local.numel()),
                               dtype=local.dtype, device=local.device)
        if not peer.can_all_gather(gathered, local):
            raise RuntimeError("GGUF FP32 prefill partial exceeds native gather layout")
        peer.all_gather(gathered, local)
        out[start:start + local.numel()].copy_(gathered.sum(dim=0))
    if not getattr(group, "_gguf_prefill_fp32_seen", False):
        group._gguf_prefill_fp32_seen = True
        logger.info("GGUF FP32 prefill reduction invoked: group=%s world=%d H2560; FP32 partials -> one BF16 model-boundary cast",
                    group.unique_name, group.world_size)
    return result


class PrefillFP32RowMethod(UnquantizedLinearMethod):
    """Keep native decode arithmetic and FP32 partials for every prefill row."""

    def __init__(self, parent, group=None):
        self.parent = parent
        self.group = group

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def apply(self, layer, x, bias=None):
        if x.shape[0] <= 4 and not is_gguf_prefill():
            return self.parent.apply(layer, x, bias)
        from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

        try:
            batch = get_forward_batch()
            mode = batch.forward_mode
        except RuntimeError:
            batch = None
            mode = None
        if mode is not None and (mode.is_decode() or mode.is_target_verify()):
            return self.parent.apply(layer, x, bias)
        weight = (self.parent.reconstruct_weight(layer)
                  if hasattr(self.parent, "reconstruct_weight") else layer.weight)
        if bias is not None or x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
            raise ValueError("GGUF FP32 prefill row projection requires bias-free BF16 GEMM")
        lengths = getattr(batch, "extend_seq_lens_cpu", None)
        if (mode is not None and mode.is_extend()
                and not mode.is_draft_extend_v2() and batch.batch_size > 1
                and lengths is not None and sum(lengths) == x.shape[0]):
            # Preserve each request's FP32 GEMM shape. A different packed M
            # selects a different reduction in oneMKL; the tiny first-layer
            # roundoff can otherwise amplify through later recurrent/MoE layers.
            partial = torch.cat([
                torch.mm(part, weight.t(), out_dtype=torch.float32)
                for part in x.split(tuple(int(n) for n in lengths))
            ])
        else:
            partial = torch.mm(x, weight.t(), out_dtype=torch.float32)
        if self.group is not None:
            return reduce_prefill_partial(partial, self.group) / self.group.world_size
        return partial


def install_prefill_fp32(model):
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.layers.quantization.gguf_ep import ResidentEPGGUFMoEMethod

    parallel = get_parallel()
    if not (parallel.tp_size == parallel.moe_ep_size
            and parallel.tp_size in (2, 4) and parallel.moe_tp_size == 1
            and parallel.attn_dp_size == parallel.attn_cp_size == 1):
        raise ValueError("GGUF FP32 prefill requires fully resident TP2/EP2 or TP4/EP4")
    separate = os.environ.get("SGLANG_XPU_GGUF_TP_PREFILL_FP32") == "separate"
    targets, experts = [], []
    for layer in model.model.layers:
        attn = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.o_proj
        method = layer.mlp.experts.quant_method
        if not isinstance(method, ResidentEPGGUFMoEMethod):
            raise ValueError("GGUF FP32 prefill requires resident expert partitions")
        if not isinstance(attn.quant_method, UnquantizedLinearMethod):
            raise ValueError("GGUF FP32 prefill requires original BF16 attention row shards")
        targets.append(attn)
        if separate:
            targets.append(layer.mlp.shared_expert.down_proj)
        experts.append(method)
    for module in targets:
        module.quant_method = PrefillFP32RowMethod(
            module.quant_method, parallel.tp_group if separate else None
        )
    for method in experts:
        method.prefill_fp32 = True
        method.prefill_separate = separate
    logger.info("GGUF FP32 prefill installed: %d row projections, %d routed branches, separate=%s; <=4-row decode delegates unchanged", len(targets), len(experts), separate)


def enabled():
    return os.environ.get("SGLANG_XPU_GGUF_TP_PREFILL_FP32") in ("1", "separate")
