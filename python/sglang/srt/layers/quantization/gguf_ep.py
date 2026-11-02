# SPDX-License-Identifier: Apache-2.0
"""GGUF expert ownership over the existing compressed expert pager.

The standard dispatcher keeps global routing weights and maps nonlocal IDs to
-1. Each partition evaluates those entries against a zero-scale dummy expert,
so the existing quantized arithmetic and static grouped graph remain usable.
Partitions may keep all local experts resident or use a smaller GPU budget
with the same byte-preserving host-mirror exchanges as the single-card path.
"""
import logging
import os
import torch

logger = logging.getLogger(__name__)


def ep_cache_capacity(real_experts, requested=None):
    """Return the physical per-rank budget, including any dummy-expert slot."""
    if requested is None:
        return real_experts + 1
    try:
        slots = int(requested)
    except (TypeError, ValueError) as error:
        raise ValueError("GGUF EP cache slots must be an integer") from error
    # The fixed four-row/top-10 graph must fit its whole expert union.
    if not 40 <= slots <= real_experts + 1:
        raise ValueError("GGUF EP cache slots must be between 40 and local experts + 1")
    return slots

from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill


def hc_norm_reuse_enabled(layer, parallel):
    """Allow HC norms across an EP block that preserves token rows.

    The caller still owns DP/CP, scattered-input and A2A guards. This opt-in
    only extends that caller's EP1 case to fixed GGUF expert ownership.
    Changing weight residency does not reorder the dispatcher token rows.
    """
    if os.environ.get("SGLANG_XPU_GGUF_HC_TP_NORM_REUSE") != "1":
        return False
    if (
        parallel.moe_ep_size not in (2, 4)
        or parallel.tp_size != parallel.moe_ep_size
        or parallel.moe_tp_size != 1
    ):
        return False
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    method = getattr(experts, "quant_method", None)
    if not (
        isinstance(method, ResidentEPGGUFMoEMethod)
        and getattr(method, "_gpu_cache", None) is not None
        and method.ep_size == parallel.moe_ep_size
        and method.real_experts * method.ep_size == 512
        and 40 <= method.cache_slots <= method.real_experts + 1
        and getattr(method, "num_experts", None) == method.real_experts + 1
    ):
        return False
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

    dispatcher = getattr(experts, "dispatcher", None)
    return (
        isinstance(dispatcher, StandardDispatcher)
        and dispatcher.num_local_shared_experts == 0
        and all(
            getattr(getattr(layer, name, None), "hc_norm", None) is not None
            and getattr(getattr(layer, name).hc_norm, "_xpu_gguf_hc_enabled", False)
            for name in ("attn_hyper_connection", "mlp_hyper_connection")
        )
    )


class ExpertPartitionRows:
    """Read only the owned contiguous experts plus one zero-encoded expert."""

    def __init__(self, source, *, start, count, total):
        if not (total > 0 and source.rows % total == 0
                and 0 <= start < total and 1 <= count <= total - start):
            raise ValueError("Invalid GGUF expert partition")
        self.source = source
        self.width, self.row_bytes, self.kind = source.width, source.row_bytes, source.kind
        self.rows_per_expert = source.rows // total
        self.real_rows = count * self.rows_per_expert
        self.first_row = start * self.rows_per_expert
        self.rows = (count + 1) * self.rows_per_expert

    def packed_row_range(self, start, stop, *, pin_memory=False):
        if not (isinstance(start, int) and isinstance(stop, int)
                and 0 <= start <= stop <= self.rows):
            raise ValueError("GGUF expert partition row range is out of bounds")
        if stop <= self.real_rows:
            return self.source.packed_row_range(
                self.first_row + start, self.first_row + stop, pin_memory=pin_memory
            )
        out = torch.zeros((stop - start, self.row_bytes), dtype=torch.uint8,
                          device="cpu", pin_memory=pin_memory)
        real_stop = min(stop, self.real_rows)
        if start < real_stop:
            out[:real_stop - start].copy_(self.source.packed_row_range(
                self.first_row + start, self.first_row + real_stop
            ))
        return out


class PartitionedGGUFMoEMethod(CompressedGGUFMoEMethod):
    """Standard-dispatch EP ownership; reuse the ordinary GGUF cache backend."""

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        self.ep_size, self.ep_rank = parallel.moe_ep_size, parallel.moe_ep_rank
        if (self.ep_size not in (2, 4) or parallel.moe_tp_size != 1
                or num_experts * self.ep_size != 512):
            raise ValueError("Resident GGUF EP requires EP2/EP4 and 512 global experts")
        self.real_experts = num_experts
        self.cache_slots = ep_cache_capacity(
            num_experts, os.environ.get("SGLANG_XPU_GGUF_EP_CACHE_SLOTS")
        )
        self.fully_resident = self.cache_slots == num_experts + 1
        super().create_weights(layer, num_experts + 1, hidden_size,
                               intermediate_size_per_partition, params_dtype, **attrs)

    def bind(self, layer, sources, device):
        from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows

        global_experts = self.real_experts * self.ep_size
        expected = [(self.hidden_size, self.intermediate, global_experts)] * 2 + [
            (self.intermediate, self.hidden_size, global_experts)
        ]
        if [tuple(t.shape) for _, t in sources] != expected:
            raise ValueError("GGUF EP source geometry does not match global experts")
        rows = [ExpertPartitionRows(
            CompressedGGUFRows(s, t), start=self.ep_rank * self.real_experts,
            count=self.real_experts, total=global_experts,
        ) for s, t in sources]
        self.bind_rows(layer, rows, device)
        if self._gpu_cache is None:
            raise ValueError("Resident GGUF EP requires the GPU cache/native expert path")
        if (not self.fully_resident
                and os.environ.get("SGLANG_XPU_GGUF_PAGED_PREP") == "1"
                and os.environ.get("SGLANG_XPU_GGUF_EP_SKIP_ZERO") == "1"):
            from sglang.kernels.ops.quantization.gguf_resident_prepare import _paged_library

            _paged_library(os.environ.get("SGLANG_XPU_GGUF_PAGED_PREP_LIB"))
            self._paged_zero_expert = self.real_experts
        if self.fully_resident and os.environ.get("SGLANG_XPU_GGUF_EP_SKIP_ZERO") == "1":
            self._ep_zero_slot = self.real_experts
        from sglang.kernels.ops.quantization.gguf_resident_prepare import configure

        # Immutable preparation cannot plan evictions. A paged partition uses
        # GPUExpertCache.prepare and computes the ordinary zero expert instead.
        self._resident_preparation = (
            configure(self._gpu_cache, self._ep_zero_slot)
            if self.fully_resident and hasattr(self, "_ep_zero_slot")
            else None
        )
        if self._resident_preparation is not None:
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

            dispatcher = getattr(layer, "dispatcher", None)
            if (isinstance(dispatcher, StandardDispatcher)
                    and dispatcher.num_experts == 512
                    and dispatcher.num_local_shared_experts == 0):
                dispatcher._xpu_gguf_resident_preparation = self._resident_preparation
        if getattr(getattr(self, "moe_runner_config", None), "layer_id", None) == 0:
            logger.info(
                "GGUF EP%d rank%d: local_real=%d cache_slots=%d mode=%s; "
                "paged slots reuse the single-card mirror exchange backend",
                self.ep_size, self.ep_rank, self.real_experts, self.cache_slots,
                "fully_resident" if self.fully_resident else "paged",
            )

    def _apply_gpu_prefill(self, x, weights, ids_cpu, experts):
        # Retain the FP32 local routed sum until the model boundary owns the
        # EP sum. Default and <=4-row native decode arithmetic stay unchanged.
        partial = super()._apply_gpu_prefill(
            x, weights, ids_cpu, experts,
            output_dtype=torch.float32 if getattr(self, "prefill_fp32", False) else None,
        )
        if getattr(self, "prefill_separate", False):
            from sglang.srt.layers.quantization.gguf_tp_prefill import reduce_prefill_partial
            from sglang.srt.runtime_context import get_parallel

            return reduce_prefill_partial(partial, get_parallel().moe_ep_group) / self.ep_size
        return partial

    def grow_cache(self, layer, slots):
        """Apply an explicit post-capture per-rank budget through stable tables.

        Immutable resident preparation remains bound as captured. A paged
        partition keeps its existing planner even when growth fills all slots.
        """
        if (
            not isinstance(slots, int)
            or not self.cache_slots <= slots <= self.real_experts + 1
        ):
            raise ValueError(
                "GGUF EP growth must stay within the local expert partition"
            )
        added = super().grow_cache(layer, slots)
        self.fully_resident = self.cache_slots == self.real_experts + 1
        return added

    def apply(self, layer, dispatch_output):
        if (
            getattr(self, "_resident_preparation", None) is not None
            and 1 <= dispatch_output.hidden_states.shape[0] <= 4
            and not is_gguf_prefill()
        ):
            # Native preparation owns negative-ID -> zero-slot mapping.
            return super().apply(layer, dispatch_output)
        topk = dispatch_output.topk_output
        local = torch.where(topk.topk_ids < 0, self.real_experts, topk.topk_ids)
        return super().apply(layer, dispatch_output._replace(
            topk_output=topk._replace(topk_ids=local)
        ))


# Preserve imports used by earlier full-resident experiments; one implementation.
ResidentEPGGUFMoEMethod = PartitionedGGUFMoEMethod
