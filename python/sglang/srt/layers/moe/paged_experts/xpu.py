"""Native XPU expert paging without per-layer host synchronization."""

import logging

import torch

from sglang.kernels.ops.moe.paged_experts_xpu import decide, gather
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _supports_expert_gemv(layer, config):
    if not envs.SGLANG_XPU_M1_FP8_EXPERT_GEMV.get():
        return False
    if (
        config.activation != "silu"
        or not config.is_gated
        or config.apply_router_weight_on_input
        or config.no_combine
        or config.gemm1_alpha is not None
        or config.gemm1_beta is not None
        or config.gemm1_clamp_limit is not None
        or config.swiglu_limit is not None
        or getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return False
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    slots = w13.shape[0]
    return (
        w13.shape == (slots, 1024, 2048)
        and w2.shape == (slots, 2048, 512)
        and s13.shape == (slots, 8, 16)
        and s2.shape == (slots, 16, 4)
        and w13.dtype == w2.dtype == torch.float8_e4m3fn
        and s13.dtype == s2.dtype == torch.float32
        and all(
            t.is_contiguous() and t.device == w13.device for t in (w13, w2, s13, s2)
        )
        and torch.xpu.get_device_properties(w13.device).device_id == 0xE223
    )


class XpuDeviceExecutor:
    def __init__(
        self, *, base_method, store, residency, contract, routed_scaling_factor, layer
    ):
        self.base_method = base_method
        self.store = store
        self.residency = residency
        self.contract = contract
        self.routed_scaling_factor = routed_scaling_factor
        self.use_expert_gemv = _supports_expert_gemv(layer, base_method.runner.config)
        if self.use_expert_gemv and base_method.runner.config.layer_id == 0:
            logger.info(
                "Using B70 single-token top-8 block-FP8 expert GEMV; "
                "prefill keeps native grouped GEMM."
            )
        device = residency.step.device
        slots = residency.slot_expert.numel()
        self.mapped_ids = torch.empty(slots, dtype=torch.int32, device=device)
        self.mapped_weights = torch.empty(slots, dtype=torch.float32, device=device)
        hosts, gpus, rows = [], [], []
        for name, host in store.host.items():
            gpu = getattr(layer, name).data
            row_bytes = host[0].numel() * host.element_size()
            if not (
                host.is_pinned()
                and host.is_contiguous()
                and gpu.is_contiguous()
                and row_bytes % 4 == 0
            ):
                raise RuntimeError(
                    f"XPU paged experts needs pinned contiguous rows of whole 4-byte words: {name}"
                )
            hosts.append(host.data_ptr())
            gpus.append(gpu.data_ptr())
            rows.append(row_bytes // 4)
        # Level Zero device addresses may set bit 63; keep pointer bits unsigned.
        self.hosts = torch.tensor(hosts, dtype=torch.uint64, device=device)
        self.gpus = torch.tensor(gpus, dtype=torch.uint64, device=device)
        self.rows = tuple(rows)

    def run(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk = dispatch_output.topk_output
        n = topk.topk_ids.numel()
        decide(
            topk.topk_ids,
            topk.topk_weights,
            self.residency,
            self.mapped_ids,
            self.mapped_weights,
        )
        gather(self.hosts, self.gpus, self.rows, self.residency)
        remapped = dispatch_output._replace(
            topk_output=topk._replace(
                topk_ids=self.mapped_ids[:n].view(topk.topk_ids.shape),
                topk_weights=self.mapped_weights[:n].view(topk.topk_weights.shape),
            )
        )
        hidden_states = dispatch_output.hidden_states
        if (
            self.use_expert_gemv
            and hidden_states.shape == (1, 2048)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and dispatch_output.hidden_states_scale is None
            and dispatch_output.hidden_states_pre_quant is None
            and topk.topk_ids.shape == (1, 8)
        ):
            from sglang.kernels.ops.moe.xpu_block_fp8_experts import (
                xpu_block_fp8_experts,
            )

            partial = xpu_block_fp8_experts(
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                remapped.topk_output.topk_ids,
                remapped.topk_output.topk_weights,
            )
        else:
            partial = self.base_method.apply(
                layer=layer, dispatch_output=remapped
            ).hidden_states
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=partial,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )
