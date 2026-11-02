"""Native XPU expert paging without per-layer host synchronization."""

import logging

import torch

from sglang.kernels.ops.moe.paged_experts_xpu import decide, gather
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _supports_expert_layout(layer, config):
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
        or getattr(layer, "w13_weight_bias", None) is not None
        or getattr(layer, "w2_weight_bias", None) is not None
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


def _supports_expert_gemv(layer, config):
    return envs.SGLANG_XPU_M1_FP8_EXPERT_GEMV.get() and _supports_expert_layout(
        layer, config
    )


def _supports_scaled_fp8(store):
    # Cover nonresident experts as well: later page-ins must obey the same bound.
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        scales = store.host[name]
        magnitude = scales.abs()
        safe = (magnitude == 0) | ((magnitude >= 2.0**-118) & (magnitude <= 2.0**118))
        if not torch.all(torch.isfinite(scales) & safe).item():
            return False
    return True


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
        self.use_scaled_fp8 = self.use_expert_gemv and _supports_scaled_fp8(store)
        self.use_compact_moe = False
        self.use_paged_compact = False
        if (
            envs.SGLANG_XPU_PAGED_DFLASH.get()
            and envs.SGLANG_XPU_DFLASH_COMPACT_MOE.get()
        ):
            from sglang.srt.layers.moe import get_moe_runner_backend
            from sglang.srt.runtime_context import get_spec

            self.use_compact_moe = (
                get_spec().speculative_algorithm == "DFLASH"
                and not get_moe_runner_backend().is_triton()
                and base_method.runner.config.routed_scaling_factor == 1.0
                and _supports_expert_layout(layer, base_method.runner.config)
            )
            if self.use_compact_moe:
                from sglang.kernels.ops.moe.xpu_compact_moe import load_compact_moe

                load_compact_moe()
                if base_method.runner.config.layer_id == 0:
                    logger.info(
                        "Using compact native W8A16 scheduling for B70 DFlash "
                        "2–8-row experts; weights retain resident-slot layout."
                    )
        self.use_paged_compact = (
            self.use_compact_moe
            and envs.SGLANG_XPU_DFLASH_PAGED_MOE.get()
            and all(
                h.is_pinned() and h.is_contiguous() and h.data_ptr() % 64 == 0
                for h in store.host.values()
            )
        )
        if self.use_paged_compact:
            from sglang.kernels.ops.moe.xpu_compact_paged import load_paged_moe

            load_paged_moe()
            if base_method.runner.config.layer_id == 0:
                logger.info(
                    "Using B70 compact native DFlash host fetch/cache fill; "
                    "expert-major scheduling, original DPAS arithmetic."
                )
        if self.use_expert_gemv and base_method.runner.config.layer_id == 0:
            logger.info(
                "Using B70 single-token top-8 block-FP8 expert GEMV; "
                "scaled FP8 decode=%s; prefill keeps native grouped GEMM.",
                self.use_scaled_fp8,
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
        if self.use_scaled_fp8 or self.use_paged_compact:
            # The fused fetch path uses an explicit order, independent of the
            # format's host-store dictionary insertion order.
            names = (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale_inv",
                "w2_weight_scale_inv",
            )
            self.expert_store = tuple(store.host[name] for name in names)
            self.expert_hosts = torch.tensor(
                [store.host[name].data_ptr() for name in names],
                dtype=torch.uint64,
                device=device,
            )
            self.expert_gpus = torch.tensor(
                [getattr(layer, name).data_ptr() for name in names],
                dtype=torch.uint64,
                device=device,
            )

        from sglang.srt.layers.moe.paged_experts.cpu40 import register
        register(self, layer)

    def run(self, layer, dispatch_output, ar_joint=None):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk = dispatch_output.topk_output
        n = topk.topk_ids.numel()
        from sglang.srt.layers.moe.paged_experts.fused37 import PlannedTopKOutput
        planned = isinstance(topk, PlannedTopKOutput)
        if planned:
            if topk.executor is not self:
                raise RuntimeError("AR37 planned output belongs to another executor")
        else:
            decide(
                topk.topk_ids,
                topk.topk_weights,
                self.residency,
                self.mapped_ids,
                self.mapped_weights,
            )
        from sglang.srt.layers.moe.paged_experts.cpu40 import run as cpu_run
        result = cpu_run(self, layer, dispatch_output, ar_shared=ar_joint)
        if result is not None:
            return result
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

            args = (
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                remapped.topk_output.topk_ids,
                remapped.topk_output.topk_weights,
            )
            if self.use_scaled_fp8:
                from sglang.kernels.ops.moe.xpu_paged_fp8_experts import (
                    xpu_paged_block_fp8_experts,
                )

                partial = xpu_paged_block_fp8_experts(
                    *args,
                    self.expert_hosts,
                    self.expert_gpus,
                    self.residency,
                    ar_joint=ar_joint,
                    scales_prepared=planned,
                )
            else:
                assert ar_joint is None, "AR101 needs scaled paged FP8"
                gather(self.hosts, self.gpus, self.rows, self.residency)
                partial = xpu_block_fp8_experts(*args)
        else:
            assert ar_joint is None, "AR101 needs M1 unquantized standard dispatch"
            if (
                self.use_compact_moe
                and hidden_states.ndim == 2
                and 2 <= hidden_states.shape[0] <= 8
                and hidden_states.shape[1] == 2048
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                and dispatch_output.hidden_states_scale is None
                and dispatch_output.hidden_states_pre_quant is None
                and topk.topk_ids.shape == (hidden_states.shape[0], 8)
            ):
                from sglang.kernels.ops.moe.xpu_compact_moe import xpu_compact_moe

                paging = None
                if self.use_paged_compact:
                    paging = (
                        self.expert_hosts,
                        self.expert_gpus,
                        self.residency,
                        self.expert_store,
                    )
                else:
                    gather(self.hosts, self.gpus, self.rows, self.residency)
                partial = xpu_compact_moe(
                    hidden_states,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale_inv,
                    layer.w2_weight_scale_inv,
                    remapped.topk_output.topk_ids,
                    remapped.topk_output.topk_weights,
                    paging=paging,
                )
            else:
                gather(self.hosts, self.gpus, self.rows, self.residency)
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
