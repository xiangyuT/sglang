"""The fused-MoE method that wires paged experts into a ``FusedMoE`` layer."""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.paged_experts.executor import DeviceExecutor, EagerExecutor
from sglang.srt.layers.moe.paged_experts.formats import (
    ExpertFormat,
    Fp8BlockFormat,
    UnquantizedFormat,
    format_for,
)
from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency, LRUPolicy
from sglang.srt.layers.moe.paged_experts.runners import XPU
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_model,
    get_parallel,
    get_platform,
    get_spec,
    process_model_config,
)

logger = logging.getLogger(__name__)


class PagedExpertsMoEMethod(FusedMoEMethodBase, torch.nn.Module):
    """Wraps a fused-MoE method so it runs over a K-slot GPU table paged from host memory.

    An ``nn.Module`` like the methods it wraps, so it can replace one as a layer's child."""

    def __init__(
        self,
        base_method,
        expert_format: ExpertFormat,
        num_experts: int,
        num_resident: int,
        graphs: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.base_method = base_method
        self.format = expert_format
        self.num_experts = num_experts
        self.num_resident = num_resident
        self.policy = LRUPolicy(num_resident)
        self.store = None  # built once the base method has created the expert tensors
        self.executor = None  # built once the store holds the paged layout
        # Some steps may run inside a CUDA graph: decide one-wave steps on the device, so
        # the graph can capture them.
        self.graphs = graphs
        self.device_executor = None
        self.routed_scaling_factor = 1.0

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra,
    ):
        # The GPU table has K rows. The loader below sends all E experts to the host store,
        # so the native expert-id remap never sees ids >= K.
        layer.num_local_experts = self.num_resident
        layer._num_local_routed = layer._num_global_routed = self.num_resident
        self.base_method.create_weights(
            layer=layer,
            num_experts=self.num_resident,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra,
        )
        # The loader follows the base method's weight layout, as it would without paging.
        layer.weight_load_method = self.base_method
        layer._derived_weight_cache_error = (
            "Online weight updates are not supported with --enable-paged-experts: "
            "experts already on the GPU would keep the old weights."
        )
        self.format.check_params(layer=layer, num_slots=self.num_resident)
        # A store that only stages the checkpoint layout for a repack need not be pinned.
        self.store = HostExpertStore(
            layer=layer,
            names=self.format.checkpoint_params,
            num_experts=self.num_experts,
            pin=not self.format.repacks_after_loading,
        )
        for name in self.format.checkpoint_params:
            getattr(layer, name).weight_loader = self._make_host_loader(
                layer=layer, name=name
            )

    def _make_host_loader(self, layer, name):
        def weight_loader(param, loaded_weight, weight_name, shard_id, expert_id):
            # Reuse the native per-expert loader (sharding, w1/w3 placement, scale layout)
            # on a one-row view of the host store, so host rows have exactly the native GPU
            # layout. The row carries the parameter's loader tags (e.g. ``quant_method``).
            row = torch.nn.Parameter(
                self.store.host[name][expert_id : expert_id + 1], requires_grad=False
            )
            row.__dict__.update(param.__dict__)
            layer._weight_loader_impl(
                param=row,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=0,
            )

        return weight_loader

    def create_moe_runner(self, layer, moe_runner_config):
        self.base_method.create_moe_runner(
            layer=layer,
            moe_runner_config=self.format.contract.runner_config(
                moe_runner_config=moe_runner_config, num_slots=self.num_resident
            ),
        )
        self.runner = self.base_method.runner
        self.format.contract.check(base_method=self.base_method)
        self.routed_scaling_factor = moe_runner_config.routed_scaling_factor or 1.0

    def process_weights_after_loading(self, layer):
        self.store = self.format.after_loading(
            layer=layer,
            base_method=self.base_method,
            store=self.store,
            num_slots=self.num_resident,
        )
        residency = None
        device = getattr(layer, self.format.paged_params[0]).device
        xpu_device_path = device.type == "xpu" and isinstance(
            self.format, Fp8BlockFormat
        )
        if self.graphs or xpu_device_path:
            residency = DeviceResidency(
                num_experts=self.num_experts,
                num_slots=self.num_resident,
                device=getattr(layer, self.format.paged_params[0]).device,
            )
            executor_args = dict(
                base_method=self.base_method,
                store=self.store,
                residency=residency,
                contract=self.format.contract,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if xpu_device_path:
                from sglang.srt.layers.moe.paged_experts.xpu import XpuDeviceExecutor

                self.device_executor = XpuDeviceExecutor(layer=layer, **executor_args)
            else:
                self.store.bind_device_gather(layer)
                self.device_executor = DeviceExecutor(**executor_args)
        prefill_priority = (
            envs.SGLANG_XPU_AR_PREFILL_PRIORITY.get()
            and xpu_device_path
            and self.num_resident == 192
            and self.num_experts == 256
            and get_spec().speculative_algorithm is None
            and torch.xpu.get_device_properties(device).device_id == 0xE223
        )
        if prefill_priority and self.base_method.runner.config.layer_id == 0:
            logger.info(
                "Using B70 AR prefill expert-cache priority: recent64/full "
                "frequency and last position; decode keeps device LRU."
            )
        self.executor = EagerExecutor(
            base_method=self.base_method,
            store=self.store,
            policy=self.policy,
            contract=self.format.contract,
            num_experts=self.num_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            device_residency=residency,
            prefill_priority=prefill_priority,
        )

    def apply(self, layer, dispatch_output, ar_joint=None):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        from sglang.srt.layers.moe.paged_experts.cpu40 import before_apply
        before_apply(dispatch_output)
        # Decided from shapes alone, so a captured graph always takes the same path.
        topk_output = dispatch_output.topk_output
        if (
            self.device_executor is not None
            and topk_output.topk_ids.numel() <= self.num_resident
        ):
            return self.device_executor.run(
                layer=layer, dispatch_output=dispatch_output,
                **({"ar_joint": ar_joint} if ar_joint is not None else {}),
            )
        assert ar_joint is None, "AR101 requires guarded XPU device executor"
        if is_in_breakable_cuda_graph():
            # The host-planned waves run as an eager break between graph segments.
            assert (
                TopKOutputChecker.format_is_standard(topk_output)
                and dispatch_output.hidden_states_scale is None
                and dispatch_output.hidden_states_pre_quant is None
            ), (
                "Paged experts: breakable CUDA graphs need standard, unquantized MoE inputs"
            )
            out = torch.empty_like(dispatch_output.hidden_states)
            self._run_eager_into(
                layer,
                dispatch_output.hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                out,
            )
            return StandardCombineInput(hidden_states=out)
        return self.executor.run(layer=layer, dispatch_output=dispatch_output)

    def _run_eager_into_impl(
        self, layer, hidden_states, topk_weights, topk_ids, router_logits, out
    ) -> None:
        # Flat tensors in, output buffer filled in place: a break replays with the tensors it
        # was captured with, rebuilt into the dispatch output here.
        from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            ),
        )
        out.copy_(
            self.executor.run(
                layer=layer, dispatch_output=dispatch_output
            ).hidden_states
        )

    _run_eager_into = eager_on_graph(True)(_run_eager_into_impl)


def make_for_layer(layer, base_method) -> PagedExpertsMoEMethod:
    expert_format = format_for(base_method)
    check_paged_experts_compat(layer=layer, expert_format=expert_format)
    if get_platform().is_xpu and not get_moe_runner_backend().is_triton():
        expert_format.contract = XPU
    num_experts = layer.num_local_experts
    num_resident = min(get_exec().moe.paged_experts_num_resident, num_experts)
    if check_cuda_graph_backend(Phase.DECODE, Backend.FULL):
        # A full decode graph cannot break out for host-planned waves.
        tokens_per_req = 1
        if get_spec().speculative_algorithm == "DFLASH":
            tokens_per_req = get_spec().speculative_num_draft_tokens
        _cap_decode_capture_batch_sizes(
            max(1, num_resident // (_router_top_k(layer) * tokens_per_req))
        )
    return PagedExpertsMoEMethod(
        base_method=base_method,
        expert_format=expert_format,
        num_experts=num_experts,
        num_resident=num_resident,
        graphs=not cuda_graph_fully_disabled(),
    )


def _cap_decode_capture_batch_sizes(max_bs: int) -> None:
    """Capture full decode graphs only at batch sizes whose routed entries fit the K slots,
    the steps the device executor serves; larger batches run without a graph. The cap itself
    is captured, so batches up to it pad to a graph. Every MoE layer calls this; only the
    first changes the config."""
    decode = get_exec().graph.cuda_graph_config.decode
    if not decode.bs:
        return
    cap = min(max_bs, max(decode.bs))
    capture_bs = sorted({bs for bs in decode.bs if bs <= cap} | {cap})
    if capture_bs != list(decode.bs):
        logger.info(
            "Paged experts: decode CUDA graphs captured at batch sizes %s "
            "(routed entries per captured step fit K)",
            capture_bs,
        )
    decode.bs = capture_bs
    decode.max_bs = cap


def check_paged_experts_compat(layer, expert_format) -> None:
    """Raise if the server or layer configuration is outside what paged experts supports;
    ``expert_format`` is None when no format handles the layer's quantization method."""

    problems = []
    if get_platform().is_xpu:
        if not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
            problems.append("XPU paged experts prefill graphs must be disabled")
        if not cuda_graph_fully_disabled():
            if not isinstance(expert_format, Fp8BlockFormat):
                problems.append("XPU paged decode graphs require block-FP8 experts")
            if get_moe_runner_backend().is_triton():
                problems.append("XPU paged decode graphs require the native MoE runner")
        if expert_format is not None and not isinstance(
            expert_format, (UnquantizedFormat, Fp8BlockFormat)
        ):
            problems.append("XPU paged experts supports BF16/FP16 and block-FP8 only")
    parallel = get_parallel()
    for name in ("tp_size", "ep_size", "pp_size", "dp_size"):
        if getattr(parallel, name) > 1:
            problems.append(f"--{name.replace('_', '-')} must be 1 (single GPU only)")
    if get_exec().moe.enable_eplb:
        problems.append("--enable-eplb is not supported")
    spec = get_spec()
    if spec.speculative_algorithm:
        paged_dflash = (
            envs.SGLANG_XPU_PAGED_DFLASH.get()
            and get_platform().is_xpu
            and isinstance(expert_format, Fp8BlockFormat)
            and spec.speculative_algorithm == "DFLASH"
            and getattr(spec, "speculative_eagle_topk", None) == 1
        )
        if not paged_dflash:
            problems.append(
                "speculative decoding requires opt-in XPU block-FP8 DFLASH "
                "with a linear draft chain (SGLANG_XPU_PAGED_DFLASH=1)"
            )
        elif (
            not isinstance(spec.speculative_num_draft_tokens, int)
            or spec.speculative_num_draft_tokens < 2
            or get_exec().moe.paged_experts_num_resident is None
            or spec.speculative_num_draft_tokens * _router_top_k(layer)
            > get_exec().moe.paged_experts_num_resident
        ):
            problems.append(
                "DFLASH verify window must contain at least 2 tokens and fit "
                "one resident expert wave (verify tokens * router top-k <= K)"
            )
    # Host-planned waves run as eager breaks in a breakable prefill graph; a full graph only
    # captures steps that fit one wave, which decode can be limited to and prefill cannot.
    if not (
        check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
        or check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
    ):
        problems.append(
            "prefill CUDA graphs must use the breakable backend or be disabled"
        )
    if not (
        check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
        or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
    ):
        problems.append("decode CUDA graphs must use the full backend or be disabled")
    if get_model().load_format == "dummy":
        problems.append(
            "--load-format dummy is not supported: the host store needs real weights"
        )
    num_resident = get_exec().moe.paged_experts_num_resident
    top_k = _router_top_k(layer)
    if num_resident is None:
        problems.append("--paged-experts-num-resident is required")
    elif num_resident < top_k:
        problems.append(
            f"--paged-experts-num-resident must be at least the router's top-k ({top_k})"
        )
    if expert_format is None:
        problems.append(
            "only unquantized (bf16/fp16), block-quantized FP8 and GPTQ experts are "
            f"supported, got {type(layer.quant_method).__name__}"
        )
    if get_exec().moe.enable_fused_moe_sum_all_reduce:
        problems.append(
            "--enable-fused-moe-sum-all-reduce is not supported: paged experts sums the "
            "top-k outputs itself"
        )
    if layer.num_fused_shared_experts:
        problems.append(
            "fused shared experts are not supported (--disable-shared-experts-fusion)"
        )
    if problems:
        raise RuntimeError("Paged experts: " + "; ".join(problems))


def _router_top_k(layer) -> int:
    # Models that route through a separate TopK module (e.g. OLMoE) leave FusedMoE's unset.
    if layer.top_k is not None:
        return layer.top_k
    return process_model_config().hf_text_config.num_experts_per_tok
