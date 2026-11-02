"""Readback and runtime updates for the scheduler's internal state.

Holds the live ``Scheduler`` because both sides answer about the state at call
time; it only reads it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.managers.io_struct import (
    GetInternalStateReq,
    GetInternalStateReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
)
from sglang.srt.managers.scheduler_components.memory_usage import build_memory_usage
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.srt.server_args import compute_world_size
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# The CUDA graph phases whose captured memory a PD router attributes to decode.
_DECODE_GRAPH_PHASES = ("decode", "target_verify", "draft_decode", "draft_extend")


class UpdateRejected(Exception):
    """A requested update is not applicable to this scheduler."""


class SchedulerInternalStateController:
    def __init__(self, scheduler: Scheduler, record_step_time: bool) -> None:
        self.scheduler = scheduler
        self.record_step_time = record_step_time

    # ---------------------------------------------------------------- readback

    def get_internal_state(self, recv_req: GetInternalStateReq):
        # Resolved config (pristine server_args + post-publish overrides) so a
        # readback reflects values changed via /set_internal_state, not startup.
        state = get_context().resolved_server_args_dict()
        for contribute in (
            self._add_topology,
            self._add_memory_usage,
            self._add_disaggregation,
            self._add_elastic_ep,
            self._add_speculative,
            self._add_step_times,
            self._add_rust_server,
            self._add_env_vars,
        ):
            contribute(state)
        # A bound signal handler is not msgpack-serializable, and no reader
        # consumes it.
        state.pop("custom_sigquit_handler", None)
        model = self.scheduler.tp_worker.model_runner.model
        config = getattr(model.config, "text_config", model.config)
        if getattr(config, "gguf_model_path", None):
            # Native GGUF does not use the older FP8 paging experiments or
            # their external fixture modules. Read only its live cache state.
            from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
                CompressedGGUFMoEMethod,
            )
            from torch import are_deterministic_algorithms_enabled

            methods = [
                (module, module.quant_method)
                for module in model.modules()
                if isinstance(
                    getattr(module, "quant_method", None), CompressedGGUFMoEMethod
                )
            ]
            state["gguf_xpu_experts"] = {
                "deterministic_torch": are_deterministic_algorithms_enabled(),
                "layers": len(methods),
                "slots_per_layer": [method.cache_slots for _, method in methods],
                "compressed_cache_bytes": sum(
                    method._gpu_cache.owned_resident_storage_bytes()
                    if method._gpu_cache is not None
                    else sum(
                        getattr(module, "gguf_" + name).numel()
                        for name in ("gate", "up", "down")
                    )
                    for module, method in methods
                ),
                "hits": sum(method.hits for _, method in methods),
                "misses": sum(method.misses for _, method in methods),
                "transferred_bytes": sum(
                    method.transferred_bytes for _, method in methods
                ),
                "legacy_counters_scope": "host-planned fallback cache",
                "gpu_cache_layers": [
                    {
                        "layer": index,
                        "ep_rank": getattr(method, "ep_rank", 0),
                        **method._gpu_cache.residency_snapshot(
                            zero_expert=getattr(method, "real_experts", None)
                        ),
                    }
                    for index, (_, method) in enumerate(methods)
                    if method._gpu_cache is not None
                ],
            }
            return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(state))
        from sglang.srt.layers.moe.paged_experts.cpu40 import readback
        state["codex_cpu67"] = readback()
        import json, os
        from pathlib import Path
        from sglang.srt.layers import ar188_metadata as meta188
        setting_root = os.environ.get('CODEX_AR_CPU40_ROOT')
        setting_path = Path(setting_root) / 'metadata-mode.json' if setting_root else None
        if meta188.enabled() and setting_path is not None and setting_path.exists():
            setting=json.loads(setting_path.read_text())
            assert set(setting)=={'mode'} and setting['mode'] in ['original','eager','ingraph']
            meta188.switch_mode(setting['mode'])
        state['codex_cpu67']['CPU_worker_count'] = 13 if state['codex_cpu67']['enabled'] else 0
        state['codex_meta188']=meta188.readback()
        from sglang.srt.mem_cache.ar231_prepare import readback as prepare_readback
        state['codex_prepare231'] = prepare_readback()
        pm = self.scheduler.profiler_manager
        state["codex_cpu67"].update(torch_profiler_active=pm.torch_profiler is not None, profile_in_progress=pm.profile_in_progress, profiler_target_forward_ct=pm.profiler_target_forward_ct)
        state["codex_cpu67"].update(
            host_ready_d2h_worker=bool(getattr(self.scheduler, "codex_cpu74_host_d2h", False)),
            async_d2h_enabled=self.scheduler.enable_async_d2h_copy,
            d2h_worker_type=type(self.scheduler.async_d2h_worker).__name__,
            overlap_enabled=self.scheduler.enable_overlap,
        )
        from sglang.srt.layers.ar86_boundary import readback as boundary_readback
        state["codex_ar86_boundary"] = boundary_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar87_inventory import readback as ar87_readback
        state["codex_ar87_inventory"] = ar87_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.xpu_tp_bounded_head import readback as tp_head_readback
        state["codex_tp_bounded_head"] = tp_head_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar91_silu import readback as silu_readback
        state["codex_ar91_silu"] = silu_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar1444_shared import readback as fused_shared_readback
        state["codex_ar1444_shared"] = fused_shared_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar1502_pregate import readback as pregate_readback
        state["codex_ar1502_pregate"] = pregate_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar1657_shared_union import readback as union_readback
        state["codex_ar1657_union"] = union_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.layers.ar1532_outnorm import readback as projection_norm_readback
        state["codex_ar1532_outnorm"] = projection_norm_readback(self.scheduler.tp_worker.model_runner.model)
        from sglang.srt.managers.ar1564_relay import readback as relay_readback
        state["codex_ar1564_relay"] = relay_readback(self.scheduler.future_map)
        from sglang.srt.layers.ar266_argmax import readback as argmax_readback
        state["codex_ar266_argmax"] = argmax_readback()
        from sglang.srt.layers.moe.paged_experts.cpu_publish362 import readback as route_readback
        state["codex_ar606_route"] = route_readback()
        from sglang.srt.layers.ar612_attention import readback as attention_readback
        state["codex_ar612_attention"] = attention_readback()
        from sglang.srt.layers.ar646_input import readback as input_readback
        state["codex_ar646_input"] = input_readback()
        from sglang.srt.layers.ar1776_inputconv import readback as inputconv_readback
        state["codex_ar1776_inputconv"] = inputconv_readback()
        from sglang.srt.layers.ar671_output import readback as output_readback
        state["codex_ar671_output"] = output_readback()
        from sglang.srt.layers.ar728_shared import readback as shared_readback
        state["codex_ar728_shared"] = shared_readback()
        from sglang.srt.layers.ar101_joint import readback as ar101_readback
        state["codex_ar101_joint"] = ar101_readback(self.scheduler.tp_worker.model_runner.model)
        return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(state))

    def _add_topology(self, state: Dict[str, Any]) -> None:
        scheduler = self.scheduler
        state["world_size"] = compute_world_size(
            dp_size=get_parallel().dp_size,
            tp_size=get_parallel().tp_size,
            pp_size=get_parallel().pp_size,
        )
        state["startup_time"] = scheduler.startup_time
        state["effective_max_running_requests_per_dp"] = scheduler.max_running_requests
        state["last_gen_throughput"] = scheduler.metrics_reporter.last_gen_throughput

    def _add_memory_usage(self, state: Dict[str, Any]) -> None:
        scheduler = self.scheduler
        draft_worker = scheduler.draft_worker
        state["memory_usage"] = build_memory_usage(
            weight_gb=scheduler.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_gb=scheduler.token_to_kv_pool_allocator.get_kvcache().mem_usage,
            startup_available_gb=scheduler.startup_available_gpu_memory_gb,
            token_capacity=scheduler.max_total_num_tokens,
            token_capacity_swa=scheduler.swa_tokens_per_layer,
            target_graph_memory_usage=scheduler.tp_worker.graph_memory_usage,
            draft_graph_memory_usage=(
                None if draft_worker is None else draft_worker.graph_memory_usage
            ),
        )

    def _add_disaggregation(self, state: Dict[str, Any]) -> None:
        # PD role switch: report this instance's role and the decode CUDA graph
        # batch sizes it captured, which a router feeds back as
        # PdRoleSwitchReqInput.decode_cuda_graph_bs. Unset until
        # init_disaggregation runs, which also re-derives it on every flip.
        mode = getattr(self.scheduler, "disaggregation_mode", None)
        if mode is None:
            return
        graph_memory = state["memory_usage"]["graph"]
        state["disaggregation_mode"] = mode.value
        state["decode_cuda_graph_bs"] = (
            self.scheduler.tp_worker.get_decode_cuda_graph_bs()
        )
        state["decode_cuda_graph_memory_gb"] = round(
            sum(graph_memory[phase] for phase in _DECODE_GRAPH_PHASES), 3
        )

    def _add_elastic_ep(self, state: Dict[str, Any]) -> None:
        if get_exec().moe.elastic_ep_backend is None:
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        state["is_scaling_elastic_ep"] = ElasticEPStateManager.is_scaling()
        state["effective_ep_size"] = ElasticEPStateManager.get_effective_ep_size()
        state["pending_ep_size"] = ElasticEPStateManager.get_pending_ep_size()
        state["scale_phase"] = ElasticEPStateManager.get_scale_phase()
        state["elastic_ep_last_error"] = ElasticEPStateManager.get_last_error()

    def _add_speculative(self, state: Dict[str, Any]) -> None:
        scheduler = self.scheduler
        if scheduler.spec_algorithm.is_none():
            return
        accept_length = self._average_accept_length()
        if accept_length is not None:
            state["avg_spec_accept_length"] = accept_length
        if scheduler.spec_algorithm.is_dspark() and scheduler.draft_worker is not None:
            info_record = scheduler.draft_worker.dump_info_records()
            if info_record is not None:
                state["dspark_info_record"] = info_record

    def _add_step_times(self, state: Dict[str, Any]) -> None:
        if self.record_step_time:
            state["step_time_dict"] = self.scheduler.metrics_reporter.step_time_dict

    def _add_rust_server(self, state: Dict[str, Any]) -> None:
        if self.scheduler.rust_server is not None:
            state["rust_mm_transport"] = self.scheduler.rust_server.mm_transport_stats()
            state["rust_server_startup_time"] = self.scheduler.rust_server.startup_time

    def _add_env_vars(self, state: Dict[str, Any]) -> None:
        if envs.SGLANG_EXPOSE_OWN_ENV_VARS.get():
            state["env_vars"] = exportable_env_vars()

    # ------------------------------------------------------------------ update

    def set_internal_state(self, recv_req: SetInternalStateReq):
        handlers = self._update_handlers()
        requested = recv_req.server_args
        try:
            for key, value in requested.items():
                handler = handlers.get(key)
                if handler is None:
                    raise UpdateRejected(f"Updating {key} is not supported.")
                handler.validate(key, value)
        except UpdateRejected as rejection:
            logger.warning(str(rejection))
            return SetInternalStateReqOutput(updated=False)

        # Worker commands apply directly; config keys land in one override.
        overrides = {}
        for key, value in requested.items():
            overrides.update(handlers[key].apply(key, value) or {})
        if overrides:
            get_context().override(source="update_server_args", **overrides)
            logger.info(f"Config updated via context override: {overrides}")
        return SetInternalStateReqOutput(updated=True)

    def _update_handlers(self) -> Dict[str, _UpdateHandler]:
        return {
            "pp_max_micro_batch_size": _UpdateHandler(
                self._validate_pp_micro_batch_size, _override
            ),
            "speculative_accept_threshold_single": _UpdateHandler(
                self._validate_accept_threshold, self._apply_accept_threshold
            ),
            "speculative_accept_threshold_acc": _UpdateHandler(
                self._validate_accept_threshold, self._apply_accept_threshold
            ),
            "dspark_force_budget_frac": _UpdateHandler(
                self._validate_dspark_budget_frac, self._apply_dspark_budget_frac
            ),
            "dspark_clear_info_records": _UpdateHandler(
                self._validate_dspark_clear_records, self._apply_dspark_clear_records
            ),
        }

    def _validate_pp_micro_batch_size(self, key: str, value: Any) -> None:
        upper = self.scheduler.max_running_requests // get_parallel().pp_size
        if value < 1 or value > upper:
            raise UpdateRejected(
                f"Updating {key} to {value} is rejected because it is out of "
                f"the valid range [1, {upper}]."
            )

    def _validate_accept_threshold(self, key: str, value: Any) -> None:
        # A relaxed threshold stops acceptance from sampling exactly from the
        # target policy, which is what a captured sampling mask describes.
        if not self.scheduler.spec_algorithm.is_dflash() or float(value) == 1.0:
            return
        if self._has_active_sampling_mask_request():
            raise UpdateRejected(
                f"Updating {key} is rejected while DFlash sampling-mask "
                "requests are active."
            )

    def _apply_accept_threshold(self, key: str, value: Any) -> Dict[str, Any]:
        # Acceptance statistics describe the previous threshold; start over.
        accept_length = self._average_accept_length()
        if accept_length is not None:
            logger.info(f"avg_spec_accept_length={accept_length}")
        metrics_reporter = self.scheduler.metrics_reporter
        metrics_reporter.spec_total_num_accept_tokens = (
            metrics_reporter.spec_total_num_forward_ct
        ) = 0
        return {key: value}

    def _validate_dspark_budget_frac(self, key: str, value: Any) -> None:
        self._require_dspark_worker(key, "set_dspark_forced_budget_frac")
        if value is not None and not (0.0 < float(value) <= 1.0):
            raise UpdateRejected(f"{key} must be in (0, 1] or null, got {value}.")

    def _apply_dspark_budget_frac(self, key: str, value: Any) -> None:
        # A worker command, not a server arg: keep it out of the override.
        self.scheduler.draft_worker.set_dspark_forced_budget_frac(
            None if value is None else float(value)
        )

    def _validate_dspark_clear_records(self, key: str, value: Any) -> None:
        self._require_dspark_worker(key, "clear_info_records")

    def _apply_dspark_clear_records(self, key: str, value: Any) -> None:
        if value:
            self.scheduler.draft_worker.clear_info_records()

    def _require_dspark_worker(self, key: str, command: str) -> None:
        if not self.scheduler.spec_algorithm.is_dspark() or not hasattr(
            self.scheduler.draft_worker, command
        ):
            raise UpdateRejected(f"{key} requires a DSpark draft worker.")

    # ------------------------------------------------------------------ shared

    def _average_accept_length(self) -> Optional[float]:
        metrics_reporter = self.scheduler.metrics_reporter
        if (
            self.scheduler.spec_algorithm.is_none()
            or metrics_reporter.spec_total_num_forward_ct <= 0
        ):
            return None
        return (
            metrics_reporter.spec_total_num_accept_tokens
            / metrics_reporter.spec_total_num_forward_ct
        )

    def _has_active_sampling_mask_request(self) -> bool:
        """Whether any unfinished request holding a sampling mask is queued or
        running, including PP micro-batches and PD decode queues."""
        scheduler = self.scheduler
        batches = [scheduler.running_batch, scheduler.last_batch]
        batches.extend(getattr(scheduler, "running_mbs", ()))
        batches.extend(getattr(scheduler, "mbs", ()))
        batches.extend(getattr(scheduler, "last_mbs", ()))
        queued = [
            *(req for batch in batches if batch is not None for req in batch.reqs),
            *scheduler.waiting_queue,
            *scheduler.grammar_manager.grammar_queue,
        ]
        if scheduler.chunked_req is not None:
            queued.append(scheduler.chunked_req)
        if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            prealloc_queue = scheduler.disagg_decode_prealloc_queue
            queued.extend(
                decode_req.req
                for decode_req in (
                    *prealloc_queue.queue,
                    *prealloc_queue.pending_reqs,
                    *scheduler.disagg_decode_transfer_queue.queue,
                )
            )
            queued.extend(prealloc_queue.retracted_queue)
            queued.extend(prealloc_queue.held_rebootstrap_reqs)
        return any(req.return_sampling_mask and not req.finished() for req in queued)


class _UpdateHandler:
    def __init__(
        self,
        validate: Callable[[str, Any], None],
        apply: Callable[[str, Any], Optional[Dict[str, Any]]],
    ) -> None:
        self.validate = validate
        self.apply = apply


def _override(key: str, value: Any) -> Dict[str, Any]:
    return {key: value}
