"""XPU compute segments with eager collective boundaries.

Reuse the common breakable capture machinery. Keep the XPU runner's output
object alive directly: it returns LogitsProcessorOutput, whereas the CUDA
backend's shared output allocator expects tensors and tuples.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
    FullXPUGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
    set_tc_piecewise_forward_context,
)

logger = logging.getLogger(__name__)


class _DebugCapture(BreakableCUDAGraphCapture):
    def _begin_new_segment(self) -> None:
        super()._begin_new_segment()
        if os.environ.get("CODEX_AR87_GRAPH_DIR"):
            self._current_graph.enable_debug_mode()


class BreakableXPUGraphBackend(FullXPUGraphBackend):
    """One sequence of compute graphs and eager calls per decode shape."""

    def __init__(self, cuda_graph_runner):
        super().__init__(cuda_graph_runner)
        self._model_runner = cuda_graph_runner.model_runner
        config = self._model_runner.model.config
        config = getattr(config, "text_config", config)
        self._native_gguf = bool(getattr(config, "gguf_model_path", None))
        self._capture_inputs = {}
        self._head_fallback_reasons = set()

    @contextmanager
    def _forward_context(self, forward_batch):
        # Qwen4 QSA/PLE eager boundaries resolve live request metadata through
        # the same context as piecewise prefill. Decode needs it on every
        # replay; retaining only capture-time CPU metadata would be stale.
        with set_tc_piecewise_forward_context(
            forward_batch,
            attention_layers=[],
            quant_config=getattr(self._model_runner, "quant_config", None),
            moe_layers=[],
            moe_fusions=[],
            num_tokens=(
                forward_batch.input_ids.numel()
                if forward_batch.input_ids is not None
                else forward_batch.positions.numel()
            ),
        ):
            yield

    def can_run(self, forward_batch, shape_key):
        from sglang.srt.layers.xpu_tp_bounded_head import enabled, graph_allowed

        if enabled() and not graph_allowed(forward_batch):
            from sglang.srt.layers.ar449_head import eligibility

            reason = eligibility(forward_batch)[1]
            if reason not in self._head_fallback_reasons:
                logger.info(
                    "TP bounded-head graph fallback: %s; original eager/full logits",
                    reason,
                )
                self._head_fallback_reasons.add(reason)
            return False
        return super().can_run(forward_batch, shape_key)

    def capture_one(
        self,
        shape_key,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        native_workspaces = self._native_moe_workspace_refs()
        graph = BreakableCUDAGraph()
        with (
            self._forward_context(capture_inputs),
            enable_breakable_cuda_graph(),
            _DebugCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
                barrier_fn=self._tp_group.barrier,
            ),
        ):
            out = forward_fn()
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        self._retain_native_moe_workspaces(shape_key, native_workspaces)
        self._capture_inputs[shape_key] = capture_inputs
        from sglang.srt.layers import ar188_metadata as meta188

        if meta188.enabled():
            # The complete capture can contain several graphs. Record its
            # selected metadata mode without treating one segment as a census.
            meta188.record_capture(None)
        if debug_root := os.environ.get("CODEX_AR87_GRAPH_DIR"):
            from sglang.srt.runtime_context import get_parallel

            target = Path(debug_root) / f"tp-rank-{get_parallel().tp_rank}"
            target = target / f"decode-{len(self._graphs) - 1}"
            target.mkdir(parents=True, exist_ok=True)
            for index, segment in enumerate(graph._segments):
                segment.debug_dump(str(target / f"segment-{index:03d}.dot"))
            (target / "shape.json").write_text(
                json.dumps(
                    dict(
                        shape_key=repr(shape_key),
                        segments=len(graph._segments),
                        eager_calls=len(graph._break_fns),
                    ),
                    indent=2,
                )
                + "\n"
            )
        logger.info(
            "XPU breakable decode captured: shape=%s segments=%d eager_calls=%d",
            shape_key,
            len(graph._segments),
            len(graph._break_fns),
        )

    def replay(self, shape_key, static_forward_batch, **kwargs):
        with self._forward_context(static_forward_batch):
            if self._native_gguf:
                # This loader has its own compressed residency manager and
                # does not use the older FP8 CPU40 pool/fixture modules.
                self._graphs[shape_key].replay()
                return self._outputs[shape_key]
            return super().replay(shape_key, static_forward_batch, **kwargs)

    @contextmanager
    def replay_session(self):
        communicator = getattr(self._tp_group, "xpu_communicator", None)
        peer = getattr(communicator, "peer_allreduce", None)
        if peer is not None:
            peer.raise_if_failed()
        with enable_breakable_cuda_graph():
            yield
        if peer is not None:
            peer.raise_if_failed()

    def cleanup(self) -> None:
        super().cleanup()
        self._capture_inputs.clear()
