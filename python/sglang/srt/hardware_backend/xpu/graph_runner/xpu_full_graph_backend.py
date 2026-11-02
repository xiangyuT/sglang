"""FullXPUGraphBackend — Intel XPU full-graph capture (torch.xpu.XPUGraph).

Mirrors FullCudaGraphBackend with XPU-specific differences:
  - Captures via torch.xpu.graph(xpu_graph=...) into torch.xpu.XPUGraph.
  - Shares the global graph memory pool with the prefill backend so that
    decode + prefill graphs occupy max(decode, prefill) rather than their sum.
  - No set_graph_pool_id: SymmetricMemoryContext is never triggered on XPU
    (oneCCL has no ncclMemAlloc equivalent; enable_symm_mem defaults False).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class FullXPUGraphBackend(BaseCudaGraphBackend):
    """One torch.xpu.XPUGraph per shape for Intel XPU devices."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
    ) -> None:
        self._graphs: Dict[Any, torch.xpu.XPUGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._native_moe_workspaces: Dict[Any, tuple[torch.Tensor, ...]] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = get_parallel().tp_group
        self._capture_stream: Optional[torch.xpu.Stream] = None
        from sglang.srt.layers import ar188_metadata as meta188

        if meta188.enabled():
            meta188.RUNNER = cuda_graph_runner

    @staticmethod
    def _native_moe_workspace_refs() -> tuple[torch.Tensor, ...]:
        # sgl-kernel-xpu grows its process-wide scratch buffers after decode
        # capture when a larger prefill arrives. The graph keeps USM addresses,
        # not these external tensors: allocator stream ordering alone cannot
        # protect a buffer for future replays after the cache replaces it.
        # GDN's wrapper has the same grow-only external-cache ownership. A
        # prefill can replace its decode allocation too; retaining only MoE
        # scratch leaves graph pointers to the old GDN buffer unowned.
        references = []
        for module_name, attribute in (
            ("sgl_kernel.moe", "_moe_ws_cache"),
            ("sgl_kernel.gdn_attn", "_gdn_ws_cache"),
        ):
            cache = getattr(sys.modules.get(module_name), attribute, None)
            if isinstance(cache, dict):
                references.extend(cache.values())
        return tuple(references)

    def _retain_native_moe_workspaces(self, shape_key, before=()) -> None:
        references = (*before, *self._native_moe_workspace_refs())
        self._native_moe_workspaces[shape_key] = tuple(
            {id(tensor): tensor for tensor in references}.values()
        )

    @contextmanager
    def capture_session(self, stream: torch.xpu.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._tp_group.world_size > 1:
            raise ValueError(
                "Full XPU TP capture has not passed collective replay validation; "
                "use --cuda-graph-backend-decode breakable."
            )
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        from sglang.srt.layers.moe.paged_experts.cpu40 import graph_boundary

        graph_boundary(capture=True)
        native_workspaces = self._native_moe_workspace_refs()
        graph = torch.xpu.XPUGraph()
        debug_root = os.environ.get("CODEX_AR87_GRAPH_DIR")
        if debug_root:
            parallel = get_parallel()
            if parallel.tp_size > 1:
                debug_root = str(Path(debug_root) / f"tp-rank-{parallel.tp_rank}")
            Path(debug_root).mkdir(parents=True, exist_ok=True)
            graph.enable_debug_mode()

        with self._device_module.graph(
            xpu_graph=graph, pool=self._pool, stream=self._capture_stream
        ):
            out = forward_fn()

        if debug_root:
            index = len(self._graphs)
            target = Path(debug_root) / f"decode-{index}.dot"
            graph.debug_dump(str(target))
            (Path(debug_root) / f"decode-{index}-shape.json").write_text(
                json.dumps(
                    dict(
                        shape_key=repr(shape_key),
                        case=os.environ.get("CODEX_AR87_CASE"),
                        boundary=os.environ.get("CODEX_AR86_BOUNDARY"),
                        census=os.environ.get("CODEX_AR86_CENSUS"),
                    ),
                    indent=2,
                )
                + "\n"
            )
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        self._retain_native_moe_workspaces(shape_key, native_workspaces)
        from sglang.srt.layers import ar188_metadata as meta188

        if meta188.enabled():
            meta188.record_capture(target if debug_root else None)

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        from sglang.srt.layers.ar188_metadata import can_use_graph

        return shape_key in self._graphs and can_use_graph(forward_batch)

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        from sglang.srt.layers.moe.paged_experts.cpu40 import graph_boundary

        graph_boundary()
        from sglang.srt.layers.ar188_metadata import validate_graph_inputs

        validate_graph_inputs(static_forward_batch)
        from sglang.srt.layers.ar449_head import prepare_replay

        prepare_replay(static_forward_batch)
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._native_moe_workspaces.clear()
        self._pool = None
