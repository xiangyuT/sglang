"""Native MoE cache growth must not invalidate an already captured graph."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
    FullXPUGraphBackend,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="xpu-unit-test")


class TestBreakableDraftForwardContext(unittest.TestCase):
    def test_qsa_graph_boundary_preserves_gpu_only_metadata(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer

        metadata = object()
        backend = SimpleNamespace(get_indexer_metadata=lambda *_: metadata)
        batch = SimpleNamespace(seq_lens_cpu=None)
        indexer = Mock(return_value=torch.zeros(1, 2, dtype=torch.int32))
        layer = SimpleNamespace(layer_id=3, indexer=indexer)
        hidden, positions = torch.empty(1, 2), torch.tensor([4])
        with (
            forward_context(ForwardContext(attn_backend=backend)),
            patch(
                "sglang.srt.models.qwen4_exp._shards_rows_over_cp", return_value=False
            ),
        ):
            Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices_eager(
                layer, hidden, positions, batch, use_host_prefill_lengths=True
            )
        indexer.assert_called_once_with(hidden, positions, batch, metadata)

    def test_qsa_break_uses_captured_step_backend_and_live_batch(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            get_attn_backend,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            set_tc_piecewise_forward_context,
        )
        from sglang.srt.models.qwen4_exp import _breakable_qsa_indexer

        live_batch = SimpleNamespace(current_request=3)
        step_backend, outer_backend = object(), object()

        def compute(**kwargs):
            self.assertIs(get_attn_backend(), step_backend)
            self.assertIs(kwargs["forward_batch"], live_batch)
            self.assertEqual(kwargs["forward_batch"].current_request, 3)
            return torch.tensor([[2, 7]], dtype=torch.int32)

        layer = SimpleNamespace(
            _qsa_prefill_topk_bridges={}, _compute_qsa_topk_indices_eager=compute
        )
        with (
            forward_context(ForwardContext(attn_backend=outer_backend)),
            set_tc_piecewise_forward_context(live_batch, [], None, [], []),
        ):
            output = _breakable_qsa_indexer.run_eager(
                layer, torch.empty(1, 2), torch.tensor([4]), step_backend
            )
            self.assertEqual(output.tolist(), [[2, 7]])
            self.assertIs(get_attn_backend(), outer_backend)

            first_pointer = output.data_ptr()
            layer._compute_qsa_topk_indices_eager = lambda **kwargs: torch.ones(
                4, 3, dtype=torch.int32
            )
            other = _breakable_qsa_indexer.run_eager(
                layer, torch.empty(4, 2), torch.arange(4), step_backend
            )
            self.assertEqual(tuple(other.shape), (4, 3))
            layer._compute_qsa_topk_indices_eager = compute
            repeated = _breakable_qsa_indexer.run_eager(
                layer, torch.empty(1, 2), torch.tensor([4]), step_backend
            )
            self.assertEqual(repeated.data_ptr(), first_pointer)
            self.assertEqual(repeated.tolist(), [[2, 7]])

    def test_context_keeps_live_batch_with_or_without_draft_input_ids(self):
        from sglang.srt.hardware_backend.xpu.graph_runner.xpu_breakable_graph_backend import (
            BreakableXPUGraphBackend,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            get_tc_piecewise_forward_context,
        )

        backend = BreakableXPUGraphBackend.__new__(BreakableXPUGraphBackend)
        quant_config = object()
        backend._model_runner = SimpleNamespace(quant_config=quant_config)
        for input_ids in (None, torch.zeros(3, dtype=torch.int64)):
            with self.subTest(input_ids_present=input_ids is not None):
                batch = SimpleNamespace(input_ids=input_ids, positions=torch.arange(4))
                with backend._forward_context(batch):
                    context = get_tc_piecewise_forward_context()
                    self.assertIs(context.forward_batch, batch)
                    self.assertIs(context.quant_config, quant_config)
                    self.assertEqual(context.num_tokens, 4 if input_ids is None else 3)
                    batch.live_metadata = 7
                    self.assertEqual(context.forward_batch.live_metadata, 7)
                self.assertIsNone(get_tc_piecewise_forward_context())


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestNativeMoeGraphWorkspace(unittest.TestCase):
    def test_gdn_prefill_growth_preserves_decode_storage_until_cleanup(self):
        from sgl_kernel import gdn_attn

        if not hasattr(gdn_attn, "_get_gdn_workspace"):
            self.skipTest("Native GDN workspace cache API unavailable")
        stream = torch.xpu.Stream()
        device = torch.device("xpu", torch.xpu.current_device())
        with torch.xpu.stream(stream), torch.inference_mode():
            key = gdn_attn._stream_key(device)
            self.assertNotIn(key, gdn_attn._gdn_ws_cache)
            backend = FullXPUGraphBackend.__new__(FullXPUGraphBackend)
            backend._graphs = {}
            backend._outputs = {}
            backend._native_moe_workspaces = {}
            backend._pool = None
            try:
                value = gdn_attn._get_gdn_workspace(4096, device)
                value.fill_(7)
                pointer = value.data_ptr()
                base_ref = weakref.ref(value)
                output = torch.empty_like(value)
                for _ in range(3):
                    torch.add(value, 1, out=output)
                stream.synchronize()
                before = backend._native_moe_workspace_refs()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    torch.add(value, 1, out=output)
                backend._graphs[0] = graph
                backend._outputs[0] = output
                backend._retain_native_moe_workspaces(0, before)
                del before, value, graph

                grown = gdn_attn._get_gdn_workspace(4096 * 512, device)
                grown.fill_(31)
                gc.collect()
                self.assertIsNotNone(base_ref())
                pressure = [
                    torch.empty(5120, dtype=torch.uint8, device=device)
                    for _ in range(64)
                ]
                self.assertTrue(all(t.data_ptr() != pointer for t in pressure))
                for tensor in pressure:
                    tensor.fill_(99)
                for _ in range(3):
                    backend._graphs[0].replay()
                stream.synchronize()
                self.assertTrue(bool(torch.all(output == 8)))

                backend.cleanup()
                gc.collect()
                self.assertIsNone(base_ref())
            finally:
                stream.synchronize()
                backend.cleanup()
                gdn_attn._gdn_ws_cache.pop(key, None)

    def test_prefill_growth_preserves_decode_storage_until_cleanup(self):
        from sgl_kernel import moe

        if not hasattr(moe, "_get_moe_ws"):
            self.skipTest("Native MoE workspace cache API unavailable")

        for dtype in (torch.bfloat16, torch.int32):
            with self.subTest(dtype=dtype), torch.inference_mode():
                name = f"test_graph_workspace_{dtype}"
                device = torch.device("xpu", torch.xpu.current_device())
                key = (name, device)
                backend = FullXPUGraphBackend.__new__(FullXPUGraphBackend)
                backend._graphs = {}
                backend._outputs = {}
                backend._native_moe_workspaces = {}
                backend._pool = None
                try:
                    value = moe._get_moe_ws(name, (4096,), dtype, device)
                    value.fill_(7)
                    pointer = value.data_ptr()
                    base_ref = weakref.ref(moe._moe_ws_cache[key])
                    output = torch.empty_like(value)
                    for _ in range(3):
                        torch.add(value, 1, out=output)
                    torch.xpu.synchronize()
                    before = backend._native_moe_workspace_refs()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        torch.add(value, 1, out=output)
                    backend._graphs[0] = graph
                    backend._outputs[0] = output
                    backend._retain_native_moe_workspaces(0, before)
                    del before, value, graph

                    # A real prefill grows the same cache names far beyond the
                    # B1 decode allocations, dropping the previous cache owner.
                    grown = moe._get_moe_ws(name, (4096 * 512,), dtype, device)
                    grown.fill_(31)
                    gc.collect()
                    self.assertIsNotNone(base_ref())
                    pressure = [
                        torch.empty(4096, dtype=dtype, device=device) for _ in range(64)
                    ]
                    self.assertTrue(all(t.data_ptr() != pointer for t in pressure))
                    for tensor in pressure:
                        tensor.fill_(99)
                    for _ in range(3):
                        backend._graphs[0].replay()
                    torch.xpu.synchronize()
                    self.assertTrue(bool(torch.all(output == 8)))

                    backend.cleanup()
                    gc.collect()
                    self.assertIsNone(base_ref())
                finally:
                    torch.xpu.synchronize()
                    backend.cleanup()
                    moe._moe_ws_view_cache.pop(key, None)
                    moe._moe_ws_cache.pop(key, None)


if __name__ == "__main__":
    unittest.main()
