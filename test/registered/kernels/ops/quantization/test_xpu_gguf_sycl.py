"""Ported GGUF formulas, partial PLE rows and graph updates on B70."""

import os
import unittest

import gguf
import numpy as np
import torch
from sglang.kernels.ops.quantization.gguf_sycl import (
    SUPPORTED_TYPES,
    dequantize_gguf,
    expert_matvec_gguf,
    matmul_gguf,
    scatter_expert_routes,
)
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=12, suite="xpu-unit-test")


def _fixture(kind, count, seed):
    rng = np.random.default_rng(seed)
    elements, size = BLOCK_GEOMETRY[GGML_TYPES[kind]]
    packed = rng.integers(0, 256, (count // elements, size), dtype=np.uint8)
    if kind == 29:
        scales = packed[:, 48:].copy().view(np.uint16)
        for i in range(4):
            scales[:, i] = (scales[:, i] & 0xFFF) | (((0x3400 >> (4 * i)) & 15) << 12)
        packed[:, 48:] = scales.view(np.uint8)
    else:
        offset = 208 if kind == 14 else 0
        packed[:, offset : offset + 2] = np.array([0.25], dtype=np.float16).view(
            np.uint8
        )
    if kind == 42:
        code = (packed[:, 2 + np.arange(64) // 4] >> ((np.arange(64) % 4) * 2)) & 3
        reference = (code.astype(np.float32) - 1) * 0.25
    else:
        reference = gguf.dequantize(packed.reshape(-1), gguf.GGMLQuantizationType(kind))
    return packed.reshape(-1), reference.reshape(-1)


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestGgufSycl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 decoder port")
        if not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"):
            raise unittest.SkipTest("Build the GGUF decoder and set its library path")

    def test_quantized_formats_match_independent_cpu(self):
        for kind in sorted(SUPPORTED_TYPES):
            with self.subTest(kind=kind):
                raw, expected = _fixture(kind, 512, kind)
                packed = torch.from_numpy(raw).to("xpu")
                for dtype in (torch.float16, torch.float32):
                    actual = dequantize_gguf(packed, kind, dtype=dtype).cpu()
                    reference = torch.from_numpy(expected).to(dtype)
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int16), reference.view(torch.int16)
                        )
                    )

    def test_partial_superblocks_and_dynamic_graph_input(self):
        for kind, count in [(20, 32), (20, 160), (8, 160), (42, 64), (42, 640)]:
            with self.subTest(kind=kind, count=count):
                raw, _ = _fixture(kind, count, 42)
                packed = torch.from_numpy(raw).to("xpu")
                storage = torch.full(
                    (count + 32,), 13.75, dtype=torch.float16, device="xpu"
                )
                output = storage[16:-16]
                dequantize_gguf(packed, kind, out=output)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    dequantize_gguf(packed, kind, out=output)
                for seed in (19, 42, 71):
                    raw, expected = _fixture(kind, count, seed)
                    packed.copy_(torch.from_numpy(raw))
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            output.cpu().view(torch.int16),
                            torch.from_numpy(expected).half().view(torch.int16),
                        )
                    )
                    self.assertTrue(bool((storage[:16] == 13.75).all()))
                    self.assertTrue(bool((storage[-16:] == 13.75).all()))

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            dequantize_gguf(torch.empty(18, dtype=torch.uint8), 42)
        packed = torch.empty(18, dtype=torch.uint8, device="xpu")
        with self.assertRaises(ValueError):
            dequantize_gguf(packed, 255)
        with self.assertRaises(ValueError):
            dequantize_gguf(packed[:17], 42)
        with self.assertRaises(ValueError):
            dequantize_gguf(packed, 42, out=torch.empty(63, device="xpu"))

    def test_matvec_and_prefill_match_fp64_reference(self):
        generator = torch.Generator().manual_seed(42)
        for kind in sorted(SUPPORTED_TYPES):
            width = 640 if kind == 42 else (160 if kind == 20 else 256)
            rows = 5
            raw, values = _fixture(kind, rows * width, kind)
            packed = torch.from_numpy(raw).to("xpu")
            for tokens in (1, 2, 8, 11):
                with self.subTest(kind=kind, tokens=tokens):
                    x = torch.randn(tokens, width, generator=generator)
                    expected = (
                        x.double()
                        @ torch.from_numpy(values).view(rows, width).double().T
                    )
                    actual = matmul_gguf(x.to("xpu"), packed, kind, rows).cpu()
                    # Random Q6_K scales produce cancellation-prone dot products.
                    # Bound forward error by |x| @ |w|, alongside relative L2.
                    condition_scale = (
                        x.double().abs()
                        @ torch.from_numpy(values).view(rows, width).double().abs().T
                    )
                    error = (actual.double() - expected).abs()
                    self.assertTrue(
                        bool((error <= 2e-6 * condition_scale + 2e-6).all())
                    )
                    self.assertLess(float(error.norm() / expected.norm()), 2e-6)
        x_gpu = x[:1].to("xpu")
        matmul_gguf(x_gpu, packed, kind, rows)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            output = matmul_gguf(x_gpu, packed, kind, rows)
        for scale in (0.5, -2.0, 1.0):
            x_gpu.copy_((x[:1] * scale))
            graph.replay()
            expected = (x[:1] * scale).double() @ torch.from_numpy(values).view(
                rows, width
            ).double().T
            torch.testing.assert_close(
                output.cpu(), expected.float(), rtol=2e-5, atol=2e-4
            )

    def test_prefill_reduction_repeats_exactly(self):
        raw, _ = _fixture(42, 640 * 64, 42)
        packed = torch.from_numpy(raw).to("xpu")
        x = torch.randn(128, 640, generator=torch.Generator().manual_seed(71)).to("xpu")
        baseline = matmul_gguf(x, packed, 42, 64).cpu()
        for _ in range(30):
            output = matmul_gguf(x, packed, 42, 64).cpu()
            self.assertTrue(
                torch.equal(baseline.view(torch.int32), output.view(torch.int32))
            )

    def test_routed_gate_up_and_down_match_single_expert_bits(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                width, rows, capacity = 256, 7, 4
                caches = []
                for seed in (19, 42):
                    raw, _ = _fixture(kind, width * rows * capacity, seed)
                    caches.append(torch.from_numpy(raw).reshape(capacity, -1).to("xpu"))
                x = torch.randn(1, width, generator=torch.Generator().manual_seed(71))
                x = x.to("xpu")
                order = [3, 1, 3, 0, 2]
                slots = torch.tensor(order, dtype=torch.int32, device="xpu")
                for _ in range(3):
                    gate, up = expert_matvec_gguf(
                        x, caches[0], kind, slots, rows, up=caches[1]
                    )
                    for actual, cache in zip((gate, up), caches):
                        expected = torch.cat(
                            [matmul_gguf(x, cache[slot], kind, rows) for slot in order]
                        )
                        self.assertTrue(
                            torch.equal(
                                actual.cpu().view(torch.int32),
                                expected.cpu().view(torch.int32),
                            )
                        )
                separate_inputs = torch.randn(
                    len(order), width, generator=torch.Generator().manual_seed(123)
                ).to("xpu")
                actual = expert_matvec_gguf(
                    separate_inputs, caches[0], kind, slots, rows
                )
                expected = torch.cat(
                    [
                        matmul_gguf(
                            separate_inputs[i : i + 1], caches[0][slot], kind, rows
                        )
                        for i, slot in enumerate(order)
                    ]
                )
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                    )
                )

    def test_routed_graph_updates_and_slot_bounds(self):
        raw, _ = _fixture(22, 4 * 7 * 256, 42)
        cache = torch.from_numpy(raw).reshape(4, -1).to("xpu")
        x = torch.randn(1, 256, generator=torch.Generator().manual_seed(19)).to("xpu")
        slots = torch.tensor([3, 1, 0], dtype=torch.int32, device="xpu")
        expert_matvec_gguf(x, cache, 22, slots, 7, up=cache)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            gate, up = expert_matvec_gguf(x, cache, 22, slots, 7, up=cache)
        for order in ([0, 2, 3], [1, 1, 0], [3, 2, 1]):
            slots.copy_(torch.tensor(order, dtype=torch.int32))
            x.mul_(-0.5)
            graph.replay()
            expected = torch.cat([matmul_gguf(x, cache[i], 22, 7) for i in order])
            for actual in (gate, up):
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                    )
                )
        slots.copy_(torch.tensor([-1, 4, 0], dtype=torch.int32))
        graph.replay()
        self.assertTrue(bool(torch.isnan(gate[:2]).all()))
        self.assertTrue(bool(torch.isfinite(gate[2]).all()))
        with self.assertRaises(ValueError):
            expert_matvec_gguf(x, cache, 22, slots.long(), 7, up=cache)
        with self.assertRaises(ValueError):
            expert_matvec_gguf(x, cache, 22, slots, 8, up=cache)

    def test_strata_q8_expert_tails_routing_and_dynamic_graph(self):
        def quantized_reference(x):
            blocks = x.numpy().reshape(-1, 32)
            scale = np.minimum(
                np.max(np.abs(blocks), axis=1) / np.float32(127), np.float32(65504)
            )
            ratio = np.divide(
                blocks,
                scale[:, None],
                out=np.zeros_like(blocks),
                where=scale[:, None] != 0,
            )
            codes = np.clip(
                np.copysign(np.floor(np.abs(ratio) + np.float32(0.5)), ratio), -127, 127
            )
            value = (
                codes.astype(np.float64)
                * scale.astype(np.float16).astype(np.float64)[:, None]
            )
            return torch.from_numpy(value.reshape(x.shape))

        generator = torch.Generator().manual_seed(42)
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                k, n, capacity = (640 if kind == 42 else 2560), 7, 4
                raw, values = _fixture(kind, k * n * capacity, kind)
                weights = torch.from_numpy(values).view(capacity, n, k).double()
                packed = torch.from_numpy(raw).reshape(capacity, -1).to("xpu")
                pair = kind != 42
                x = torch.randn((4 if pair else 5, k), generator=generator)
                x[:, :32] = 0
                xgpu = x.to("xpu")
                order = [3, 1, 3, 0, 2]
                mapping = [0, 1, 3, 2, 1] if pair else list(range(5))
                slots = torch.tensor(order, dtype=torch.int32, device="xpu")
                tokens = (
                    torch.tensor(mapping, dtype=torch.int32, device="xpu")
                    if pair
                    else None
                )
                scratch = torch.empty(
                    (x.shape[0], k // 32 * 36), dtype=torch.uint8, device="xpu"
                )

                def run():
                    return expert_matvec_gguf(
                        xgpu,
                        packed,
                        kind,
                        slots,
                        n,
                        up=packed if pair else None,
                        tokens=tokens,
                        q8_activations=True,
                        q8_workspace=scratch,
                    )

                def check(actual, scale):
                    decoded = quantized_reference(x * scale)
                    expected = torch.stack(
                        [weights[e] @ decoded[t] for e, t in zip(order, mapping)]
                    )
                    condition = torch.stack(
                        [
                            weights[e].abs() @ decoded[t].abs()
                            for e, t in zip(order, mapping)
                        ]
                    )
                    for output in actual if pair else (actual,):
                        error = (output.cpu().double() - expected).abs()
                        self.assertTrue(bool((error <= condition * 1e-5 + 2e-5).all()))
                        self.assertLess(float(error.norm() / expected.norm()), 5e-4)

                check(run(), 1)
                fresh = expert_matvec_gguf(
                    xgpu,
                    packed,
                    kind,
                    slots,
                    n,
                    up=packed if pair else None,
                    tokens=tokens,
                    q8_activations=True,
                )
                reused = run()
                for a, b in zip(
                    fresh if pair else (fresh,), reused if pair else (reused,)
                ):
                    self.assertTrue(
                        torch.equal(
                            a.cpu().view(torch.int32), b.cpu().view(torch.int32)
                        )
                    )
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = run()
                for scale, order in (
                    (0.5, [0, 2, 3, 1, 0]),
                    (-2.0, [1, 1, 0, 2, 3]),
                    (1.0, [3, 2, 1, 0, 1]),
                ):
                    slots.copy_(torch.tensor(order, dtype=torch.int32))
                    xgpu.copy_(x * scale)
                    graph.replay()
                    check(output, scale)
                slots.copy_(torch.tensor([-1, 4, 0, 1, 2], dtype=torch.int32))
                graph.replay()
                for value in output if pair else (output,):
                    self.assertTrue(bool(torch.isnan(value[:2]).all()))
                    self.assertTrue(bool(torch.isfinite(value[2:]).all()))

    def test_decode_q8_workspace_is_owned_and_stream_specific(self):
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )

        method = CompressedGGUFMoEMethod(3, decode_q8=True)
        device = torch.device("xpu", 0)
        first = method._decode_q8_workspace(device, 1, 2560)
        self.assertIs(first, method._decode_q8_workspace(device, 1, 2560))
        down = method._decode_q8_workspace(device, 10, 640)
        self.assertIsNot(first, down)
        with torch.xpu.stream(torch.xpu.Stream()):
            other_stream = method._decode_q8_workspace(device, 1, 2560)
        self.assertIsNot(first, other_stream)
        self.assertEqual(len(method._decode_q8_workspaces), 3)

    def test_moe_dispatch_type_survives_breakable_replay(self):
        from collections import OrderedDict
        from types import SimpleNamespace

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            enable_breakable_cuda_graph,
        )

        layer = torch.nn.Module()
        method = CompressedGGUFMoEMethod(2)
        method.create_weights(layer, 2, 256, 64, torch.bfloat16)
        method.sources = [SimpleNamespace(kind=42)] * 3
        method.cache = OrderedDict(((0, 0), (1, 1)))
        for seed, name in enumerate(("gate", "up", "down")):
            raw, _ = _fixture(42, 2 * 256 * 64, seed)
            layer.register_buffer(
                "gguf_" + name, torch.from_numpy(raw).reshape(2, -1).to("xpu")
            )
        x = torch.randn(1, 256, generator=torch.Generator().manual_seed(42))
        x = x.to(device="xpu", dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="xpu")
        weights = torch.tensor([[0.25, 0.75]], device="xpu")

        def forward():
            dispatch = StandardDispatchOutput(
                x + 0, None, StandardTopKOutput(weights, ids + 0, weights)
            )
            combined = method.apply(layer, dispatch)
            self.assertIsInstance(combined, StandardCombineInput)
            return combined.hidden_states + 0

        for _ in range(3):
            forward()
        graph = BreakableCUDAGraph()
        with (
            enable_breakable_cuda_graph(),
            BreakableCUDAGraphCapture(graph, pool=torch.xpu.graph_pool_handle()),
        ):
            output = forward()
        for order in ([[1, 0]], [[0, 0]], [[0, 1]]):
            ids.copy_(torch.tensor(order, dtype=torch.int32))
            x.mul_(-0.5)
            with enable_breakable_cuda_graph():
                graph.replay()
            expected = forward()
            self.assertTrue(
                torch.equal(
                    output.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                )
            )

    def test_async_expert_replacements_keep_slot_bytes_and_selected_hits(self):
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )
        from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows

        layer = torch.nn.Module()
        method = CompressedGGUFMoEMethod(3)
        method.create_weights(layer, 5, 256, 64, torch.bfloat16)
        method.sources = []
        for seed, (name, rows, width) in enumerate(
            (("gate", 64, 256), ("up", 64, 256), ("down", 256, 64))
        ):
            raw, _ = _fixture(42, 5 * rows * width, seed)
            source = CompressedGGUFRows.__new__(CompressedGGUFRows)
            source.row_bytes, source.rows = width // 64 * 18, rows * 5
            source._mapping = raw.reshape(source.rows, source.row_bytes)
            source._mapping.flags.writeable = False
            method.sources.append(source)
            layer.register_buffer(
                "gguf_" + name,
                torch.empty(
                    (3, rows * source.row_bytes), dtype=torch.uint8, device="xpu"
                ),
            )
        snapshots = []
        for _ in range(6):
            for group in ([1, 2, 3], [0, 1], [1, 2, 4], [0, 4], [1, 2, 3], [4]):
                resident = method._ensure_group(layer, group)
                self.assertEqual(set(resident), set(group))
                self.assertEqual(len(set(resident.values())), len(group))
                for name, source in zip(("gate", "up", "down"), method.sources):
                    # Queue snapshots without a host wait: newly released pinned
                    # allocations must not be recycled while DMA still reads them.
                    snapshot = getattr(layer, "gguf_" + name).clone()
                    snapshots.append((snapshot, source, dict(method.cache)))
        for snapshot, source, residents in snapshots:
            actual = snapshot.cpu()
            rows = source.rows // method.num_experts
            for expert, slot in residents.items():
                expected = source.packed_row_range(expert * rows, (expert + 1) * rows)
                self.assertTrue(torch.equal(actual[slot], expected.reshape(-1)))

    def test_qsa_graph_uses_replay_request_slots(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
            QwenSparseAttnMetadata,
        )

        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.req_to_token = torch.tensor(
            [[0, 0, 0, 0, 0, 0], [7, 19, 4, 30, 5, 2], [33, 9, 7, 20, 14, 4]],
            dtype=torch.int32,
            device="xpu",
        )
        lengths = torch.tensor([4, 3], dtype=torch.int32, device="xpu")
        request_rows = torch.tensor([2, 1], dtype=torch.int32, device="xpu")
        logical = torch.tensor(
            [[0, 2, -1, 3, 4], [2, 0, 1, -1, 3]],
            dtype=torch.int32,
            device="xpu",
        )
        metadata = QwenSparseAttnMetadata(
            sequence_lengths=lengths,
            token_to_batch_idx=torch.arange(2, device="xpu"),
            token_slot_table=torch.zeros((2, 1), dtype=torch.int32, device="xpu"),
            indexer_metadata=None,  # Address mapping does not consume the indexer.
            row_req_pool_indices=request_rows,
            is_cuda_graph=True,
        )
        expected = torch.tensor([[33, 7, -1, 20, -1], [4, 7, 19, -1, -1]])
        actual = backend._logical_to_physical(logical, metadata)
        self.assertTrue(torch.equal(actual.cpu(), expected))
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = backend._logical_to_physical(logical, metadata)
        request_rows.copy_(torch.tensor([1, 2], dtype=torch.int32))
        lengths.copy_(torch.tensor([3, 4], dtype=torch.int32))
        graph.replay()
        expected = torch.tensor([[7, 4, -1, -1, -1], [7, 33, 9, -1, 20]])
        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_general_route_rows_and_scatter(self):
        raw, _ = _fixture(22, 3 * 7 * 256, 42)
        cache = torch.from_numpy(raw).reshape(3, -1).to("xpu")
        x = torch.randn(4, 256, generator=torch.Generator().manual_seed(19)).to("xpu")
        slots = torch.tensor([2, 0, 1, 2, 0], dtype=torch.int32, device="xpu")
        tokens = torch.tensor([1, 0, 3, 2, 1], dtype=torch.int32, device="xpu")
        gate, _ = expert_matvec_gguf(x, cache, 22, slots, 7, up=cache, tokens=tokens)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            gate, up = expert_matvec_gguf(
                x, cache, 22, slots, 7, up=cache, tokens=tokens
            )
        for token_rows in ([3, 2, 1, 0, 3], [0, 0, 2, 1, 2]):
            tokens.copy_(torch.tensor(token_rows, dtype=torch.int32))
            graph.replay()
            expected = torch.cat(
                [
                    matmul_gguf(x[t : t + 1], cache[e], 22, 7)
                    for t, e in zip(token_rows, [2, 0, 1, 2, 0])
                ]
            ).cpu()
            for actual in (gate, up):
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int32), expected.view(torch.int32)
                    )
                )
        storage = torch.full((9, 7), 13.75, device="xpu")
        target = storage[1:-1]
        destinations = torch.tensor([3, 1, 4, 0, 2], dtype=torch.int32, device="xpu")
        scatter_expert_routes(gate, destinations, target)
        reference = torch.full((9, 7), 13.75)
        reference[torch.tensor([3, 1, 4, 0, 2]) + 1] = expected
        self.assertTrue(
            torch.equal(storage.cpu().view(torch.int32), reference.view(torch.int32))
        )
        tokens.copy_(torch.tensor([-1, 4, 0, 1, 2], dtype=torch.int32))
        graph.replay()
        self.assertTrue(bool(torch.isnan(gate[:2]).all()))
        self.assertTrue(bool(torch.isfinite(gate[2:]).all()))

    def test_resident_hits_consumed_before_eviction(self):
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )

        class Source:
            kind = 42

            def __init__(self, packed):
                self.packed = packed
                self.rows = packed.shape[0]

            def packed_rows(self, indices):
                return self.packed[torch.as_tensor(indices)]

            def packed_row_range(self, start, stop, *, pin_memory=False):
                value = self.packed[start:stop].clone()
                return value.pin_memory() if pin_memory else value

        method = CompressedGGUFMoEMethod(2)
        layer = torch.nn.Module()
        method.create_weights(layer, 3, 256, 64, torch.bfloat16)
        method.sources = []
        full_weights = []
        for seed, (name, rows, width) in enumerate(
            (("gate", 64, 256), ("up", 64, 256), ("down", 256, 64))
        ):
            raw, _ = _fixture(42, 3 * rows * width, seed)
            packed = torch.from_numpy(raw).reshape(3 * rows, -1)
            method.sources.append(Source(packed))
            full_weights.append(packed.reshape(3, -1).to("xpu"))
            layer.register_buffer(
                "gguf_" + name,
                torch.empty((2, packed.numel() // 3), dtype=torch.uint8, device="xpu"),
            )
        x = torch.randn(4, 256, generator=torch.Generator().manual_seed(42)).to(
            device="xpu", dtype=torch.bfloat16
        )
        ids_cpu = [[0, 1], [2, 0], [1, 2], [2, 1]]
        ids = torch.tensor(ids_cpu, dtype=torch.int64, device="xpu")
        weights = torch.full((4, 2), 0.5, device="xpu")
        reference_parts = []
        for token, order in enumerate(ids_cpu):
            parts = []
            for expert in order:
                activation = x[token : token + 1].float()
                gate = matmul_gguf(activation, full_weights[0][expert], 42, 64)
                up = matmul_gguf(activation, full_weights[1][expert], 42, 64)
                parts.append(
                    matmul_gguf(
                        torch.nn.functional.silu(gate) * up,
                        full_weights[2][expert],
                        42,
                        256,
                    )
                )
            reference_parts.append(torch.cat(parts))
        expected = (
            (torch.stack(reference_parts) * weights.unsqueeze(-1))
            .sum(dim=1)
            .to(x.dtype)
            .cpu()
        )
        for index in range(4):
            previous_hits, previous_misses = method.hits, method.misses
            actual = method._apply_routes(layer, x, weights, ids).cpu()
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
            if index:
                self.assertEqual(method.hits - previous_hits, 2)
                self.assertEqual(method.misses - previous_misses, 1)
            self.assertEqual(len(set(method.cache.values())), len(method.cache))
            self.assertLessEqual(len(method.cache), 2)


if __name__ == "__main__":
    unittest.main()
