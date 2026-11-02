"""GPU LRU byte ownership, graph growth and unchanged GGUF MoE arithmetic."""

from collections import OrderedDict
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sglang.kernels.ops.quantization.gguf_gpu_cache import (
    GPUExpertCache,
    encode_host_experts,
)
from sglang.kernels.ops.quantization.gguf_iq2s_pvr import IQ2SGroupedRoutes
from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts
from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="xpu-unit-test")


def pack_gu_cpu(raw):
    e = raw.shape[0]
    x = raw.numpy().reshape(e, 40, 16, 10, 82)
    y = np.empty((e, 40, 10, 1312), dtype=np.uint8)
    y[..., 1280:] = x[..., :2].transpose(0, 1, 3, 2, 4).reshape(e, 40, 10, 32)
    for p in range(8):
        y[..., p * 160 : p * 160 + 64] = (
            x[..., 2 + 4 * p : 6 + 4 * p]
            .transpose(0, 1, 3, 2, 4)
            .reshape(e, 40, 10, 64)
        )
        y[..., p * 160 + 64 : p * 160 + 128] = (
            x[..., 34 + 4 * p : 38 + 4 * p]
            .transpose(0, 1, 3, 2, 4)
            .reshape(e, 40, 10, 64)
        )
        qh = x[..., 66 + p].transpose(0, 1, 3, 2)
        scale = x[..., 74 + p].transpose(0, 1, 3, 2)
        y[..., p * 160 + 128 : p * 160 + 160] = np.stack((qh, scale), axis=-1).reshape(
            e, 40, 10, 32
        )
    return torch.from_numpy(y.reshape(raw.shape))


def pack_down_cpu(raw):
    e = raw.shape[0]
    x = raw.numpy().reshape(e, 160, 16, 10, 18)
    y = np.empty((e, 160, 10, 288), dtype=np.uint8)
    for p in range(2):
        y[..., p * 128 : p * 128 + 64] = (
            x[..., 2 + 8 * p : 6 + 8 * p]
            .transpose(0, 1, 3, 2, 4)
            .reshape(e, 160, 10, 64)
        )
        y[..., p * 128 + 64 : p * 128 + 128] = (
            x[..., 6 + 8 * p : 10 + 8 * p]
            .transpose(0, 1, 3, 2, 4)
            .reshape(e, 160, 10, 64)
        )
    y[..., 256:] = x[..., :2].transpose(0, 1, 3, 2, 4).reshape(e, 160, 10, 32)
    return torch.from_numpy(y.reshape(raw.shape))


class DynamicFixture:
    def __init__(self, library, kind, experts=80, capacity=48, *, packed_gu=False):
        self.lib, self.kind, self.experts, self.cap = library, kind, experts, capacity
        self.gu = {16: 422400, 22: 524800, 29: 358400, 42: 460800}[kind]
        self.sizes = (self.gu, self.gu, 460800)
        generator = torch.Generator().manual_seed(kind)
        raw = [
            torch.randint(0, 256, (experts, n), dtype=torch.uint8, generator=generator)
            for n in self.sizes
        ]
        self.original = raw
        self.expected = (
            [encode_host_experts(x, kind, i, packed_gu=True) for i, x in enumerate(raw)]
            if packed_gu
            else (
                [
                    pack_gu_cpu(x) if i < 2 else pack_down_cpu(x)
                    for i, x in enumerate(raw)
                ]
                if kind == 22
                else raw
            )
        )
        initial = [x[:capacity].to("xpu") for x in self.expected]
        self.chunks = [initial]
        self.host = [x[capacity:].pin_memory() for x in self.expected]
        self.state = GPUExpertCache(
            kind,
            range(capacity),
            initial,
            range(capacity, experts),
            self.host,
            layout="lowbit_gu_packed"
            if packed_gu
            else "iq2s_q2_pvr"
            if kind == 22
            else "original",
        )
        self.chunks = self.state.resident_chunks
        self.mirror = self.state.mirror_addresses
        self.expert_slot = self.state.expert_slots
        self.slot_expert = self.state.slot_experts
        self.last = [e + 1 if e < capacity else 0 for e in range(experts)]
        self.used = self.state.last_used
        self.clock = capacity
        self.clocks = self.state.counters
        self.capacity = self.state.capacity
        self.addresses = self.state.slot_addresses
        self.meta = None
        self.routes = None
        self.cache = OrderedDict((e, e) for e in range(capacity))
        self.last_cold = 0

    def reference(self, ids):
        selected = sorted(set(ids.reshape(-1).tolist()))
        self.last_cold = sum(e not in self.cache for e in selected)
        order = [e for e in selected if e in self.cache] + [
            e for e in selected if e not in self.cache
        ]
        for e in order:
            if e in self.cache:
                slot = self.cache.pop(e)
            else:
                _, slot = self.cache.popitem(last=False)
            self.cache[e] = slot
            self.clock += 1
            self.last[e] = self.clock

    def call(self, ids):
        self.routes = self.state.prepare(ids)
        self.meta = self.routes.grouping.data

    def verify(self, ids):
        routes = ids.numel()
        meta = self.meta[: 6 * routes + 3].cpu().tolist()
        assert meta[6 * routes + 2] == 0, meta[6 * routes + 2]
        selected = sorted(set(ids.reshape(-1).tolist()))
        assert meta[:routes] == [self.cache[e] for e in ids.reshape(-1).tolist()]
        assert meta[routes : routes + len(selected)] == [
            self.cache[e] for e in selected
        ]
        slots = self.expert_slot.cpu().tolist()
        assert slots == [self.cache.get(e, -1) for e in range(self.experts)]
        inverse = [-1] * self.experts
        for e, slot in self.cache.items():
            inverse[slot] = e
        assert self.slot_expert.cpu().tolist() == inverse
        assert self.used.cpu().tolist() == self.last
        assert self.clocks.cpu().tolist() == [self.clock, self.last_cold]
        expected_indices = torch.tensor(inverse[: self.cap], dtype=torch.long)
        for field in range(3):
            actual = torch.cat([chunk[field].cpu() for chunk in self.chunks])
            assert torch.equal(
                actual, self.expected[field].index_select(0, expected_indices)
            ), (self.kind, field, "GPU bytes")
        table = self.mirror.cpu().tolist()
        outside = [e for e in range(self.experts) if e not in self.cache]
        for e in self.cache:
            assert table[e] == [0, 0, 0]
        for field, size in enumerate(self.sizes):
            base = self.host[field].data_ptr()
            positions = []
            for e in outside:
                delta = table[e][field] - base
                assert delta >= 0 and delta % size == 0
                position = delta // size
                assert position < self.host[field].shape[0]
                positions.append(position)
            assert len(set(positions)) == len(positions)
            actual = self.host[field].index_select(
                0, torch.tensor(positions, dtype=torch.long)
            )
            expected = self.expected[field].index_select(
                0, torch.tensor(outside, dtype=torch.long)
            )
            assert torch.equal(actual, expected), (
                self.kind,
                field,
                "host mirror bytes",
            )

    def grow(self, count):
        selected = [e for e in range(self.experts) if e not in self.cache][:count]
        chunk = [
            torch.empty((count, n), dtype=torch.uint8, device="xpu") for n in self.sizes
        ]
        self.state.grow(chunk, selected)
        self.host = self.state.mirrors
        for row, e in enumerate(selected):
            self.cache[e] = self.cap + row
            self.clock += 1
            self.last[e] = self.clock
        self.cap += count


def numeric_fixture(library, kind, *, packed_gu=False):
    f = DynamicFixture(None, kind, packed_gu=packed_gu)
    sources = []
    raws = []
    for field, (ty, rows, width) in enumerate(
        ((kind, 640, 2560), (kind, 640, 2560), (42, 2560, 640))
    ):
        block, size = {16: (256, 66), 22: (256, 82), 29: (256, 56), 42: (64, 18)}[ty]
        data = np.random.default_rng(field).integers(
            0, 256, (f.experts * rows * width // block, size), dtype=np.uint8
        )
        if ty == 29:
            scales = data[:, 48:].copy().view(np.uint16)
            for i in range(4):
                scales[:, i] = (scales[:, i] & 0xFFF) | (
                    ((0x2400 >> (i * 4)) & 15) << 12
                )
            data[:, 48:] = scales.view(np.uint8)
        else:
            data[:, :2] = np.array([0.001953125], dtype=np.float16).view(np.uint8)
        raw = torch.from_numpy(data.reshape(f.experts, -1))
        raws.append(raw)
        source = CompressedGGUFRows.__new__(CompressedGGUFRows)
        source.width, source.rows, source.kind = width, rows * f.experts, ty
        source.row_bytes = width // block * size
        source._mapping = data.reshape(source.rows, source.row_bytes)
        source._mapping.flags.writeable = False
        sources.append(source)
    f.expected = (
        [encode_host_experts(x, kind, i, packed_gu=True) for i, x in enumerate(raws)]
        if packed_gu
        else (
            [pack_gu_cpu(x) if i < 2 else pack_down_cpu(x) for i, x in enumerate(raws)]
            if kind == 22
            else raws
        )
    )
    f.original = raws
    for field in range(3):
        f.chunks[0][field].copy_(f.expected[field][: f.cap])
        f.host[field].copy_(f.expected[field][f.cap :])
    method = CompressedGGUFMoEMethod(f.cap, decode_q8=True)
    layer = torch.nn.Module()
    method.create_weights(layer, f.experts, 2560, 640, torch.bfloat16)
    method.bind_rows(layer, sources, torch.device("xpu:0"))
    method.fused_q8 = True
    method.iq2s_pvr = True
    method.cache = OrderedDict((e, e) for e in range(f.cap))
    for name, raw in zip(("gate", "up", "down"), raws):
        getattr(layer, "gguf_" + name).copy_(raw[: f.cap])
    return f, method, layer


def candidate(f, x, weights, ids, workspaces):
    f.call(ids)
    routes = ids.numel()
    grouping = None
    if f.kind == 22 or f.state.packed_gu:
        grouping = IQ2SGroupedRoutes(
            data=f.meta,
            route_slots=f.meta[:routes],
            group_slots=f.meta[routes : 2 * routes],
            starts=f.meta[2 * routes : 3 * routes + 1],
            token_rows=f.meta[3 * routes + 1 : 4 * routes + 1],
            destinations=f.meta[4 * routes + 1 : 5 * routes + 1],
            groups=routes,
        )
    return fused_q8_experts(
        x,
        *f.chunks[0],
        f.kind,
        f.meta[:routes],
        weights,
        workspaces,
        grouping=grouping,
        slot_addresses=f.state.slot_addresses,
    )


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestGPUCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 native cache")
        path = os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")
        if not path or not hasattr(_library(path), "xpu_gguf_gpu_cache_prepare"):
            raise unittest.SkipTest("Build/select the GPU cache GGUF library")
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        os.environ["SGLANG_XPU_GGUF_NATIVE_STAGE"] = "1"

    def test_lru_slot_identity_and_every_expert_byte_after_exchanges(self):
        for kind in (16, 22, 29, 42):
            f = DynamicFixture(None, kind)
            self.addCleanup(f.state.close)
            for dtype in (torch.int32, torch.int64):
                for step, tokens in enumerate((1, 4, 2, 4, 3, 4)):
                    with self.subTest(kind=kind, dtype=dtype, tokens=tokens, step=step):
                        ids = torch.randint(
                            0,
                            f.experts,
                            (tokens, 10),
                            generator=torch.Generator().manual_seed(step + 31),
                            dtype=dtype,
                        )
                        f.reference(ids)
                        f.call(ids.to("xpu"))
                        f.verify(ids)

    def test_stateful_graph_replay_and_post_capture_growth(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                f = DynamicFixture(None, kind)
                self.addCleanup(f.state.close)
                ids = torch.randint(
                    0,
                    f.experts,
                    (4, 10),
                    generator=torch.Generator().manual_seed(42),
                    dtype=torch.int32,
                )
                gpu = ids.to("xpu")
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    f.call(gpu)
                for seed in (7, 19, 31):
                    changed = torch.randint(
                        0,
                        f.experts,
                        (4, 10),
                        generator=torch.Generator().manual_seed(seed),
                        dtype=torch.int32,
                    )
                    gpu.copy_(changed)
                    f.reference(changed)
                    graph.replay()
                    f.verify(changed)
                pointers = (
                    f.mirror.data_ptr(),
                    f.addresses.data_ptr(),
                    f.meta.data_ptr(),
                )
                f.grow(8)
                assert all(v.shape[0] == f.experts - f.cap for v in f.host)
                assert sum(v.untyped_storage().nbytes() for v in f.host) == (f.experts - f.cap) * sum(f.sizes)
                assert pointers == (
                    f.mirror.data_ptr(),
                    f.addresses.data_ptr(),
                    f.meta.data_ptr(),
                )
                for seed in (43, 71, 89):
                    changed = torch.randint(
                        0,
                        f.experts,
                        (4, 10),
                        generator=torch.Generator().manual_seed(seed),
                        dtype=torch.int32,
                    )
                    gpu.copy_(changed)
                    f.reference(changed)
                    graph.replay()
                    f.verify(changed)

    def test_full_outputs_preserve_existing_native_arithmetic(self):
        for kind in (16, 22, 29, 42):
            f, method, layer = numeric_fixture(None, kind)
            self.addCleanup(f.state.close)
            workspaces = {}
            for index, tokens in enumerate((4, 1, 2, 3, 4, 4)):
                with self.subTest(kind=kind, tokens=tokens, index=index):
                    generator = torch.Generator().manual_seed(17 + index)
                    ids = torch.randint(
                        0,
                        f.experts,
                        (tokens, 10),
                        generator=generator,
                        dtype=torch.int32,
                    )
                    ids_gpu = ids.to("xpu")
                    x = (
                        (torch.randn(tokens, 2560, generator=generator) * 0.1)
                        .bfloat16()
                        .to("xpu")
                    )
                    weights = torch.rand(tokens, 10, generator=generator)
                    weights = (weights / weights.sum(dim=1, keepdim=True)).to("xpu")
                    expected = method._apply_routes.run_eager(
                        layer, x, weights, ids_gpu
                    )
                    f.reference(ids)
                    actual = candidate(f, x, weights, ids_gpu, workspaces)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                    self.assertEqual(f.cache, method.cache)
                    f.verify(ids)

    def test_captured_full_outputs_after_adding_physical_slot_chunks(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                f, method, layer = numeric_fixture(None, kind)
                self.addCleanup(f.state.close)
                generator = torch.Generator().manual_seed(409)
                ids = torch.randint(
                    0, f.experts, (4, 10), generator=generator, dtype=torch.int32
                )
                gpu = ids.to("xpu")
                x = (
                    (torch.randn(4, 2560, generator=generator) * 0.1)
                    .bfloat16()
                    .to("xpu")
                )
                weights = torch.rand(4, 10, generator=generator)
                weights = (weights / weights.sum(dim=1, keepdim=True)).to("xpu")
                workspaces = {}
                expected = method._apply_routes.run_eager(layer, x, weights, gpu)
                f.reference(ids)
                actual = candidate(f, x, weights, gpu, workspaces)
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                    )
                )
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = candidate(f, x, weights, gpu, workspaces)

                def check(changed, recorded_graph=graph):
                    gpu.copy_(changed)
                    expected = method._apply_routes.run_eager(layer, x, weights, gpu)
                    f.reference(changed)
                    recorded_graph.replay()
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                    self.assertEqual(f.cache, method.cache)
                    f.verify(changed)

                for seed in (61, 73):
                    check(
                        torch.randint(
                            0,
                            f.experts,
                            (4, 10),
                            generator=torch.Generator().manual_seed(seed),
                            dtype=torch.int32,
                        )
                    )
                promoted = [e for e in range(f.experts) if e not in f.cache][:8]
                address = f.state.slot_addresses.data_ptr()
                f.grow(8)
                method.grow_cache(layer, 56)
                method._ensure_group(layer, promoted, packed_gu=kind == 22)
                self.assertEqual(f.state.slot_addresses.data_ptr(), address)
                self.assertEqual(f.cache, method.cache)
                # Every selected row is in a chunk that did not exist at capture.
                check(torch.tensor((promoted * 5), dtype=torch.int32).reshape(4, 10))
                for seed in (89, 103):
                    check(
                        torch.randint(
                            0,
                            f.experts,
                            (4, 10),
                            generator=torch.Generator().manual_seed(seed),
                            dtype=torch.int32,
                        )
                    )
                del graph

    def test_original_prefill_staging_after_exchanges_and_growth(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                f = DynamicFixture(None, kind)
                self.addCleanup(f.state.close)
                for field in range(3):
                    self.assertTrue(
                        torch.equal(
                            encode_host_experts(f.original[field], kind, field),
                            f.expected[field],
                        )
                    )
                ids = torch.randint(
                    0,
                    f.experts,
                    (4, 10),
                    dtype=torch.int32,
                    generator=torch.Generator().manual_seed(17),
                )
                f.reference(ids)
                f.call(ids.to("xpu"))
                f.grow(8)
                before = [
                    v.cpu().clone()
                    for v in (f.expert_slot, f.slot_expert, f.clocks, f.mirror)
                ]
                for start in range(0, f.experts, 32):
                    experts = list(range(start, min(start + 32, f.experts)))
                    buffers = [
                        torch.empty((len(experts), n), dtype=torch.uint8, device="xpu")
                        for n in f.sizes
                    ]
                    f.state.stage_original(experts, buffers)
                    for field in range(3):
                        self.assertTrue(
                            torch.equal(
                                buffers[field].cpu(), f.original[field][experts]
                            )
                        )
                for old, new in zip(
                    before, (f.expert_slot, f.slot_expert, f.clocks, f.mirror)
                ):
                    self.assertTrue(torch.equal(old, new.cpu()))

    def test_model_method_prefill_decode_and_post_capture_growth(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                f, baseline, old_layer = numeric_fixture(None, kind)
                f.state.close()
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_XPU_GGUF_GPU_CACHE": "1",
                        "SGLANG_XPU_GGUF_MOE_Q8_FUSED": "1",
                        "SGLANG_XPU_GGUF_IQ2S_PVR": "1",
                    },
                ):
                    method = CompressedGGUFMoEMethod(48, decode_q8=True)
                    layer = torch.nn.Module()
                    method.create_weights(layer, f.experts, 2560, 640, torch.bfloat16)
                    method.bind_rows(layer, baseline.sources, torch.device("xpu:0"))
                self.addCleanup(method._gpu_cache.close)
                for index, tokens in enumerate((7, 4, 2, 9, 1, 4)):
                    generator = torch.Generator().manual_seed(701 + index)
                    ids = torch.randint(
                        0,
                        f.experts,
                        (tokens, 10),
                        generator=generator,
                        dtype=torch.int32,
                    ).to("xpu")
                    x = (
                        (torch.randn(tokens, 2560, generator=generator) * 0.1)
                        .bfloat16()
                        .to("xpu")
                    )
                    weights = torch.rand(tokens, 10, generator=generator)
                    weights = (weights / weights.sum(dim=1, keepdim=True)).to("xpu")
                    dispatch = SimpleNamespace(
                        hidden_states=x, topk_output=(weights, ids, None)
                    )
                    expected = baseline._apply_routes.run_eager(
                        old_layer, x, weights, ids
                    )
                    actual = method.apply(layer, dispatch).hidden_states
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = method.apply(layer, dispatch).hidden_states
                method.grow_cache(layer, 56)
                baseline.grow_cache(old_layer, 56)
                for seed in (911, 919):
                    ids.copy_(
                        torch.randint(
                            0,
                            f.experts,
                            ids.shape,
                            generator=torch.Generator().manual_seed(seed),
                            dtype=torch.int32,
                        )
                    )
                    expected = baseline._apply_routes.run_eager(
                        old_layer, x, weights, ids
                    )
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                del graph

    def test_invalid_device_ids_preserve_experts_and_flag_output(self):
        f = DynamicFixture(None, 22, experts=12, capacity=6)
        self.addCleanup(f.state.close)
        before = f.expert_slot.cpu().clone()
        for invalid in (-1, 12):
            f.call(torch.full((4, 10), invalid, dtype=torch.int64, device="xpu"))
            self.assertEqual(f.routes.error.item(), 1)
            self.assertTrue(torch.equal(f.expert_slot.cpu(), before))
            self.assertTrue(bool((f.routes.slots == -1).all()))
        f.state.close()
        with self.assertRaises(RuntimeError):
            f.call(torch.zeros((1, 1), dtype=torch.int32, device="xpu"))

    def test_lowbit_packed_byte_ownership_and_original_prefill(self):
        for kind in (16, 29):
            with self.subTest(kind=kind):
                f = DynamicFixture(None, kind, packed_gu=True)
                self.addCleanup(f.state.close)
                ids = torch.zeros((4, 10), device="xpu", dtype=torch.int32)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    f.call(ids)
                for iteration, seed in enumerate((317, 331, 347)):
                    host_ids = torch.randint(
                        0,
                        f.experts,
                        ids.shape,
                        generator=torch.Generator().manual_seed(seed),
                        dtype=torch.int32,
                    )
                    if iteration == 1:
                        address = f.state.slot_addresses.data_ptr()
                        f.grow(8)
                        self.assertEqual(f.state.slot_addresses.data_ptr(), address)
                    ids.copy_(host_ids)
                    f.reference(host_ids)
                    graph.replay()
                    f.verify(host_ids)
                    before = [
                        v.cpu().clone()
                        for v in (f.expert_slot, f.slot_expert, f.clocks, f.mirror)
                    ]
                    for begin in range(0, f.experts, 32):
                        experts = list(range(begin, min(begin + 32, f.experts)))
                        out = [
                            torch.empty(
                                (len(experts), size), device="xpu", dtype=torch.uint8
                            )
                            for size in f.sizes
                        ]
                        f.state.stage_original(experts, out)
                        for field in range(3):
                            self.assertTrue(
                                torch.equal(
                                    out[field].cpu(), f.original[field][experts]
                                )
                            )
                    for old, new in zip(
                        before, (f.expert_slot, f.slot_expert, f.clocks, f.mirror)
                    ):
                        self.assertTrue(torch.equal(old, new.cpu()))
                del graph

    def test_lowbit_model_prefill_fp32_dots_and_captured_growth(self):
        for kind in (16, 29):
            with self.subTest(kind=kind):
                f, baseline, old_layer = numeric_fixture(None, kind)
                f.state.close()
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_XPU_GGUF_GPU_CACHE": "1",
                        "SGLANG_XPU_GGUF_MOE_Q8_FUSED": "1",
                        "SGLANG_XPU_GGUF_LOWBIT_GU": "1",
                    },
                ):
                    method = CompressedGGUFMoEMethod(48, decode_q8=True)
                    layer = torch.nn.Module()
                    method.create_weights(layer, f.experts, 2560, 640, torch.bfloat16)
                    method.bind_rows(layer, baseline.sources, torch.device("xpu:0"))
                self.addCleanup(method._gpu_cache.close)
                self.assertTrue(method._gpu_cache.packed_gu)
                for index, tokens in enumerate((7, 4, 1, 2, 3, 9, 4)):
                    gen = torch.Generator().manual_seed(1901 + index)
                    ids = torch.randint(
                        0, f.experts, (tokens, 10), generator=gen, dtype=torch.int32
                    ).to("xpu")
                    x = (
                        (torch.randn(tokens, 2560, generator=gen) * 0.1)
                        .bfloat16()
                        .to("xpu")
                    )
                    weights = torch.rand(tokens, 10, generator=gen)
                    weights = (weights / weights.sum(-1, keepdim=True)).to("xpu")
                    dispatch = SimpleNamespace(
                        hidden_states=x, topk_output=(weights, ids, None)
                    )
                    expected = baseline._apply_routes.run_eager(
                        old_layer, x, weights, ids
                    )
                    actual = method.apply(layer, dispatch).hidden_states
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                    if tokens <= 4:
                        key = (
                            x.device,
                            torch.xpu.current_stream(x.device).sycl_queue,
                            tokens,
                            10,
                        )
                        for field in ("gate", "up", "activation", "down"):
                            self.assertTrue(
                                torch.equal(
                                    method._fused_q8_workspaces[key][field]
                                    .cpu()
                                    .view(torch.int32),
                                    baseline._fused_q8_workspaces[key][field]
                                    .cpu()
                                    .view(torch.int32),
                                ),
                                (kind, tokens, field),
                            )
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = method.apply(layer, dispatch).hidden_states
                address = method._gpu_cache.slot_addresses.data_ptr()
                method.grow_cache(layer, 56)
                baseline.grow_cache(old_layer, 56)
                self.assertEqual(method._gpu_cache.slot_addresses.data_ptr(), address)
                for seed in (2027, 2039, 2053):
                    ids.copy_(
                        torch.randint(
                            0,
                            f.experts,
                            ids.shape,
                            generator=torch.Generator().manual_seed(seed),
                            dtype=torch.int32,
                        )
                    )
                    x.mul_(-0.5)
                    expected = baseline._apply_routes.run_eager(
                        old_layer, x, weights, ids
                    )
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            actual.cpu().view(torch.int16),
                            expected.cpu().view(torch.int16),
                        )
                    )
                del graph

    def test_raw_q2_dpas_exact_after_capture_and_slot_growth(self):
        for kind in (16, 22, 29, 42):
            with self.subTest(kind=kind):
                f, _, _ = numeric_fixture(None, kind, packed_gu=kind in (16, 29))
                self.addCleanup(f.state.close)
                buffers = f.state.resident_chunks[0]
                x = (torch.randn(4, 2560, generator=torch.Generator().manual_seed(2617)) * 0.1).bfloat16().to("xpu")
                weights = torch.full((4, 10), 0.1, device="xpu", dtype=torch.float32)
                live = torch.zeros((4, 10), dtype=torch.int32, device="xpu")
                workspaces = {}
                def call():
                    routes = f.state.prepare(live)
                    return fused_q8_experts(
                        x, *buffers, kind, routes.slots, weights, workspaces,
                        grouping=routes.grouping if kind in (16, 22, 29) else None,
                        slot_addresses=f.state.slot_addresses,
                    )
                with patch.dict(os.environ, {
                    "SGLANG_XPU_GGUF_RAW_Q2_DPAS": "1",
                    "SGLANG_XPU_GGUF_FUSED_QUANT": "1",
                }):
                    call()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        output = call()
                captured_ws = next(v for v in workspaces.values() if v["output"].data_ptr() == output.data_ptr())
                for index, seed in enumerate((2621, 2633, 2647)):
                    if index == 1:
                        # Baseline and graph replays have changed residency.
                        # Grow using current mirrored IDs and original encoded bytes.
                        resident = f.state.expert_slots.cpu().tolist()
                        selected = [e for e, slot in enumerate(resident) if slot < 0][:8]
                        selection = torch.tensor(selected, dtype=torch.long)
                        f.state.grow(
                            [v.index_select(0, selection).to("xpu") for v in f.expected],
                            selected,
                        )
                    ids = torch.randint(0, f.experts, live.shape, dtype=torch.int32, generator=torch.Generator().manual_seed(seed))
                    live.copy_(ids)
                    x.mul_(-0.5)
                    with patch.dict(os.environ, {
                        "SGLANG_XPU_GGUF_RAW_Q2_DPAS": "0",
                        "SGLANG_XPU_GGUF_FUSED_QUANT": "0",
                    }):
                        reference = call()
                    baseline_ws = next(v for v in workspaces.values() if v["output"].data_ptr() == reference.data_ptr())
                    saved = [reference.cpu().clone()] + [baseline_ws[n].cpu().clone() for n in ("gate", "up", "activation", "down")]
                    graph.replay()
                    actual = [output.cpu()] + [captured_ws[n].cpu() for n in ("gate", "up", "activation", "down")]
                    for a, b in zip(actual, saved):
                        view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                        self.assertTrue(torch.equal(a.view(view), b.view(view)))
                del graph

    def test_parallel_planner_unsigned_tied_ages_and_all_hits(self):
        for wg in (64, 128, 256):
            for full in (False, True):
                with (
                    self.subTest(workgroup=wg, full=full),
                    patch.dict(os.environ, {"SGLANG_XPU_GGUF_PARALLEL_PLAN": str(wg)}),
                ):
                    f = DynamicFixture(None, 42, experts=12, capacity=12 if full else 6)
                    self.addCleanup(f.state.close)
                    # Preserve unsigned comparisons and the serial planner's
                    # lowest-slot tie break; don't pack timestamps into a key.
                    base = 1 << 63
                    f.clock = base + 20
                    f.last = [base + 13 if e < f.cap else 0 for e in range(f.experts)]
                    f.used.copy_(torch.tensor(f.last, dtype=torch.uint64))
                    f.clocks.copy_(torch.tensor([f.clock, 0], dtype=torch.uint64))
                    live = torch.zeros((1, 4), device="xpu", dtype=torch.int32)
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        f.call(live)
                    for index, values in enumerate(
                        ([4, 6, 7, 4], [9, 1, 3, 9], [10, 11, 10, 11])
                    ):
                        if index == 1 and not full:
                            f.grow(2)
                        ids = torch.tensor([values], dtype=torch.int32)
                        live.copy_(ids)
                        f.reference(ids)
                        graph.replay()
                        f.verify(ids)
                    del graph


if __name__ == "__main__":
    unittest.main()
