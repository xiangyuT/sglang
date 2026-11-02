"""IQ2S grouping, in-place cache transitions and raw/prefill interoperability."""

from collections import OrderedDict
import os
import unittest

import numpy as np
import torch

from sglang.kernels.ops.quantization.gguf_iq2s_pvr import (
    group_iq2s_routes,
    repack_iq2s_cache,
)
from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="xpu-unit-test")


class TestIQ2SGrouping(unittest.TestCase):
    def test_grouped_routes_preserve_original_destinations_and_logical_experts(self):
        ids = np.array([[7, 2, 4], [4, 7, 2]], dtype=np.int32)
        resident = {7: 0, 2: 2, 4: 1}
        value = group_iq2s_routes(ids, resident, "cpu")
        self.assertEqual(value.groups, 3)
        self.assertEqual(value.route_slots.tolist(), [0, 2, 1, 1, 0, 2])
        self.assertEqual(value.group_slots.tolist(), [2, 1, 0])
        self.assertEqual(value.starts.tolist(), [0, 2, 4, 6])
        self.assertEqual(value.destinations.tolist(), [1, 5, 2, 3, 0, 4])
        self.assertEqual(value.token_rows.tolist(), [0, 1, 0, 1, 0, 1])
        for group in range(value.groups):
            begin, end = value.starts[group : group + 2].tolist()
            destinations = value.destinations[begin:end]
            expected = value.route_slots[destinations]
            self.assertTrue(bool(torch.all(expected == value.group_slots[group])))
        with self.assertRaises(ValueError):
            group_iq2s_routes(ids, {7: 0, 2: 0, 4: 1}, "cpu")


def sources_and_layer(slots=4):
    layer = torch.nn.Module()
    method = CompressedGGUFMoEMethod(slots, decode_q8=True)
    method.create_weights(layer, 6, 2560, 640, torch.bfloat16)
    method.sources = []
    for seed, (name, kind, rows, width, block, size) in enumerate(
        (
            ("gate", 22, 640, 2560, 256, 82),
            ("up", 22, 640, 2560, 256, 82),
            ("down", 42, 2560, 640, 64, 18),
        )
    ):
        data = np.random.default_rng(seed).integers(
            0, 256, (6 * rows * width // block, size), dtype=np.uint8
        )
        data[:, :2] = np.array([0.25], dtype=np.float16).view(np.uint8)
        source = CompressedGGUFRows.__new__(CompressedGGUFRows)
        source.rows, source.width, source.kind = 6 * rows, width, kind
        source.row_bytes = width // block * size
        source._mapping = data.reshape(source.rows, source.row_bytes)
        source._mapping.flags.writeable = False
        method.sources.append(source)
        layer.register_buffer(
            "gguf_" + name,
            torch.empty(
                slots, rows * source.row_bytes, dtype=torch.uint8, device="xpu"
            ),
        )
    method.fused_q8 = True
    return method, layer


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestIQ2SPackedCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 kernel")
        path = os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")
        if not path or not hasattr(_library(path), "xpu_gguf_moe_q8_pvr"):
            raise unittest.SkipTest("Build the IQ2S grouped GGUF library")

    def test_replacements_and_queued_format_transitions_preserve_original_bytes(self):
        method, layer = sources_and_layer(slots=3)
        snapshots = []
        for repeat in range(2):
            for index, experts in enumerate(
                ([1, 2, 3], [0, 1], [1, 2, 4], [0, 4], [4, 5])
            ):
                packed = (index + repeat) % 2 == 0
                method._ensure_group(layer, experts, packed_gu=packed)
                self.assertTrue(method._iq2s_packed_slots <= set(method.cache.values()))
                self.assertTrue(
                    all(
                        (method.cache[e] in method._iq2s_packed_slots) == packed
                        for e in experts
                    )
                )
                copies = [
                    getattr(layer, "gguf_" + name).clone()
                    for name in ("gate", "up", "down")
                ]
                residents = OrderedDict(method.cache)
                repack_iq2s_cache(
                    copies[0],
                    copies[1],
                    method._iq2s_packed_slots,
                    to_packed=False,
                    down=copies[2],
                )
                snapshots.append((copies, residents))
        for copies, residents in snapshots:
            for actual, source in zip(copies, method.sources):
                actual = actual.cpu()
                rows = source.rows // method.num_experts
                for expert, slot in residents.items():
                    expected = source.packed_row_range(
                        expert * rows, (expert + 1) * rows
                    )
                    self.assertTrue(torch.equal(actual[slot], expected.reshape(-1)))
        with self.assertRaises(ValueError):
            repack_iq2s_cache(layer.gguf_gate, layer.gguf_up, [1, 1], to_packed=True)

    def test_native_packed_decode_and_raw_prefill_match_original_q8_path(self):
        original, old_layer = sources_and_layer()
        method, layer = sources_and_layer()
        original.iq2s_pvr = False
        method.iq2s_pvr = True
        generator = torch.Generator().manual_seed(42)
        cases = (
            [[2, 0]],
            [[2, 3], [1, 0], [3, 2], [0, 1]],
            [[i % 6, (i + 1) % 6] for i in range(8)],
            [[4, 2], [1, 5], [2, 4]],
            [[4, 5], [5, 2], [2, 4], [5, 4]],
            [[0, 1], [1, 0]],
        )
        for order in cases:
            with self.subTest(rows=len(order)), torch.inference_mode():
                x = (torch.randn(len(order), 2560, generator=generator) * 0.01).to(
                    device="xpu", dtype=torch.bfloat16
                )
                ids = torch.tensor(order, dtype=torch.int32, device="xpu")
                weights = torch.tensor([[0.3, 0.7]], device="xpu").repeat(len(order), 1)
                expected = original._apply_routes.run_eager(old_layer, x, weights, ids)
                actual = method._apply_routes.run_eager(layer, x, weights, ids)
                if len(order) == 4:
                    self.assertTrue(method._iq2s_packed_slots)
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                    )
                )
        self.assertGreater(method.misses, method.cache_slots)


if __name__ == "__main__":
    unittest.main()
