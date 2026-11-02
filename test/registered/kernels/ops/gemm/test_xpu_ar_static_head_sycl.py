"""Static-head lossless bits, arithmetic controls, and changing-input graphs."""

import os
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.gemm import xpu_ar_static_head_sycl as op
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuArStaticHead(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 static head")
        if not os.environ.get("SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL_LIB"):
            raise unittest.SkipTest("Build and set the static head SYCL library")

    def test_all_bf16_payloads_and_fallback_layout(self):
        # Covers every sign/exponent/mantissa pattern, including NaN payloads.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        bits = bits.reshape(256, 256).t().contiguous()
        weight = bits.view(torch.bfloat16).reshape(32, 2048).to("xpu")
        prepared = op.PackedBf16Head(weight)
        self.assertEqual(prepared.verify(), 0)
        self.assertGreater(prepared.bad_blocks, 0)
        weight = torch.full((32, 2048), -0.125, dtype=torch.bfloat16, device="xpu")
        prepared = op.PackedBf16Head(weight)
        self.assertEqual(prepared.verify(), 0)
        self.assertEqual(prepared.bad_blocks, 0)
        self.assertTrue(prepared.storage()["raw_weight_retained"])

    def test_raw_packed_arithmetic_control_and_native_tolerance(self):
        for seed in (19, 42, 71):
            torch.manual_seed(seed)
            weight = torch.randn(64, 2048, device="xpu").bfloat16() * 0.02
            hidden = torch.randn(1, 2048, device="xpu").bfloat16()
            prepared = op.PackedBf16Head(weight)
            raw = prepared.gemv(hidden, packed=False)
            packed = prepared.gemv(hidden)
            self.assertTrue(
                torch.equal(raw.view(torch.int16), packed.view(torch.int16))
            )
            torch.testing.assert_close(
                packed, hidden @ weight.T, rtol=0.008, atol=0.001
            )

    def test_dynamic_graph_inputs_and_output_poison(self):
        torch.manual_seed(42)
        weight = torch.randn(64, 2048, device="xpu").bfloat16() * 0.02
        hidden = torch.randn(1, 2048, device="xpu").bfloat16()
        prepared = op.PackedBf16Head(weight)
        prepared.gemv(hidden)
        prepared.gemv(hidden, packed=False)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            packed = prepared.gemv(hidden)
            raw = prepared.gemv(hidden, packed=False)
        for seed in range(12):
            torch.manual_seed(seed)
            hidden.copy_(torch.randn_like(hidden))
            packed.fill_(float("nan"))
            raw.fill_(float("inf"))
            graph.replay()
            self.assertTrue(
                torch.equal(raw.view(torch.int16), packed.view(torch.int16))
            )
            torch.testing.assert_close(
                packed, hidden @ weight.T, rtol=0.008, atol=0.001
            )

    def test_guards_and_eager_mutation_diagnostic(self):
        weight = torch.zeros((64, 2048), dtype=torch.bfloat16, device="xpu")
        hidden = torch.zeros((1, 2048), dtype=torch.bfloat16, device="xpu")
        self.assertFalse(op.can_use_ar_static_head(hidden, weight))
        self.assertFalse(op.can_use_ar_static_head(hidden.cpu(), weight.cpu()))
        for bad in (weight[:1], weight.float(), weight[:, ::2]):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                op.PackedBf16Head(bad)
        prepared = op.PackedBf16Head(weight)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            prepared.gemv(hidden.expand(2, -1))
        weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "weight changed"):
            prepared.gemv(hidden)

    def test_preparation_must_precede_capture(self):
        op._load_library.cache_clear()
        with self.assertRaisesRegex(RuntimeError, "Set SGLANG"):
            op._load_library("")
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                op._load_library(
                    os.environ["SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL_LIB"]
                )
        op._load_library(os.environ["SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL_LIB"])


if __name__ == "__main__":
    unittest.main()
