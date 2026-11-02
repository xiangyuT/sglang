"""Preserve one-row routing when concurrent rows are added or graph inputs change."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe import gguf_topk
from sglang.srt.layers.moe.topk import fused_topk_torch_native
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestGGUFTopK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 selector")
        if not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"):
            raise unittest.SkipTest("Build the GGUF library")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

    def route(self, x, renorm, enabled, native_norm=True):
        hidden = torch.empty(x.shape[0], 2560, dtype=torch.bfloat16, device=x.device)
        with patch.dict(os.environ, {
            "SGLANG_XPU_GGUF_TOPK": str(int(enabled)),
            "SGLANG_XPU_GGUF_TOPK_NORM": str(int(native_norm)),
        }):
            return fused_topk_torch_native(hidden, x, 10, renorm)

    def one_row_reference(self, x, renorm):
        pairs = [self.route(row, renorm, False) for row in x.split(1)]
        return tuple(torch.cat([pair[i] for pair in pairs]) for i in range(2))

    def exact(self, actual, expected):
        self.assertTrue(torch.equal(
            actual[0].cpu().view(torch.int32), expected[0].cpu().view(torch.int32)
        ))
        self.assertEqual(actual[1].dtype, torch.int64)
        self.assertTrue(torch.equal(actual[1].cpu(), expected[1].cpu()))

    def test_dtypes_ties_nonfinite_and_dynamic_graph(self):
        for tokens in range(1, 9):
            for dtype in (torch.bfloat16, torch.float32):
                for renorm in (False, True):
                    with self.subTest(tokens=tokens, dtype=dtype, renorm=renorm):
                        x = torch.randn(
                            tokens, 512, generator=torch.Generator().manual_seed(2707)
                        ).to(dtype).to("xpu")
                        self.exact(self.route(x, renorm, True), self.one_row_reference(x, renorm))
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            actual = self.route(x, renorm, True)
                        for label in ("random", "zero", "tie", "inf", "nan"):
                            values = torch.randn(
                                tokens, 512, generator=torch.Generator().manual_seed(2711)
                            ).to(dtype)
                            if label == "zero":
                                values.zero_()
                            elif label == "tie":
                                values.zero_()
                                values[:, 8:24] = 1
                            elif label in ("inf", "nan"):
                                values.fill_(float(label))
                            x.copy_(values)
                            graph.replay()
                            self.exact(actual, self.one_row_reference(x, renorm))
                        del graph

    def test_adding_neighbors_preserves_selected_weights(self):
        values = torch.randn(1, 512, generator=torch.Generator().manual_seed(932))
        values = values.bfloat16().to("xpu")
        expected = self.route(values, True, True)
        for tokens in range(1, 9):
            actual = self.route(values.repeat(tokens, 1), True, True)
            self.exact(actual, tuple(value.repeat(tokens, 1) for value in expected))

    def test_torch_normalization_remains_available(self):
        for tokens in (1, 4, 8):
            x = torch.randn(tokens, 512, device="xpu")
            self.exact(self.route(x, True, True, native_norm=False),
                       self.route(x, True, False))

    def test_guard_and_noneligible_fallback(self):
        for shape in ((9, 512), (1, 256), (1, 512)):
            x = torch.randn(shape, device="xpu")
            if shape == (1, 512):
                x.requires_grad_(True)
            self.assertFalse(gguf_topk.supports(x))
            self.exact(self.route(x, True, True), self.route(x, True, False))
        self.assertFalse(gguf_topk.supports(torch.zeros(1, 512)))
        self.assertFalse(gguf_topk.supports(torch.zeros(1, 512, dtype=torch.bfloat16, device="xpu")))


if __name__ == "__main__":
    unittest.main()
