import unittest

import torch
from sglang.kernels.ops.speculative.dflash_topk_xpu import (
    can_use_xpu_dflash_topk,
    xpu_dflash_topk,
)


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestDFlashTopkXpu(unittest.TestCase):
    def test_values_indices_and_changing_input_graph(self):
        for n, v in [(1, 17), (3, 4097), (7, 248320), (8, 262144)]:
            with self.subTest(rows=n, vocab=v):
                torch.manual_seed(42)
                x = torch.randn(n, v, device="xpu").bfloat16()
                self.assertTrue(can_use_xpu_dflash_topk(x, 16))
                eager = xpu_dflash_topk(x)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    captured = xpu_dflash_topk(x)
                for variant in range(4):
                    if variant == 1:
                        x.fill_(1)  # ties across every tile and padding boundary
                    elif variant == 2:
                        x.normal_()
                        x[:, 0] = float("-inf")
                        x[:, -1] = float("inf")
                    elif variant == 3:
                        x.normal_()
                        x[:, 0] = float("nan")
                        x[:, -1] = float("nan")
                    expected_ids = torch.argsort(
                        x.float(), dim=-1, descending=True, stable=True
                    )[:, :16]
                    expected_vals = x.gather(-1, expected_ids)
                    graph.replay()
                    for vals, ids in [eager, captured] if variant == 0 else [captured]:
                        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                        torch.testing.assert_close(
                            vals, expected_vals, rtol=0, atol=0, equal_nan=True
                        )
                del graph

    def test_dispatch_guards(self):
        x = torch.empty(7, 248320, device="xpu", dtype=torch.bfloat16)
        self.assertFalse(can_use_xpu_dflash_topk(x, 8))
        self.assertFalse(can_use_xpu_dflash_topk(x.float(), 16))
        self.assertFalse(can_use_xpu_dflash_topk(x[:, ::2], 16))
        self.assertFalse(can_use_xpu_dflash_topk(torch.empty(16), 16))

    def test_selector_dispatch(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.dflash import _radix_topk

        x = torch.randn(7, 248320, device="xpu").bfloat16()
        expected = xpu_dflash_topk(x)
        with envs.SGLANG_XPU_PAGED_DFLASH.override(True):
            with envs.SGLANG_XPU_DFLASH_TOPK.override(True):
                actual = _radix_topk(x, 16)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with envs.SGLANG_XPU_DFLASH_TOPK.override(False):
                actual = _radix_topk(x, 16)
                native = torch.topk(x, 16, dim=-1)
                for a, b in zip(actual, native):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
