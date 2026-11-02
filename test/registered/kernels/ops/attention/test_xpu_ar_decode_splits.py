"""B70 ordinary AR split selection and changing paged-attention metadata."""

import unittest

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.srt.layers.attention.xpu_backend import ar_decode_split_count
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestArDecodeSplits(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 ordinary AR attention")

    def fixture(self):
        torch.manual_seed(42)
        q = torch.randn((1, 16, 256), device="xpu", dtype=torch.bfloat16)
        k = torch.randn((40, 128, 2, 256), device="xpu", dtype=q.dtype)
        v = torch.randn_like(k)
        pages = torch.randperm(40, device="xpu")[:32].int().view(1, 32)
        return q, k, v, pages

    def test_explicit_default_and_unmeasured_layouts_keep_native_dispatch(self):
        q, k, v, pages = self.fixture()
        self.assertEqual(ar_decode_split_count(q, k, v, pages, 8, 0), 8)
        self.assertEqual(ar_decode_split_count(q, k, v, pages, 0, 0), 0)
        self.assertEqual(ar_decode_split_count(q, k, v, pages, 8, 1), 1)
        for tensors in (
            (q.expand(2, -1, -1).contiguous(), k, v, pages),
            (q.half(), k, v, pages),
            (q[:, :, :128], k, v, pages),
            (q, k[:, ::2], v[:, ::2], pages),
            (q, k, v, None),
            (q, k, v, pages.long()),
            (q, k, v, pages.expand(2, -1)),
            (q, k, v, torch.zeros((1, 33), device="xpu", dtype=torch.int32)),
        ):
            with self.subTest(shapes=[getattr(t, "shape", None) for t in tensors]):
                self.assertEqual(ar_decode_split_count(*tensors, 8, 0), 0)

    def test_graph_uses_changing_lengths_pages_and_queries(self):
        q, k, v, pages = self.fixture()
        length = torch.tensor([1056], device="xpu", dtype=torch.int32)
        cuq = torch.tensor([0, 1], device="xpu", dtype=torch.int32)

        def run(splits):
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                page_table=pages,
                cache_seqlens=length,
                cu_seqlens_q=cuq,
                max_seqlen_q=1,
                softmax_scale=0.0625,
                causal=True,
                num_splits=splits,
            )

        selected = ar_decode_split_count(q, k, v, pages, 8, 0)
        run(selected)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = run(selected)
        for n in (1, 127, 128, 129, 1056, 1279, 4096):
            length.fill_(n)
            q.normal_()
            pages.copy_(torch.randperm(40, device="xpu")[:32].int().view(1, 32))
            graph.replay()
            expected = run(0)
            self.assertTrue(bool(torch.isfinite(actual).all().item()))
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)


if __name__ == "__main__":
    unittest.main()
