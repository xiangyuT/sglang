import unittest

import torch

from sglang.kernels.ops.attention.xpu_dflash_prefill import try_native_dflash_prefill
from sglang.srt.environ import envs


@unittest.skipUnless(
    torch.xpu.is_available() and envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get(),
    "XPU and explicitly built native DFlash library required",
)
class TestNativeDFlashPrefill(unittest.TestCase):
    def test_exact_native_prefill_changing_graph_and_fallback(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        torch.manual_seed(42)
        for m in range(2, 9):
            q = torch.randn(m, 16, 256, device="xpu", dtype=torch.bfloat16)
            k = torch.randn(64, 128, 2, 256, device="xpu", dtype=q.dtype)
            v = torch.randn_like(k)
            table = torch.randperm(64, device="xpu", dtype=torch.int32)[:32][None]
            lens = torch.tensor([m], device="xpu", dtype=torch.int32)
            cuq = torch.tensor([0, m], device="xpu", dtype=torch.int32)
            out = torch.empty_like(q)

            def op(**overrides):
                args = dict(
                    q=q,
                    key_cache=k,
                    value_cache=v,
                    page_table=table,
                    cache_seqlens=lens,
                    cu_seqlens_q=cuq,
                    max_seqlen_q=m,
                    softmax_scale=256**-0.5,
                    out=out,
                )
                args.update(overrides)
                return try_native_dflash_prefill(**args)

            self.assertIs(op(), out)
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                self.assertIs(op(), out)
            for prefix in (0, 127, 128, 1024, 4096 - m):
                with self.subTest(tokens=m, prefix=prefix):
                    q.normal_()
                    k.normal_()
                    v.normal_()
                    lens.fill_(prefix + m)
                    table.copy_(
                        torch.randperm(64, device="xpu", dtype=torch.int32)[:32][None]
                    )
                    reference = flash_attn_with_kvcache(
                        q=q,
                        k_cache=k,
                        v_cache=v,
                        page_table=table,
                        cache_seqlens=lens,
                        cu_seqlens_q=cuq,
                        max_seqlen_q=m,
                        softmax_scale=256**-0.5,
                        causal=True,
                        num_splits=0,
                    )
                    graph.replay()
                    torch.testing.assert_close(out, reference, rtol=0, atol=0)
            del graph
            out.fill_(float("nan"))
            self.assertIsNone(op(max_seqlen_q=1))
            self.assertIsNone(op(page_table=table.expand(2, -1)))
            self.assertIsNone(op(page_table=table.repeat(1, 2)))
            self.assertIsNone(op(softmax_scale=1.0))
            self.assertIsNone(op(cu_seqlens_q=cuq.long()))
            self.assertTrue(torch.isnan(out).all().item())


if __name__ == "__main__":
    unittest.main()
