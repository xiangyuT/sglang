import unittest

import torch

from sglang.kernels.ops.speculative.dflash_attention_xpu import (
    try_xpu_dflash_attention,
)


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestDFlashAttentionXpu(unittest.TestCase):
    def test_shuffled_prefix_window_and_changing_input_graph(self):
        torch.manual_seed(42)
        for m in range(2, 9):
            q = torch.randn(m, 48, 128, device="xpu", dtype=torch.bfloat16)[:, :32]
            k = torch.randn(m, 8, 128, device="xpu", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            kb = torch.randn(4096, 8, 128, device="xpu", dtype=torch.bfloat16)
            vb = torch.randn_like(kb)
            ids = torch.randperm(4096, device="xpu", dtype=torch.int64)
            ptr = torch.zeros(2, device="xpu", dtype=torch.int32)
            qp = torch.tensor([0, m], device="xpu", dtype=torch.int64)
            out = torch.empty_like(q)

            def op(window=2047, page_size=128, **kwargs):
                return try_xpu_dflash_attention(
                    q,
                    k,
                    v,
                    out,
                    kb,
                    vb,
                    qp,
                    ptr,
                    ids,
                    None,
                    False,
                    None,
                    m,
                    1.0,
                    1.0,
                    128**-0.5,
                    sliding_window_size=window,
                    page_size=page_size,
                    **kwargs,
                )

            self.assertTrue(op())
            self.assertTrue(op(page_size=1))
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                handled = op()
            self.assertTrue(handled)
            for length in (0, 37, 1024, 2047):
                with self.subTest(tokens=m, prefix=length):
                    ptr[1] = length
                    q.normal_()
                    ids.copy_(torch.randperm(4096, device="xpu", dtype=torch.int32))
                    keys = (
                        torch.cat([kb[ids[:length].long()], k])
                        .repeat_interleave(4, dim=1)
                        .float()
                    )
                    values = (
                        torch.cat([vb[ids[:length].long()], v])
                        .repeat_interleave(4, dim=1)
                        .float()
                    )
                    logits = torch.einsum("thd,shd->hts", q.float(), keys) * (128**-0.5)
                    col = torch.arange(length + m, device="xpu")
                    row = torch.arange(m, device="xpu")
                    mask = (col[None, :] >= length) | (
                        length + row[:, None] <= col[None, :] + 2047
                    )
                    logits.masked_fill_(~mask[None, :, :], float("-inf"))
                    ref = torch.einsum(
                        "hts,shd->thd", logits.softmax(-1), values
                    ).bfloat16()
                    graph.replay()
                    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.003)
                    relative = torch.linalg.vector_norm(
                        (out - ref).float()
                    ) / torch.linalg.vector_norm(ref.float())
                    self.assertLess(relative.item(), 0.005)
                    self.assertTrue(torch.isfinite(out).all().item())
            # At the exact left-window boundary only query 0 sees prefix slot 0.
            # Its dominant key makes an off-by-one mask observable without tolerance.
            q.zero_()
            q[:, :, 0] = 32
            k.zero_()
            v.zero_()
            kb.zero_()
            vb.zero_()
            kb[ids[0].long(), :, 0] = 32
            vb[ids[0].long()] = 10
            ptr[1] = 2047
            graph.replay()
            torch.testing.assert_close(
                out[0], torch.full_like(out[0], 10), rtol=0, atol=0
            )
            torch.testing.assert_close(
                out[1:], torch.zeros_like(out[1:]), rtol=0, atol=0
            )
            # Unsupported contracts must leave the destination untouched.
            out.fill_(float("nan"))
            self.assertFalse(op(page_size=2))
            self.assertFalse(op(window=2048))
            self.assertTrue(torch.isnan(out).all().item())
            del graph
