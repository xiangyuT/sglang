"""Draft FP8 head preserves target storage and replays on changing hiddens."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dflash_head_xpu import (
    XpuDflashCandidateHead,
    can_prepare_candidate_head,
)
from sglang.kernels.ops.speculative.dflash_topk_xpu import xpu_dflash_topk
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestDflashCandidateHead(CustomTestCase):
    def test_private_copy_quantization_and_graph(self):
        torch.manual_seed(42)
        w = torch.randn(256, 2048, device="xpu", dtype=torch.bfloat16)
        w[:128, :128] = 0
        original = w.clone()
        pointer, version = w.data_ptr(), w._version
        head = XpuDflashCandidateHead(w)
        self.assertNotEqual(w.data_ptr(), head.weight.data_ptr())
        # Independent CPU block quantization, with a full zero block included.
        blocks = w.cpu().float().view(2, 128, 16, 128).permute(0, 2, 1, 3)
        scales = blocks.abs().amax((-1, -2)).clamp_min(1e-12) / 448.0
        expected = (
            (blocks / scales[:, :, None, None])
            .to(torch.float8_e4m3fn)
            .permute(0, 2, 1, 3)
            .reshape(w.shape)
        )
        torch.testing.assert_close(head.scale.cpu(), scales, rtol=0, atol=0)
        self.assertTrue(
            torch.equal(head.weight.cpu().view(torch.uint8), expected.view(torch.uint8))
        )
        x = torch.randn(3, 2048, device="xpu", dtype=torch.bfloat16)
        head(x)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            y = head(x)
        for step in range(3):
            x.copy_(torch.randn_like(x))
            if step == 0:
                x.zero_()
            graph.replay()
            torch.testing.assert_close(y, head(x), rtol=0, atol=0)
            torch.testing.assert_close(
                y,
                torch_w8a8_block_fp8_linear(x, head.weight, [128, 128], head.scale),
                rtol=0,
                atol=0,
            )
            self.assertTrue(head.matches(x, w))
        self.assertEqual((w.data_ptr(), w._version), (pointer, version))
        torch.testing.assert_close(w, original, rtol=0, atol=0)
        w.add_(1)
        self.assertFalse(head.matches(x, w))

    def test_prepare_guard(self):
        w = torch.empty(248320, 2048, device="xpu", dtype=torch.bfloat16)
        h = SimpleNamespace(weight=w, org_vocab_size=248320)
        self.assertTrue(can_prepare_candidate_head(h, 1))
        self.assertFalse(can_prepare_candidate_head(h, 2))
        self.assertFalse(
            can_prepare_candidate_head(
                SimpleNamespace(weight=w, org_vocab_size=248319), 1
            )
        )
        self.assertFalse(
            can_prepare_candidate_head(
                SimpleNamespace(weight=w[:128], org_vocab_size=128), 1
            )
        )
        self.assertFalse(can_prepare_candidate_head(None, 1))

    def test_shortlist_rerank_and_graph(self):
        torch.manual_seed(42)
        w = torch.randn(1024, 2048, device="xpu", dtype=torch.bfloat16)
        x = torch.randn(3, 2048, device="xpu", dtype=torch.bfloat16)
        source = w.clone()
        head = XpuDflashCandidateHead(w)
        head.refined_top16(x, w)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            vals, ids = head.refined_top16(x, w)
        for step in range(4):
            x.copy_(torch.randn_like(x))
            if step == 0:
                x.zero_()
            graph.replay()
            expected_values, expected_ids = xpu_dflash_topk(torch.matmul(x, w.T), 16)
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(vals, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(w, source, rtol=0, atol=0)

    def test_top64_ties_and_cpu_reference(self):
        torch.manual_seed(42)
        for n in (64, 4095, 4096, 4097, 248320):
            x = torch.randn(3, n, device="xpu", dtype=torch.bfloat16)
            x[0].zero_()
            values, ids = xpu_dflash_topk(x, 64)
            cpu = x.cpu()
            expected = torch.argsort(cpu.float(), descending=True, stable=True, dim=-1)[
                :, :64
            ]
            torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                values.cpu(), torch.gather(cpu, 1, expected), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
