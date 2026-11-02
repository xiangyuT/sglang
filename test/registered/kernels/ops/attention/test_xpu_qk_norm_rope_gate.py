"""B70 QK norm/RoPE fusion preserves native BF16 product boundaries."""

import unittest

import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuQkNormRopeGate(CustomTestCase):
    def _inputs(self, tokens):
        torch.manual_seed(42)
        rope = MRotaryEmbedding(
            head_size=256,
            rotary_dim=64,
            max_position_embeddings=4096,
            base=10000000,
            is_neox_style=True,
            dtype=torch.bfloat16,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to("xpu", torch.bfloat16)
        qkv = torch.randn(tokens, 9216, device="xpu", dtype=torch.bfloat16)
        qw = torch.randn(256, device="xpu", dtype=torch.bfloat16) * 0.1
        kw = torch.randn_like(qw) * 0.1
        pos = torch.randint(0, 4096, (tokens,), device="xpu")
        return qkv, qw, kw, pos, rope

    def _fused(self, qkv, qw, kw, pos, rope, rounded=True):
        qg, key, _ = qkv.split([8192, 512, 512], dim=-1)
        return fused_qk_gemma_rmsnorm_rope_gate(
            qg,
            key,
            qw,
            kw,
            rope.cos_sin_cache,
            pos,
            1e-6,
            16,
            2,
            256,
            64,
            round_rope_products=rounded,
        )

    def _native(self, qkv, qw, kw, pos, rope):
        qg, key, _ = qkv.split([8192, 512, 512], dim=-1)
        q, k, gate = fused_qk_gemma_rmsnorm_with_gate(qg, key, qw, kw, 1e-6, 256, 16)
        q, k = rope(pos, q.view(qkv.shape[0], -1), k.view(qkv.shape[0], -1))
        return q, k, gate

    def _reference(self, qkv, qw, kw, pos, rope, rounded=True):
        qkv, qw, kw, pos = [t.cpu() for t in (qkv, qw, kw, pos)]
        tokens = qkv.shape[0]
        qg = qkv[:, :8192].reshape(tokens, 16, 512)
        q, gate = qg.chunk(2, -1)
        k = qkv[:, 8192:8704].reshape(tokens, 2, 256)
        cos, sin = rope.cos_sin_cache.cpu()[pos].chunk(2, -1)
        outputs = []
        for x, weight in ((q, qw), (k, kw)):
            value = x.float()
            value = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * (weight.float() + 1)
            ).to(torch.bfloat16)
            first, second = value[..., :64].chunk(2, -1)
            a = first.float() * cos[:, None].float()
            b = second.float() * sin[:, None].float()
            c = second.float() * cos[:, None].float()
            d = first.float() * sin[:, None].float()
            if rounded:
                a, b, c, d = [v.to(torch.bfloat16).float() for v in (a, b, c, d)]
            rot = torch.cat(
                ((a - b).to(torch.bfloat16), (c + d).to(torch.bfloat16)), -1
            )
            outputs.append(torch.cat((rot, value[..., 64:]), -1))
        return (*outputs, gate)

    def test_native_and_independent_reference(self):
        for tokens in (1, 3, 17):
            args = self._inputs(tokens)
            for scale in (0.0, 1e-4, 1.0, 64.0):
                args[0].normal_().mul_(scale)
                original = args[0].clone()
                actual = self._fused(*args)
                native = self._native(*args)
                reference = self._reference(*args)
                with self.subTest(tokens=tokens, scale=scale):
                    for a, n, r in zip(actual, native, reference):
                        torch.testing.assert_close(
                            a.reshape(n.shape), n, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            a.cpu().reshape(r.shape), r, rtol=0.008, atol=0.0001
                        )
                    self.assertTrue(torch.equal(original, args[0]))

    def test_dynamic_graph_replay(self):
        args = self._inputs(1)
        for _ in range(2):
            self._fused(*args)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = self._fused(*args)
        for position in (0, 1, 1023, 1087, 4095):
            args[0].normal_()
            args[3].fill_(position)
            graph.replay()
            native = self._native(*args)
            with self.subTest(position=position):
                for a, n in zip(actual, native):
                    torch.testing.assert_close(a.reshape(n.shape), n, rtol=0, atol=0)

    def test_default_keeps_fp32_rope_products(self):
        args = self._inputs(3)
        actual = self._fused(*args, rounded=False)
        reference = self._reference(*args, rounded=False)
        for a, r in zip(actual, reference):
            torch.testing.assert_close(
                a.cpu().reshape(r.shape), r, rtol=0.008, atol=0.0001
            )


if __name__ == "__main__":
    unittest.main()
