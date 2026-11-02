"""GGUF GDN decode normalization must retain the one-token BF16 result."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm, _layer_norm_fwd
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=15, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestGGUFGDNDecodeNorm(unittest.TestCase):
    def test_distinct_tokens_match_one_token_with_graph_replay(self):
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            self.skipTest("B70 one-token reduction contract")
        torch.manual_seed(975)
        norm = RMSNorm(
            128, eps=1e-6, activation="sigmoid", device="xpu", dtype=torch.bfloat16
        ).eval()
        with torch.no_grad():
            norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + 1)
        tokens = torch.randn(8, 24, 128, device="xpu", dtype=torch.bfloat16)
        gates = torch.randn_like(tokens)

        with torch.inference_mode():
            for batch in range(1, 9):
                x = tokens[:batch].reshape(-1, 128).clone()
                z = gates[:batch].reshape(-1, 128).clone()

                def run():
                    return _layer_norm_fwd(
                        x, norm.weight, None, norm.eps,
                        z=z, is_rms_norm=True, activation="sigmoid",
                        rows_per_block=1,
                    )[0]

                eager = run()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    captured = run()
                for replay in range(2):
                    if replay:
                        x.copy_(x.roll(1, dims=0))
                        z.copy_(z.roll(-1, dims=0))
                    expected = torch.cat([
                        norm(a, b)
                        for a, b in zip(x.split(24), z.split(24))
                    ])
                    graph.replay()
                    torch.xpu.synchronize()
                    actual = captured if replay else eager
                    with self.subTest(batch=batch, replay=replay):
                        self.assertTrue(torch.equal(
                            actual.view(torch.int16), expected.view(torch.int16)
                        ))
                        self.assertTrue(torch.equal(
                            captured.view(torch.int16), expected.view(torch.int16)
                        ))


if __name__ == "__main__":
    unittest.main()
