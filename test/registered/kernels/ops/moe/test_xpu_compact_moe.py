import unittest
import torch
from sgl_kernel import fused_experts
import os
from sglang.kernels.ops.moe.xpu_compact_moe import xpu_compact_moe as run


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "XPU and explicitly built compact MoE library required",
)
class TestCompactNative(unittest.TestCase):
    def test_replay_routes_padding_and_duplicate_rows(self):
        torch.manual_seed(42)
        w1 = (torch.randn(192, 1024, 2048, device="xpu") * 16).to(torch.float8_e4m3fn)
        w2 = (torch.randn(192, 2048, 512, device="xpu") * 16).to(torch.float8_e4m3fn)
        s1 = torch.rand(192, 8, 16, device="xpu") * 0.002 + 0.0001
        s2 = torch.rand(192, 16, 4, device="xpu") * 0.002 + 0.0001
        for m in range(2, 9):
            x = torch.randn(m, 2048, device="xpu", dtype=torch.bfloat16)
            ids = torch.arange(m * 8, device="xpu", dtype=torch.int32).view(m, 8)
            weights = torch.rand(m, 8, device="xpu")
            weights /= weights.sum(-1, keepdim=True)
            torch.testing.assert_close(
                run(x, w1, w2, s1, s2, ids, weights),
                fused_experts(
                    x,
                    w1,
                    w2,
                    weights,
                    ids,
                    use_fp8_w8a8=True,
                    w1_scale=s1,
                    w2_scale=s2,
                ),
                rtol=0,
                atol=0,
            )
            torch.xpu.synchronize()
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                out = run(x, w1, w2, s1, s2, ids, weights)
            for variant in (
                "high_slots",
                "overlap",
                "shared",
                "duplicate_one_expert",
                "all_zero_weights",
                "zero_input",
            ):
                with self.subTest(m=m, variant=variant):
                    x.normal_()
                    weights.uniform_()
                    weights /= weights.sum(-1, keepdim=True)
                    route = torch.arange(m * 8, device="xpu", dtype=torch.int32).view(
                        m, 8
                    )
                    if variant == "high_slots":
                        route += 128
                    if variant == "overlap":
                        route %= max(8, int(m * 8 * 0.6))
                    if variant == "shared":
                        route = route[:1].expand(m, -1).contiguous()
                    if variant == "duplicate_one_expert":
                        route.fill_(191)
                    if variant == "all_zero_weights":
                        route.fill_(0)
                        weights.zero_()
                    if variant == "zero_input":
                        x.zero_()
                    ids.copy_(route)
                    reference = fused_experts(
                        x,
                        w1,
                        w2,
                        weights,
                        ids,
                        use_fp8_w8a8=True,
                        w1_scale=s1,
                        w2_scale=s2,
                    )
                    for _ in range(3):
                        g.replay()
                    torch.testing.assert_close(out, reference, rtol=0, atol=0)
            del g


if __name__ == "__main__":
    unittest.main()
