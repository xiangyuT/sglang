"""EP2 experts preserve analytic results and dropped routes during graph replay."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.moe.xpu_tp_expert_gu_sycl import (
    tp_expert_gu,
    try_ep_local_ids,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU required")
class TestXpuEpExperts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 EP2 expert kernels")
        for name in (
            "SGLANG_XPU_TP_EXPERT_GU_SYCL_LIB",
            "SGLANG_XPU_TP_EXPERT_DOWN_SYCL_LIB",
        ):
            if not os.environ.get(name):
                raise unittest.SkipTest(f"Build and set {name}")
        # The serving runtime loads this package before the expert path;
        # the standalone test also needs its native SiLU op registration.
        importlib.import_module("sgl_kernel")

    def test_fused_local_map_preserves_dropped_routes(self):
        with (
            torch.inference_mode(),
            patch.dict(os.environ, {"SGLANG_XPU_TP_EXPERT_GU_SYCL": "1"}),
        ):
            ids = torch.zeros((1, 8), dtype=torch.int32, device="xpu")
            for rank in (0, 1):
                mapping = torch.full((256,), -1, dtype=torch.int32, device="xpu")
                mapping[rank * 128 : (rank + 1) * 128] = torch.arange(
                    128, dtype=torch.int32, device="xpu"
                )
                output = try_ep_local_ids(ids, mapping)
                if output is None:
                    self.skipTest("GU library lacks optional fused local-map entry")
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = try_ep_local_ids(ids, mapping)
                for values in (
                    [0, 1, 127, 128, 129, 255, -1, -4],
                    [-1] * 8,
                    [255, 254, 253, 128, 127, 4, 0, -128],
                    list(range(8)),
                ):
                    ids.copy_(torch.tensor([values], dtype=torch.int32))
                    expected = torch.where(ids < 0, -1, mapping[ids.clamp_min(0)])
                    eager = try_ep_local_ids(ids, mapping)
                    graph.replay()
                    self.assertTrue(torch.equal(eager, expected))
                    self.assertTrue(torch.equal(output, expected))

    def test_changing_local_routes_and_masked_nan_coefficients(self):
        with torch.inference_mode():
            device = torch.device("xpu")
            layer = SimpleNamespace(
                w13_weight=torch.zeros(
                    (128, 1024, 2048), dtype=torch.uint8, device=device
                ).view(torch.float8_e4m3fn),
                w2_weight=torch.zeros(
                    (128, 2048, 512), dtype=torch.uint8, device=device
                ).view(torch.float8_e4m3fn),
                w13_weight_scale_inv=torch.full((128, 8, 16), 0.125, device=device),
                w2_weight_scale_inv=torch.full((128, 16, 4), 0.125, device=device),
            )
            # Gate/up become 128, 64 and 256 respectively. SiLU rounds to
            # identity at these positive values. With down weight 1/32,
            # full expert outputs are exactly 262144, 65536 and 1048576.
            for expert, code in ((0, 0x30), (13, 0x28), (127, 0x38)):
                layer.w13_weight[expert].view(torch.uint8).fill_(code)
                layer.w2_weight[expert].view(torch.uint8).fill_(0x28)
            x = torch.ones((1, 2048), dtype=torch.bfloat16, device=device)
            ids = torch.zeros((1, 8), dtype=torch.int32, device=device)
            weights = torch.full((1, 8), 0.125, device=device)
            for _ in range(3):
                tp_expert_gu(layer, x, ids, weights)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                graph_output = tp_expert_gu(layer, x, ids, weights)

            nan = float("nan")
            cases = (
                ([-1] * 8, [nan] * 8, 0),
                ([0] + [-1] * 7, [0.5] + [nan] * 7, 131072),
                ([13, 127] + [-1] * 6, [0.25, 0.125] + [nan] * 6, 147456),
                ([0, 13, 127, 0, 13, 127, -1, -1], [0.125] * 6 + [nan] * 2, 344064),
            )
            for replay in (False, True):
                for route_ids, coefficients, expected in cases:
                    with self.subTest(replay=replay, routes=route_ids):
                        ids.copy_(torch.tensor([route_ids], dtype=torch.int32))
                        weights.copy_(torch.tensor([coefficients], dtype=torch.float32))
                        if replay:
                            graph.replay()
                            actual = graph_output
                        else:
                            actual = tp_expert_gu(layer, x, ids, weights)
                        torch.xpu.synchronize()
                        reference = torch.full_like(actual, expected)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int16), reference.view(torch.int16)
                            )
                        )
                        raw, activated, _ = layer._xpu_tp_gu_workspace
                        dropped = ids.view(-1) < 0
                        self.assertTrue(bool((raw[dropped] == 0).all()))
                        self.assertTrue(bool((activated[dropped] == 0).all()))

            # A valid route must still propagate NaN; masking is specific to
            # the negative local-ID sentinel, not a general NaN sanitizer.
            ids.fill_(-1)
            ids[0, 0] = 0
            weights.fill_(nan)
            graph.replay()
            self.assertTrue(bool(torch.isnan(graph_output).all()))

            # Native BF16 SiLU rounds its intermediate before multiplying up.
            # Small signed gate/up values detect a full-FP32 fusion that would
            # pass the large positive identity cases above.
            gates = torch.tensor(
                [0.125, 0.5, 1.5, -0.5, -1.25, -4.0, 0.03125, 3.5],
                dtype=torch.bfloat16,
                device=device,
            ).repeat(64)
            ups = torch.tensor(
                [0.375, 1.25, -0.75, 0.625, -1.5, 0.75, 3.0, -0.25],
                dtype=torch.bfloat16,
                device=device,
            ).repeat(64)
            rows = torch.cat((gates, ups))[:, None].expand(-1, 2048)
            layer.w13_weight[0].copy_(rows.to(torch.float8_e4m3fn))
            layer.w13_weight_scale_inv.fill_(1.0)
            x.fill_(1 / 2048)
            ids.zero_()
            weights.fill_(0.125)
            for replay in (False, True):
                if replay:
                    graph.replay()
                else:
                    tp_expert_gu(layer, x, ids, weights)
                raw, activated, _ = layer._xpu_tp_gu_workspace
                self.assertTrue(torch.equal(raw[0, :512], gates))
                self.assertTrue(torch.equal(raw[0, 512:], ups))
                reference = torch.nn.functional.silu(raw[:, :512]) * raw[:, 512:]
                without_rounding = (
                    torch.nn.functional.silu(raw[:, :512].float())
                    * raw[:, 512:].float()
                ).bfloat16()
                self.assertFalse(torch.equal(reference, without_rounding))
                self.assertTrue(
                    torch.equal(
                        activated.view(torch.int16), reference.view(torch.int16)
                    )
                )


if __name__ == "__main__":
    unittest.main()
