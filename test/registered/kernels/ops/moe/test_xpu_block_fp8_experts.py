"""Native and independent-reference checks for B70 single-token experts."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.xpu_block_fp8_experts import (
    _dequant_weight,
    xpu_block_fp8_experts,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.paged_experts.xpu import _supports_expert_gemv
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


@triton.jit
def _check_scaled_fp32(X, S, Y, SMALL: tl.constexpr):
    column = tl.arange(0, 256)
    value = tl.load(X + column)
    scale = tl.load(S + tl.program_id(0))
    output = _dequant_weight(value, scale, SMALL)
    tl.store(Y + tl.program_id(0) * 256 + column, output)


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuBlockFp8Experts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 layout and native rounding contract")

    def _inputs(self):
        torch.manual_seed(43)
        w13 = (torch.randn(16, 1024, 2048, device="xpu") * 16).to(torch.float8_e4m3fn)
        w2 = (torch.randn(16, 2048, 512, device="xpu") * 16).to(torch.float8_e4m3fn)
        s13 = torch.rand(16, 8, 16, device="xpu") * 0.002
        s2 = torch.rand(16, 16, 4, device="xpu") * 0.002
        x = torch.randn(1, 2048, dtype=torch.bfloat16, device="xpu")
        ids = torch.tensor(
            [[11, 2, 6, 3, 15, 1, 8, 0]], dtype=torch.int32, device="xpu"
        )
        weights = torch.rand(1, 8, device="xpu")
        weights /= weights.sum(-1, keepdim=True)
        return x, w13, w2, s13, s2, ids, weights

    def _native(self, x, w13, w2, s13, s2, ids, weights):
        from sgl_kernel import fused_experts

        return fused_experts(
            x,
            w13,
            w2,
            weights,
            ids,
            use_fp8_w8a8=True,
            w1_scale=s13,
            w2_scale=s2,
            block_shape=[128, 128],
        )

    def _reference(self, x, w13, w2, s13, s2, ids, weights):
        # FP64 dot products over the exact BF16-truncated dequantized weights.
        # Native SiLU rounds to BF16 before multiplying by the up output.
        x = x.cpu().double()
        result = torch.zeros(1, 2048)
        for expert, weight in zip(ids.cpu()[0].tolist(), weights.cpu()[0].tolist()):
            matrices = []
            for w, s in ((w13[expert], s13[expert]), (w2[expert], s2[expert])):
                w, s = w.cpu().float(), s.cpu().float()
                n, k = w.shape
                scaled = (
                    w.reshape(n // 128, 128, k // 128, 128) * s[:, None, :, None]
                ).reshape(n, k)
                truncated = (scaled.view(torch.int32) & -65536).view(torch.float32)
                matrices.append(truncated.double())
            projected = (x @ matrices[0].t()).bfloat16().float()
            gate, up = projected.chunk(2, -1)
            activated = (
                (gate * torch.sigmoid(gate)).bfloat16().float() * up
            ).bfloat16()
            partial = (activated.double() @ matrices[1].t()).bfloat16().float()
            result += partial * weight
        return result.bfloat16()

    def _assert_close(self, actual, reference):
        torch.testing.assert_close(
            actual.cpu(), reference.cpu(), rtol=0.008, atol=0.0002
        )

    def test_native_and_independent_reference(self):
        args = self._inputs()
        for mode in ("random", "nonuniform", "zero"):
            if mode == "nonuniform":
                args[0][:, :128] *= 0.001
                args[0][:, 128:256] *= 4
            elif mode == "zero":
                args[0].zero_()
            actual = xpu_block_fp8_experts(*args, scaled_fp8=True)
            original = xpu_block_fp8_experts(*args)
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            with self.subTest(mode=mode):
                self._assert_close(actual, self._native(*args))
                self._assert_close(actual, self._reference(*args))
                self.assertTrue(torch.isfinite(actual).all().item())

    def test_dynamic_graph_duplicate_routes_and_zero_weights(self):
        args = self._inputs()
        x, _, _, _, _, ids, weights = args
        for _ in range(2):
            xpu_block_fp8_experts(*args)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = xpu_block_fp8_experts(*args, scaled_fp8=True)
        for step in range(6):
            x.normal_()
            ids.copy_(torch.randint(0, 16, ids.shape, dtype=ids.dtype, device="xpu"))
            weights.uniform_()
            weights /= weights.sum(-1, keepdim=True)
            if step == 0:
                weights.zero_()
            elif step == 1:
                ids.fill_(3)
            elif step == 2:
                weights[0, ::2] = 0
            graph.replay()
            with self.subTest(step=step):
                torch.testing.assert_close(
                    actual, xpu_block_fp8_experts(*args), rtol=0, atol=0
                )
                self._assert_close(actual, self._native(*args))
                if step in (0, 5):
                    self._assert_close(actual, self._reference(*args))
                if step == 0:
                    self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_scaled_decode_fp32_exponents_and_subnormals(self):
        codes = torch.arange(256, device="xpu", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        powers = torch.exp2(torch.arange(-118, 119, device="xpu").float())
        scales = torch.cat(
            (
                powers,
                -powers,
                powers * 1.5,
                -powers * 1.5,
                torch.zeros(1, device="xpu"),
                torch.rand(4096, device="xpu") * 0.002,
            )
        )
        scales = scales[scales.abs() <= 2.0**118]
        expected = torch.empty((len(scales), 256), device="xpu")
        actual = torch.empty_like(expected)
        _check_scaled_fp32[(len(scales),)](codes, scales, expected, SMALL=False)
        _check_scaled_fp32[(len(scales),)](codes, scales, actual, SMALL=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        self.assertTrue(
            torch.equal(
                actual[finite].view(torch.int32), expected[finite].view(torch.int32)
            )
        )

    def test_dispatch_guards(self):
        _, w13, w2, s13, s2, _, _ = self._inputs()
        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale_inv=s13,
            w2_weight_scale_inv=s2,
        )
        config = MoeRunnerConfig(top_k=8)
        with envs.SGLANG_XPU_M1_FP8_EXPERT_GEMV.override(True):
            self.assertTrue(_supports_expert_gemv(layer, config))
            for override in (
                {"activation": "gelu"},
                {"apply_router_weight_on_input": True},
                {"no_combine": True},
                {"swiglu_limit": 10},
                {"gemm1_alpha": 1.7},
            ):
                with self.subTest(override=override):
                    self.assertFalse(
                        _supports_expert_gemv(layer, replace(config, **override))
                    )
            layer.w13_bias = torch.zeros(16, 1024, device="xpu")
            self.assertFalse(_supports_expert_gemv(layer, config))
            layer.w13_bias = None
            layer.w13_weight_scale_inv = s13.transpose(1, 2)
            self.assertFalse(_supports_expert_gemv(layer, config))
        with envs.SGLANG_XPU_M1_FP8_EXPERT_GEMV.override(False):
            self.assertFalse(_supports_expert_gemv(layer, config))


if __name__ == "__main__":
    unittest.main()
