"""Block-FP8 single-token GEMV versus native scaled_mm and dynamic graph replay."""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")

SHAPES = ((12288, 2048), (9216, 2048), (2048, 4096), (1024, 2048), (2048, 512))


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuBlockFp8Gemv(CustomTestCase):
    def _inputs(self, n, k, m=1, dtype=torch.bfloat16):
        torch.manual_seed(19)
        x = torch.randn((m, k), dtype=dtype, device="xpu")
        w = (torch.randn((n, k), device="xpu") * 16).to(torch.float8_e4m3fn)
        # Logical [N/128,K/128], with the post-load column-major storage.
        scales = torch.rand((k // 128, n // 128), device="xpu").t() * 0.02
        return x, w, scales

    def _run(self, x, w, scales, enabled, input_scale=None):
        with envs.SGLANG_XPU_M1_BLOCK_FP8_GEMV.override(enabled):
            out = torch_w8a8_block_fp8_linear(
                x, w, [128, 128], scales, input_scale=input_scale
            )
        return out

    def _check_reference(self, actual, x, w, scales, input_scale=None):
        if input_scale is None:
            q, xs = per_token_group_quant_fp8(x, 128)
        else:
            q, xs = x, input_scale
        # Independent CPU reference for oneDNN Xe2's internal FP16 scale
        # conversion and dequantization, followed by FP32 accumulation.
        q, xs, w, scales = [v.cpu().float() for v in (q, xs, w, scales)]
        n, k = w.shape
        fp16_dequant = (n, k) != (1024, 2048)
        if fp16_dequant:
            xs, scales = xs.half().float(), scales.half().float()
        a = q.reshape(1, k // 128, 128) * xs[:, :, None]
        b = w.reshape(n // 128, 128, k // 128, 128) * scales[:, None, :, None]
        if fp16_dequant:
            a, b = a.half().float(), b.half().float()
        a, b = a.reshape(1, k), b.reshape(n, k)
        reference = (a @ b.t()).to(torch.bfloat16)
        torch.testing.assert_close(actual.cpu(), reference, rtol=0.008, atol=0.0001)

    def _check_native(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0001)

    def test_native_parity_for_qwen_projections(self):
        for n, k in SHAPES:
            x, w, scales = self._inputs(n, k)
            expected = self._run(x, w, scales, False)
            actual = self._run(x, w, scales, True)
            with self.subTest(n=n, k=k):
                self.assertEqual(actual.dtype, expected.dtype)
                self._check_reference(actual, x, w, scales)
                self._check_native(actual, expected)

    def test_zero_and_nonuniform_activation_groups(self):
        x, w, scales = self._inputs(1024, 2048)
        x.zero_()
        torch.testing.assert_close(
            self._run(x, w, scales, True),
            torch.zeros((1, 1024), device="xpu", dtype=x.dtype),
        )
        x.normal_()
        x[:, :128] *= 0.001
        x[:, 128:256] *= 64
        x[:, 256:384] = 0
        actual = self._run(x, w, scales, True)
        self._check_reference(actual, x, w, scales)
        self._check_native(actual, self._run(x, w, scales, False))

    def test_prequantized_input_graph_replay(self):
        x, w, scales = self._inputs(2048, 512)
        q, xs = per_token_group_quant_fp8(x, 128)
        for _ in range(2):
            self._run(q, w, scales, True, input_scale=xs)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = self._run(q, w, scales, True, input_scale=xs)
        for step in range(4):
            x.normal_()
            if step == 0:
                x.zero_()
            new_q, new_xs = per_token_group_quant_fp8(x, 128)
            q.copy_(new_q)
            xs.copy_(new_xs)
            graph.replay()
            expected = self._run(q, w, scales, False, input_scale=xs)
            with self.subTest(step=step):
                self._check_reference(actual, q, w, scales, input_scale=xs)
                self._check_native(actual, expected)

    def test_larger_batch_keeps_native_path(self):
        x, w, scales = self._inputs(1024, 2048, m=2)
        torch.testing.assert_close(
            self._run(x, w, scales, True),
            self._run(x, w, scales, False),
            rtol=0,
            atol=0,
        )

    def test_unsupported_shape_keeps_native_path(self):
        x, w, scales = self._inputs(512, 2048)
        torch.testing.assert_close(
            self._run(x, w, scales, True),
            self._run(x, w, scales, False),
            rtol=0,
            atol=0,
        )

    def test_fp16_output_dtype_is_preserved(self):
        x, w, scales = self._inputs(1024, 2048, dtype=torch.float16)
        actual = self._run(x, w, scales, True)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(
            actual, self._run(x, w, scales, False), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
