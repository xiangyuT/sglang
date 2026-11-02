"""Block-FP8 single-token GEMV versus native scaled_mm and dynamic graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_decode_xpu import fp8_e4m3_div_256
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")

SHAPES = ((12288, 2048), (9216, 2048), (2048, 4096), (1024, 2048), (2048, 512))


@triton.jit
def _check_scaled_decode(X, S, Y, SMALL: tl.constexpr):
    column = tl.arange(0, 256)
    value = tl.load(X + column)
    scale = tl.load(S + tl.program_id(0)).to(tl.float16).to(tl.float32)
    if SMALL:
        output = fp8_e4m3_div_256(value) * (scale * 256)
    else:
        output = value.to(tl.float32) * scale
    tl.store(Y + tl.program_id(0) * 256 + column, output.to(tl.float16))


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

    def test_scaled_decode_all_fp8_codes_and_fp16_scales(self):
        codes = torch.arange(256, device="xpu", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        scales = (
            torch.arange(65536, device="xpu", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.float16)
            .float()
        )
        expected = torch.empty((65536, 256), device="xpu", dtype=torch.float16)
        actual = torch.empty_like(expected)
        _check_scaled_decode[(65536,)](codes, scales, expected, SMALL=False)
        _check_scaled_decode[(65536,)](codes, scales, actual, SMALL=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        self.assertTrue(
            torch.equal(
                actual[finite].view(torch.int16), expected[finite].view(torch.int16)
            )
        )

    def test_native_parity_for_qwen_projections(self):
        for n, k in SHAPES:
            x, w, scales = self._inputs(n, k)
            expected = self._run(x, w, scales, False)
            actual = self._run(x, w, scales, True)
            with self.subTest(n=n, k=k):
                self.assertEqual(actual.dtype, expected.dtype)
                self._check_reference(actual, x, w, scales)
                self._check_native(actual, expected)

    def test_long_k_reduction_modes(self):
        x, w, scales = self._inputs(2048, 4096)
        expected = self._run(x, w, scales, False)
        for enabled in (False, True):
            with self.subTest(warp8=enabled):
                with envs.SGLANG_XPU_M1_BLOCK_FP8_LONG_K_WARP8.override(enabled):
                    actual = self._run(x, w, scales, True)
                self._check_reference(actual, x, w, scales)
                self._check_native(actual, expected)

    def test_ar_scaled_long_k_keeps_current_warp8_bits(self):
        from sglang.srt.layers.quantization import fp8_utils

        if torch.xpu.get_device_properties().device_id != 0xE223:
            self.skipTest("B70 AR long-K opt-in")
        x, w, scales = self._inputs(2048, 4096)
        with (
            envs.SGLANG_XPU_M1_BLOCK_FP8_LONG_K_WARP8.override(True),
            patch.object(
                fp8_utils,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(
                fp8_utils,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(
                        enable_paged_experts=True, paged_experts_num_resident=192
                    )
                ),
            ),
        ):
            for step in range(6):
                x.normal_()
                if step == 0:
                    x.zero_()
                if step == 1:
                    x[:, :128] *= 0.0001
                    x[:, 128:256] *= 64
                    x[:, 256:384] = 0
                with envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(False):
                    expected = self._run(x, w, scales, True)
                with envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(True):
                    actual = self._run(x, w, scales, True)
                self.assertTrue(torch.equal(actual, expected), f"step={step}")

    def test_ar_scaled_long_k_changing_graph_inputs_are_exact(self):
        from sglang.srt.layers.quantization import fp8_utils

        if torch.xpu.get_device_properties().device_id != 0xE223:
            self.skipTest("B70 AR long-K opt-in")
        x, w, scales = self._inputs(2048, 4096)
        q, xs = per_token_group_quant_fp8(x, 128)
        with (
            envs.SGLANG_XPU_M1_BLOCK_FP8_LONG_K_WARP8.override(True),
            patch.object(
                fp8_utils,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(
                fp8_utils,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(
                        enable_paged_experts=True, paged_experts_num_resident=192
                    )
                ),
            ),
        ):
            with envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(True):
                self._run(q, w, scales, True, input_scale=xs)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = self._run(q, w, scales, True, input_scale=xs)
            for step in range(8):
                x.normal_()
                if step == 0:
                    x.zero_()
                scales.uniform_(0.0001, 0.03)
                new_q, new_xs = per_token_group_quant_fp8(x, 128)
                q.copy_(new_q)
                xs.copy_(new_xs)
                graph.replay()
                with envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(False):
                    expected = self._run(q, w, scales, True, input_scale=xs)
                self.assertTrue(torch.equal(actual, expected), f"step={step}")
            del graph

    def test_ar_scaled_long_k_excludes_speculation_and_other_residency(self):
        from sglang.kernels.ops.gemm import xpu_block_fp8_gemv as gemv
        from sglang.srt.layers.quantization import fp8_utils

        if torch.xpu.get_device_properties().device_id != 0xE223:
            self.skipTest("B70 AR long-K opt-in")
        x, w, scales = self._inputs(2048, 4096)
        for algorithm, paged, resident, warp8 in (
            ("DFLASH", True, 192, True),
            (None, False, 192, True),
            (None, True, 128, True),
            (None, True, 192, False),
        ):
            with (
                envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(True),
                envs.SGLANG_XPU_M1_BLOCK_FP8_LONG_K_WARP8.override(warp8),
                patch.object(
                    fp8_utils,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=algorithm),
                ),
                patch.object(
                    fp8_utils,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(
                            enable_paged_experts=paged,
                            paged_experts_num_resident=resident,
                        )
                    ),
                ),
                patch.object(
                    gemv, "xpu_block_fp8_gemv", wraps=gemv.xpu_block_fp8_gemv
                ) as spy,
            ):
                actual = self._run(x, w, scales, True)
                self.assertFalse(spy.call_args.kwargs["compensate_long_k"])
                with envs.SGLANG_XPU_AR_LONG_K_SCALED_FP8.override(False):
                    expected = self._run(x, w, scales, True)
                self.assertTrue(torch.equal(actual, expected))

    def test_zero_and_nonuniform_activation_groups(self):
        for n, k in ((1024, 2048), (2048, 4096), (2048, 512)):
            x, w, scales = self._inputs(n, k)
            x.zero_()
            torch.testing.assert_close(
                self._run(x, w, scales, True),
                torch.zeros((1, n), device="xpu", dtype=x.dtype),
            )
            x.normal_()
            x[:, :128] *= 0.001
            x[:, 128:256] *= 64
            x[:, 256:384] = 0
            actual = self._run(x, w, scales, True)
            self._check_reference(actual, x, w, scales)
            self._check_native(actual, self._run(x, w, scales, False))

    def test_prequantized_input_graph_replay(self):
        for n, k in ((2048, 512), (2048, 4096)):
            with self.subTest(n=n, k=k):
                self._check_prequantized_graph(n, k)

    def _check_prequantized_graph(self, n, k):
        x, w, scales = self._inputs(n, k)
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
