"""B70 DFlash quantization bits, dynamic replay and actual dispatch guards."""

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from sglang.kernels.ops.quantization.dflash_quant_sycl import (
    can_use_dflash_quant,
    dflash_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")


class TestDflashQuantGuards(unittest.TestCase):
    def test_cpu_unsupported_without_library(self):
        x = torch.zeros((4, 2048), dtype=torch.bfloat16)
        self.assertFalse(can_use_dflash_quant(x))
        with self.assertRaisesRegex(ValueError, "Unsupported B70"):
            dflash_quant(x)


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_SMALL_TLA_LIB"),
    "requires B70 and explicit small TLA library",
)
class TestDflashQuantSycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("requires B70")

    def test_bits_and_dynamic_graph_all_selected_shapes(self):
        for shape in ((4, 2048), (4, 4096), (3, 2048), (8, 2048), (8, 4096), (7, 2048)):
            x = torch.zeros(shape, device="xpu", dtype=torch.bfloat16)
            dflash_quant(x)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                q, scales = dflash_quant(x)
            for seed in (42, 43, 44):
                gen = torch.Generator().manual_seed(seed)
                value = torch.randn(shape, generator=gen)
                value *= torch.exp2(torch.linspace(-40, 40, shape[1]))
                if seed == 44:
                    value[0].zero_()
                    value[1, :128] = float("nan")
                    value[1, 128:256] = float("inf")
                    value[1, 256:384] = -float("inf")
                x.copy_(value.bfloat16().to("xpu"))
                graph.replay()
                eq, es = per_token_group_quant_fp8(x, 128)
                self.assertTrue(
                    torch.equal(q.cpu().view(torch.uint8), eq.cpu().view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(
                        scales.cpu().view(torch.int32), es.cpu().view(torch.int32)
                    )
                )

    def test_dispatch_output_bits_opt_in_and_prequantized_fallback(self):
        target = "sglang.kernels.ops.quantization.dflash_quant_sycl.dflash_quant"
        for shape in ((4, 2048), (4, 4096), (3, 2048), (8, 2048), (8, 4096), (7, 2048)):
            gen = torch.Generator().manual_seed(42)
            x = torch.randn(shape, generator=gen, dtype=torch.bfloat16).to("xpu")
            w = (
                (torch.randn((128, shape[1]), generator=gen) * 8)
                .to(torch.float8_e4m3fn)
                .to("xpu")
            )
            scales = torch.full((1, shape[1] // 128), 0.01, device="xpu")

            def forward(master, enabled, **kwargs):
                with ExitStack() as stack:
                    for setting, value in (
                        (envs.SGLANG_XPU_PAGED_DFLASH, master),
                        (envs.SGLANG_XPU_DFLASH_QUANT_SYCL, enabled),
                        (envs.SGLANG_XPU_DFLASH_DENSE_DPAS, False),
                        (envs.SGLANG_XPU_DFLASH_SMALL_TLA, False),
                        (envs.SGLANG_XPU_M1_BLOCK_FP8_GEMV, False),
                    ):
                        stack.enter_context(setting.override(value))
                    return torch_w8a8_block_fp8_linear(
                        x, w, [128, 128], scales, **kwargs
                    )

            expected = forward(True, False)
            with patch(
                "sglang.srt.layers.quantization.fp8_utils.per_token_group_quant_fp8",
                side_effect=AssertionError("must use selected SYCL quantizer"),
            ):
                actual = forward(True, True)
            self.assertTrue(
                torch.equal(
                    actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                )
            )
            with patch(
                target, side_effect=AssertionError("must retain quant fallback")
            ):
                disabled = forward(True, False)
                no_master = forward(False, True)
            self.assertTrue(torch.equal(disabled.cpu(), expected.cpu()))
            self.assertTrue(torch.equal(no_master.cpu(), expected.cpu()))
            q, xs = per_token_group_quant_fp8(x, 128)
            x = q
            with patch(target, side_effect=AssertionError("prequantized must bypass")):
                prequantized = forward(True, True, input_scale=xs)
            self.assertTrue(torch.equal(prequantized.cpu(), expected.cpu()))

        for shape in ((1, 2048), (2, 2048), (4, 512), (4, 1024), (3, 4096)):
            self.assertFalse(
                can_use_dflash_quant(
                    torch.zeros(shape, device="xpu", dtype=torch.bfloat16)
                )
            )
        self.assertFalse(can_use_dflash_quant(torch.zeros((4, 2048), device="xpu")))
        self.assertFalse(
            can_use_dflash_quant(
                torch.zeros((4, 4096), device="xpu", dtype=torch.bfloat16)[:, ::2]
            )
        )


if __name__ == "__main__":
    unittest.main()
