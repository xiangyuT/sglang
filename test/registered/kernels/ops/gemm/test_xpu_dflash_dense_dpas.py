"""B70 block-verify projections: native/CPU references and live graph inputs."""

import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.xpu_dflash_dense_dpas import (
    _half_operand,
    can_use_dense_dpas,
)
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")
SHAPES = ((12288, 2048), (9216, 2048), (2048, 512))


@triton.jit
def _decode_half(Q, S, Y):
    row = tl.program_id(0)
    c = tl.arange(0, 256)
    q = tl.load(Q + c).to(tl.float8e4nv, bitcast=True)
    tl.store(Y + row * 256 + c, _half_operand(q, tl.load(S + row)))


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestDflashDenseDpas(CustomTestCase):
    def inputs(self, n, k, m=4):
        torch.manual_seed(42)
        x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
        x[:, :128] *= 0.001
        x[:, 128:256] *= 32
        q, xs = per_token_group_quant_fp8(x, 128)
        w = (torch.randn(n, k, device="xpu") * 16).to(q.dtype)
        ws = (torch.rand(k // 128, n // 128, device="xpu") * 0.002 + 0.0001).t()
        return q, w, xs, ws

    def run_linear(self, q, w, xs, ws, enabled, master=True):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_DENSE_DPAS.override(enabled),
        ):
            return torch_w8a8_block_fp8_linear(q, w, [128, 128], ws, input_scale=xs)

    def check_reference(self, y, native, q, w, xs, ws):
        self.assertTrue(bool(torch.isfinite(y).all()))
        rel = torch.linalg.vector_norm((y - native).float()) / torch.linalg.vector_norm(
            native.float()
        ).clamp_min(1e-20)
        self.assertLess(rel.item(), 2e-4)
        selected = [0, 127, 128, 255, w.shape[0] - 1]
        m, k = q.shape
        a = (
            (
                q.cpu().double().view(m, k // 128, 128)
                * xs.cpu().half().double()[:, :, None]
            )
            .reshape(m, k)
            .half()
            .double()
        )
        b = (
            (
                w[selected].cpu().double().view(len(selected), k // 128, 128)
                * ws[[r // 128 for r in selected]].cpu().half().double()[:, :, None]
            )
            .reshape(len(selected), k)
            .half()
            .double()
        )
        reference = (a @ b.T).bfloat16()
        torch.testing.assert_close(
            y[:, selected].cpu(), reference, rtol=0.008, atol=0.0001
        )

    def test_all_fp8_codes_and_fp16_scales(self):
        codes = torch.arange(256, dtype=torch.uint8)
        scales = (
            torch.arange(65536, dtype=torch.int32).short().view(torch.float16).float()
        )
        y = torch.empty((65536, 256), device="xpu", dtype=torch.float16)
        _decode_half[(65536,)](codes.to("xpu"), scales.to("xpu"), y, num_warps=4)
        actual = y.cpu()
        expected = (
            scales[:, None] * codes.view(torch.float8_e4m3fn).float()[None, :]
        ).half()
        same = (actual.view(torch.int16) == expected.view(torch.int16)) | (
            torch.isnan(actual) & torch.isnan(expected)
        )
        self.assertTrue(bool(same.all()))

    def test_native_reference_and_changing_graph_inputs(self):
        for n, k in SHAPES:
            for contiguous_scale in (False, True):
                q, w, xs, ws = self.inputs(n, k)
                if contiguous_scale:
                    ws = ws.contiguous()
                self.assertTrue(can_use_dense_dpas(q, w, xs, ws, [128, 128], None))
                self.run_linear(q, w, xs, ws, True)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    y = self.run_linear(q, w, xs, ws, True)
                for step in range(3):
                    if step == 0:
                        q.zero_()
                    else:
                        new = self.inputs(n, k)
                        q.copy_(new[0])
                        xs.copy_(new[2] * (1 + step * 0.03))
                        w[:128].copy_(new[1][:128])
                        ws.mul_(1.05)
                    graph.replay()
                    with self.subTest(
                        n=n, k=k, contiguous_scale=contiguous_scale, step=step
                    ):
                        torch.testing.assert_close(
                            y, self.run_linear(q, w, xs, ws, True), rtol=0, atol=0
                        )
                        self.check_reference(
                            y, self.run_linear(q, w, xs, ws, False), q, w, xs, ws
                        )

    def test_dispatch_and_master_guards(self):
        target = "sglang.kernels.ops.gemm.xpu_dflash_dense_dpas.dense_dpas"
        q, w, xs, ws = self.inputs(9216, 2048)
        with patch(target, side_effect=AssertionError("must stay native")):
            torch.testing.assert_close(
                self.run_linear(q, w, xs, ws, True, master=False),
                self.run_linear(q, w, xs, ws, False),
                rtol=0,
                atol=0,
            )
            for n, k, m in (
                (9216, 2048, 3),
                (9216, 2048, 8),
                (2048, 4096, 4),
                (1024, 2048, 4),
            ):
                q, w, xs, ws = self.inputs(n, k, m)
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws, True),
                    self.run_linear(q, w, xs, ws, False),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
