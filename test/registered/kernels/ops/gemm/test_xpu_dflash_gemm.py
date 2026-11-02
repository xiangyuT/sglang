"""Experimental small-M projections: native and independent operand references."""

import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.xpu_dflash_gemm import _decode_fp8
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")
SHAPES = ((1024, 2048),)


@triton.jit
def _decode_codes(X, Y):
    r = tl.arange(0, 256)
    tl.store(Y + r, _decode_fp8(tl.load(X + r)))


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuDflashGemm(CustomTestCase):
    def _inputs(self, m, n, k):
        torch.manual_seed(42)
        x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
        x[:, :128] *= 0.001
        x[:, 128:256] *= 32
        x[:, 256:384] = 0
        q, xs = per_token_group_quant_fp8(x, 128)
        w = (torch.randn(n, k, device="xpu") * 16).to(torch.float8_e4m3fn)
        ws = (torch.rand(k // 128, n // 128, device="xpu") * 0.002 + 0.0001).t()
        return q, w, xs, ws

    def _run(self, q, w, xs, ws, enabled, master=True):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_SHARED_GEMV.override(enabled),
        ):
            return torch_w8a8_block_fp8_linear(q, w, [128, 128], ws, input_scale=xs)

    def _check(self, actual, native, q, w, xs, ws):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, native, rtol=0.008, atol=0.0001)
        relative = torch.linalg.vector_norm((actual - native).float()) / (
            torch.linalg.vector_norm(native.float()) + 1e-20
        )
        self.assertLess(relative.item(), 0.0005)
        # Independent FP64 dot on selected rows after the documented native
        # operand rounding. Avoid reconstructing a full large weight matrix.
        selected = [0, 127, 128, 255, w.shape[0] - 1]
        m, k = q.shape
        a = q.cpu().double()
        b = w[selected].cpu().double()
        sa = xs.cpu().double()
        sb = ws[[r // 128 for r in selected]].cpu().double()
        half = w.shape != (1024, 2048)
        if half:
            sa, sb = sa.half().double(), sb.half().double()
        a = (a.view(m, k // 128, 128) * sa[:, :, None]).reshape(m, k)
        b = (b.view(len(selected), k // 128, 128) * sb[:, :, None]).reshape(
            len(selected), k
        )
        if half:
            a, b = a.half().double(), b.half().double()
        else:
            a, b = a.float().double(), b.float().double()
        reference = (a @ b.t()).bfloat16()
        torch.testing.assert_close(
            actual[:, selected].cpu(), reference, rtol=0.008, atol=0.0001
        )

    def test_all_fp8_codes_decode_to_fp32(self):
        q = torch.arange(256, device="xpu", dtype=torch.uint8).view(torch.float8_e4m3fn)
        actual = torch.empty(256, device="xpu", dtype=torch.float32)
        _decode_codes[(1,)](q, actual)
        expected = q.float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        self.assertTrue(
            torch.equal(
                actual[finite].view(torch.int32), expected[finite].view(torch.int32)
            )
        )

    def test_native_and_independent_reference(self):
        for m in range(2, 9):
            for n, k in SHAPES:
                with self.subTest(m=m, n=n, k=k):
                    q, w, xs, ws = self._inputs(m, n, k)
                    self._check(
                        self._run(q, w, xs, ws, True),
                        self._run(q, w, xs, ws, False),
                        q,
                        w,
                        xs,
                        ws,
                    )

    def test_shared_large_scale_keeps_finite_products(self):
        # scale*256 would overflow here, although the dequantized operands and
        # their one-column dot are finite. The FP32 decode must not move that
        # compensating factor into an arbitrary scale.
        q = torch.zeros(4, 2048, device="xpu", dtype=torch.float8_e4m3fn)
        w = torch.zeros(1024, 2048, device="xpu", dtype=q.dtype)
        q[:, 0] = 2.0**-9
        w[:, 0] = 2.0**-9
        xs = torch.full((4, 16), 2.0**-100, device="xpu", dtype=torch.float32)
        ws = torch.full((16, 8), 2.0**127, device="xpu", dtype=torch.float32).t()
        expected = torch.full((4, 1024), 512.0, device="xpu", dtype=torch.bfloat16)
        torch.testing.assert_close(
            self._run(q, w, xs, ws, True), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self._run(q, w, xs, ws, False), expected, rtol=0, atol=0
        )

    def test_changing_graph_inputs_weights_and_scales(self):
        for m in (3, 4, 5, 8):
            for n, k in SHAPES:
                q, w, xs, ws = self._inputs(m, n, k)
                self._run(q, w, xs, ws, True)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = self._run(q, w, xs, ws, True)
                for step in range(4):
                    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
                    if step == 0:
                        x.zero_()
                    new_q, new_xs = per_token_group_quant_fp8(x, 128)
                    q.copy_(new_q)
                    xs.copy_(new_xs)
                    w[:128].copy_((torch.randn(128, k, device="xpu") * 16).to(w.dtype))
                    ws.mul_(1.05)
                    graph.replay()
                    with self.subTest(m=m, n=n, k=k, step=step):
                        self._check(
                            actual, self._run(q, w, xs, ws, False), q, w, xs, ws
                        )

    def test_master_and_fallback_guards(self):
        for m, n, k in [
            (1, 2048, 4096),
            (9, 1024, 2048),
            (4, 512, 2048),
            (4, 12288, 2048),
            (4, 9216, 2048),
            (4, 2048, 4096),
        ]:
            q, w, xs, ws = self._inputs(m, n, k)
            with self.subTest(m=m, n=n, k=k):
                torch.testing.assert_close(
                    self._run(q, w, xs, ws, True),
                    self._run(q, w, xs, ws, False),
                    rtol=0,
                    atol=0,
                )
        q, w, xs, ws = self._inputs(4, 2048, 4096)
        torch.testing.assert_close(
            self._run(q, w, xs, ws, True, master=False),
            self._run(q, w, xs, ws, False),
            rtol=0,
            atol=0,
        )
        # A different scale layout stays on the original native route.
        ws = ws.contiguous()
        torch.testing.assert_close(
            self._run(q, w, xs, ws, True),
            self._run(q, w, xs, ws, False),
            rtol=0,
            atol=0,
        )
        q, w, xs, ws = self._inputs(4, 1024, 2048)
        storage = torch.empty((4, 4096), device="xpu", dtype=q.dtype)
        strided = storage[:, :2048]
        strided.copy_(q)
        self.assertFalse(strided.is_contiguous())
        torch.testing.assert_close(
            self._run(strided, w, xs, ws, True),
            self._run(strided, w, xs, ws, False),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
