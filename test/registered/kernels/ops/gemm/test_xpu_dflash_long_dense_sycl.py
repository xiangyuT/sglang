"""Long SYCL projection: full FP64 bounds, dynamic graph inputs and opt-in guards."""

import os
import unittest
from unittest.mock import patch
import test_xpu_dflash_dense_dpas as fixtures
import torch
from sglang.kernels.ops.gemm.xpu_dflash_long_dense_sycl import (
    can_use_long_dense,
    load_long_dense,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "requires B70 and an explicitly built native library",
)
class TestDflashLongDenseSycl(CustomTestCase):
    inputs = fixtures.TestDflashDenseDpas.inputs

    def setUp(self):
        super().setUp()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            self.skipTest("B70 long projection contract")

    def run_linear(self, q, w, xs, ws, child=True, master=True, parent=True, sycl=True):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_DENSE_DPAS.override(parent),
            envs.SGLANG_XPU_DFLASH_DENSE_SYCL.override(sycl),
            envs.SGLANG_XPU_DFLASH_LONG_DENSE_SYCL.override(child),
        ):
            return torch_w8a8_block_fp8_linear(q, w, [128, 128], ws, input_scale=xs)

    def check_full_fp64_bound(self, y, q, w, xs, ws):
        self.assertTrue(bool(y.isfinite().all()))
        a = (
            (q.cpu().double().view(4, 32, 128) * xs.cpu().half().double()[:, :, None])
            .reshape(4, 4096)
            .half()
            .double()
        )
        b = (
            (
                w.cpu().double().view(16, 128, 32, 128)
                * ws.cpu().half().double()[:, None, :, None]
            )
            .reshape(2048, 4096)
            .half()
            .double()
        )
        exact = a @ b.T
        sumabs = a.abs() @ b.abs().T
        yy = y.cpu()
        bits = yy.abs().view(torch.int16).to(torch.int32) & 65535
        ulp = ((bits + 1) << 16).view(torch.float32).double() - yy.double().abs()
        gamma = (4096 * 2**-24) / (1 - 4096 * 2**-24)
        self.assertTrue(
            bool(
                (
                    (yy.double() - exact).abs() <= gamma * sumabs + 0.5 * ulp + 2**-120
                ).all()
            )
        )

    def test_changing_graph_inputs_and_full_fp64_bound(self):
        for contiguous in (False, True):
            q, w, xs, ws = self.inputs(2048, 4096)
            if contiguous:
                ws = ws.contiguous()
            self.run_linear(q, w, xs, ws)
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = self.run_linear(q, w, xs, ws)
            saved = q.clone()
            for step in range(3):
                if step == 0:
                    q.zero_()
                else:
                    q.copy_(saved)
                    xs.mul_(1.03)
                    ws.mul_(1.05)
                graph.replay()
                torch.testing.assert_close(
                    actual, self.run_linear(q, w, xs, ws), rtol=0, atol=0
                )
                self.check_full_fp64_bound(actual, q, w, xs, ws)
                self.check_full_fp64_bound(
                    self.run_linear(q, w, xs, ws, child=False), q, w, xs, ws
                )
            del graph

    def test_weight_bytes_rebuild_and_recapture(self):
        from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import _CACHE_ATTRIBUTE

        q, w, xs, ws = self.inputs(2048, 4096)
        self.run_linear(q, w, xs, ws)
        previous = getattr(w, _CACHE_ATTRIBUTE)[1]
        w[:128].copy_((w[:128].float() * 1.125).to(w.dtype))
        y = self.run_linear(q, w, xs, ws)
        packed = getattr(w, _CACHE_ATTRIBUTE)[1]
        self.assertNotEqual(previous.data_ptr(), packed.data_ptr())
        restored = (
            packed.view(torch.uint8)
            .view(128, 128, 8, 16, 4)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(2048, 4096)
        )
        self.assertTrue(torch.equal(restored, w.view(torch.uint8)))
        self.check_full_fp64_bound(y, q, w, xs, ws)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = self.run_linear(q, w, xs, ws)
        graph.replay()
        torch.testing.assert_close(actual, y, rtol=0, atol=0)

    def test_master_parent_sycl_child_and_shape_guards(self):
        target = "sglang.kernels.ops.gemm.xpu_dflash_long_dense_sycl.long_dense_sycl"
        q, w, xs, ws = self.inputs(2048, 4096)
        with patch(target, side_effect=AssertionError("must not select long SYCL")):
            for master, parent, sycl, child in (
                (False, True, True, True),
                (True, False, True, True),
                (True, True, False, True),
                (True, True, True, False),
            ):
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws, child, master, parent, sycl),
                    self.run_linear(q, w, xs, ws, False, master, parent, sycl),
                    rtol=0,
                    atol=0,
                )
            for n, k, m in ((2048, 4096, 3), (2048, 512, 4), (1024, 2048, 4)):
                q, w, xs, ws = self.inputs(n, k, m)
                self.assertFalse(can_use_long_dense(q, w, xs, ws, [128, 128], None))
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws),
                    self.run_linear(q, w, xs, ws, child=False),
                    rtol=0,
                    atol=0,
                )
        q, w, xs, ws = self.inputs(2048, 4096)
        self.assertFalse(can_use_long_dense(q, w, xs, ws, [1, 128], None))
        self.assertFalse(
            can_use_long_dense(
                q, w, xs, ws, [128, 128], torch.zeros(2048, device="xpu")
            )
        )
        self.assertFalse(can_use_long_dense(q, w, xs.half(), ws, [128, 128], None))

    def test_missing_native_library_fails(self):
        load_long_dense.cache_clear()
        try:
            with envs.SGLANG_XPU_DFLASH_NATIVE_LIB.override(""):
                with self.assertRaisesRegex(RuntimeError, "requires.*NATIVE_LIB"):
                    load_long_dense()
        finally:
            load_long_dense.cache_clear()


if __name__ == "__main__":
    unittest.main()
