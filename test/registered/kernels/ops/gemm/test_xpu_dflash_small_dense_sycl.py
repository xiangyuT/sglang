"""Small SYCL projection: live graph inputs, FP64 checks and dispatch guards."""

import os
import unittest
from unittest.mock import patch

import test_xpu_dflash_dense_dpas as fixtures
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "requires B70 and an explicitly built native library",
)
class TestDflashSmallDenseSycl(CustomTestCase):
    inputs = fixtures.TestDflashDenseDpas.inputs
    check_reference = fixtures.TestDflashDenseDpas.check_reference

    def run_linear(self, q, w, xs, ws, child=True, master=True, parent=True):
        with envs.SGLANG_XPU_DFLASH_SMALL_DENSE_SYCL.override(child):
            return fixtures.TestDflashDenseDpas.run_linear(
                self, q, w, xs, ws, parent, master
            )

    def test_native_reference_and_live_graph_inputs(self):
        for contiguous in (False, True):
            q, w, xs, ws = self.inputs(2048, 512)
            if contiguous:
                ws = ws.contiguous()
            self.run_linear(q, w, xs, ws)
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = self.run_linear(q, w, xs, ws)
            for step in range(4):
                if step == 0:
                    q.zero_()
                else:
                    new = self.inputs(2048, 512)
                    q.copy_(new[0])
                    xs.copy_(new[2] * (1 + step * 0.03))
                    if step == 2:
                        codes = torch.arange(256, device="xpu", dtype=torch.int32)
                        codes = codes[(codes & 127) != 127].to(torch.uint8)
                        codes = codes.repeat(3)[:512]
                        w.view(torch.uint8).copy_(codes[None, :].expand(2048, 512))
                    else:
                        w.copy_(new[1])
                    ws.mul_(1.05)
                graph.replay()
                with self.subTest(contiguous=contiguous, step=step):
                    torch.testing.assert_close(
                        actual, self.run_linear(q, w, xs, ws), rtol=0, atol=0
                    )
                    self.check_reference(
                        actual,
                        self.run_linear(q, w, xs, ws, parent=False),
                        q,
                        w,
                        xs,
                        ws,
                    )
                    self.check_reference(
                        actual,
                        self.run_linear(q, w, xs, ws, child=False),
                        q,
                        w,
                        xs,
                        ws,
                    )
            del graph

    def test_master_parent_child_and_shape_guards(self):
        target = "sglang.kernels.ops.gemm.xpu_dflash_small_dense_sycl.small_dense_sycl"
        q, w, xs, ws = self.inputs(2048, 512)
        with patch(target, side_effect=AssertionError("must not select small SYCL")):
            for master, parent, child in (
                (False, True, True),
                (True, False, True),
                (True, True, False),
            ):
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws, child, master, parent),
                    self.run_linear(q, w, xs, ws, False, master, parent),
                    rtol=0,
                    atol=0,
                )
            for n, k, m in ((2048, 512, 3), (2048, 4096, 4), (1024, 2048, 4)):
                q, w, xs, ws = self.inputs(n, k, m)
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws),
                    self.run_linear(q, w, xs, ws, child=False),
                    rtol=0,
                    atol=0,
                )

    def test_selected_missing_library_fails(self):
        from sglang.kernels.ops.gemm.xpu_dflash_small_dense_sycl import load_small_dense

        load_small_dense.cache_clear()
        try:
            with envs.SGLANG_XPU_DFLASH_NATIVE_LIB.override(""):
                with self.assertRaisesRegex(RuntimeError, "requires.*NATIVE_LIB"):
                    load_small_dense()
        finally:
            load_small_dense.cache_clear()


if __name__ == "__main__":
    unittest.main()
