"""Selected native dense shapes, immutable weight packing, and graph guards."""

import os
import unittest
from unittest.mock import patch

import test_xpu_dflash_dense_dpas as fixtures
import torch

from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import (
    _CACHE_ATTRIBUTE,
    packed_weight,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")
WINDOW = int(os.environ.get("SGLANG_TEST_DFLASH_NATIVE_WINDOW", "4"))


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "requires B70 and an explicitly built native library",
)
class TestDflashDenseSycl(CustomTestCase):
    def inputs(self, n, k, m=None):
        return fixtures.TestDflashDenseDpas.inputs(
            self, n, k, WINDOW if m is None else m
        )

    check_reference = fixtures.TestDflashDenseDpas.check_reference

    def run_linear(self, q, w, xs, ws, child=True, master=True, parent=True):
        with envs.SGLANG_XPU_DFLASH_DENSE_SYCL.override(child):
            return fixtures.TestDflashDenseDpas.run_linear(
                self, q, w, xs, ws, parent, master
            )

    def test_native_reference_and_changing_graph_inputs(self):
        for n, k in fixtures.SHAPES:
            for contiguous in (False, True):
                q, w, xs, ws = self.inputs(n, k)
                if contiguous:
                    ws = ws.contiguous()
                self.run_linear(q, w, xs, ws)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = self.run_linear(q, w, xs, ws)
                original = q.clone()
                for step in range(3):
                    if step == 0:
                        q.zero_()
                    else:
                        q.copy_(original)
                        xs.mul_(1.03)
                        ws.mul_(1.05)
                    graph.replay()
                    with self.subTest(n=n, k=k, contiguous=contiguous, step=step):
                        torch.testing.assert_close(
                            actual, self.run_linear(q, w, xs, ws), rtol=0, atol=0
                        )
                        self.check_reference(
                            actual,
                            self.run_linear(q, w, xs, ws, child=False, parent=False),
                            q,
                            w,
                            xs,
                            ws,
                        )
                del graph

    def test_weight_bytes_and_rebuild_before_recapture(self):
        for n, k in ((12288, 2048), (9216, 2048)):
            q, w, xs, ws = self.inputs(n, k)
            self.run_linear(q, w, xs, ws)
            previous = getattr(w, _CACHE_ATTRIBUTE)[1]
            w[:128].copy_((w[:128].float() * 1.125).to(w.dtype))
            result = self.run_linear(q, w, xs, ws)
            current = getattr(w, _CACHE_ATTRIBUTE)[1]
            self.assertNotEqual(previous.data_ptr(), current.data_ptr())
            restored = (
                current.view(torch.uint8)
                .view(n // 16, k // 32, 8, 16, 4)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
                .view(n, k)
            )
            self.assertTrue(torch.equal(restored, w.view(torch.uint8)))
            self.check_reference(
                result,
                self.run_linear(q, w, xs, ws, child=False, parent=False),
                q,
                w,
                xs,
                ws,
            )
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = self.run_linear(q, w, xs, ws)
            graph.replay()
            torch.testing.assert_close(actual, result, rtol=0, atol=0)
            del graph

    def test_invalid_native_geometry_fails_before_submission(self):
        from sglang.kernels.ops.gemm.xpu_dflash_dense_sycl import load_dense_sycl

        library = load_dense_sycl()
        # Null pointers make any attempted queue/data access fail. Each invalid
        # geometry must return an error before the activation prepare is queued.
        for n, k, mode, message in (
            (12288, 512, 0, "Unsupported matrix K"),
            (12288, 2048, 17, "Unknown sequential production geometry"),
            (12345, 2048, 0, "N must be positive"),
        ):
            with self.subTest(n=n, k=k, mode=mode):
                rc = library.dflash_seq_dense_gemm(*([None] * 7), n, k, 16, 1, mode)
                self.assertNotEqual(rc, 0)
                self.assertIn(message, library.dflash_seq_dense_error().decode())

    def test_master_parent_child_and_shape_guards(self):
        target = "sglang.kernels.ops.gemm.xpu_dflash_dense_sycl.dense_sycl"
        q, w, xs, ws = self.inputs(9216, 2048)
        with patch(target, side_effect=AssertionError("must not select SYCL")):
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
            for n, k, m in ((2048, 512, 4), (9216, 2048, 3), (2048, 4096, 4)):
                q, w, xs, ws = self.inputs(n, k, m)
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws),
                    self.run_linear(q, w, xs, ws, child=False),
                    rtol=0,
                    atol=0,
                )
            if WINDOW == 8:
                q, w, xs, ws = self.inputs(9216, 2048, 4)
                torch.testing.assert_close(
                    self.run_linear(q, w, xs, ws),
                    self.run_linear(q, w, xs, ws, child=False, parent=False),
                    rtol=0,
                    atol=0,
                )

    def test_pack_requires_warmup_before_capture(self):
        _, weight, _, _ = self.inputs(2048, 512)
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm SYCL dense"):
                packed_weight(weight)
        packed = packed_weight(weight)
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            self.assertIs(packed_weight(weight), packed)
        weight[:128].zero_()
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm SYCL dense"):
                packed_weight(weight)


if __name__ == "__main__":
    unittest.main()
