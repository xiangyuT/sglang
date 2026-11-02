"""AR small-dense dispatch, exact rounding, and changing-weight graph replay."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.gemm import xpu_ar_small_dense_sycl as sycl_op
from sglang.kernels.ops.gemm import xpu_block_fp8_gemv as op
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuArSmallDense(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 AR small dense")
        if not os.environ.get("SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB"):
            raise unittest.SkipTest("Build and set SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB")

    def fixture(self, n, k, transpose=False):
        return (
            torch.empty((1, k), dtype=torch.float8_e4m3fn, device="xpu"),
            torch.empty((n, k), dtype=torch.float8_e4m3fn, device="xpu"),
            torch.empty((1, k // 128), dtype=torch.float32, device="xpu"),
            torch.empty((k // 128, n // 128), device="xpu").t()
            if transpose
            else torch.empty((n // 128, k // 128), device="xpu"),
        )

    def fill(self, tensors, seed):
        torch.manual_seed(seed)
        for q in tensors[:2]:
            bits = torch.randint(0, 256, q.shape, dtype=torch.int16, device="xpu")
            bits = torch.where((bits & 127) == 127, bits - 1, bits)
            q.view(torch.uint8).copy_(bits.to(torch.uint8))
        for q in tensors[2:]:
            powers = torch.randint(-12, -3, q.shape, device="xpu")
            sign = torch.randint(0, 2, q.shape, device="xpu") * 2 - 1
            q.copy_(torch.exp2(powers.float()) * sign)
            q[0, 0] = 0

    def invoke(self, tensors, enabled, *, speculative=None, rows=4):
        with (
            envs.SGLANG_XPU_AR_SMALL_DENSE_SYCL.override(enabled),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=speculative),
            ),
        ):
            return op.xpu_block_fp8_gemv(*tensors, rows=rows)

    def assert_bits(self, left, right):
        self.assertTrue(torch.equal(left.view(torch.int16), right.view(torch.int16)))

    def test_exact_rounding_transposed_scales_and_selected_dispatch(self):
        for n, k in ((1024, 2048), (2048, 512)):
            for transpose in (False, True):
                tensors = self.fixture(n, k, transpose)
                for seed in (19, 42, 71):
                    self.fill(tensors, seed)
                    expected = self.invoke(tensors, False)
                    # An eligible opt-in must use the native library, not pass
                    # merely because both sides silently used the old kernel.
                    with patch.object(op, "_gemv_kernel") as native:
                        native.__getitem__.side_effect = AssertionError(
                            "unexpected fallback"
                        )
                        actual = self.invoke(tensors, True)
                    self.assert_bits(expected, actual)

    def test_graph_replay_uses_live_weights_scales_and_poisoned_outputs(self):
        for n, k in ((1024, 2048), (2048, 512)):
            tensors = self.fixture(n, k, transpose=True)
            self.fill(tensors, 42)
            self.invoke(tensors, False)
            self.invoke(tensors, True)
            torch.xpu.synchronize()
            graphs, outputs = [], []
            for enabled in (False, True):
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = self.invoke(tensors, enabled)
                graphs.append(graph)
                outputs.append(output)
            for seed in range(12):
                self.fill(tensors, seed)
                outputs[0].fill_(float("nan"))
                outputs[1].fill_(float("inf"))
                for graph in graphs:
                    graph.replay()
                self.assert_bits(*outputs)

    def test_metadata_guards(self):
        q, w, xs, ws = self.fixture(1024, 2048, transpose=True)
        self.assertTrue(sycl_op.can_use_ar_small_dense(q, w, xs, ws))
        self.assertFalse(sycl_op.can_use_ar_small_dense(q.expand(2, -1), w, xs, ws))
        self.assertFalse(sycl_op.can_use_ar_small_dense(q.half(), w, xs, ws))
        self.assertFalse(sycl_op.can_use_ar_small_dense(q.cpu(), w, xs, ws))
        self.assertFalse(sycl_op.can_use_ar_small_dense(q, w, xs, ws[:, :-1]))
        self.assertFalse(sycl_op.can_use_ar_small_dense(q, w, xs, ws[:1].expand_as(ws)))
        unaligned = torch.empty(2049, dtype=q.dtype, device=q.device)[1:].view(1, 2048)
        self.assertFalse(sycl_op.can_use_ar_small_dense(unaligned, w, xs, ws))
        with torch.enable_grad():
            xs.requires_grad_(True)
            self.assertFalse(sycl_op.can_use_ar_small_dense(q, w, xs, ws))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            sycl_op.xpu_ar_small_dense(q, w, xs, ws[:, :-1])

    def test_default_speculation_rows_and_compiler_exclusions(self):
        tensors = self.fixture(1024, 2048)
        self.fill(tensors, 42)
        reference = self.invoke(tensors, False)
        with patch.object(
            sycl_op,
            "xpu_ar_small_dense",
            side_effect=AssertionError("unexpected dispatch"),
        ):
            self.assert_bits(reference, self.invoke(tensors, False))
            self.assert_bits(
                reference, self.invoke(tensors, True, speculative="DFLASH")
            )
            expected_rows2 = self.invoke(tensors, False, rows=2)
            self.assert_bits(expected_rows2, self.invoke(tensors, True, rows=2))
            with patch("torch.compiler.is_compiling", return_value=True):
                self.assert_bits(reference, self.invoke(tensors, True))

    def test_library_must_be_warmed_before_capture(self):
        sycl_op._load_library.cache_clear()
        with self.assertRaisesRegex(RuntimeError, "Set SGLANG"):
            sycl_op._load_library("")
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                sycl_op._load_library(os.environ["SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB"])
        sycl_op._load_library(os.environ["SGLANG_XPU_AR_SMALL_DENSE_SYCL_LIB"])


if __name__ == "__main__":
    unittest.main()
