"""B70 AR native routing equivalence, changing graph inputs and dispatch guards."""

import ctypes.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sgl_kernel import topk_softmax

from sglang.kernels.ops.moe.xpu_ar_topk_sycl import (
    _load_library,
    can_use_ar_topk,
    xpu_ar_topk,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuArTopk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 ordinary AR top-k")
        if not os.environ.get("SGLANG_XPU_AR_TOPK_SYCL_LIB"):
            raise unittest.SkipTest("Build and set SGLANG_XPU_AR_TOPK_SYCL_LIB")

    def fixture(self):
        return (
            torch.randn((1, 256), device="xpu", dtype=torch.bfloat16),
            torch.empty((1, 8), device="xpu"),
            torch.empty((1, 8), device="xpu", dtype=torch.int32),
        )

    def assert_native(self, x, weights, ids, renorm):
        expected_weights, expected_ids = (
            torch.empty_like(weights),
            torch.empty_like(ids),
        )
        topk_softmax(expected_weights, expected_ids, x, renorm)
        self.assertTrue(torch.equal(ids, expected_ids))
        torch.testing.assert_close(
            weights, expected_weights, rtol=0, atol=0, equal_nan=True
        )

    def test_adversarial_and_nonfinite_inputs_match_native(self):
        for seed in (19, 42, 71):
            torch.manual_seed(seed)
            x, weights, ids = self.fixture()
            normal = x.clone()
            signed_zero = torch.zeros_like(x)
            signed_zero[:, 1::2] = -0.0
            ties = torch.zeros_like(x)
            ties[:, 8:18] = 1
            extreme = torch.full_like(x, -1024)
            extreme[:, 200] = 1024
            cases = [normal, normal * 1e-8, signed_zero, ties, extreme]
            cases += [
                torch.full_like(x, value)
                for value in (float("nan"), float("inf"), -float("inf"))
            ]
            for values in cases:
                x.copy_(values)
                for renorm in (False, True):
                    with self.subTest(
                        seed=seed, renorm=renorm, finite=bool(torch.isfinite(x).all())
                    ):
                        xpu_ar_topk(x, weights, ids, renorm)
                        self.assert_native(x, weights, ids, renorm)

    def test_changing_graph_inputs_keep_exact_weights_and_ties(self):
        x, weights, ids = self.fixture()
        for renorm in (False, True):
            xpu_ar_topk(x, weights, ids, renorm)
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                xpu_ar_topk(x, weights, ids, renorm)
            for step in range(12):
                x.normal_()
                if step == 0:
                    x.zero_()
                if step == 1:
                    x.mul_(1e-8)
                if step == 2:
                    x.fill_(-1024)
                    x[:, 200] = 1024
                graph.replay()
                self.assert_native(x, weights, ids, renorm)
            del graph

    def test_layout_dtype_and_autograd_guards(self):
        x, weights, ids = self.fixture()
        self.assertTrue(can_use_ar_topk(x, weights, ids, True))
        for tensors in (
            (x.float(), weights, ids),
            (x.expand(2, -1), weights, ids),
            (x, weights.half(), ids),
            (x, weights, ids.long()),
            (x, weights.expand(2, -1), ids),
            (x, weights, ids.cpu()),
        ):
            self.assertFalse(can_use_ar_topk(*tensors, True))
        with torch.enable_grad():
            self.assertFalse(can_use_ar_topk(x.requires_grad_(), weights, ids, True))

    def test_missing_abi_and_capture_before_warmup_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "Set SGLANG"):
            _load_library("")
        with patch.object(torch.xpu, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                _load_library("unwarmed-library")
        libc = ctypes.util.find_library("c")
        if libc:
            with self.assertRaisesRegex(RuntimeError, "lacks AR top-k ABI"):
                _load_library(libc)

    def test_runtime_ar_scope_keeps_speculation_and_other_k_native(self):
        from sglang.srt.layers.moe import topk

        x, weights, ids = self.fixture()
        hidden = torch.zeros((1, 2048), device="xpu", dtype=torch.bfloat16)
        for algorithm, enabled, resident in (
            ("DFLASH", True, 192),
            (None, False, 192),
            (None, True, 128),
        ):
            with (
                patch.dict(
                    os.environ,
                    {
                        "SGLANG_XPU_AR_TOPK_SYCL": "1",
                        "SGLANG_XPU_AR_TOPK_SYCL_LIB": "nonexistent-ar-library",
                    },
                ),
                patch.object(
                    topk,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=algorithm),
                ),
                patch.object(
                    topk,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(
                            enable_paged_experts=enabled,
                            paged_experts_num_resident=resident,
                        )
                    ),
                ),
            ):
                weights, ids = topk.fused_topk(hidden, x, 8, True)
                self.assert_native(x, weights, ids, True)


if __name__ == "__main__":
    unittest.main()
