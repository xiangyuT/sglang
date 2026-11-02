"""Pure SYCL metadata parity against the retained Triton/native pager path."""

import os
import unittest
from unittest.mock import patch

import test_xpu_dflash_fused_routes as fixtures
import torch

from sglang.kernels.ops.moe.xpu_compact_moe import xpu_compact_moe
from sglang.kernels.ops.moe.xpu_compact_paged import _PagedMatmul
from sglang.srt.environ import envs


class TestDflashSyclRoutes(fixtures.TestDflashFusedRoutes):
    sycl_backend = True
    window = int(os.environ.get("SGLANG_TEST_DFLASH_NATIVE_WINDOW", "4"))

    def bind_candidate(self, matmul, *args):
        matmul.bind_sycl_routes(*args)

    def test_sycl_flag_without_triton_fusion_flag(self):
        self.state.count.zero_()
        self.reset_caches()
        x = torch.randn(self.window, 2048, device="xpu", dtype=torch.bfloat16)
        ids = torch.arange(self.window * 8, device="xpu", dtype=torch.int32).view(
            self.window, 8
        )
        weights = torch.full((self.window, 8), 0.125, device="xpu")
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_FUSED_ROUTES.override(False),
            envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.override(True),
            patch.object(
                _PagedMatmul,
                "bind_sycl_routes",
                autospec=True,
                side_effect=_PagedMatmul.bind_sycl_routes,
            ) as selected,
        ):
            actual = xpu_compact_moe(
                x, *self.caches[1], ids, weights, paging=self.paging(1)
            )
            self.assertEqual(selected.call_count, 1)
            with envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.override(False):
                expected = xpu_compact_moe(
                    x, *self.caches[0], ids, weights, paging=self.paging(0)
                )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.check_caches()

    def test_four_row_prompt_with_selected_window_library(self):
        self.state.count.zero_()
        self.reset_caches()
        x = torch.randn(4, 2048, device="xpu", dtype=torch.bfloat16)
        ids = torch.arange(32, device="xpu", dtype=torch.int32).view(4, 8)
        weights = torch.full((4, 8), 0.125, device="xpu")
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.override(True),
            patch.object(
                _PagedMatmul,
                "bind_fused_routes",
                autospec=True,
                side_effect=_PagedMatmul.bind_fused_routes,
            ) as retained,
        ):
            actual = xpu_compact_moe(
                x, *self.caches[1], ids, weights, paging=self.paging(1)
            )
            self.assertEqual(retained.call_count, int(self.window == 8))
            with (
                envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.override(False),
                envs.SGLANG_XPU_DFLASH_FUSED_ROUTES.override(False),
            ):
                expected = xpu_compact_moe(
                    x, *self.caches[0], ids, weights, paging=self.paging(0)
                )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.check_caches()


if __name__ == "__main__":
    unittest.main()
