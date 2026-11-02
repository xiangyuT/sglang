"""GDN norm/block quant: numerical boundaries, dynamic graph and dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import dflash_gdn_norm_quant_sycl as native
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.gemm.xpu_dflash_small_tla import quantize_bf16_block128
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires B70 and native libraries")
class TestGdnNormQuantSycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 norm/block quant")
        native.load_gdn_norm_quant()

    def fixture(self):
        torch.manual_seed(42)
        x = torch.randn(256, 128, device="xpu", dtype=torch.bfloat16)
        return (
            x,
            torch.randn_like(x),
            torch.randn(128, device="xpu", dtype=torch.bfloat16),
        )

    def reference(self, x, z, w):
        y, _, rs = _layer_norm_fwd(
            x, w, None, 1e-6, z=z, is_rms_norm=True, activation="swish"
        )
        return y, rs, quantize_bf16_block128(y.view(8, 4096))

    def assert_bits(self, a, b):
        dtype = torch.int32 if a.dtype == torch.float32 else torch.uint8
        self.assertTrue(torch.equal(a.view(dtype), b.view(dtype)))

    def test_live_inputs_and_weight_under_graph(self):
        x, z, w = self.fixture()
        native.gdn_norm_quant(x, z, w, 1e-6)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = native.gdn_norm_quant(x, z, w, 1e-6)
        for pattern in range(4):
            if pattern == 1:
                x.zero_()
            elif pattern == 2:
                x.normal_().mul_(torch.exp2(torch.linspace(-12, 12, 128, device="xpu")))
                z.normal_().mul_(8)
                w.add_(0.125)
            elif pattern == 3:
                z.zero_()
            _, _, expected = self.reference(x, z, w)
            graph.replay()
            torch.xpu.synchronize()
            for a, b in zip(actual, expected):
                self.assert_bits(a, b)

    def test_debug_norm_and_rstd_match_retained(self):
        x, z, w = self.fixture()
        q = torch.empty((8, 4096), device="xpu", dtype=torch.float8_e4m3fn)
        s = torch.empty((8, 32), device="xpu")
        y, rs = torch.empty_like(x), torch.empty(256, device="xpu")
        library = native.load_gdn_norm_quant()
        self.assertEqual(
            library.dflash_gdn_norm_quant_run(
                torch.xpu.current_stream().sycl_queue,
                x.data_ptr(),
                z.data_ptr(),
                w.data_ptr(),
                q.data_ptr(),
                s.data_ptr(),
                y.data_ptr(),
                rs.data_ptr(),
                256,
                1e-6,
                1,
                448.0,
            ),
            0,
        )
        expected_y, expected_rs, (expected_q, expected_s) = self.reference(x, z, w)
        for a, b in (
            (y, expected_y),
            (rs, expected_rs),
            (q, expected_q),
            (s, expected_s),
        ):
            self.assert_bits(a, b)

    def test_tensor_and_numeric_guards(self):
        x, z, w = self.fixture()
        self.assertTrue(native.can_use_gdn_norm_quant(x, z, w, 1e-6))
        for args in (
            (x[:128], z[:128], w, 1e-6),
            (x, z.float(), w, 1e-6),
            (x, z.t().contiguous().t(), w, 1e-6),
            (x, z, w[:64], 1e-6),
            (x, z, w, float("nan")),
            (x, z, w, float("inf")),
            (x, z, w, 0),
        ):
            self.assertFalse(native.can_use_gdn_norm_quant(*args))
            with self.assertRaises(ValueError):
                native.gdn_norm_quant(*args)
        x.requires_grad_(True)
        self.assertFalse(native.can_use_gdn_norm_quant(x, z, w, 1e-6))

    def test_flags_and_projection_contract(self):
        x, z, w = self.fixture()
        norm = SimpleNamespace(
            weight=w, eps=1e-6, activation="swish", norm_before_gate=True
        )
        method = type("Fp8LinearMethod", (), {})()
        method.block_quant, method.weight_block_size = True, [128, 128]
        projection = SimpleNamespace(
            quant_method=method,
            weight=torch.empty((2048, 4096), device="xpu", dtype=torch.float8_e4m3fn),
            bias=None,
        )
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_QUANT_SYCL.override(True),
            envs.SGLANG_XPU_DFLASH_GDN_NORM_QUANT_SYCL.override(True),
        ):
            self.assertTrue(native.can_fuse_gdn_norm_quant(x, z, norm, projection))
            for flag in (
                envs.SGLANG_XPU_PAGED_DFLASH,
                envs.SGLANG_XPU_DFLASH_QUANT_SYCL,
                envs.SGLANG_XPU_DFLASH_GDN_NORM_QUANT_SYCL,
            ):
                with flag.override(False):
                    self.assertFalse(
                        native.can_fuse_gdn_norm_quant(x, z, norm, projection)
                    )
            method.use_mxfp8 = True
            self.assertFalse(native.can_fuse_gdn_norm_quant(x, z, norm, projection))
            method.use_mxfp8 = False
            method.weight_block_size = [1, 128]
            self.assertFalse(native.can_fuse_gdn_norm_quant(x, z, norm, projection))
            method.weight_block_size = [128, 128]
            norm.group_size = 64
            self.assertFalse(native.can_fuse_gdn_norm_quant(x, z, norm, projection))
            norm.group_size = 128
            norm.bias = w
            self.assertFalse(native.can_fuse_gdn_norm_quant(x, z, norm, projection))
            norm.bias = None
            norm.norm_before_gate = False
            self.assertFalse(native.can_fuse_gdn_norm_quant(x, z, norm, projection))

    def test_missing_library_and_capture_warmup(self):
        native.load_gdn_norm_quant.cache_clear()
        try:
            with envs.SGLANG_XPU_DFLASH_GDN_NORM_QUANT_LIB.override(""):
                with self.assertRaisesRegex(RuntimeError, "NORM_QUANT_LIB"):
                    native.load_gdn_norm_quant()
            with patch("torch.xpu.is_current_stream_capturing", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "before graph capture"):
                    native.load_gdn_norm_quant()
        finally:
            native.load_gdn_norm_quant.cache_clear()
            native.load_gdn_norm_quant()
