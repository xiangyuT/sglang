"""B70 graph-input bit copies, allocator lifetime and unchanged fallbacks."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dflash_gdn_sycl import load_gdn
from sglang.kernels.ops.memory import xpu_small_copy_sycl as copy
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_buffer_registry import _grouped_foreach_copy_
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires B70 auxiliary SYCL library")
class TestXpuSmallCopySycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 input copies")
        copy.load_small_copy()

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.enterContext(envs.SGLANG_XPU_PAGED_DFLASH.override(True))
        self.enterContext(envs.SGLANG_XPU_DFLASH_SMALL_COPY_SYCL.override(True))

    def inputs(self, lengths=(4, 4, 4, 1, 1), dtype=torch.int64):
        source = [
            torch.randint(-(2**40), 2**40, (n,), device="xpu", dtype=torch.int64).to(
                dtype
            )
            for n in lengths
        ]
        return [torch.empty_like(t) for t in source], source

    def assert_copied(self, destination, source):
        self.assertTrue(all(torch.equal(d, s) for d, s in zip(destination, source)))

    def test_int64_extremes_and_changing_captured_inputs(self):
        self.check_dynamic_inputs(4)

    def test_m8_int64_extremes_and_changing_captured_inputs(self):
        self.check_dynamic_inputs(8)

    def check_dynamic_inputs(self, window):
        dst, src = self.inputs((window, window, window, 1, 1))
        src[0][:4].copy_(torch.tensor([-(2**63), 2**63 - 1, -1, 0], device="xpu"))
        self.assertTrue(copy.try_small_copy(dst, src))
        self.assert_copied(dst, src)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            self.assertTrue(copy.try_small_copy(dst, src))
        for step in range(12):
            for i, tensor in enumerate(src):
                tensor.add_(step + i + 1)
            graph.replay()
            torch.xpu.synchronize()
            self.assert_copied(dst, src)

    def test_source_release_keeps_stream_lifetime(self):
        for i in range(16):
            window = 4 if i % 2 else 8
            dst, src = self.inputs((window, window, window, 1, 1))
            expected = [t.clone() for t in src]
            self.assertTrue(copy.try_small_copy(dst, src))
            del src
            pressure = [torch.full((4,), i, device="xpu") for _ in range(64)]
            torch.xpu.synchronize()
            self.assert_copied(dst, expected)
            del pressure

    def test_master_child_shape_dtype_and_stride_fallbacks(self):
        for master, child in ((False, True), (True, False), (False, False)):
            dst, src = self.inputs()
            with (
                envs.SGLANG_XPU_PAGED_DFLASH.override(master),
                envs.SGLANG_XPU_DFLASH_SMALL_COPY_SYCL.override(child),
                patch.object(copy, "submit") as submitted,
            ):
                _grouped_foreach_copy_(dst, src)
                submitted.assert_not_called()
            self.assert_copied(dst, src)
        for dst, src in (
            self.inputs((512, 512, 512, 1, 1)),
            self.inputs(dtype=torch.int32),
            self.inputs((1, 4, 4, 4, 1)),
        ):
            with patch.object(copy, "submit") as submitted:
                _grouped_foreach_copy_(dst, src)
                submitted.assert_not_called()
            self.assert_copied(dst, src)
        dst, src = self.inputs()
        src[0] = torch.arange(8, device="xpu")[::2]
        with patch.object(copy, "submit") as submitted:
            _grouped_foreach_copy_(dst, src)
            submitted.assert_not_called()
        self.assert_copied(dst, src)

    def test_registry_dispatch_and_alias_guards(self):
        dst, src = self.inputs()
        with patch.object(copy, "submit", wraps=copy.submit) as submitted:
            _grouped_foreach_copy_(dst, src)
            submitted.assert_called_once()
        self.assert_copied(dst, src)
        dst[0] = src[0]
        self.assertTrue(copy.try_small_copy(dst, src))
        self.assert_copied(dst, src)
        dst, src = self.inputs()
        dst[1] = dst[0]
        self.assertFalse(copy.eligible(dst, src))

        dst, src = self.inputs()
        src[0] = dst[1]
        self.assertFalse(copy.eligible(dst, src))
        dst, src = self.inputs()
        base = torch.arange(5, device="xpu")
        dst[0], src[0] = base[:4], base[1:]
        self.assertFalse(copy.eligible(dst, src))

    def test_m8_registry_dispatch_and_alias_guards(self):
        dst, src = self.inputs((8, 8, 8, 1, 1))
        with patch.object(copy, "submit", wraps=copy.submit) as submitted:
            _grouped_foreach_copy_(dst, src)
            submitted.assert_called_once()
        self.assert_copied(dst, src)
        dst[0] = src[0]
        self.assertTrue(copy.try_small_copy(dst, src))
        self.assert_copied(dst, src)
        dst[1] = dst[0]
        self.assertFalse(copy.eligible(dst, src))
        dst, src = self.inputs((8, 8, 8, 1, 1))
        src[0] = dst[1]
        self.assertFalse(copy.eligible(dst, src))
        dst, src = self.inputs((8, 8, 8, 1, 1))
        base = torch.arange(9, device="xpu")
        dst[0], src[0] = base[:8], base[1:]
        self.assertFalse(copy.eligible(dst, src))

    def test_m8_missing_symbol_rejects_before_writes(self):
        from types import SimpleNamespace

        dst, src = self.inputs((8, 8, 8, 1, 1))
        for tensor in dst:
            tensor.fill_(123456)
        expected = [t.clone() for t in dst]
        with (
            patch.object(copy, "load_small_copy", return_value=SimpleNamespace()),
            self.assertRaisesRegex(RuntimeError, "lacks M8 support"),
        ):
            copy.try_small_copy(dst, src)
        self.assert_copied(dst, expected)

    def test_missing_library_or_copy_symbols_is_explicit(self):
        copy.load_small_copy.cache_clear()
        load_gdn.cache_clear()
        try:
            with (
                envs.SGLANG_XPU_DFLASH_GDN_LIB.override(""),
                self.assertRaises(RuntimeError),
            ):
                copy.load_small_copy()
            with (
                patch.object(copy, "load_gdn", return_value=object()),
                self.assertRaisesRegex(RuntimeError, "symbols missing"),
            ):
                copy.load_small_copy()
        finally:
            copy.load_small_copy.cache_clear()
            load_gdn.cache_clear()


if __name__ == "__main__":
    unittest.main()
