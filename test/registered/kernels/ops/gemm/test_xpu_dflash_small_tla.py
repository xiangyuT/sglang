"""B70 small TLA: quant bits, native outputs, cache ownership and graph guards."""

import gc
import os
import unittest
import weakref
from contextlib import ExitStack
from unittest.mock import patch

import torch
from sglang.kernels.ops.gemm import xpu_dflash_small_tla as tla
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")


class TestSmallTlaLoader(unittest.TestCase):
    def tearDown(self):
        tla.load_small_tla.cache_clear()

    def test_missing_library_and_symbols_fail(self):
        tla.load_small_tla.cache_clear()
        with envs.SGLANG_XPU_DFLASH_SMALL_TLA_LIB.override(""):
            with self.assertRaisesRegex(RuntimeError, "requires.*SMALL_TLA_LIB"):
                tla.load_small_tla()
        with (
            envs.SGLANG_XPU_DFLASH_SMALL_TLA_LIB.override("mock-small-tla.so"),
            patch.object(tla.ctypes, "CDLL", return_value=object()),
        ):
            with self.assertRaisesRegex(RuntimeError, "symbols missing"):
                tla.load_small_tla()

    def test_wrong_abi_fails_before_submission(self):
        class Function:
            def __call__(self):
                return 0

        class Library:
            dflash_small_tla_abi = Function()

        tla.load_small_tla.cache_clear()
        with (
            envs.SGLANG_XPU_DFLASH_SMALL_TLA_LIB.override("mock-small-tla.so"),
            patch.object(tla.ctypes, "CDLL", return_value=Library()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Unsupported.*ABI"):
                tla.load_small_tla()


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_SMALL_TLA_LIB"),
    "requires B70 and an explicitly built small TLA library",
)
class TestDflashSmallTla(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("requires B70")

    @staticmethod
    def inputs(seed=42, transposed=True, m=4, n=2048):
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn((m, 512), generator=gen, dtype=torch.bfloat16).to("xpu")
        w = (torch.randn((n, 512), generator=gen) * 8).to(torch.float8_e4m3fn).to("xpu")
        scales = (torch.rand((n // 128, 4), generator=gen) * 0.02 + 0.005).to("xpu")
        if transposed:
            scales = scales.T.contiguous().T
        return x, w, scales

    @staticmethod
    def linear(x, w, scales, enabled=True, master=True, parent=True, **kwargs):
        with ExitStack() as stack:
            for setting, value in (
                (envs.SGLANG_XPU_PAGED_DFLASH, master),
                (envs.SGLANG_XPU_DFLASH_DENSE_DPAS, parent),
                (envs.SGLANG_XPU_DFLASH_DENSE_SYCL, False),
                (envs.SGLANG_XPU_DFLASH_SMALL_DENSE_SYCL, False),
                (envs.SGLANG_XPU_DFLASH_SMALL_TLA, enabled),
            ):
                stack.enter_context(setting.override(value))
            return torch_w8a8_block_fp8_linear(x, w, [128, 128], scales, **kwargs)

    def assert_quant_bits(self, x):
        q, s = tla._quantize(x)
        eq, es = per_token_group_quant_fp8(x, 128)
        self.assertTrue(
            torch.equal(q.cpu().view(torch.uint8), eq.cpu().view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(s.cpu().view(torch.int32), es.cpu().view(torch.int32))
        )

    @staticmethod
    def capture_on_warmed_stream(forward):
        original = torch.xpu.current_stream()
        stream = torch.xpu.Stream()
        stream.wait_stream(original)
        with torch.xpu.stream(stream):
            forward()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                output = forward()
        original.wait_stream(stream)
        return graph, output, stream

    def test_all_quant_bits_midpoints_and_dynamic_graph(self):
        x = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
        self.assert_quant_bits(x.reshape(-1, 128).to("xpu"))
        values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        ties = ((values[:-1] + values[1:]) / 2).bfloat16()
        levels = [
            ties,
            torch.nextafter(ties, torch.full_like(ties, float("inf"))),
            torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
        ]
        entries = torch.cat(levels + [-v for v in levels])
        entries = torch.cat([entries, torch.zeros((-entries.numel()) % 127).bfloat16()])
        tx = torch.cat(
            [
                entries.reshape(-1, 127),
                torch.full((entries.numel() // 127, 1), 448).bfloat16(),
            ],
            1,
        ).to("xpu")
        self.assert_quant_bits(tx)
        eq, _ = per_token_group_quant_fp8(tx, 128)
        self.assertTrue(
            torch.equal(
                eq.cpu().view(torch.uint8),
                tx.cpu().to(torch.float8_e4m3fn).view(torch.uint8),
            )
        )
        x = torch.zeros((4, 512), device="xpu", dtype=torch.bfloat16)
        tla._quantize(x)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            q, s = tla._quantize(x)
        for seed in (42, 43, 44):
            gen = torch.Generator().manual_seed(seed)
            values = torch.randn((16, 128), generator=gen)
            values *= torch.exp2(torch.linspace(-40, 40, 16))[:, None]
            x.copy_(values.reshape(4, 512).bfloat16().to("xpu"))
            graph.replay()
            eq, es = per_token_group_quant_fp8(x, 128)
            self.assertTrue(
                torch.equal(q.cpu().view(torch.uint8), eq.cpu().view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(s.cpu().view(torch.int32), es.cpu().view(torch.int32))
            )
        x.zero_()
        graph.replay()
        eq, es = per_token_group_quant_fp8(x, 128)
        self.assertTrue(
            torch.equal(q.cpu().view(torch.uint8), eq.cpu().view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(s.cpu().view(torch.int32), es.cpu().view(torch.int32))
        )

    def test_full_outputs_cache_bits_graph_and_fp64(self):
        self.check_full_outputs_cache_bits_graph_and_fp64(4)

    def test_m8_full_outputs_cache_bits_graph_and_fp64(self):
        self.check_full_outputs_cache_bits_graph_and_fp64(8)

    def check_full_outputs_cache_bits_graph_and_fp64(self, rows):
        for layout in (False, True):
            x, w, scales = self.inputs(transposed=layout, m=rows)
            self.linear(x, w, scales)
            entry = getattr(w, tla._CACHE_ATTRIBUTE)
            expected_w = (
                w.cpu().float()
                * scales.cpu()
                .half()
                .float()
                .repeat_interleave(128, 0)
                .repeat_interleave(128, 1)
            ).half()
            self.assertTrue(
                torch.equal(
                    entry.packed.cpu().view(torch.int16), expected_w.view(torch.int16)
                )
            )
            self.assertEqual(
                entry.packed.numel() * entry.packed.element_size(), 2 * 1024**2
            )
            graph, actual, capture_stream = self.capture_on_warmed_stream(
                lambda: self.linear(x, w, scales)
            )
            for seed in (42, 43, 44):
                x.copy_(self.inputs(seed, layout, m=rows)[0])
                graph.replay()
                y = actual.cpu()
                expected = self.linear(x, w, scales, enabled=False).cpu()
                self.assertTrue(
                    torch.equal(y.view(torch.int16), expected.view(torch.int16))
                )
                self.assertTrue(
                    torch.equal(
                        y.view(torch.int16),
                        self.linear(x, w, scales).cpu().view(torch.int16),
                    )
                )
                q, xs = per_token_group_quant_fp8(x, 128)
                a = (
                    (
                        q.cpu().float()
                        * xs.cpu().half().float().repeat_interleave(128, 1)
                    )
                    .half()
                    .double()
                )
                b = expected_w.double()
                ref, absolute = a @ b.T, a.abs() @ b.abs().T
                gamma = 512 * 2**-24 / (1 - 512 * 2**-24)
                self.assertTrue(
                    bool(
                        (
                            (y.double() - ref).abs()
                            <= gamma * absolute + ref.abs() * 2**-8 + 2**-133
                        ).all()
                    )
                )
            del graph

    def test_m4_and_m8_reuse_one_static_weight_cache(self):
        x4, w, scales = self.inputs()
        self.linear(x4, w, scales)
        entry = getattr(w, tla._CACHE_ATTRIBUTE)
        x8 = self.inputs(m=8)[0]
        actual = self.linear(x8, w, scales)
        self.assertIs(getattr(w, tla._CACHE_ATTRIBUTE), entry)
        expected = self.linear(x8, w, scales, enabled=False)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def test_selected_m8_missing_symbol_rejects_before_packing(self):
        from types import SimpleNamespace

        x, w, scales = self.inputs(m=8)
        with (
            patch.object(tla, "load_small_tla", return_value=SimpleNamespace()),
            patch.object(
                tla, "_weight_cache", side_effect=AssertionError("must reject")
            ),
            self.assertRaisesRegex(RuntimeError, "lacks M8 support"),
        ):
            tla.small_tla(x, w, scales)

    def test_dispatch_master_parent_child_and_layout_guards(self):
        x, w, scales = self.inputs()
        target = "sglang.kernels.ops.gemm.xpu_dflash_small_tla.small_tla"
        with patch(target, side_effect=AssertionError("must use retained path")):
            for master, parent, child in (
                (False, True, True),
                (True, False, True),
                (True, True, False),
            ):
                actual = self.linear(x, w, scales, child, master, parent)
                expected = self.linear(x, w, scales, False, master, parent)
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                    )
                )
            for m, n in ((3, 2048), (4, 1024)):
                args = self.inputs(m=m, n=n)
                self.assertTrue(
                    torch.equal(
                        self.linear(*args).cpu(),
                        self.linear(*args, enabled=False).cpu(),
                    )
                )
            q, xs = per_token_group_quant_fp8(x, 128)
            self.assertTrue(
                torch.equal(
                    self.linear(q, w, scales, input_scale=xs).cpu(),
                    self.linear(q, w, scales, False, input_scale=xs).cpu(),
                )
            )
            bias = torch.zeros(2048, device="xpu", dtype=torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    self.linear(x, w, scales, bias=bias).cpu(),
                    self.linear(x, w, scales, False, bias=bias).cpu(),
                )
            )
        marker = torch.full((4, 2048), 3, device="xpu", dtype=torch.bfloat16)
        with patch(target, return_value=marker) as dispatch:
            self.assertTrue(torch.equal(self.linear(x, w, scales).cpu(), marker.cpu()))
            dispatch.assert_called_once()
        noncontiguous = x.T.contiguous().T
        self.assertFalse(tla.can_use_small_tla(noncontiguous, w, scales, (128, 128)))
        self.assertFalse(tla.can_use_small_tla(x.float(), w, scales, (128, 128)))
        self.assertFalse(
            tla.can_use_small_tla(x.cpu(), w.cpu(), scales.cpu(), (128, 128))
        )
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(tla.can_use_small_tla(x, w, scales, (128, 128)))

    def test_eager_repack_and_captured_static_weight_guard(self):
        x, w, scales = self.inputs()
        self.linear(x, w, scales)
        previous = getattr(w, tla._CACHE_ATTRIBUTE)
        w.copy_(self.inputs(43)[1])
        self.linear(x, w, scales)
        current = getattr(w, tla._CACHE_ATTRIBUTE)
        self.assertIsNot(previous, current)
        scales.mul_(1.05)
        self.linear(x, w, scales)
        self.assertIsNot(current, getattr(w, tla._CACHE_ATTRIBUTE))
        graph, _, capture_stream = self.capture_on_warmed_stream(
            lambda: self.linear(x, w, scales)
        )
        sealed = getattr(w, tla._CACHE_ATTRIBUTE)
        scales.mul_(1.01)
        with self.assertRaisesRegex(RuntimeError, "changed after graph capture"):
            self.linear(x, w, scales)
        self.assertIs(getattr(w, tla._CACHE_ATTRIBUTE), sealed)
        cold = self.inputs(44)
        with patch.object(torch.xpu, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm.*before XPU graph"):
                self.linear(*cold)

    def test_cache_owners_and_allocator_pressure(self):
        x, w, scales = self.inputs()
        self.linear(x, w, scales)
        entry = getattr(w, tla._CACHE_ATTRIBUTE)
        scale_ref, packed_ref = weakref.ref(scales), weakref.ref(entry.packed)
        graph, actual, capture_stream = self.capture_on_warmed_stream(
            lambda saved_scales=scales: self.linear(x, w, saved_scales)
        )
        del scales, entry
        gc.collect()
        self.assertIsNotNone(scale_ref())
        self.assertIsNotNone(packed_ref())
        for seed in (43, 44):
            pressure = [
                torch.empty(1024**2, device="xpu", dtype=torch.int32) for _ in range(8)
            ]
            x.copy_(self.inputs(seed)[0])
            graph.replay()
            expected = self.linear(x, w, scale_ref(), False)
            self.assertTrue(
                torch.equal(
                    actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
                )
            )
            del pressure

    def test_cross_stream_ready_event_and_capture_warmup(self):
        x, w, scales = self.inputs()
        original = torch.xpu.current_stream()
        first, second = torch.xpu.Stream(), torch.xpu.Stream()
        first.wait_stream(original)
        second.wait_stream(original)
        with torch.xpu.stream(first):
            self.linear(x, w, scales)
        with torch.xpu.stream(second):
            with patch.object(
                torch.xpu, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "capture stream first"):
                    self.linear(x, w, scales)
            self.linear(x, w, scales)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=second):
                actual = self.linear(x, w, scales)
            graph.replay()
        original.wait_stream(second)
        expected = self.linear(x, w, scales, False)
        self.assertTrue(
            torch.equal(
                actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
            )
        )


if __name__ == "__main__":
    unittest.main()
