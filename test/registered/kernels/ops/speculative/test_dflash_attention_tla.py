"""Guarded draft attention dispatch, dynamic graph and exact mask semantics."""

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from sglang.kernels.ops.speculative.dflash_attention_tla import (
    can_use_attention_tla,
    load_attention_tla,
    try_attention_tla,
)
from sglang.kernels.ops.speculative.dflash_attention_xpu import (
    _run,
    try_xpu_dflash_attention,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")


class Fixture:
    def __init__(self, m=4, qstride=6144, device="xpu"):
        self.q = torch.empty_strided(
            (m, 32, 128), (qstride, 128, 1), device=device, dtype=torch.bfloat16
        ).normal_()
        self.k = torch.randn((m, 8, 128), device=device, dtype=torch.bfloat16)
        self.v = torch.randn_like(self.k)
        self.kb = torch.randn((4096, 8, 128), device=device, dtype=torch.bfloat16)
        self.vb = torch.randn_like(self.kb)
        self.ptr = torch.tensor([7, 7], device=device, dtype=torch.int32)
        self.ids = torch.randperm(4096, device=device, dtype=torch.int64)
        self.qptr = torch.tensor([0, m], device=device, dtype=torch.int32)
        self.out = torch.empty((m, 32, 128), device=device, dtype=torch.bfloat16)

    def arguments(self):
        return self.q, self.k, self.v, self.kb, self.vb, self.ptr, self.ids, self.out

    def reference(self):
        return _run(*self.arguments()[:-1], self.out)

    def dispatch(self, **kwargs):
        return try_xpu_dflash_attention(
            self.q,
            self.k,
            self.v,
            self.out,
            self.kb,
            self.vb,
            self.qptr,
            self.ptr,
            self.ids,
            None,
            False,
            None,
            self.q.shape[0],
            1.0,
            1.0,
            128**-0.5,
            sliding_window_size=2047,
            **kwargs,
        )


class TestAttentionTlaCpu(unittest.TestCase):
    def test_cpu_rejected_without_loading_library(self):
        c = Fixture(device="cpu")
        c.out.fill_(float("nan"))
        with patch(
            "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_tla",
            side_effect=AssertionError("unsupported input must not load a library"),
        ):
            self.assertFalse(try_attention_tla(*c.arguments()))
        self.assertTrue(torch.isnan(c.out).all().item())


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB"),
    "requires B70 and an explicit attention TLA library",
)
class TestAttentionTla(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("requires B70")

    def enabled(self, master=True, parent=True, child=True):
        stack = ExitStack()
        for setting, value in (
            (envs.SGLANG_XPU_PAGED_DFLASH, master),
            (envs.SGLANG_XPU_DFLASH_ATTENTION, parent),
            (envs.SGLANG_XPU_DFLASH_ATTENTION_TLA, child),
        ):
            stack.enter_context(setting.override(value))
        return stack

    def test_flags_select_native_and_preserve_existing_path(self):
        c = Fixture()
        c.ptr[1] = 1287
        expected = c.reference().clone()
        for master, parent, child in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with (
                self.enabled(master, parent, child),
                patch(
                    "sglang.kernels.ops.speculative.dflash_attention_tla.try_attention_tla",
                    side_effect=AssertionError(
                        "disabled child must retain the existing path"
                    ),
                ),
            ):
                self.assertTrue(c.dispatch())
            torch.testing.assert_close(c.out, expected, rtol=0, atol=0)
        with (
            self.enabled(),
            patch(
                "sglang.kernels.ops.speculative.dflash_attention_xpu._run",
                side_effect=AssertionError("selected supported input must use TLA"),
            ),
        ):
            self.assertTrue(c.dispatch(page_size=128))
        torch.testing.assert_close(c.out, expected, rtol=0.02, atol=0.003)

    def test_dynamic_graph_with_both_q_strides(self):
        for qstride in (4096, 6144):
            torch.manual_seed(42)
            c = Fixture(qstride=qstride)
            with self.enabled():
                self.assertTrue(c.dispatch())
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    self.assertTrue(c.dispatch())
                for seed in (42, 43, 44):
                    torch.manual_seed(seed)
                    for length in (
                        0,
                        1,
                        63,
                        64,
                        65,
                        1024,
                        1280,
                        2044,
                        2045,
                        2046,
                        2047,
                    ):
                        c.ptr[1] = 7 + length
                        c.q.normal_()
                        c.k.normal_()
                        c.v.normal_()
                        c.ids.copy_(
                            torch.randperm(4096, device="xpu", dtype=torch.int64)
                        )
                        expected = c.reference().clone()
                        graph.replay()
                        torch.testing.assert_close(
                            c.out, expected, rtol=0.02, atol=0.003
                        )
                        relative = torch.linalg.vector_norm(
                            (c.out - expected).float()
                        ) / torch.linalg.vector_norm(expected.float())
                        self.assertLess(relative.item(), 0.005)
                        self.assertTrue(torch.isfinite(c.out).all().item())
                del graph

    def test_exact_left_boundary_and_bidirectional_current_block(self):
        c = Fixture()
        c.ptr[1] = 2054
        c.q.zero_()
        c.q[:, :, 0] = 32
        c.k.zero_()
        c.v.zero_()
        c.kb.zero_()
        c.vb.zero_()
        c.kb[c.ids[7], :, 0] = 32
        c.vb[c.ids[7]] = 10
        with self.enabled():
            self.assertTrue(c.dispatch())
            torch.testing.assert_close(
                c.out[0], torch.full_like(c.out[0], 10), rtol=0, atol=0
            )
            torch.testing.assert_close(
                c.out[1:], torch.zeros_like(c.out[1:]), rtol=0, atol=0
            )
            c.kb.zero_()
            c.vb.zero_()
            c.k[3, :, 0] = 32
            c.v[3] = 10
            self.assertTrue(c.dispatch())
            torch.testing.assert_close(
                c.out, torch.full_like(c.out, 10), rtol=0, atol=0
            )

    def test_layout_guards_and_other_width_fallback(self):
        for m in (2, 3, 5, 8):
            c = Fixture(m=m)
            self.assertFalse(can_use_attention_tla(*c.arguments()))
            with (
                self.enabled(),
                patch(
                    "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_tla",
                    side_effect=AssertionError(
                        "other draft widths must retain existing attention"
                    ),
                ),
            ):
                self.assertTrue(c.dispatch())
        c = Fixture()
        for field, value in (
            ("q", c.q.float()),
            (
                "q",
                torch.zeros((4, 64, 128), device="xpu", dtype=torch.bfloat16)[:, ::2],
            ),
            (
                "k",
                torch.zeros((4, 16, 128), device="xpu", dtype=torch.bfloat16)[:, ::2],
            ),
            ("ptr", c.ptr.long()),
            ("ids", c.ids.int()),
            ("out", c.q),
        ):
            original = getattr(c, field)
            setattr(c, field, value)
            self.assertFalse(can_use_attention_tla(*c.arguments()))
            setattr(c, field, original)
        with (
            self.enabled(),
            patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_tla",
                side_effect=AssertionError(
                    "unsupported semantic parameters must not enter TLA"
                ),
            ),
        ):
            c.out.fill_(float("nan"))
            self.assertFalse(c.dispatch(page_size=2))
            self.assertTrue(torch.isnan(c.out).all().item())

    def test_selected_missing_library_fails_explicitly(self):
        c = Fixture()
        load_attention_tla.cache_clear()
        try:
            with self.enabled(), envs.SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB.override(""):
                with self.assertRaisesRegex(RuntimeError, "ATTENTION_TLA_LIB"):
                    c.dispatch()
        finally:
            load_attention_tla.cache_clear()


if __name__ == "__main__":
    unittest.main()
