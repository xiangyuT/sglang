"""Saved page128 KV, aligned sliding windows and graph-safe native dispatch."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.speculative.dflash_attention_tla import (
    can_use_attention_paged_tla,
    load_attention_paged_tla,
    try_attention_paged_tla,
)
from sglang.kernels.ops.speculative.dflash_attention_xpu import (
    _run,
    try_xpu_dflash_attention,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="xpu-unit-test")


class PagedFixture:
    def __init__(self, n=0, stride=6144, device="xpu", m=4):
        self.device = device
        self.m = m
        self.q = torch.empty_strided(
            (m, 32, 128), (stride, 128, 1), device=device, dtype=torch.bfloat16
        )
        self.k = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)
        self.kb = torch.empty((8320, 8, 128), device=device, dtype=torch.bfloat16)
        self.vb = torch.empty_like(self.kb)
        self.fullids = torch.zeros(16384, device=device, dtype=torch.int64)
        self.ids = torch.zeros(8188, device=device, dtype=torch.int64)
        self.fullptr = torch.tensor([7, 7], device=device, dtype=torch.int32)
        self.ptr = self.fullptr.clone()
        self.loc = torch.empty(m, device=device, dtype=torch.int64)
        self.out = torch.empty((m, 32, 128), device=device, dtype=torch.bfloat16)
        self.qptr = torch.tensor([0, m], device=device, dtype=torch.int32)
        self.configure(n)

    def configure(self, n):
        self.q.normal_()
        self.k.normal_()
        self.v.normal_()
        pages = torch.randperm(64) + 1
        self.logical = (pages[:, None] * 128 + torch.arange(128)[None, :]).reshape(-1)
        slots = self.logical[: n + self.m].to(self.device)
        self.kb.fill_(float("nan"))
        self.vb.fill_(float("nan"))
        for pool in (self.kb, self.vb):
            pool.index_copy_(
                0,
                slots[:n],
                torch.randn(n, 8, 128, device=self.device, dtype=torch.bfloat16),
            )
        self.kb.index_copy_(0, slots[n:], self.k)
        self.vb.index_copy_(0, slots[n:], self.v)
        self.fullids[7 : 7 + n].copy_(slots[:n])
        self.fullptr[1] = 7 + n
        clipped = min(n, 2047)
        self.ids[7 : 7 + clipped].copy_(slots[n - clipped : n])
        self.ptr[1] = 7 + clipped
        self.loc.copy_(slots[n:])
        self.n = n

    def saved(self):
        return self.fullptr, self.fullids, self.loc

    def arguments(self):
        return (
            self.q,
            self.k,
            self.v,
            self.kb,
            self.vb,
            self.ptr,
            self.ids,
            self.out,
            self.saved(),
        )

    def reference(self):
        return _run(
            self.q, self.k, self.v, self.kb, self.vb, self.ptr, self.ids, self.out
        )

    def dispatch(self, *, saved=True, page_size=128):
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
            self.m,
            1.0,
            1.0,
            128**-0.5,
            sliding_window_size=2047,
            page_size=page_size,
            saved_paged_kv=self.saved() if saved else None,
        )


class TestPagedTlaCpu(unittest.TestCase):
    def test_cpu_does_not_load_library(self):
        for m in (4, 8):
            c = PagedFixture(device="cpu", m=m)
            with patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_paged_tla",
                side_effect=AssertionError("unsupported"),
            ):
                self.assertFalse(try_attention_paged_tla(*c.arguments(), page_size=128))


@unittest.skipUnless(
    torch.xpu.is_available()
    and os.environ.get("SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB")
    and os.environ.get("SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB"),
    "B70 and both explicit native attention libraries required",
)
class TestPagedTla(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 required")

    def enabled(self, master=True, parent=True, tla=True, paged=True):
        stack = ExitStack()
        for setting, value in [
            (envs.SGLANG_XPU_PAGED_DFLASH, master),
            (envs.SGLANG_XPU_DFLASH_ATTENTION, parent),
            (envs.SGLANG_XPU_DFLASH_ATTENTION_TLA, tla),
            (envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA, paged),
        ]:
            stack.enter_context(setting.override(value))
        return stack

    def test_dispatch_flags_and_saved_metadata_fallback(self):
        c = PagedFixture(1280)
        expected = c.reference().clone()
        for args in [
            (False, True, True, True),
            (True, False, True, True),
            (True, True, False, True),
            (True, True, True, False),
        ]:
            with (
                self.enabled(*args),
                patch(
                    "sglang.kernels.ops.speculative.dflash_attention_tla.try_attention_paged_tla",
                    side_effect=AssertionError("disabled"),
                ),
            ):
                self.assertTrue(c.dispatch())
            torch.testing.assert_close(c.out, expected, rtol=0.02, atol=0.003)
        with (
            self.enabled(),
            patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.try_attention_tla",
                side_effect=AssertionError("must use saved pages"),
            ),
        ):
            self.assertTrue(c.dispatch())
        torch.testing.assert_close(c.out, expected, rtol=0.02, atol=0.003)
        with (
            self.enabled(),
            patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.try_attention_paged_tla",
                side_effect=AssertionError("missing metadata"),
            ),
        ):
            self.assertTrue(c.dispatch(saved=False))

    def test_backend_dflash_target_verify_metadata_and_fallbacks(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        from sglang.srt.layers.radix_attention import AttentionType
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        c = PagedFixture()
        pool = object.__new__(MHATokenToKVPool)
        backend = SimpleNamespace(
            is_draft_runner=True,
            page_size=128,
            token_to_kv_pool=pool,
            forward_metadata=SimpleNamespace(kv_indptr=c.fullptr, kv_indices=c.fullids),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            out_cache_loc=c.loc,
        )
        layer = SimpleNamespace(attn_type=AttentionType.ENCODER_ONLY)
        get = TritonAttnBackend._saved_dflash_paged_kv
        with self.enabled():
            saved = get(backend, batch, layer, True)
            self.assertTrue(all(a is b for a, b in zip(saved, c.saved())))
            for obj, name, value in (
                (backend, "is_draft_runner", False),
                (backend, "page_size", 1),
                (backend, "token_to_kv_pool", object()),
                (batch, "forward_mode", ForwardMode.DRAFT_EXTEND_V2),
                (batch, "forward_mode", ForwardMode.EXTEND),
                (batch, "spec_algorithm", SpeculativeAlgorithm.DSPARK),
                (layer, "attn_type", AttentionType.DECODER),
            ):
                original = getattr(obj, name)
                setattr(obj, name, value)
                self.assertIsNone(get(backend, batch, layer, True))
                setattr(obj, name, original)
            self.assertIsNone(get(backend, batch, layer, False))
        with self.enabled(paged=False):
            self.assertIsNone(get(backend, batch, layer, True))

    def test_dynamic_graph_pages_and_long_prefix(self):
        lengths = (
            0,
            1,
            63,
            64,
            65,
            127,
            128,
            129,
            1024,
            1280,
            2044,
            2045,
            2046,
            2047,
            2048,
            2051,
            2174,
            2175,
            2176,
            3072,
            4092,
            4096,
            6144,
            8188,
        )
        for stride in (4096, 6144):
            c = PagedFixture(stride=stride)
            with self.enabled():
                c.dispatch()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    self.assertTrue(c.dispatch())
                for seed in (42, 43, 44):
                    torch.manual_seed(seed)
                    for n in lengths:
                        c.configure(n)
                        before_k, before_v = c.kb.clone(), c.vb.clone()
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
                        last = int(c.logical[n + 3])
                        end = (last // 128 + 1) * 128
                        before_k[last + 1 : end].zero_()
                        before_v[last + 1 : end].zero_()
                        torch.testing.assert_close(
                            c.kb, before_k, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(
                            c.vb, before_v, rtol=0, atol=0, equal_nan=True
                        )
                del graph

    def test_exact_left_boundary_after_page_clipping_and_future(self):
        c = PagedFixture(3072)
        for kind in ("left", "future"):
            c.q.zero_()
            c.q[:, :, 0] = 32
            c.k.zero_()
            c.v.zero_()
            c.kb.zero_()
            c.vb.zero_()
            if kind == "left":
                slot = int(c.logical[c.n - 2047])
                c.kb[slot, :, 0] = 32
                c.vb[slot] = 10
            else:
                c.k[3, :, 0] = 32
                c.v[3] = 10
            c.kb.index_copy_(0, c.loc, c.k)
            c.vb.index_copy_(0, c.loc, c.v)
            with self.enabled():
                self.assertTrue(c.dispatch())
            expected = (
                torch.full_like(c.out, 10)
                if kind == "future"
                else torch.zeros_like(c.out)
            )
            if kind == "left":
                expected[0] = 10
            torch.testing.assert_close(c.out, expected, rtol=0, atol=0)

    def test_layout_and_saved_metadata_guards(self):
        c = PagedFixture()
        for saved, page in [
            (None, 128),
            ((), 128),
            ((c.fullptr, c.fullids.int(), c.loc), 128),
            ((c.fullptr, c.fullids, c.loc.int()), 128),
            (c.saved(), 1),
        ]:
            with patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_paged_tla",
                side_effect=AssertionError("unsupported"),
            ):
                self.assertFalse(
                    try_attention_paged_tla(*c.arguments()[:-1], saved, page_size=page)
                )
        self.assertTrue(can_use_attention_paged_tla(*c.arguments(), page_size=128))

    def test_missing_library_and_wrong_abi_fail(self):
        c = PagedFixture()
        load_attention_paged_tla.cache_clear()
        try:
            with (
                self.enabled(),
                envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.override(""),
            ):
                with self.assertRaisesRegex(RuntimeError, "PAGED_TLA_LIB"):
                    c.dispatch()
            load_attention_paged_tla.cache_clear()
            with (
                self.enabled(),
                envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.override(
                    os.environ["SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB"]
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "symbols missing"):
                    c.dispatch()
        finally:
            load_attention_paged_tla.cache_clear()


@unittest.skipUnless(
    torch.xpu.is_available()
    and os.environ.get("SGLANG_TEST_DFLASH_PAGED_TLA_M8_LIB")
    and os.environ.get("SGLANG_TEST_DFLASH_PAGED_TLA_LEGACY_LIB"),
    "B70 and explicit M8/legacy M4 paged libraries required",
)
class TestPagedTlaM8(CustomTestCase):
    enabled = TestPagedTla.enabled

    def setUp(self):
        super().setUp()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            self.skipTest("B70 required")
        self.m4_library = envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.get()
        selected = envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.override(
            os.environ["SGLANG_TEST_DFLASH_PAGED_TLA_M8_LIB"]
        )
        selected.__enter__()
        self.addCleanup(selected.__exit__, None, None, None)
        self.addCleanup(load_attention_paged_tla.cache_clear)
        load_attention_paged_tla.cache_clear()

    def test_dynamic_m8_graph_pages_prefixes_and_cache_preservation(self):
        lengths = (
            0,
            1,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            1024,
            1280,
            2040,
            2047,
            2048,
            2055,
            2174,
            2175,
            2176,
            3072,
            4096,
            6144,
            8184,
        )
        self.assertEqual(load_attention_paged_tla().window_size, 8)
        for stride in (4096, 6144):
            c = PagedFixture(stride=stride, m=8)
            with self.enabled():
                self.assertTrue(c.dispatch())
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    self.assertTrue(c.dispatch())
                for seed in (42, 43, 44):
                    torch.manual_seed(seed)
                    for n in lengths:
                        c.configure(n)
                        before_k, before_v = c.kb.clone(), c.vb.clone()
                        expected = c.reference().clone()
                        graph.replay()
                        torch.testing.assert_close(
                            c.out, expected, rtol=0.02, atol=0.003
                        )
                        relative = torch.linalg.vector_norm((c.out - expected).float())
                        relative /= torch.linalg.vector_norm(expected.float())
                        self.assertLess(relative.item(), 0.005)
                        self.assertTrue(torch.isfinite(c.out).all().item())
                        last = int(c.logical[n + c.m - 1])
                        end = (last // 128 + 1) * 128
                        before_k[last + 1 : end].zero_()
                        before_v[last + 1 : end].zero_()
                        torch.testing.assert_close(
                            c.kb, before_k, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(
                            c.vb, before_v, rtol=0, atol=0, equal_nan=True
                        )
                del graph

    def test_m8_exact_left_boundary_and_future_token_visibility(self):
        c = PagedFixture(3072, m=8)
        for kind in ("left", "future"):
            c.q.zero_()
            c.q[:, :, 0] = 32
            c.k.zero_()
            c.v.zero_()
            c.kb.zero_()
            c.vb.zero_()
            if kind == "left":
                slot = int(c.logical[c.n - 2047])
                c.kb[slot, :, 0] = 32
                c.vb[slot] = 10
            else:
                c.k[-1, :, 0] = 32
                c.v[-1] = 10
            c.kb.index_copy_(0, c.loc, c.k)
            c.vb.index_copy_(0, c.loc, c.v)
            with self.enabled():
                self.assertTrue(c.dispatch())
            expected = (
                torch.full_like(c.out, 10)
                if kind == "future"
                else torch.zeros_like(c.out)
            )
            if kind == "left":
                expected[0] = 10
            torch.testing.assert_close(c.out, expected, rtol=0, atol=0)

    def test_m8_flags_and_missing_saved_metadata_use_retained_attention(self):
        c = PagedFixture(1280, m=8)
        expected = c.reference().clone()
        for flags in (
            (False, True, True, True),
            (True, False, True, True),
            (True, True, False, True),
            (True, True, True, False),
        ):
            with (
                self.enabled(*flags),
                patch(
                    "sglang.kernels.ops.speculative.dflash_attention_tla.try_attention_paged_tla",
                    side_effect=AssertionError("disabled"),
                ),
            ):
                self.assertTrue(c.dispatch())
            torch.testing.assert_close(c.out, expected, rtol=0, atol=0)
        with (
            self.enabled(),
            patch(
                "sglang.kernels.ops.speculative.dflash_attention_tla.load_attention_paged_tla",
                side_effect=AssertionError("missing saved metadata"),
            ),
        ):
            self.assertTrue(c.dispatch(saved=False))
        torch.testing.assert_close(c.out, expected, rtol=0, atol=0)

    def test_library_window_mismatch_rejects_before_output_or_cache_writes(self):
        c = PagedFixture(1280, m=4)
        c.out.fill_(float("nan"))
        before_k, before_v = c.kb.clone(), c.vb.clone()
        with self.enabled(), self.assertRaisesRegex(RuntimeError, "window mismatch"):
            c.dispatch()
        self.assertTrue(c.out.isnan().all().item())
        torch.testing.assert_close(c.kb, before_k, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(c.vb, before_v, rtol=0, atol=0, equal_nan=True)

        c = PagedFixture(1280, m=8)
        c.out.fill_(float("nan"))
        before_k, before_v = c.kb.clone(), c.vb.clone()
        with envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.override(
            os.environ["SGLANG_TEST_DFLASH_PAGED_TLA_LEGACY_LIB"]
        ):
            load_attention_paged_tla.cache_clear()
            self.assertEqual(load_attention_paged_tla().window_size, 4)
            with (
                self.enabled(),
                self.assertRaisesRegex(RuntimeError, "window mismatch"),
            ):
                c.dispatch()
        self.assertTrue(c.out.isnan().all().item())
        torch.testing.assert_close(c.kb, before_k, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(c.vb, before_v, rtol=0, atol=0, equal_nan=True)

    def test_new_m4_library_retains_legacy_complete_output_bits(self):
        c = PagedFixture(2048)
        expected = None
        for path in (
            os.environ["SGLANG_TEST_DFLASH_PAGED_TLA_LEGACY_LIB"],
            self.m4_library,
        ):
            with envs.SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB.override(path):
                load_attention_paged_tla.cache_clear()
                with self.enabled():
                    self.assertTrue(c.dispatch())
                if expected is None:
                    expected = c.out.clone()
                else:
                    self.assertTrue(
                        torch.equal(c.out.view(torch.int16), expected.view(torch.int16))
                    )


if __name__ == "__main__":
    unittest.main()
