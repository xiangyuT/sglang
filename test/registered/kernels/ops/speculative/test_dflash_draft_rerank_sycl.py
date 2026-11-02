"""Draft native rescore/selection: dynamic graph, ties, ownership and dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative import dflash_head_xpu as head
from sglang.kernels.ops.speculative.dflash_draft_rerank_sycl import (
    DraftRerankState,
    can_use_rerank,
)
from sglang.srt.environ import envs
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=35, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires B70 and draft rerank library")
class TestDraftRerankSycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 draft rerank")
        torch.manual_seed(42)
        cls.weight = torch.empty(
            (248320, 2048), device="xpu", dtype=torch.bfloat16
        ).normal_(0, 0.02)
        # Missing selected library must fail this suite, rather than skip it.
        DraftRerankState(cls.weight)

    def fixture(self, rows=7):
        hidden = torch.randn(rows, 2048, device="xpu", dtype=torch.bfloat16)
        ids = torch.randperm(248320, device="xpu")[: rows * 64].view(rows, 64)
        return hidden, ids

    def reference(self, hidden, ids):
        rows = hidden.shape[0]
        scores = torch.empty((rows, 64), device="xpu", dtype=torch.bfloat16)
        values = torch.empty((rows, 16), device="xpu", dtype=torch.bfloat16)
        output = torch.empty((rows, 16), device="xpu", dtype=torch.int64)
        head._rescore_shortlist[(rows, 16)](
            hidden, self.weight, ids, scores, K=2048, num_warps=4, grf_mode="256"
        )
        head._select_shortlist[(rows,)](scores, ids, values, output, num_warps=4)
        return scores, values, output

    def assert_bits(self, a, b):
        if a.dtype == torch.bfloat16:
            a, b = a.view(torch.int16), b.view(torch.int16)
        self.assertTrue(torch.equal(a, b))

    def test_dynamic_graph_full_scores_and_global_ties(self):
        state = DraftRerankState(self.weight)
        for rows in (3, 7):
            hidden, ids = self.fixture(rows)
            state.run(hidden, self.weight, ids)
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = state.run(hidden, self.weight, ids)
            for pattern in range(3):
                if pattern == 1:
                    hidden.zero_()
                elif pattern == 2:
                    hidden.copy_(
                        torch.randn_like(hidden)
                        * torch.exp2(torch.linspace(-6, 6, 2048, device="xpu"))
                    )
                    ids.copy_(ids.flip(1))
                scores, values, expected_ids = self.reference(hidden, ids)
                graph.replay()
                torch.xpu.synchronize()
                self.assert_bits(state.scores[:rows], scores)
                self.assert_bits(actual[0], values)
                self.assert_bits(actual[1], expected_ids)
                self.assertTrue(bool((state.counters == 0).all()))
                if pattern == 1:
                    self.assert_bits(actual[1], ids.sort(dim=1).values[:, :16])

    def test_independent_states_preserve_source_and_prior_output(self):
        first, second = DraftRerankState(self.weight), DraftRerankState(self.weight)
        self.assertNotEqual(first.counters.data_ptr(), second.counters.data_ptr())
        hidden, ids = self.fixture()
        before = self.weight.clone()
        a = tuple(t.clone() for t in first.run(hidden, self.weight, ids))
        second.run(hidden + 1, self.weight, ids)
        self.assert_bits(first.values, a[0])
        self.assert_bits(first.ids, a[1])
        self.assert_bits(self.weight, before)

    def test_layout_guards_and_stale_source(self):
        hidden, ids = self.fixture()
        state = DraftRerankState(self.weight)
        self.assertTrue(can_use_rerank(hidden, self.weight, ids))
        self.assertFalse(can_use_rerank(hidden[:4], self.weight, ids[:4]))
        self.assertFalse(can_use_rerank(hidden, self.weight, ids.int()))
        self.assertFalse(can_use_rerank(hidden, self.weight, ids.t().contiguous().t()))
        self.assertFalse(can_use_rerank(hidden, self.weight[:128], ids))
        self.weight.add_(0.000001)
        with self.assertRaisesRegex(ValueError, "source changed"):
            state.run(hidden, self.weight, ids)

    def test_missing_library_and_pre_capture_requirement(self):
        with envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_LIB.override(""):
            with self.assertRaisesRegex(RuntimeError, "DRAFT_RERANK_LIB"):
                DraftRerankState(self.weight)
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before graph capture"):
                DraftRerankState(self.weight)

    def test_disabled_and_unsupported_dispatch_and_unprepared_error(self):
        class FakeHead:
            _sycl_rerank = None

            def __call__(self, hidden):
                return None

        for enabled, rows in ((False, 3), (True, 8), (True, 3)):
            hidden, ids = self.fixture(rows)
            with (
                envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_SYCL.override(enabled),
                envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_LIB.override(""),
                patch.object(head, "xpu_dflash_topk", return_value=(None, ids)),
            ):
                if enabled and rows == 3:
                    with self.assertRaisesRegex(RuntimeError, "before capture"):
                        head.XpuDflashCandidateHead.refined_top16(
                            FakeHead(), hidden, self.weight
                        )
                else:
                    actual = head.XpuDflashCandidateHead.refined_top16(
                        FakeHead(), hidden, self.weight
                    )
                    expected = self.reference(hidden, ids)
                    self.assert_bits(actual[0], expected[1])
                    self.assert_bits(actual[1], expected[2])

    def test_model_rejects_incompatible_rerank_setting_before_allocation(self):
        model = SimpleNamespace(
            _xpu_candidate_head=None,
            lm_head=SimpleNamespace(weight=self.weight, org_vocab_size=248320),
        )
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_FP8_DRAFT_HEAD.override(True),
            envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_SYCL.override(True),
            envs.SGLANG_XPU_DFLASH_DRAFT_HEAD_RERANK.override(0),
            patch(
                "sglang.srt.models.dflash.get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "RERANK=64"):
                DFlash2DraftModel.prepare_xpu_candidate_head(model)
        self.assertIsNone(model._xpu_candidate_head)
