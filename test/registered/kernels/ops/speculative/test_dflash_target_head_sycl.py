"""SYCL target head: graph flags, FP64 envelopes, eligibility and state guards."""

import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.speculative.dflash_head_xpu import shared_candidate_head
from sglang.kernels.ops.speculative.dflash_target_head_sycl import (
    TargetHeadPlan,
    TargetHeadState,
    can_use_target_head,
    load_target_head,
    native_call,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="xpu-unit-test")

WINDOW = int(os.environ.get("SGLANG_TEST_DFLASH_NATIVE_WINDOW", "4"))

_NAMES = dict(
    rescore_run="dflash_head_rescore",
    target_finish="dflash_head_finish",
    target_fill="dflash_head_fill",
)


def has_native_b70():
    return (
        torch.xpu.is_available()
        and bool(os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"))
        and torch.xpu.get_device_properties().device_id == 0xE223
    )


@unittest.skipUnless(
    has_native_b70(), "requires B70 and explicitly built native library"
)
class TestTargetHeadGpu(CustomTestCase):
    def test_confidence_edges_and_changing_graph_inputs(self):
        torch.manual_seed(42)
        native = SimpleNamespace(
            call=lambda name, *args: native_call(_NAMES[name], *args)
        )
        device = "xpu"
        n = 248320
        h = torch.randn(WINDOW, 2048, device=device, dtype=torch.bfloat16)
        q = torch.randn(WINDOW, 2048, device=device).to(torch.float8_e4m3fn)
        xs = torch.rand(WINDOW, 16, device=device) * 0.02
        coarse = torch.randn(WINDOW, n, device=device, dtype=torch.bfloat16) * 0.1
        ids = torch.arange(64, device=device).repeat(WINDOW, 1)
        coeff = torch.full((3, n), 0.1, device=device)
        coeff[1].fill_(0.01)
        score = torch.randn(WINDOW, 64, device=device) * 5
        absolute = torch.full_like(score, 200)
        state = torch.empty(WINDOW, 4, device=device)
        partial = torch.empty(WINDOW, 970, device=device)
        prediction = torch.empty(WINDOW, device=device, dtype=torch.int64)
        flags = torch.empty(WINDOW, device=device, dtype=torch.int32)
        logits = torch.empty(WINDOW, n, device=device)

        # Rescore ABI/numeric smoke in the rebuilt multi-unit library.
        w = torch.randn(128, 2048, device=device, dtype=torch.bfloat16) * 0.02
        native.call("rescore_run", h, w, ids, score, absolute)
        hc, wc = h.cpu().double(), w.cpu().double()
        exact = hc @ wc[:64].t()
        absolute_exact = hc.abs() @ wc[:64].abs().t()
        gamma23 = (23 * 2**-24) / (1 - 23 * 2**-24)
        assert bool(
            ((score.cpu().double() - exact).abs() <= gamma23 * absolute_exact).all()
        )

        def op():
            native.call(
                "target_finish",
                h,
                q,
                xs,
                coarse,
                ids,
                coeff,
                score,
                absolute,
                state,
                partial,
                prediction,
                flags,
            )
            native.call("target_fill", prediction, logits)

        op()
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            op()

        def oracle(seed):
            hh, qq, ss = h.cpu().float(), q.cpu().float(), xs.cpu()
            ns, es = [], []
            for sd in [torch.float32, torch.float16, torch.bfloat16]:
                for od in [torch.float32, torch.float16, torch.bfloat16]:
                    a = (
                        (qq * ss.to(sd).float().repeat_interleave(128, 1))
                        .to(od)
                        .float()
                    )
                    ns.append(a.square().sum(1).sqrt())
                    es.append((a - hh).square().sum(1).sqrt())
            norm = hh.square().sum(1).sqrt() * 1.001 + 1e-10
            an = torch.stack(ns).amax(0) * 1.001 + 1e-10
            error = torch.stack(es).amax(0) * 1.001 + 1e-10
            gamma = (2048 * 2**-24) / (1 - 2048 * 2**-24)
            u = 2**-8
            c = coeff.cpu()
            limit = (
                (1 + u)
                * (
                    error[:, None] * c[0]
                    + an[:, None] * c[1]
                    + gamma * (an[:, None] * c[2] + norm[:, None] * c[0])
                    + u / (1 - u) * seed.float().abs()
                    + 1e-10
                )
                + u * seed.float().abs()
                + 1e-10
            )
            bound = seed.float() + limit
            idc = ids.cpu()
            bound.scatter_(1, idc, -float("inf"))
            sc, ab = score.cpu(), absolute.cpu()
            e = (gamma + gamma23) * ab * 1.001 + 1e-30
            lower, upper = (sc - e).bfloat16().float(), (sc + e).bfloat16().float()
            pred, ok = [], []
            for t in range(WINDOW):
                j = min(
                    range(64),
                    key=lambda x: (-float(sc[t, x].bfloat16()), int(idc[t, x])),
                )
                pred.append(int(idc[t, j]))
                internal = all(
                    j == k
                    or lower[t, j] > upper[t, k]
                    or (lower[t, j] == upper[t, k] and idc[t, j] < idc[t, k])
                    for k in range(64)
                )
                ok.append(bool(internal and lower[t, j] > bound[t].max()))
            return (
                pred,
                ok,
                torch.stack([norm, an, error], 1),
                bound.view(WINDOW, 970, 256).amax(2),
            )

        rows = []
        for case in [
            "ordinary",
            "lowest_id_tie",
            "rounding_ambiguity",
            "outside_winner",
            "zero",
            "all_fp8_codes",
            "nonfinite_hidden",
            "nonfinite_coarse",
        ]:
            h.normal_()
            q.copy_(torch.randn(WINDOW, 2048, device=device).to(q.dtype))
            xs.uniform_(0.0001, 0.0201)
            coarse.normal_().mul_(0.1)
            ids.copy_(torch.arange(64, device=device).repeat(WINDOW, 1))
            score.normal_().mul_(5)
            score[:, 0] = 50
            absolute.fill_(200)
            if case in ["lowest_id_tie", "rounding_ambiguity"]:
                ids[:, 0] = 100
                score[:, :2] = 50 if case == "lowest_id_tie" else 50.125
            elif case == "outside_winner":
                coarse[:, 1000] = 100
            elif case == "zero":
                h.zero_()
                q.zero_()
                coarse.zero_()
                score.zero_()
                absolute.zero_()
            elif case == "all_fp8_codes":
                raw = torch.arange(256, device=device, dtype=torch.int32)
                raw = raw[(raw & 127) != 127].to(torch.uint8).repeat(9)[:2048]
                q.view(torch.uint8).copy_(raw[None, :].expand(WINDOW, 2048))
            elif case == "nonfinite_hidden":
                h[:, 0] = float("nan")
            elif case == "nonfinite_coarse":
                coarse[:, 1000] = float("nan")
            seed = coarse.cpu()
            g.replay()
            torch.xpu.synchronize()
            actual_pred, actual_flags = (
                prediction.cpu().tolist(),
                [bool(x) for x in flags.cpu().tolist()],
            )
            torch.testing.assert_close(logits.argmax(1), prediction, rtol=0, atol=0)
            if case.startswith("nonfinite"):
                assert not any(actual_flags)
            else:
                expected_pred, expected_flags, expected_state, expected_partial = (
                    oracle(seed)
                )
                assert actual_pred == expected_pred, (case, actual_pred, expected_pred)
                assert actual_flags == expected_flags, (
                    case,
                    actual_flags,
                    expected_flags,
                )
                torch.testing.assert_close(
                    state[:, :3].cpu(), expected_state, rtol=2e-5, atol=2e-5
                )
                torch.testing.assert_close(
                    partial.cpu(), expected_partial, rtol=2e-5, atol=2e-5
                )
            rows.append(
                dict(
                    case=case, predictions=actual_pred, flags=actual_flags, passed=True
                )
            )

    def test_conditional_bf16_rows_and_graph(self):
        torch.manual_seed(42)
        weight = torch.randn((248320, 2048), device="xpu", dtype=torch.bfloat16) * 0.02
        hidden = torch.randn((WINDOW, 2048), device="xpu", dtype=torch.bfloat16)
        flags = torch.ones(WINDOW, device="xpu", dtype=torch.int32)
        predictions = torch.tensor(([0, 1, 4096, 248319] * (WINDOW // 4)), device="xpu")
        output = torch.empty((WINDOW, 248320), device="xpu", dtype=torch.float32)
        counts = torch.zeros(3, device="xpu", dtype=torch.int32)

        def run():
            native_call(
                "dflash_head_complete",
                hidden,
                weight,
                predictions,
                flags,
                output,
                counts,
            )
            return output

        run()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            run()
        for mask in [
            v * (WINDOW // 4)
            for v in [[1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]]
        ] + [[int(i != j) for i in range(WINDOW)] for j in range(WINDOW)]:
            hidden.normal_()
            flags.copy_(torch.tensor(mask, device="xpu", dtype=torch.int32))
            eager = run().clone()
            graph.replay()
            torch.xpu.synchronize()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
            reference = (hidden @ weight.t()).float()
            for token, fast in enumerate(mask):
                with self.subTest(mask=mask, token=token):
                    if fast:
                        expected = torch.full((248320,), float("-inf"), device="xpu")
                        expected[predictions[token]] = 0
                        torch.testing.assert_close(
                            output[token], expected, rtol=0, atol=0
                        )
                    else:
                        actual, truth = output[token], reference[token]
                        self.assertLess(
                            float((actual - truth).norm() / truth.norm()), 1e-4
                        )
                        self.assertEqual(int(actual.argmax()), int(truth.argmax()))
                        ids = torch.cat(
                            [truth.topk(8).indices, torch.arange(8, device="xpu")]
                        )
                        products = (
                            weight[ids].cpu().double() * hidden[token].cpu().double()
                        )
                        exact, absolute = products.sum(1), products.abs().sum(1)
                        error = (23 / (2**24 - 23)) * absolute
                        low, high = (
                            (exact - error).bfloat16().float().double(),
                            (exact + error).bfloat16().float().double(),
                        )
                        observed = actual[ids].cpu().double()
                        self.assertTrue(
                            bool(((observed >= low) & (observed <= high)).all())
                        )
        del graph

    def test_dispatch_guards(self):
        hidden = torch.empty((WINDOW, 2048), device="xpu", dtype=torch.bfloat16)
        head = SimpleNamespace(
            weight=torch.empty((248320, 2048), device="xpu", dtype=torch.bfloat16),
            org_vocab_size=248320,
        )
        lp = SimpleNamespace(
            vocab_size=248320,
            do_tensor_parallel_all_gather=False,
            do_tensor_parallel_all_gather_dp_attn=False,
            use_fp32_lm_head=False,
            logit_scale=None,
            final_logit_softcapping=None,
            return_full_logits=False,
            rl_on_policy_target=None,
        )
        metadata = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: True)
        )
        spec = SimpleNamespace(
            speculative_algorithm="DFLASH", speculative_num_draft_tokens=WINDOW
        )

        def eligible(h=hidden, bias=None):
            return can_use_target_head(lp, h, head, metadata, bias)

        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL.override(True),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ) as parallel,
            patch("sglang.srt.runtime_context.get_spec", return_value=spec),
        ):
            self.assertTrue(eligible())
            for flag in (
                envs.SGLANG_XPU_PAGED_DFLASH,
                envs.SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL,
            ):
                with flag.override(False):
                    self.assertFalse(eligible())
            for field, value in [
                ("vocab_size", 248319),
                ("do_tensor_parallel_all_gather", True),
                ("do_tensor_parallel_all_gather_dp_attn", True),
                ("use_fp32_lm_head", True),
                ("logit_scale", 1.0),
                ("final_logit_softcapping", 30.0),
                ("return_full_logits", True),
                ("rl_on_policy_target", "x"),
            ]:
                before = getattr(lp, field)
                setattr(lp, field, value)
                self.assertFalse(eligible(), field)
                setattr(lp, field, before)
            self.assertFalse(eligible(hidden.float()))
            self.assertFalse(eligible(hidden[:2]))
            self.assertFalse(eligible(hidden[:, ::2]))
            self.assertFalse(eligible(hidden.cpu()))
            self.assertFalse(eligible(bias=torch.zeros(1)))
            head.org_vocab_size = 248319
            self.assertFalse(eligible())
            head.org_vocab_size = 248320
            head.set_lora = True
            self.assertFalse(eligible())
            del head.set_lora
            metadata.forward_mode.is_target_verify = lambda: False
            self.assertFalse(eligible())
            metadata.forward_mode.is_target_verify = lambda: True
            spec.speculative_algorithm = "EAGLE"
            self.assertFalse(eligible())
            spec.speculative_algorithm = "DFLASH"
            spec.speculative_num_draft_tokens = 3
            self.assertFalse(eligible())
            spec.speculative_num_draft_tokens = WINDOW
            parallel.return_value = SimpleNamespace(tp_size=2)
            self.assertFalse(eligible())

    def test_candidate_sharing_source_version_and_lifetime(self):
        weight = torch.randn((128, 128), device="xpu", dtype=torch.bfloat16)
        a = shared_candidate_head(weight)
        b = shared_candidate_head(weight)
        self.assertIs(a, b)
        weight.add_(0.125)
        new = shared_candidate_head(weight)
        self.assertIsNot(new, a)
        ref = weakref.ref(a)
        del a, b
        gc.collect()
        self.assertIsNone(ref())


class TestTargetHeadState(CustomTestCase):
    def test_missing_and_incompatible_library(self):
        load_target_head.cache_clear()
        try:
            with envs.SGLANG_XPU_DFLASH_NATIVE_LIB.override(""):
                with self.assertRaisesRegex(RuntimeError, "requires.*NATIVE_LIB"):
                    load_target_head()
            with (
                envs.SGLANG_XPU_DFLASH_NATIVE_LIB.override("unused"),
                patch(
                    "sglang.kernels.ops.speculative.dflash_target_head_sycl.ctypes.CDLL",
                    return_value=SimpleNamespace(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "symbols missing"):
                    load_target_head()
        finally:
            load_target_head.cache_clear()

    def test_graph_and_eager_plans_and_source_changes(self):
        state = object.__new__(TargetHeadState)
        state.graph_plans = {}
        state.eager_plan = None
        hidden = torch.empty(4, 2048, dtype=torch.bfloat16)
        head = SimpleNamespace(weight=torch.empty(128, 2048, dtype=torch.bfloat16))
        plan = TargetHeadPlan(state, hidden, head, None, False)
        out = torch.empty(8)
        with patch("torch.xpu.is_current_stream_capturing", return_value=True):
            state.bind(plan, out)
        other = TargetHeadPlan(state, hidden, head, None, False)
        other_out = torch.empty(8)
        with patch("torch.xpu.is_current_stream_capturing", return_value=False):
            state.bind(other, other_out)
        self.assertIs(state.plan_for(out, True), plan)
        self.assertIs(state.plan_for(other_out, False), other)
        self.assertIsNone(state.plan_for(out, False))
        self.assertIsNone(state.plan_for(None, True))
        self.assertTrue(plan.source_matches())
        head.weight.add_(1)
        self.assertFalse(plan.source_matches())
        changed = TargetHeadPlan(state, hidden, head, None, False)
        head.weight = torch.empty_like(head.weight)
        self.assertFalse(changed.source_matches())

    def test_original_request_fallback_uses_matching_normalized_hidden(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        output = torch.empty(8)
        plan = SimpleNamespace(
            hidden=object(),
            head=object(),
            metadata=object(),
            use_buffer=False,
            source_matches=lambda: True,
        )
        state = SimpleNamespace(plan_for=Mock(return_value=plan))
        fallback = object()
        lp = SimpleNamespace(
            _xpu_dflash_target_state=state, _get_logits=Mock(return_value=fallback)
        )
        self.assertIs(
            LogitsProcessor.finish_xpu_dflash_target_head(lp, output, True, True),
            output,
        )
        lp._get_logits.assert_not_called()
        self.assertIs(
            LogitsProcessor.finish_xpu_dflash_target_head(lp, output, True, False),
            fallback,
        )
        lp._get_logits.assert_called_once_with(
            plan.hidden,
            plan.head,
            plan.metadata,
            None,
            False,
            skip_xpu_target_head=True,
        )
        lp._get_logits.reset_mock()
        plan.source_matches = lambda: False
        self.assertIs(
            LogitsProcessor.finish_xpu_dflash_target_head(lp, output, True, True),
            fallback,
        )
        lp._get_logits.assert_called_once()

    def test_unsupported_eager_call_clears_previous_plan(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        state = SimpleNamespace(eager_plan=object())
        lp = SimpleNamespace(
            _xpu_dflash_target_state=state,
            do_tensor_parallel_all_gather_dp_attn=False,
            do_tensor_parallel_all_gather=False,
            _gather_dp_attn_hidden_states=Mock(
                side_effect=RuntimeError("original path")
            ),
        )
        hidden = SimpleNamespace(device=SimpleNamespace(type="xpu"), shape=(4, 2048))
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL.override(True),
            patch("torch.xpu.is_current_stream_capturing", return_value=False),
            patch(
                "sglang.kernels.ops.speculative.dflash_target_head_sycl.can_use_target_head",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "original path"):
                LogitsProcessor._get_logits(lp, hidden, None, None)
        self.assertIsNone(state.eager_plan)


if __name__ == "__main__":
    unittest.main()
