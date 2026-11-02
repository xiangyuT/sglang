"""Existing recurrent verify: exact checkpoints, live graphs and fallback."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.xpu.kernels.fla import (
    fused_sigmoid_gating_recurrent as recurrent,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires B70")
class TestDflashGdnVerify(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 recurrent verify")

    def inputs(self, steps=4):
        torch.manual_seed(42)
        mixed = torch.randn(1, steps, 8192, device="xpu", dtype=torch.bfloat16) * 0.2
        q, k, v = mixed.split((2048, 2048, 4096), dim=-1)
        gates = torch.randn(steps, 64, device="xpu", dtype=torch.bfloat16)
        return dict(
            A_log=torch.rand(32, device="xpu") * 4 - 2,
            a=gates[:, 32:],
            dt_bias=torch.rand(32, device="xpu", dtype=torch.bfloat16),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q.view(1, steps, 16, 128),
            k=k.view(1, steps, 16, 128),
            v=v.view(1, steps, 32, 128),
            b=gates[:, :32],
            initial_state_source=torch.randn(5, 32, 128, 128, device="xpu") * 0.1,
            initial_state_indices=torch.tensor([2], device="xpu", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, steps], device="xpu", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=torch.full(
                (2, steps, 32, 128, 128), 0.125, device="xpu"
            ),
            intermediate_state_indices=torch.tensor(
                [1], device="xpu", dtype=torch.int32
            ),
            cache_steps=steps,
            round_gdn_gate_to_bf16=True,
        )

    def run_verify(self, kwargs, child=True, master=True):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_GDN_NARROW.override(child),
        ):
            return recurrent.fused_sigmoid_gating_delta_rule_update(**kwargs)

    def test_all_checkpoints_and_live_graph_metadata(self):
        kwargs = self.inputs()
        state_before = kwargs["initial_state_source"].clone()
        self.run_verify(kwargs)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = self.run_verify(kwargs)
        cache = kwargs["intermediate_states_buffer"]
        # Check every accepted prefix's state, plus untouched cache slots.
        for state_slot, cache_slot in ((2, 1), (4, 0), (-1, 1), (1, -1)):
            kwargs["q"].add_(0.0078125)
            kwargs["a"].mul_(0.95)
            kwargs["initial_state_indices"].fill_(state_slot)
            kwargs["intermediate_state_indices"].fill_(cache_slot)
            cache.fill_(0.125)
            expected = self.run_verify(kwargs, child=False)
            expected_cache = cache.clone()
            cache.fill_(0.125)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
            torch.testing.assert_close(
                kwargs["initial_state_source"], state_before, rtol=0, atol=0
            )
        del graph

    def test_master_child_and_unsupported_verify_fallbacks(self):
        target = "fused_sigmoid_gating_delta_rule_update_kernel"
        with patch.object(recurrent, target) as kernel:
            for master, child, expected in (
                (False, True, 16),
                (True, False, 16),
                (True, True, 4),
            ):
                self.run_verify(self.inputs(), child, master)
                self.assertEqual(
                    kernel.__getitem__.return_value.call_args.kwargs["BV"], expected
                )
            kwargs = self.inputs()
            for updates in (
                {"round_gdn_gate_to_bf16": False},
                {"disable_state_update": False},
                {"intermediate_states_buffer": None},
                {
                    "retrieve_parent_token": torch.zeros(
                        1, 4, device="xpu", dtype=torch.int64
                    )
                },
            ):
                self.run_verify({**kwargs, **updates})
                self.assertEqual(
                    kernel.__getitem__.return_value.call_args.kwargs["BV"], 16
                )
            self.run_verify(self.inputs(3))
            self.assertEqual(kernel.__getitem__.return_value.call_args.kwargs["BV"], 16)
