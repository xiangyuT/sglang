"""Exact pool-owned GDN windows, state lifetime, conv rollback and opt-in guards."""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention import dflash_gdn_sycl as native
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires B70 and precise GDN library")
class TestDflashGdnSycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 GDN delta replay")
        native.load_gdn()

    def fixture(self, stride=32, layers=2, steps=4):
        torch.manual_seed(42)
        self.layers = layers
        self.steps = steps
        source = torch.randn(layers, 5, 32, 128, 128, device="xpu") * 0.1
        with patch.object(native, "validate_runtime"):
            scratch = native.allocate_scratch(
                source, steps, is_kda=False, replayssm=False, envelope=False
            )
        self.pool = MambaPool.SpeculativeState(
            temporal=source,
            conv=[torch.randn(layers, 5, 16, 3, device="xpu", dtype=torch.bfloat16)],
            intermediate_ssm=torch.full(
                (layers, 2, steps, 32, 128, 128), float("nan"), device="xpu"
            ),
            intermediate_conv_window=[
                torch.randn(layers, 2, steps, 16, 3, device="xpu", dtype=torch.bfloat16)
            ],
            dflash_sycl=scratch,
        )
        self.layer = [self.pool.at_layer_idx(i) for i in range(layers)]
        self.qkv = (
            torch.randn(layers, 1, steps, 8192, device="xpu", dtype=torch.bfloat16)
            * 0.2
        )
        gates = torch.randn(layers, steps, 64, device="xpu", dtype=torch.bfloat16)
        self.a, self.b = gates[:, :, 32:], gates[:, :, :32]
        if stride == 32:
            self.a, self.b = self.a.contiguous(), self.b.contiguous()
        self.alog = torch.rand(layers, 32, device="xpu") * 4 - 2
        self.bias = torch.rand(layers, 32, device="xpu", dtype=torch.bfloat16)
        self.indices = torch.tensor([2], device="xpu", dtype=torch.int32)
        self.destination = torch.tensor([2], device="xpu", dtype=torch.int64)
        self.last = torch.tensor([steps - 1], device="xpu", dtype=torch.int64)
        self.cu = torch.tensor([0, steps], device="xpu", dtype=torch.int32)
        self.cache_index = torch.tensor([1], device="xpu", dtype=torch.int32)
        self.ref_cache = torch.empty_like(self.pool.intermediate_ssm)
        req_pool = SimpleNamespace(
            get_speculative_mamba2_params_all_layers=lambda: self.pool,
            mamba_pool=SimpleNamespace(replayssm_is_kda=False),
        )
        self.backend = hybrid.HybridLinearAttnBackend.__new__(
            hybrid.HybridLinearAttnBackend
        )
        self.backend.linear_attn_backend = SimpleNamespace(
            req_to_token_pool=req_pool,
            forward_metadata=SimpleNamespace(mamba_cache_indices=self.destination),
            accept_lens_pool=None,
            _translate_mamba_indices=lambda indices: indices,
        )
        self.backend._update_ple_state_after_mtp_verify = Mock()

    def params(self, layer):
        q, k, v = self.qkv[layer].split((2048, 2048, 4096), dim=-1)
        return dict(
            q=q.view(1, self.steps, 16, 128),
            k=k.view(1, self.steps, 16, 128),
            v=v.view(1, self.steps, 32, 128),
            a=self.a[layer],
            b=self.b[layer],
            A_log=self.alog[layer],
            dt_bias=self.bias[layer],
            indices=self.indices,
            cu_seqlens=self.cu,
            steps=self.steps,
            native_bf16_gdn=True,
            retrieve_parent_token=None,
            stable_rows=False,
        )

    def verify(self):
        return [
            native.verify_gdn(self.layer[i], **self.params(i))
            for i in range(self.layers)
        ]

    def reference(self):
        out = []
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_GDN_NARROW.override(True),
        ):
            for i in range(self.layers):
                p = self.params(i)
                for key in ("steps", "native_bf16_gdn", "stable_rows"):
                    p.pop(key)
                p["initial_state_indices"] = p.pop("indices")
                out.append(
                    fused_sigmoid_gating_delta_rule_update(
                        **p,
                        initial_state_source=self.pool.temporal[i],
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                        use_qk_l2norm_in_kernel=True,
                        round_gdn_gate_to_bf16=True,
                        disable_state_update=True,
                        cache_steps=self.steps,
                        intermediate_states_buffer=self.ref_cache[i],
                        intermediate_state_indices=self.cache_index,
                    )
                )
        return torch.stack(out)

    def update(self, track=None, track_steps=None):
        with patch.object(hybrid, "pp_spec_stable_rows_enabled", return_value=False):
            self.backend.update_mamba_state_after_mtp_verify(
                self.last, track, track_steps, None
            )

    def assert_bits(self, a, b):
        dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        self.assertTrue(
            torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))
        )

    def test_dynamic_graph_every_prefix_and_original_conv_rollback(self):
        self.check_dynamic_graph(4)

    def test_m8_dynamic_graph_every_prefix_and_original_conv_rollback(self):
        self.check_dynamic_graph(8)

    def check_dynamic_graph(self, steps):
        for stride in (32, 64):
            self.fixture(stride, steps=steps)
            self.verify()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                out = self.verify()
                self.update()
            cases = [(2, 2, prefix) for prefix in range(1, steps + 1)] + [
                (1, 3, steps),
                (3, 1, steps - 1),
                (2, 4, steps),
                (-1, 2, 2),
                (2, 3, 0),
                (2, -1, steps),
            ]
            for src, dst, prefix in cases:
                self.qkv.add_(0.0078125)
                self.a.add_(0.015625)
                self.pool.temporal.add_(0.001)
                self.indices.fill_(src)
                self.destination.fill_(dst)
                self.last.fill_(prefix - 1)
                expected_out = self.reference()
                expected = self.pool.temporal.clone()
                expected_conv = self.pool.conv[0].clone()
                if prefix and dst >= 0:
                    expected[:, dst] = self.ref_cache[:, 1, prefix - 1]
                    expected_conv[:, dst] = self.pool.intermediate_conv_window[0][
                        :, 1, prefix - 1
                    ]
                # The backend's conv rollback uses scratch row0 without PP-stable rows.
                self.pool.intermediate_conv_window[0][:, 0].copy_(
                    self.pool.intermediate_conv_window[0][:, 1]
                )
                graph.replay()
                torch.xpu.synchronize()
                self.assert_bits(torch.stack(out), expected_out)
                self.assert_bits(self.pool.temporal, expected)
                self.assert_bits(self.pool.conv[0], expected_conv)
                self.assertTrue(bool(self.pool.intermediate_ssm.isnan().all()))
            self.backend._update_ple_state_after_mtp_verify.assert_called()

    def test_saved_source_survives_input_reuse(self):
        self.check_saved_source_input_reuse(4)

    def test_m8_saved_source_survives_input_reuse(self):
        self.check_saved_source_input_reuse(8)

    def check_saved_source_input_reuse(self, steps):
        self.fixture(steps=steps)
        expected_out = self.reference()
        self.verify()
        expected = self.pool.temporal.clone()
        expected[:, 2] = self.ref_cache[:, 1, steps - 1]
        self.qkv.fill_(float("nan"))
        self.a.fill_(float("nan"))
        self.b.fill_(float("nan"))
        self.indices.fill_(123456)
        self.update()
        self.assert_bits(self.pool.temporal, expected)
        self.assert_bits(self.pool.dflash_sycl.output, expected_out)

    def test_independent_pool_and_layer_ownership_and_accounting(self):
        self.fixture()
        first = self.pool
        self.verify()
        before = first.temporal.clone()
        records_before = first.dflash_sycl.records.clone()
        ordinary = dataclasses.replace(first, dflash_sycl=None)
        expected = sum(t.numel() * t.element_size() for t in first.dflash_sycl.buffers)
        self.assertEqual(first.mem_usage_bytes() - ordinary.mem_usage_bytes(), expected)
        for i in range(self.layers):
            self.assertTrue(self.layer[i].dflash_sycl.matches_source(first.temporal[i]))
            self.assertEqual(
                self.layer[i].dflash_sycl.records.data_ptr(),
                first.dflash_sycl.records[i].data_ptr(),
            )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.dflash_sycl = None
        self.fixture()
        self.assertNotEqual(
            self.pool.dflash_sycl.records.data_ptr(),
            first.dflash_sycl.records.data_ptr(),
        )
        self.verify()
        self.update()
        self.assert_bits(first.temporal, before)
        self.assert_bits(first.dflash_sycl.records, records_before)

    def test_complete_thirty_layer_states_and_snapshot_fallback(self):
        self.check_complete_thirty_layers(4)

    def test_m8_complete_thirty_layer_states_and_snapshot_fallback(self):
        self.check_complete_thirty_layers(8)

    def check_complete_thirty_layers(self, steps):
        self.fixture(layers=30, steps=steps)
        for prefix in range(1, steps + 1):
            expected_out = self.reference()
            self.verify()
            expected = self.pool.temporal.clone()
            expected[:, 2] = self.ref_cache[:, 1, prefix - 1]
            self.last.fill_(prefix - 1)
            self.update()
            self.assert_bits(self.pool.temporal, expected)
            self.assert_bits(self.pool.dflash_sycl.output, expected_out)
        # A disabled/unallocated child still takes the original snapshot route.
        self.reference()
        self.ref_cache[:, 0].copy_(self.ref_cache[:, 1])
        expected = self.pool.temporal.clone()
        expected[:, 2] = self.ref_cache[:, 1, steps - 1]
        self.pool = dataclasses.replace(
            self.pool, dflash_sycl=None, intermediate_ssm=self.ref_cache
        )
        self.last.fill_(steps - 1)
        self.update()
        self.assert_bits(self.pool.temporal, expected)

    def test_mapping_and_tracking_reject_before_state_mutation(self):
        self.fixture()
        self.verify()
        before, conv_before = self.pool.temporal.clone(), self.pool.conv[0].clone()
        with self.assertRaises(ValueError):
            self.update(self.destination, self.last)
        self.assert_bits(self.pool.temporal, before)
        self.assert_bits(self.pool.conv[0], conv_before)
        wrong = dataclasses.replace(self.pool, temporal=self.pool.temporal.clone())
        with self.assertRaisesRegex(ValueError, "mapping changed"):
            native.commit_gdn(wrong, self.destination, self.last, None, None, None)
        wrong_layer = dataclasses.replace(
            self.layer[0], temporal=self.layer[1].temporal
        )
        with self.assertRaisesRegex(ValueError, "mapping changed"):
            native.verify_gdn(wrong_layer, **self.params(0))
        self.assert_bits(self.pool.temporal, before)

    def test_selected_verify_rejects_incompatible_contract(self):
        self.fixture()
        before = self.pool.temporal.clone()
        for changes in (
            {"steps": 3},
            {"steps": 8},  # A selected M8 verify cannot reuse an M4 pool.
            {"native_bf16_gdn": False},
            {"stable_rows": True},
            {"retrieve_parent_token": self.cu},
        ):
            with self.assertRaises(ValueError):
                native.verify_gdn(self.layer[0], **(self.params(0) | changes))
        self.assert_bits(self.pool.temporal, before)

    def test_m8_rejects_legacy_m4_library_before_allocating_scratch(self):
        self.fixture()
        with (
            patch.object(native, "validate_runtime"),
            patch.object(native, "load_gdn", return_value=SimpleNamespace()),
            self.assertRaisesRegex(RuntimeError, "lacks M8 support"),
        ):
            native.allocate_scratch(
                self.pool.temporal, 8, is_kda=False, replayssm=False, envelope=False
            )

    def test_backend_metadata_selects_matching_m4_and_m8_strided_qkv(self):
        from sglang.srt.layers.attention.linear import gdn_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        for steps in (4, 8):
            self.fixture(steps=steps)
            backend = SimpleNamespace(
                forward_metadata=SimpleNamespace(retrieve_parent_token=None),
                req_to_token_pool=SimpleNamespace(
                    mamba_pool=SimpleNamespace(mamba_cache=self.pool)
                ),
            )
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(draft_token_num=steps),
            )
            route = gdn_backend.GDNAttnBackend._init_target_verify_qkv_routing
            with patch.object(
                gdn_backend, "pp_spec_stable_rows_enabled", return_value=False
            ):
                route(backend, batch)
                self.assertTrue(backend._use_strided_target_verify_qkv)
                batch.spec_info.draft_token_num = 12 - steps
                with self.assertRaisesRegex(ValueError, "matching M4/M8"):
                    route(backend, batch)
                batch.forward_mode = ForwardMode.EXTEND
                route(backend, batch)
                self.assertFalse(backend._use_strided_target_verify_qkv)

    def test_runtime_master_and_topology_guards(self):
        from sglang.srt import runtime_context as rc

        base = dict(tp_size=1, pp_size=1)
        with (
            patch.object(
                rc, "get_parallel", return_value=SimpleNamespace(**base)
            ) as par,
            patch.object(
                rc,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm="DFLASH", speculative_num_draft_tokens=4
                ),
            ) as spec,
            patch.object(
                rc, "get_memory", return_value=SimpleNamespace(disable_radix_cache=True)
            ) as memory,
            patch.object(
                rc, "get_schedule", return_value=SimpleNamespace(max_running_requests=1)
            ) as sched,
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
        ):
            native.validate_runtime()
            spec.return_value = SimpleNamespace(
                speculative_algorithm="DFLASH", speculative_num_draft_tokens=8
            )
            native.validate_runtime()
            spec.return_value = SimpleNamespace(
                speculative_algorithm="DFLASH", speculative_num_draft_tokens=4
            )
            for obj, good, bad in [
                (par, SimpleNamespace(**base), SimpleNamespace(tp_size=2, pp_size=1)),
                (par, SimpleNamespace(**base), SimpleNamespace(tp_size=1, pp_size=2)),
                (
                    spec,
                    SimpleNamespace(
                        speculative_algorithm="DFLASH", speculative_num_draft_tokens=4
                    ),
                    SimpleNamespace(
                        speculative_algorithm="DFLASH", speculative_num_draft_tokens=5
                    ),
                ),
                (
                    memory,
                    SimpleNamespace(disable_radix_cache=True),
                    SimpleNamespace(disable_radix_cache=False),
                ),
                (
                    sched,
                    SimpleNamespace(max_running_requests=1),
                    SimpleNamespace(max_running_requests=2),
                ),
            ]:
                obj.return_value = bad
                with self.assertRaises(ValueError):
                    native.validate_runtime()
                obj.return_value = good
            with (
                envs.SGLANG_XPU_PAGED_DFLASH.override(False),
                self.assertRaises(ValueError),
            ):
                native.validate_runtime()

    def test_missing_library_is_explicit(self):
        native.load_gdn.cache_clear()
        try:
            with (
                envs.SGLANG_XPU_DFLASH_GDN_LIB.override(""),
                self.assertRaisesRegex(RuntimeError, "requires"),
            ):
                native.load_gdn()
        finally:
            native.load_gdn.cache_clear()


if __name__ == "__main__":
    unittest.main()
