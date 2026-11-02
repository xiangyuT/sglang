"""Compare native XPU expert paging with a fully resident layer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.paged_experts import PagedExpertsMoEMethod
from sglang.srt.layers.moe.paged_experts.formats import (
    Fp8BlockFormat,
    UnquantizedFormat,
)
from sglang.srt.layers.moe.paged_experts.runners import XpuContract
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="xpu-unit-test")


class _NativeMethod:
    def __init__(self, fp8):
        self.fp8 = fp8

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **kwargs,
    ):
        dtype = torch.float8_e4m3fn if self.fp8 else params_dtype
        inter = intermediate_size_per_partition
        for name, shape in (
            ("w13_weight", (num_experts, 2 * inter, hidden_size)),
            ("w2_weight", (num_experts, hidden_size, inter)),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype, device="xpu"), requires_grad=False
                ),
            )
            if self.fp8:
                scale_shape = (
                    shape[0],
                    (shape[1] + 127) // 128,
                    (shape[2] + 127) // 128,
                )
                layer.register_parameter(
                    name + "_scale_inv",
                    torch.nn.Parameter(
                        torch.empty(scale_shape, device="xpu"), requires_grad=False
                    ),
                )

    def create_moe_runner(self, layer, moe_runner_config):
        self.runner = SimpleNamespace(config=moe_runner_config)

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, dispatch_output):
        from sgl_kernel import fused_experts

        topk = dispatch_output.topk_output
        out = fused_experts(
            dispatch_output.hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            topk.topk_weights,
            topk.topk_ids,
            activation="silu",
            use_fp8_w8a8=self.fp8,
            w1_scale=getattr(layer, "w13_weight_scale_inv", None),
            w2_scale=getattr(layer, "w2_weight_scale_inv", None),
            routed_scaling_factor=self.runner.config.routed_scaling_factor,
        )
        return SimpleNamespace(hidden_states=out)


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuPagedExperts(CustomTestCase):
    def _setup(
        self,
        fp8,
        hidden,
        inter,
        top_k,
        graphs=False,
        scale_override=None,
        experts=16,
        slots=8,
    ):
        torch.manual_seed(7)
        fmt = Fp8BlockFormat() if fp8 else UnquantizedFormat()
        fmt.contract = XpuContract()
        base = _NativeMethod(fp8)
        method = PagedExpertsMoEMethod(base, fmt, experts, slots, graphs=graphs)
        layer = torch.nn.Module()
        method.create_weights(layer, experts, hidden, inter, torch.bfloat16)
        method.create_moe_runner(
            layer,
            MoeRunnerConfig(
                num_experts=experts,
                num_local_experts=experts,
                top_k=top_k,
                routed_scaling_factor=2.5,
            ),
        )
        for name, weights in method.store.host.items():
            if name.endswith("_scale_inv"):
                weights.uniform_(0.1, 0.9)
            else:
                weights.copy_((torch.randn(weights.shape) * 0.02).to(weights.dtype))
        if scale_override is not None:
            # This expert starts outside the resident cache.
            method.store.host["w2_weight_scale_inv"][-1].fill_(scale_override)
        method.process_weights_after_loading(layer)
        full = SimpleNamespace(
            **{name: value.to("xpu") for name, value in method.store.host.items()}
        )
        reference = _NativeMethod(fp8)
        reference.runner = SimpleNamespace(
            config=MoeRunnerConfig(routed_scaling_factor=2.5)
        )
        return method, layer, full, reference

    def _check(self, fp8, hidden=256, inter=128, top_k=4):
        experts = 16
        method, layer, full, reference = self._setup(fp8, hidden, inter, top_k)
        for tokens, padded in (
            (1, False),
            (3, False),
            (17, False),
            (3, True),
            (1, True),
        ):
            for step in range(4):
                ids = torch.stack(
                    [torch.randperm(experts)[:top_k] for _ in range(tokens)]
                ).to("xpu")
                if padded:
                    ids[0] = -1
                weights = torch.rand(tokens, top_k, device="xpu")
                weights /= weights.sum(-1, keepdim=True)
                d = StandardDispatchOutput(
                    hidden_states=torch.randn(
                        tokens, hidden, device="xpu", dtype=torch.bfloat16
                    ),
                    hidden_states_scale=None,
                    topk_output=StandardTopKOutput(weights, ids, None),
                )
                # Native kernels expect valid ids; weight-zero padding is equivalent.
                ref_d = d._replace(
                    topk_output=d.topk_output._replace(
                        topk_ids=ids.clamp(min=0),
                        topk_weights=weights.masked_fill(ids < 0, 0),
                    )
                )
                expected = reference.apply(full, ref_d).hidden_states
                actual = method.apply(layer, d).hidden_states
                torch.xpu.synchronize()
                with self.subTest(fp8=fp8, tokens=tokens, padded=padded, step=step):
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    if padded:
                        self.assertTrue(
                            torch.equal(actual[0], torch.zeros_like(actual[0]))
                        )

    def test_nonresident_extreme_scales_keep_original_decode(self):
        for unsafe in (float("inf"), 2.0**119, 2.0**-119):
            method, layer, full, reference = self._setup(
                True, 2048, 512, 8, scale_override=unsafe
            )
            self.assertTrue(method.device_executor.use_expert_gemv)
            self.assertFalse(method.device_executor.use_scaled_fp8)
            ids = torch.arange(8, device="xpu", dtype=torch.int32).view(1, 8)
            weights = torch.full((1, 8), 1 / 8, device="xpu")
            dispatch = StandardDispatchOutput(
                hidden_states=torch.randn(1, 2048, device="xpu", dtype=torch.bfloat16),
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(weights, ids, None),
            )
            with self.subTest(nonresident_scale=unsafe):
                actual = method.apply(layer, dispatch).hidden_states
                expected = reference.apply(full, dispatch).hidden_states
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)

    def test_bf16(self):
        self._check(fp8=False)

    def test_block_fp8(self):
        self._check(fp8=True)

    def test_qwen36_bf16(self):
        self._check(fp8=False, hidden=2048, inter=512, top_k=8)

    def test_qwen36_block_fp8(self):
        self._check(fp8=True, hidden=2048, inter=512, top_k=8)

    def test_qwen36_verify_block_dynamic_paging_graph(self):
        self._check_verify_block_dynamic_paging_graph()

    @unittest.skipUnless(
        envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get(),
        "explicitly built compact MoE library required",
    )
    def test_compact_verify_block_dynamic_paging_graph(self):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_COMPACT_MOE.override(True),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm="DFLASH"),
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=SimpleNamespace(is_triton=lambda: False),
            ),
        ):
            for tokens in (3, 5, 6, 7, 8):
                self._check_verify_block_dynamic_paging_graph(
                    compact=True, tokens=tokens
                )

    @unittest.skipUnless(
        envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get(),
        "explicitly built native paged MoE library required",
    )
    def test_compact_native_paged_verify_graph(self):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_COMPACT_MOE.override(True),
            envs.SGLANG_XPU_DFLASH_PAGED_MOE.override(True),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm="DFLASH"),
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=SimpleNamespace(is_triton=lambda: False),
            ),
        ):
            for tokens in range(2, 9):
                self._check_verify_block_dynamic_paging_graph(
                    compact=True, paged=True, tokens=tokens
                )

    def _check_verify_block_dynamic_paging_graph(
        self, compact=False, tokens=8, paged=False
    ):
        method, layer, full, reference = self._setup(
            True, 2048, 512, 8, graphs=True, experts=256, slots=192
        )
        self.assertEqual(method.device_executor.use_compact_moe, compact)
        self.assertEqual(method.device_executor.use_paged_compact, paged)
        # XpuContract applies the layer factor after the native BF16 top-k sum.
        # Match that boundary rather than the native op's internal scaled sum.
        reference.runner.config.routed_scaling_factor = 1.0
        routes = tokens * 8
        x = torch.randn(tokens, 2048, device="xpu", dtype=torch.bfloat16)
        ids = torch.arange(routes, device="xpu", dtype=torch.int32).view(tokens, 8)
        weights = torch.full((tokens, 8), 1 / 8, device="xpu")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        method.apply(layer, dispatch)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = method.apply(layer, dispatch).hidden_states
        # Include 64 distinct misses, repeated routes, and padded verify rows.
        offsets = (192, 64, 128, 0, 224, 32) + ((255, -1) if paged else ())
        for step, offset in enumerate(offsets):
            route = (torch.arange(routes, device="xpu", dtype=ids.dtype) + offset) % 256
            ids.copy_(route.view(tokens, 8))
            if step == 4:
                ids[0].fill_(255)
                ids[-1].fill_(-1)
            if step == 6:
                # More than 16 duplicate routes exercises a second M tile;
                # only the first tile may fill each raw cache word.
                missing = (
                    (method.device_executor.residency.expert_slot < 0)
                    .nonzero()[0]
                    .item()
                )
                ids.fill_(missing)
            elif step == 7:
                ids.fill_(-1)
            x.normal_()
            graph.replay()
            clean = dispatch._replace(
                topk_output=dispatch.topk_output._replace(
                    topk_ids=ids.clamp(min=0),
                    topk_weights=weights.masked_fill(ids < 0, 0),
                )
            )
            expected = reference.apply(full, clean).hidden_states * 2.5
            with self.subTest(tokens=tokens, step=step):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                resident = method.device_executor.residency.slot_expert.cpu().long()
                for name, host in method.store.host.items():
                    torch.testing.assert_close(
                        getattr(layer, name).view(torch.uint8).cpu(),
                        host[resident].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )

    def test_qwen36_page_fill_matches_resident_gemv_exactly(self):
        from sglang.kernels.ops.moe.xpu_block_fp8_experts import xpu_block_fp8_experts

        method, layer, full, _ = self._setup(True, 2048, 512, 8, graphs=True)
        self.assertTrue(method.device_executor.use_scaled_fp8)
        ids = torch.arange(8, device="xpu", dtype=torch.int32).view(1, 8)
        weights = torch.full((1, 8), 1 / 8, device="xpu")
        x = torch.randn(1, 2048, dtype=torch.bfloat16, device="xpu")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        method.apply(layer, dispatch)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = method.apply(layer, dispatch).hidden_states
        cases = (
            list(range(8, 16)),
            [-1, 0, 8, 0, -1, 8, 0, 8],
            list(range(8)),
            [-1] * 8,
            [15] * 8,
            [15, 14, 15, 13, 12, 14, 11, 10],
        )
        for step, route in enumerate(cases):
            ids.copy_(torch.tensor([route], device="xpu", dtype=ids.dtype))
            x.normal_()
            if step == 4:
                x.zero_()
            weights.uniform_()
            weights.div_(weights.sum(-1, keepdim=True))
            graph.replay()
            expected = (
                xpu_block_fp8_experts(
                    x,
                    full.w13_weight,
                    full.w2_weight,
                    full.w13_weight_scale_inv,
                    full.w2_weight_scale_inv,
                    ids.clamp(min=0),
                    weights.masked_fill(ids < 0, 0),
                    scaled_fp8=True,
                )
                * 2.5
            )
            with self.subTest(step=step):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                resident = method.device_executor.residency.slot_expert.cpu().long()
                for name, host in method.store.host.items():
                    torch.testing.assert_close(
                        getattr(layer, name).detach().cpu().view(torch.uint8),
                        host[resident].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )

    def _check_graph(self, hidden, inter, top_k):
        method, layer, full, reference = self._setup(
            True, hidden, inter, top_k, graphs=True
        )
        ids = torch.arange(top_k, device="xpu").view(1, top_k)
        weights = torch.full((1, top_k), 1 / top_k, device="xpu")
        d = StandardDispatchOutput(
            hidden_states=torch.randn(1, hidden, device="xpu", dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        for _ in range(2):
            method.apply(layer, d)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = method.apply(layer, d).hidden_states

        for step in range(12):
            # Change routes and inputs after capture, including hits, misses,
            # duplicates and all-padding batches. A new multi-wave prefill must
            # also hand its updated residency back to the captured decode path.
            ids.copy_(torch.randint(0, 16, ids.shape, device="xpu"))
            if step % 4 == 0:
                ids.fill_(-1)
            elif step % 4 == 1:
                ids[0, 0] = -1
            d.hidden_states.normal_()
            weights.uniform_()
            weights.div_(weights.sum(-1, keepdim=True))
            graph.replay()
            ref_d = d._replace(
                topk_output=d.topk_output._replace(
                    topk_ids=ids.clamp(min=0),
                    topk_weights=weights.masked_fill(ids < 0, 0),
                )
            )
            expected = reference.apply(full, ref_d).hidden_states
            torch.xpu.synchronize()
            with self.subTest(hidden=hidden, replay=step):
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
                state = method.device_executor.residency
                resident = state.slot_expert.cpu().long()
                self.assertEqual(len(set(resident.tolist())), len(resident))
                for name, host in method.store.host.items():
                    self.assertTrue(
                        torch.equal(
                            getattr(layer, name).detach().cpu().view(torch.uint8),
                            host[resident].view(torch.uint8),
                        ),
                        name,
                    )
            if step % 4 == 3:
                prefill_ids = torch.randint(0, 16, (17, top_k), device="xpu")
                prefill = StandardDispatchOutput(
                    hidden_states=torch.randn(
                        17, hidden, device="xpu", dtype=torch.bfloat16
                    ),
                    hidden_states_scale=None,
                    topk_output=StandardTopKOutput(
                        torch.full((17, top_k), 1 / top_k, device="xpu"),
                        prefill_ids,
                        None,
                    ),
                )
                torch.testing.assert_close(
                    method.apply(layer, prefill).hidden_states,
                    reference.apply(full, prefill).hidden_states,
                    rtol=0.03,
                    atol=0.002,
                )

    def test_block_fp8_graph_replay(self):
        self._check_graph(hidden=256, inter=128, top_k=4)

    def test_qwen36_block_fp8_graph_replay(self):
        self._check_graph(hidden=2048, inter=512, top_k=8)


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuPagerPlanner(CustomTestCase):
    def test_hit_miss_lru_and_graph_replay(self):
        from sglang.kernels.ops.moe.paged_experts_xpu import decide
        from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency

        generator = torch.Generator().manual_seed(42)
        routes = [torch.randint(0, 192, (8,), generator=generator) for _ in range(16)]
        routes += [torch.randint(0, 256, (8,), generator=generator) for _ in range(48)]
        routes += [torch.tensor([-1, 256, 5, 5, -1, 255, 0, 255]), torch.full((8,), -1)]
        for captured in (False, True):
            state = DeviceResidency(256, 192, "xpu")
            ids = torch.zeros(8, dtype=torch.int32, device="xpu")
            weights = torch.arange(8, dtype=torch.float32, device="xpu") / 8
            mapped = torch.empty_like(ids)
            mapped_weights = torch.empty_like(weights)
            if captured:
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    decide(ids, weights, state, mapped, mapped_weights)
            slots = list(range(192))
            age = [0] * 192
            for step, route in enumerate(routes, 1):
                logical = route.tolist()
                wanted = {e for e in logical if 0 <= e < 256}
                for slot, expert in enumerate(slots):
                    if expert in wanted:
                        age[slot] = step
                loads = []
                for expert in logical:
                    if 0 <= expert < 256 and expert not in slots:
                        victim = min(
                            (slot for slot in range(192) if age[slot] != step),
                            key=lambda slot: (age[slot], slot),
                        )
                        slots[victim] = expert
                        age[victim] = step
                        loads.append((expert, victim))
                ids.copy_(route)
                if captured:
                    graph.replay()
                else:
                    decide(ids, weights, state, mapped, mapped_weights)
                with self.subTest(captured=captured, step=step):
                    self.assertEqual(state.slot_expert.tolist(), slots)
                    self.assertEqual(state.slot_lastuse.tolist(), age)
                    self.assertEqual(state.step.item(), step)
                    inverse = [-1] * 256
                    for slot, expert in enumerate(slots):
                        inverse[expert] = slot
                    self.assertEqual(state.expert_slot.tolist(), inverse)
                    self.assertEqual(state.count.item(), len(loads))
                    self.assertEqual(
                        state.src[: len(loads)].tolist(), [e for e, _ in loads]
                    )
                    self.assertEqual(
                        state.dst[: len(loads)].tolist(), [v for _, v in loads]
                    )
                    self.assertEqual(
                        mapped.tolist(),
                        [inverse[e] if 0 <= e < 256 else 0 for e in logical],
                    )
                    self.assertEqual(
                        mapped_weights.tolist(),
                        [i / 8 if 0 <= e < 256 else 0 for i, e in enumerate(logical)],
                    )


if __name__ == "__main__":
    unittest.main()
