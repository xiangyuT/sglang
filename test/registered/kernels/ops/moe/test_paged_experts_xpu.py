"""Compare native XPU expert paging with a fully resident layer."""

import unittest
from types import SimpleNamespace

import torch

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
    def _setup(self, fp8, hidden, inter, top_k, graphs=False):
        torch.manual_seed(7)
        experts, slots = 16, 8
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

    def test_bf16(self):
        self._check(fp8=False)

    def test_block_fp8(self):
        self._check(fp8=True)

    def test_qwen36_bf16(self):
        self._check(fp8=False, hidden=2048, inter=512, top_k=8)

    def test_qwen36_block_fp8(self):
        self._check(fp8=True, hidden=2048, inter=512, top_k=8)

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


if __name__ == "__main__":
    unittest.main()
