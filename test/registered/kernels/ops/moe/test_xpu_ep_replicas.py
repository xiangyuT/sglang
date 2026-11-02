"""Replicated EP routes have one owner and preserve checkpoint expert identity."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.moe.xpu_ep_replica_sycl import (
    _local_ids_by_rank,
    replica_local_ids,
)
from sglang.kernels.ops.moe.xpu_tp_expert_down_sycl import can_use_tp_expert_down
from sglang.kernels.ops.moe.xpu_tp_expert_gu_sycl import tp_expert_gu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="xpu-unit-test")


class TestReplicaLoaderLayout(unittest.TestCase):
    def test_inverse_indices_match_physical_checkpoint_slots(self):
        physical = torch.arange(384) % 256
        table = _local_ids_by_rank(physical)
        self.assertEqual(int((table >= 0).sum()), 384)
        self.assertEqual(int((table >= 0).all(dim=1).sum()), 128)
        for rank in range(2):
            logical = torch.where(table[:, rank] >= 0)[0]
            reconstructed = physical[rank * 192 + table[logical, rank].long()]
            self.assertTrue(torch.equal(reconstructed, logical))
        with self.assertRaises(ValueError):
            _local_ids_by_rank(torch.zeros(384, dtype=torch.int64))


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU required")
class TestXpuEpReplicas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 replica dispatch")
        if not os.environ.get("SGLANG_XPU_TP_EXPERT_GU_SYCL_LIB"):
            raise unittest.SkipTest("Build and set the GU library with replica mapping")

    def _assert_ownership(self, ids_cpu, outputs, cpu_table):
        valid = ids_cpu >= 0
        local = [output.cpu() for output in outputs]
        coverage = (local[0] >= 0).int() + (local[1] >= 0).int()
        self.assertTrue(torch.equal(coverage, valid.int()))
        for rank in range(2):
            owned = local[rank] >= 0
            physical = rank * 192 + local[rank][owned]
            self.assertTrue(torch.equal(physical % 256, ids_cpu[owned]))
        clipped = ids_cpu.clamp_min(0).long()
        available = cpu_table[clipped] >= 0
        forced0 = (available[..., 0] & ~available[..., 1] & valid).sum(-1)
        forced1 = (available[..., 1] & ~available[..., 0] & valid).sum(-1)
        lower_bound = torch.maximum(
            (valid.sum(-1) + 1) // 2, torch.maximum(forced0, forced1)
        )
        actual = torch.maximum((local[0] >= 0).sum(-1), (local[1] >= 0).sum(-1))
        self.assertTrue(torch.equal(actual, lower_bound))

    def test_single_and_prefill_batches_eager_and_graph(self):
        generator = torch.Generator().manual_seed(42)
        cpu_table = _local_ids_by_rank(torch.arange(384) % 256)
        edge_cases = torch.tensor(
            [
                list(range(8)),
                list(range(128, 136)),
                list(range(192, 200)),
                [0, 1, 127, 128, 129, 191, 192, 255],
                [128, 129, 130, 131, 132, 0, 1, 2],
                [0, 0, 0, -1, -4, -128, 192, 255],
                [-1] * 8,
            ],
            dtype=torch.int32,
        )
        with torch.inference_mode():
            table = cpu_table.to("xpu")
            for size in (1, 512):
                host = torch.stack(
                    [torch.randperm(256, generator=generator)[:8] for _ in range(size)]
                ).int()
                ids = host.to("xpu")
                for rank in range(2):
                    replica_local_ids(ids, table, rank)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    outputs = [replica_local_ids(ids, table, rank) for rank in range(2)]
                for row in edge_cases:
                    host[0] = row
                    ids.copy_(host)
                    eager = [replica_local_ids(ids, table, rank) for rank in range(2)]
                    graph.replay()
                    self._assert_ownership(host, outputs, cpu_table)
                    for left, right in zip(outputs, eager):
                        self.assertTrue(torch.equal(left, right))

    def test_replicated_weights_sum_to_analytic_logical_experts(self):
        if not os.environ.get("SGLANG_XPU_TP_EXPERT_DOWN_SYCL_LIB"):
            self.skipTest("Build and set the down library")
        selected = [0, 1, 127, 128, 129, 191, 192, 255]
        config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            no_combine=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        parallel = SimpleNamespace(
            tp_size=2,
            moe_ep_size=2,
            pp_size=1,
            attn_cp_size=1,
            attn_dp_enabled=False,
        )
        with (
            torch.inference_mode(),
            patch.dict(os.environ, {"SGLANG_XPU_EP_REPLICA_BALANCE": "1"}),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            table = _local_ids_by_rank(torch.arange(384) % 256).to("xpu")
            ids = torch.tensor([selected], dtype=torch.int32, device="xpu")
            x = torch.ones((1, 2048), dtype=torch.bfloat16, device="xpu")
            contributions = []
            for rank in range(2):
                layer = SimpleNamespace(
                    w13_weight=torch.zeros(
                        (192, 1024, 2048), dtype=torch.uint8, device="xpu"
                    ).view(torch.float8_e4m3fn),
                    w2_weight=torch.zeros(
                        (192, 2048, 512), dtype=torch.uint8, device="xpu"
                    ).view(torch.float8_e4m3fn),
                    w13_weight_scale_inv=torch.full((192, 8, 16), 0.125, device="xpu"),
                    w2_weight_scale_inv=torch.full((192, 16, 4), 0.125, device="xpu"),
                )
                for slot in range(192):
                    logical = (rank * 192 + slot) % 256
                    if logical in selected:
                        layer.w13_weight[slot].view(torch.uint8).fill_(
                            (0x28, 0x30, 0x38)[logical % 3]
                        )
                        layer.w2_weight[slot].view(torch.uint8).fill_(0x28)
                local_ids = replica_local_ids(ids, table, rank)
                weights = torch.where(local_ids >= 0, 0.125, float("nan"))
                self.assertTrue(
                    can_use_tp_expert_down(layer, config, x, local_ids, weights)
                )
                contributions.append(tp_expert_gu(layer, x, local_ids, weights))
                if os.environ.get("SGLANG_XPU_EP_REPLICA_FUSED_GU") == "1":
                    fused = tp_expert_gu(
                        layer, x, ids, weights, replica_routing=(table, rank)
                    )
                    self.assertTrue(torch.equal(fused, contributions[-1]))
                    self.assertTrue(
                        torch.equal(layer._xpu_replica_gu_local_ids, local_ids)
                    )
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        fused = tp_expert_gu(
                            layer, x, ids, weights, replica_routing=(table, rank)
                        )
                    for routes in (
                        selected[::-1],
                        [128, 129, 191, 128, 129, 0, 1, -1],
                        [192, 255, 192, 255, 192, 0, 1, -1],
                        [-1] * 8,
                        selected,
                    ):
                        ids.copy_(torch.tensor([routes], device="xpu"))
                        weights.copy_(torch.where(ids >= 0, 0.125, float("nan")))
                        reference_ids = replica_local_ids(ids, table, rank)
                        reference = tp_expert_gu(layer, x, reference_ids, weights)
                        graph.replay()
                        self.assertTrue(torch.equal(fused, reference))
                        self.assertTrue(
                            torch.equal(layer._xpu_replica_gu_local_ids, reference_ids)
                        )
            expected = sum(
                16 * (64 * 2 ** (logical % 3)) ** 2 / 8 for logical in selected
            )
            combined = (contributions[0].float() + contributions[1].float()).bfloat16()
            self.assertTrue(torch.equal(combined, torch.full_like(combined, expected)))


if __name__ == "__main__":
    unittest.main()
