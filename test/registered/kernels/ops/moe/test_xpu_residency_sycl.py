"""Native expert residency against scalar LRU and captured live inputs."""

import os
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.moe.paged_experts_xpu import decide
from sglang.srt.environ import envs
from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")
WINDOW = int(os.environ.get("SGLANG_TEST_DFLASH_NATIVE_WINDOW", "4"))


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "requires B70 and an explicitly built native library",
)
class TestSyclResidency(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 planner")

    def run_decide(
        self, ids, weights, state, mapped, outweights, child=True, master=True
    ):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_SYCL_PLANNER.override(child),
        ):
            decide(ids, weights, state, mapped, outweights)

    def test_scalar_lru_and_graph_replay(self):
        generator = torch.Generator().manual_seed(42)
        for captured in (False, True):
            for tied in (False, True):
                state = DeviceResidency(256, 192, "xpu")
                slots = torch.randperm(256, generator=generator)[:192].tolist()
                ages = (
                    [0] * 192
                    if tied
                    else torch.randint(0, 128, (192,), generator=generator).tolist()
                )
                inverse = [-1] * 256
                for slot, expert in enumerate(slots):
                    inverse[expert] = slot
                state.step.fill_(128)
                state.slot_expert.copy_(torch.tensor(slots))
                state.slot_lastuse.copy_(torch.tensor(ages))
                state.expert_slot.copy_(torch.tensor(inverse))
                ids = torch.zeros(WINDOW, 8, dtype=torch.int32, device="xpu")
                bits = torch.randint(
                    -2147483648,
                    2147483647,
                    (WINDOW, 8),
                    generator=generator,
                    dtype=torch.int32,
                )
                weights = bits.to("xpu").view(torch.float32)
                mapped = torch.empty(WINDOW * 8, device="xpu", dtype=torch.int32)
                outweights = torch.empty(WINDOW * 8, device="xpu")
                sources, destinations = [0] * 192, [0] * 192
                if captured:
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        self.run_decide(ids, weights, state, mapped, outweights)
                routes = [
                    torch.randint(-4, 260, (WINDOW * 8,), generator=generator)
                    for _ in range(32)
                ]
                routes[0] = torch.tensor([slots[i % 20] for i in range(WINDOW * 8)])
                routes[1] = torch.tensor(
                    [e for e in range(256) if e not in slots][: WINDOW * 8]
                )
                routes[2] = torch.full((WINDOW * 8,), -1)
                routes[3] = torch.tensor([-1, 256, 5, 5, -1, 255, 0, 255] * WINDOW)
                for step, route in enumerate(routes, 129):
                    logical = route.tolist()
                    wanted = {e for e in logical if 0 <= e < 256}
                    for slot, expert in enumerate(slots):
                        if expert in wanted:
                            ages[slot] = step
                    loads = []
                    for expert in logical:
                        if 0 <= expert < 256 and expert not in slots:
                            victim = min(
                                (s for s in range(192) if ages[s] != step),
                                key=lambda s: (ages[s], s),
                            )
                            slots[victim] = expert
                            ages[victim] = step
                            loads.append((expert, victim))
                    inverse = [-1] * 256
                    for slot, expert in enumerate(slots):
                        inverse[expert] = slot
                    for i, (expert, slot) in enumerate(loads):
                        sources[i], destinations[i] = expert, slot
                    ids.copy_(route.view(WINDOW, 8))
                    if captured:
                        graph.replay()
                    else:
                        self.run_decide(ids, weights, state, mapped, outweights)
                    self.assertEqual(state.step.item(), step)
                    self.assertEqual(state.slot_expert.tolist(), slots)
                    self.assertEqual(state.expert_slot.tolist(), inverse)
                    self.assertEqual(state.slot_lastuse.tolist(), ages)
                    self.assertEqual(state.count.item(), len(loads))
                    self.assertEqual(state.src.tolist(), sources)
                    self.assertEqual(state.dst.tolist(), destinations)
                    self.assertEqual(
                        mapped.tolist(),
                        [inverse[e] if 0 <= e < 256 else 0 for e in logical],
                    )
                    self.assertEqual(
                        outweights.view(torch.int32).tolist(),
                        [
                            w if 0 <= e < 256 else 0
                            for e, w in zip(logical, bits.flatten().tolist())
                        ],
                    )
                if captured:
                    del graph

    def test_master_child_dtype_shape_and_capacity_fallbacks(self):
        target = "sglang.kernels.ops.moe.xpu_residency_sycl.decide_sycl"
        with patch(target, side_effect=AssertionError("must use retained planner")):
            for master, child, rows, dtype, experts, capacity in (
                (False, True, 4, torch.int32, 256, 192),
                (True, False, 4, torch.int32, 256, 192),
                (True, True, 1, torch.int32, 256, 192),
                (True, True, 4, torch.int64, 256, 192),
                (True, True, 4, torch.int32, 512, 192),
                (True, True, 4, torch.int32, 256, 191),
                *(([(True, True, 4, torch.int32, 256, 192)]) if WINDOW == 8 else []),
            ):
                ids = torch.zeros(rows, 8, device="xpu", dtype=dtype)
                weights = torch.ones(rows, 8, device="xpu")
                state = DeviceResidency(experts, capacity, "xpu")
                mapped = torch.empty(rows * 8, device="xpu", dtype=torch.int32)
                outweights = torch.empty(rows * 8, device="xpu")
                self.run_decide(ids, weights, state, mapped, outweights, child, master)
        with patch(target) as native:
            ids = torch.zeros(WINDOW, 8, device="xpu", dtype=torch.int32)
            self.run_decide(
                ids,
                torch.ones(WINDOW, 8, device="xpu"),
                DeviceResidency(256, 192, "xpu"),
                torch.empty(WINDOW * 8, device="xpu", dtype=torch.int32),
                torch.empty(WINDOW * 8, device="xpu"),
            )
            native.assert_called_once()
