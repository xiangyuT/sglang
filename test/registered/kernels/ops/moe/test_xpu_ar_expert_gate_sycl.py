"""Native AR equivalence, changing miss plans under graph, and scope guards."""

import ctypes.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.xpu_ar_expert_gate_sycl import (
    _load_library,
    can_use_ar_expert_gate,
    xpu_ar_expert_gate,
)
from sglang.kernels.ops.moe.xpu_paged_fp8_experts import xpu_paged_block_fp8_experts
from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=60, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuArExpertGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 ordinary AR expert gate")
        if not os.environ.get("SGLANG_XPU_AR_EXPERT_GATE_SYCL_LIB"):
            raise unittest.SkipTest("Build and set SGLANG_XPU_AR_EXPERT_GATE_SYCL_LIB")

    def fixture(self, seed=42):
        torch.manual_seed(seed)
        host = [
            torch.randn((16, 1024, 2048)).to(torch.float8_e4m3fn).pin_memory(),
            torch.randn((16, 2048, 512)).to(torch.float8_e4m3fn).pin_memory(),
            torch.full((16, 8, 16), 0.02, pin_memory=True),
            torch.full((16, 16, 4), 0.02, pin_memory=True),
        ]
        hp = torch.tensor(
            [t.data_ptr() for t in host], dtype=torch.uint64, device="xpu"
        )
        caches = []
        for _ in range(2):
            cache = [
                torch.zeros((192, 1024, 2048), dtype=torch.float8_e4m3fn, device="xpu"),
                torch.zeros((192, 2048, 512), dtype=torch.float8_e4m3fn, device="xpu"),
                torch.full((192, 8, 16), 0.02, device="xpu"),
                torch.full((192, 16, 4), 0.02, device="xpu"),
            ]
            for target, source in zip(cache, host):
                target[:16].copy_(source)
            gp = torch.tensor(
                [t.data_ptr() for t in cache], dtype=torch.uint64, device="xpu"
            )
            caches.append((cache, gp))
        x = torch.randn((1, 2048), dtype=torch.bfloat16, device="xpu") * 0.25
        ids = torch.arange(8, dtype=torch.int32, device="xpu").view(1, 8)
        weights = torch.full((1, 8), 0.125, device="xpu")
        state = DeviceResidency(256, 192, x.device)
        state.src[:8].copy_(torch.arange(8, dtype=torch.int32, device="xpu"))
        state.dst[:8].copy_(torch.arange(8, dtype=torch.int32, device="xpu"))
        return host, hp, caches, x, ids, weights, state

    def invoke(self, fixture, side, enabled):
        _, hp, caches, x, ids, weights, state = fixture
        cache, gp = caches[side]
        with (
            patch.dict(
                os.environ, {"SGLANG_XPU_AR_EXPERT_GATE_SYCL": str(int(enabled))}
            ),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            return xpu_paged_block_fp8_experts(x, *cache, ids, weights, hp, gp, state)

    def assert_bits(self, left, right):
        self.assertTrue(torch.equal(left.view(torch.int16), right.view(torch.int16)))

    def test_native_cache_fill_and_nonlinear_boundaries(self):
        for seed in (19, 42, 71):
            f = self.fixture(seed)
            host, _, caches, x, ids, weights, state = f
            for step in range(9):
                state.count.fill_(step)
                x.normal_().mul_(0.25)
                if step == 4:
                    ids[0, -2:] = ids[0, :2]
                else:
                    ids.copy_(
                        torch.arange(8, dtype=torch.int32, device="xpu").view(1, 8)
                    )
                weights.fill_(0 if step == 8 else 0.125)
                scale = (0.02, 0.125, 1.0)[step % 3]
                for tensor in host[2:]:
                    tensor.fill_(scale)
                for cache, _ in caches:
                    for tensor in cache[2:]:
                        tensor.fill_(scale)
                reference = self.invoke(f, 0, False)
                candidate = self.invoke(f, 1, True)
                self.assert_bits(reference, candidate)
            for left, right in zip(caches[0][0], caches[1][0]):
                self.assertTrue(
                    torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                )

    def test_changing_graph_plans_inputs_and_output_poison(self):
        f = self.fixture()
        host, _, caches, x, ids, weights, state = f
        self.invoke(f, 0, False)
        self.invoke(f, 1, True)
        torch.xpu.synchronize()
        graphs, outputs = [], []
        for side in (0, 1):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output = self.invoke(f, side, bool(side))
            graphs.append(graph)
            outputs.append(output)
        for step in range(24):
            state.count.fill_(step % 9)
            state.src[:8].copy_(
                (torch.arange(8, dtype=torch.int32, device="xpu") + step) % 16
            )
            x.normal_().mul_(0.25)
            weights.uniform_().div_(weights.sum())
            ids.copy_(torch.arange(8, dtype=torch.int32, device="xpu").view(1, 8))
            if step % 4 == 0:
                ids[0, -2:] = ids[0, :2]
            outputs[0].fill_(float("nan"))
            outputs[1].fill_(float("inf"))
            for graph in graphs:
                graph.replay()
            self.assert_bits(*outputs)
        for left, right in zip(caches[0][0], caches[1][0]):
            self.assertTrue(
                torch.equal(left.view(torch.uint8), right.view(torch.uint8))
            )

    def test_layout_dtype_autograd_and_runtime_exclusions(self):
        f = self.fixture()
        _, hp, caches, x, ids, weights, state = f
        w1, _, s1, _ = caches[0][0]
        a = torch.empty((8, 512), dtype=x.dtype, device="xpu")
        self.assertEqual(state.src.shape, (192,))
        self.assertTrue(can_use_ar_expert_gate(x, w1, s1, ids, a, hp, state))
        short_plan = SimpleNamespace(
            src=state.src[:7], dst=state.dst[:7], count=state.count
        )
        self.assertFalse(can_use_ar_expert_gate(x, w1, s1, ids, a, hp, short_plan))
        with self.assertRaisesRegex(ValueError, "Unsupported B70"):
            xpu_ar_expert_gate(x.float(), w1, s1, ids, a, hp, state)
        for args in (
            (x.float(), w1, s1, ids, a, hp, state),
            (x, w1[:128], s1[:128], ids, a, hp, state),
            (x, w1, s1, ids.long(), a, hp, state),
            (x, w1, s1, ids, a, hp.cpu(), state),
        ):
            self.assertFalse(can_use_ar_expert_gate(*args))
        with torch.enable_grad():
            self.assertFalse(
                can_use_ar_expert_gate(
                    x.detach().requires_grad_(), w1, s1, ids, a, hp, state
                )
            )
        reference = self.invoke(f, 0, False)
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_XPU_AR_EXPERT_GATE_SYCL": "1",
                    "SGLANG_XPU_AR_EXPERT_GATE_SYCL_LIB": "nonexistent-library",
                },
            ),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm="DFLASH"),
            ),
        ):
            cache, gp = caches[1]
            candidate = xpu_paged_block_fp8_experts(
                x, *cache, ids, weights, hp, gp, state
            )
        self.assert_bits(reference, candidate)

    def test_abi_and_capture_warmup_errors(self):
        with self.assertRaisesRegex(RuntimeError, "Set SGLANG"):
            _load_library("")
        with patch.object(torch.xpu, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                _load_library("unwarmed-library")
        libc = ctypes.util.find_library("c")
        if libc:
            with self.assertRaisesRegex(RuntimeError, "lacks AR expert gate ABI"):
                _load_library(libc)


if __name__ == "__main__":
    unittest.main()
