"""CPU-only XPU kernel inventory and router operand guards."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels import CapabilityRequirement, DeviceType, PlatformInfo, registry
from sglang.kernels.ops.moe.xpu_ar_topk_sycl import can_use_ar_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestXpuArTopkMetadata(CustomTestCase):
    def test_inventory_is_xpu_only(self):
        specs = registry.get("moe.xpu_ar_topk")
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertTrue(spec.is_available(PlatformInfo(device_type="xpu")))
        self.assertFalse(spec.is_available(PlatformInfo(device_type="cpu")))
        self.assertFalse(spec.is_available(PlatformInfo(device_type="cuda")))
        self.assertEqual(spec.capabilities, frozenset({CapabilityRequirement.XPU}))
        self.assertEqual(
            spec.target, "sglang.kernels.ops.moe.xpu_ar_topk_sycl:xpu_ar_topk"
        )

    def test_detection_distinguishes_xpu_and_cpu(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch, "npu", SimpleNamespace(is_available=lambda: False), create=True
            ),
            patch.object(torch.xpu, "is_available", return_value=True),
        ):
            self.assertEqual(PlatformInfo.detect().device, DeviceType.XPU)
            with patch.object(torch.xpu, "is_available", return_value=False):
                self.assertEqual(PlatformInfo.detect().device, DeviceType.CPU)

    def test_cpu_operands_keep_native_path(self):
        x = torch.zeros((1, 256), dtype=torch.bfloat16)
        weights = torch.empty((1, 8))
        ids = torch.empty((1, 8), dtype=torch.int32)
        self.assertFalse(can_use_ar_topk(x, weights, ids, True))
        self.assertFalse(can_use_ar_topk(x, weights, ids, 1))


if __name__ == "__main__":
    unittest.main()
