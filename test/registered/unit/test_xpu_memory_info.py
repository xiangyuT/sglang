"""XPU pool sizing must include allocations outside Torch's allocator."""

import unittest
from unittest.mock import patch

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestXPUMemoryInfo(unittest.TestCase):
    def test_driver_free_memory_bounds_capacity_with_native_allocations(self):
        with (
            patch.object(common.torch.xpu, "device_count", return_value=1),
            patch.object(common.torch.xpu, "current_device", return_value=0),
            patch.object(
                common.torch.xpu,
                "mem_get_info",
                return_value=(1536 * 2**20, 32 * 2**30),
            ) as driver_info,
            patch.object(
                common.torch.xpu, "memory_allocated", return_value=0
            ) as allocated,
            patch.object(common, "empty_device_cache") as empty_cache,
        ):
            self.assertEqual(common.get_available_gpu_memory("xpu", 0), 1.5)
            driver_info.assert_called_once_with(0)
            allocated.assert_not_called()
            empty_cache.assert_called_once_with(common.torch.xpu)

    def test_no_empty_cache_request_preserves_allocator_state(self):
        with (
            patch.object(common.torch.xpu, "device_count", return_value=1),
            patch.object(common.torch.xpu, "current_device", return_value=0),
            patch.object(
                common.torch.xpu, "mem_get_info", return_value=(2**30, 32 * 2**30)
            ),
            patch.object(common, "empty_device_cache") as empty_cache,
        ):
            self.assertEqual(
                common.get_available_gpu_memory("xpu", 0, empty_cache=False), 1
            )
            empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
