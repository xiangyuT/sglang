"""Static head rejects weight writes before warmup and before session restore."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)


class TestStaticHeadWeightUpdates(CustomTestCase):
    def updater(self, model=None):
        return WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model if model is not None else torch.nn.Linear(2, 2),
            update_model_fields=Mock(),
            recapture_cuda_graph=Mock(),
            get_model_runner=Mock(),
        )

    def test_all_update_entrypoints_reject_before_loading(self):
        updater = self.updater()
        with (
            envs.SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL.override(True),
            patch.object(WeightUpdater, "_assert_weight_cache_inactive"),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.restore_weights_before_loading"
            ) as restore,
        ):
            self.assertIn("static packed", updater.derived_weight_cache_error())
            for result in (
                updater.update_weights_from_disk("unused", None),
                updater.load_weights_from_distributed([]),
                updater.update_weights_from_tensor([]),
                updater.update_weights_from_ipc(SimpleNamespace()),
            ):
                self.assertFalse(result[0])
                self.assertIn("SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL=0", result[1])
            with self.assertRaisesRegex(RuntimeError, "static packed"):
                updater.begin_weight_update()
            restore.assert_not_called()

    def test_scheduler_rejects_without_opening_session_or_restoring_weights(self):
        updater = self.updater()
        worker = SimpleNamespace(
            weight_update_runners=lambda: [
                ("target", SimpleNamespace(weight_updater=updater))
            ]
        )
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None)),
        ):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=worker,
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=Mock(),
                is_fully_idle=Mock(),
            )
        with (
            envs.SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL.override(True),
            patch.object(WeightUpdater, "begin_weight_update") as begin,
            patch("torch.distributed.barrier") as barrier,
        ):
            response = manager.begin_weight_update(SimpleNamespace(selector="all"))
            self.assertFalse(response.success)
            self.assertIn("static packed", response.message)
            self.assertIsNone(manager._session)
            begin.assert_not_called()
            barrier.assert_not_called()

    def test_default_off_preserves_the_existing_weight_update_path(self):
        updater = self.updater()
        with (
            envs.SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL.override(False),
            patch(
                "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.restore_weights_before_loading"
            ) as restore,
        ):
            self.assertIsNone(updater.derived_weight_cache_error())
            updater.begin_weight_update()
            restore.assert_called_once()


if __name__ == "__main__":
    unittest.main()
