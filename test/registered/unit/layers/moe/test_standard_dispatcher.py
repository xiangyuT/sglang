import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dispatch_preserves_negative_routing_sentinel():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None
    dispatcher.xpu_replica_balance = False

    topk_ids = torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
        topk_ids=topk_ids,
        router_logits=None,
    )

    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=False,
    ):
        output = dispatcher.dispatch(torch.ones((2, 4)), topk_output)

    expected = torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32)
    assert torch.equal(output.topk_output.topk_ids, expected)


def test_gguf_resident_dispatch_defers_only_to_its_registered_consumer():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None
    dispatcher.xpu_replica_balance = False
    preparation = SimpleNamespace(supports_deferred_mapping=lambda *args: True)
    dispatcher._xpu_gguf_resident_preparation = preparation
    ids = torch.tensor([[-1, 3]], dtype=torch.int64)
    topk = StandardTopKOutput(torch.tensor([[0.3, 0.7]]), ids, None)
    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather", return_value=False
    ), patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "_map_topk_ids_to_local_experts", side_effect=AssertionError("mapping launched")
    ):
        output = dispatcher.dispatch(torch.ones((1, 4)), topk)
    assert output.topk_output is topk
    assert output.xpu_gguf_resident_routing[0] is preparation
    assert output.xpu_gguf_resident_routing[1] is dispatcher.local_expert_mapping


def test_gguf_resident_unsupported_shape_retains_standard_mapping():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None
    dispatcher.xpu_replica_balance = False
    dispatcher._xpu_gguf_resident_preparation = SimpleNamespace(
        supports_deferred_mapping=lambda *args: False
    )
    topk = StandardTopKOutput(torch.ones((1, 2)), torch.tensor([[3, -1]]), None)
    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather", return_value=False
    ):
        output = dispatcher.dispatch(torch.ones((1, 4)), topk)
    assert output.xpu_gguf_resident_routing is None
    assert torch.equal(output.topk_output.topk_ids, torch.tensor([[1, -1]]))


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
