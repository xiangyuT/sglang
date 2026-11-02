# SPDX-License-Identifier: Apache-2.0
"""Strict FP32/BF16 parity with scatter -> multiply -> Omix Torch sum."""
import os

import numpy as np
import pytest
import torch

from sglang.kernels.ops.quantization.gguf_prefill_reduce import reduce_routes, route_metadata
from sglang.kernels.ops.quantization.gguf_sycl import scatter_expert_routes


@pytest.mark.parametrize("order", [[-1], [2], [0, 0], [[0]]])
def test_invalid_route_order(order):
    with pytest.raises(ValueError):
        route_metadata(order, 2, "cpu")


@pytest.mark.parametrize("T", [8, 1024])
@pytest.mark.parametrize("K", range(1, 11))
@pytest.mark.parametrize("ghosts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_route_reduction(T, K, ghosts, dtype):
    if not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_PREFILL_REDUCE_LIB"):
        pytest.skip("Requires the B70 routed-prefill SYCL library")
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70-only kernel")
    torch.manual_seed(42 + T + K)
    order = torch.randperm(T * K).numpy().astype(np.int32)
    if ghosts:
        order = order[order % 3 != 0]
    x = torch.randn((len(order), 2560), device="xpu")
    x.flatten()[::17] = -0.0
    weights = torch.softmax(torch.randn((T, K), device="xpu"), dim=1)
    gpu_order, inverse = route_metadata(order, T * K, x.device)
    parts = torch.zeros((T * K, 2560), device="xpu") if ghosts else torch.empty((T * K, 2560), device="xpu")
    scatter_expert_routes(x, gpu_order, parts)
    reference = (parts.view(T, K, 2560) * weights.unsqueeze(-1)).sum(dim=1).to(dtype)
    actual = reduce_routes(x, inverse, weights, dtype)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), reference.view(bits))


def test_nondefault_stream():
    if not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_PREFILL_REDUCE_LIB"):
        pytest.skip("Requires the B70 routed-prefill SYCL library")
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70-only kernel")
    with torch.xpu.stream(torch.xpu.Stream()):
        x = torch.randn((80, 2560), device="xpu")
        weights = torch.ones((8, 10), device="xpu") / 10
        _, inverse = route_metadata(np.arange(80), 80, x.device)
        actual = reduce_routes(x, inverse, weights, torch.bfloat16)
        reference = (x.view(8, 10, 2560) * weights.unsqueeze(-1)).sum(dim=1).bfloat16()
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
