# SPDX-License-Identifier: Apache-2.0
"""CPU checks for delegation and stream ownership; device arithmetic has a real-weight test."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.kernels.ops.quantization.gguf_iq4_bf16_row import (
    _RowCacheMethod,
    install_iq4_bf16_rows,
    pack_iq4_row_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.mark.parametrize("shape", [(1, 3072), (4, 3072), (16, 3072)])
def test_cpu_input_delegates_unchanged(shape):
    parent = Mock()
    parent.apply.return_value = object()
    layer = SimpleNamespace(weight=torch.empty((2560, 3072)))
    x = torch.empty(shape, dtype=torch.bfloat16)
    bias = torch.empty(2560)
    method = _RowCacheMethod(parent, object())
    assert method.apply(layer, x, bias) is parent.apply.return_value
    parent.apply.assert_called_once_with(layer, x, bias)


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 2), (4, -1)])
def test_rejects_invalid_tp_identity(world, rank):
    with pytest.raises(ValueError, match="TP2/4"):
        pack_iq4_row_cache(np.zeros(2560 * 24 * 136, np.uint8), world, rank, gdn=True)


def test_unvalidated_tp4_is_rejected_before_device_access(monkeypatch):
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=4, moe_ep_size=4),
    )
    monkeypatch.setattr(
        torch.xpu, "get_device_properties", lambda *args: pytest.fail("device access")
    )
    with pytest.raises(ValueError, match="TP2"):
        install_iq4_bf16_rows(object(), [], SimpleNamespace(), torch.device("cpu"))


def test_scratch_uses_actual_consumer_queue(monkeypatch):
    pointer = iter(range(1000, 100000))
    device = torch.device("xpu:0")

    class Tensor:
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape, self.dtype, self.device = shape, dtype, device
            self.ndim, self.pointer, self.streams = len(shape), next(pointer), []

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return self.pointer

        def record_stream(self, stream):
            self.streams.append(stream.sycl_queue)

    queue = [11]
    monkeypatch.setattr(
        torch.xpu, "current_stream", lambda *args: SimpleNamespace(sycl_queue=queue[0])
    )
    monkeypatch.setattr(
        torch, "empty", lambda shape, *, dtype, device: Tensor(shape, dtype)
    )
    native = Mock(return_value=0)
    method = _RowCacheMethod(Mock(), SimpleNamespace(xpu_iq4_bf16_row=native))
    layer = SimpleNamespace(
        weight=Tensor((2560, 3072)),
        gguf_iq4_row_codes=Tensor((1,)),
        gguf_iq4_row_gains=Tensor((1,)),
    )
    x = Tensor((4, 3072))
    method.apply(layer, x)
    queue[0] = 22
    method.apply(layer, x)
    queue[0] = 11
    method.apply(layer, x)
    calls = [row.args for row in native.call_args_list]
    assert [row[0] for row in calls] == [11, 22, 11]
    assert calls[0][6] != calls[1][6]
    assert calls[0][6] == calls[2][6]
    assert [row[-4:] for row in calls] == [(4, 2560, 3072, 8)] * 3
    assert x.streams == [11, 22, 11]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
