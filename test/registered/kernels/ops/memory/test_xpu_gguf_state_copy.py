# SPDX-License-Identifier: Apache-2.0
"""Accepted-prefix state semantics, including physical padding and graph inputs."""
import os
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.memory.gguf_state_copy_sycl import eligible, try_scatter


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_STATE_COPY_LIB"),
    reason="Requires B70 and the built GGUF state-copy library",
)


def bits(tensor):
    dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int64: torch.int64}[tensor.dtype]
    return tensor.detach().cpu().contiguous().view(dtype)


def states(dtype, padded):
    width, layers, slots, steps = 128, 2, 3, 4
    gap = 16 if padded else 0
    dst_pitch = width + gap
    dst_layer = slots * dst_pitch + gap
    src_step = width + gap
    src_slot = steps * src_step + gap
    src_layer = 2 * src_slot + gap
    dst_storage = torch.arange(layers * dst_layer, dtype=torch.float32).to(dtype).to("xpu")
    src_storage = (-torch.arange(layers * src_layer, dtype=torch.float32)-17).to(dtype).to("xpu")
    dst = dst_storage.as_strided((layers, slots, width), (dst_layer, dst_pitch, 1))
    src = src_storage.as_strided((layers, 2, steps, width), (src_layer, src_slot, src_step, 1))
    return dst_storage, src_storage, dst, src


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_masked_prefix_and_padding(dtype, padded, index_dtype, monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_GGUF_STATE_COPY", "1")
    storage, source_storage, dst, src = states(dtype, padded)
    original = storage.clone()
    source_before = bits(source_storage)
    index = torch.zeros(1, device="xpu", dtype=index_dtype)
    step = torch.zeros(1, device="xpu", dtype=index_dtype)
    assert eligible(dst, src, index, step, None)
    for slot, prefix in [(-1, 0), (3, 0), (1, -1), (1, 4), (0, 0), (1, 1), (2, 2), (1, 3)]:
        storage.copy_(original)
        expected = original.cpu().clone()
        view = expected.as_strided(dst.shape, dst.stride())
        if 0 <= slot < 3 and 0 <= prefix < 4:
            view[:, slot].copy_(src[:, 0, prefix].cpu())
        index.fill_(slot)
        step.fill_(prefix)
        assert try_scatter(dst, src, index, step)
        assert torch.equal(bits(storage), bits(expected))
        assert torch.equal(bits(source_storage), source_before)


def test_graph_dynamic_prefix_and_native_integration(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_GGUF_STATE_COPY", "1")
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend

    storage, _, dst, src = states(torch.bfloat16, True)
    original = storage.clone()
    index = torch.tensor([1], device="xpu", dtype=torch.int64)
    step = torch.tensor([0], device="xpu", dtype=torch.int64)
    commit = HybridLinearAttnBackend._scatter_speculative_state_with_mask
    commit(dst, src, index, step)
    graph = torch.xpu.XPUGraph()
    with patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("CPU mask fallback used")):
        with torch.xpu.graph(graph):
            commit(dst, src, index, step)
    for slot, prefix in [(2, 3), (0, 1), (-1, 2), (1, -1), (1, 0)]:
        storage.copy_(original)
        index.fill_(slot)
        step.fill_(prefix)
        graph.replay()
        expected = original.cpu().clone()
        if slot >= 0 and prefix >= 0:
            expected.as_strided(dst.shape, dst.stride())[:, slot].copy_(src[:, 0, prefix].cpu())
        assert torch.equal(bits(storage), bits(expected))


def test_original_fallback_guards(monkeypatch):
    _, _, dst, src = states(torch.int64, False)
    index = torch.zeros(1, device="xpu", dtype=torch.int64)
    step = index.clone()
    monkeypatch.setenv("SGLANG_XPU_GGUF_STATE_COPY", "0")
    assert not try_scatter(dst, src, index, step)
    monkeypatch.setenv("SGLANG_XPU_GGUF_STATE_COPY", "1")
    assert not eligible(dst, src, index, step, index)
    assert not eligible(dst, src, index.expand(2), step.expand(2), None)
    assert not eligible(dst[..., ::2], src[..., ::2], index, step, None)
    assert not eligible(src[:, :, 0], src, index, step, None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
@pytest.mark.parametrize("batch", [2, 8])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_batch_masked_state_copy_replays_bit_exact(
    dtype, batch, padded, index_dtype, monkeypatch
):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    monkeypatch.setenv("SGLANG_XPU_GGUF_STATE_COPY", "1")
    layers, slots, steps, width = 2, batch + 3, 4, 128
    gap = 16 if padded else 0
    dst_pitch = width + gap
    dst_layer = slots * dst_pitch + gap
    src_step = width + gap
    src_slot = steps * src_step + gap
    src_layer = (batch + 1) * src_slot + gap
    storage = torch.arange(layers * dst_layer).to(dtype=dtype, device="xpu")
    source = (-torch.arange(layers * src_layer) - 17).to(dtype=dtype, device="xpu")
    dst = storage.as_strided((layers, slots, width), (dst_layer, dst_pitch, 1))
    src = source.as_strided(
        (layers, batch + 1, steps, width), (src_layer, src_slot, src_step, 1)
    )
    original, source_before = storage.clone(), bits(source)
    index = torch.arange(batch, dtype=index_dtype, device="xpu")
    step = index % steps
    assert eligible(dst, src, index, step, None)
    commit = HybridLinearAttnBackend._scatter_speculative_state_with_mask
    assert try_scatter(dst, src, index, step)
    graph = torch.xpu.XPUGraph()
    with patch.object(
        torch.Tensor, "nonzero", side_effect=AssertionError("host mask fallback")
    ):
        with torch.xpu.graph(graph):
            commit(dst, src, index, step)
    cases = [
        (list(range(batch)), [i % steps for i in range(batch)]),
        ([i + 1 for i in range(batch)], [-1 if i % 2 else 3 for i in range(batch)]),
        ([-1 if i % 2 else slots for i in range(batch)], [2] * batch),
        ([0] * batch, [i % steps for i in range(batch)]),
    ]
    for destinations, prefixes in cases:
        storage.copy_(original)
        index.copy_(torch.tensor(destinations, dtype=index_dtype))
        step.copy_(torch.tensor(prefixes, dtype=index_dtype))
        graph.replay()
        expected = original.cpu().clone()
        view = expected.as_strided(dst.shape, dst.stride())
        for row, (slot, prefix) in enumerate(zip(destinations, prefixes)):
            if 0 <= slot < slots and 0 <= prefix < steps:
                view[:, slot].copy_(src[:, row, prefix].cpu())
        assert torch.equal(bits(storage), bits(expected))
        assert torch.equal(bits(source), source_before)
