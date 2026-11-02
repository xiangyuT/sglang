# SPDX-License-Identifier: Apache-2.0
"""Exact QSA index selection against the installed Torch XPU stable sort."""

import os

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import try_decode_topk
from sglang.kernels.ops.attention.xpu_qsa_topk_sycl import load_library


@pytest.fixture(params=range(torch.xpu.device_count()))
def device(request, monkeypatch):
    if torch.xpu.get_device_properties(request.param).device_id != 0xE223:
        pytest.skip("requires B70")
    path = os.environ.get("SGLANG_XPU_QSA_TOPK_LIB")
    if not path:
        pytest.skip(
            "build with build_xpu_qsa.py --topk-only and set SGLANG_XPU_QSA_TOPK_LIB"
        )
    torch.xpu.set_device(request.param)
    load_library(path)
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_NATIVE_TOPK", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_LONG_INDEXER_GRAPH", "1")
    return torch.device("xpu", request.param)


def reference(logits, lengths):
    width = min(512, logits.shape[1])
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)
    selected = selected[:, :width].to(torch.int32)
    selected.masked_fill_(selected >= lengths[:, None], -1)
    output = torch.full(
        (logits.shape[0], 512), -1, dtype=torch.int32, device=logits.device
    )
    output[:, :width] = selected
    return output


def masked_scores(device, rows, columns, full, nonfinite=False, strided=False):
    torch.manual_seed(1137)
    logits = torch.randn(rows, columns + (8 if strided else 0), device=device)[
        :, :columns
    ]
    values = [
        0,
        1,
        min(columns, 511),
        min(columns, 512),
        min(columns, 513),
        columns // 2,
        columns,
    ]
    lengths = (
        torch.full((rows,), columns, dtype=torch.int64, device=device)
        if full
        else torch.tensor(
            [values[i % len(values)] for i in range(rows)],
            dtype=torch.int32,
            device=device,
        )
    )
    if nonfinite:
        logits.copy_((torch.arange(columns, device=device) % 9 - 4).float()[None, :])
        bits = torch.tensor(
            [
                -8388608,
                -2147483648,
                0,
                2139095040,
                2143289345,
                2143289346,
                -4194303,
                -4194302,
                -1082130432,
                1065353216,
            ],
            dtype=torch.int32,
            device=device,
        )
        logits[:, :10] = bits.view(torch.float32)
    logits.masked_fill_(
        torch.arange(columns, device=device)[None, :] >= lengths[:, None], -torch.inf
    )
    return logits, lengths


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
@pytest.mark.parametrize("columns", [1, 63, 512, 513, 2048, 2049, 3073, 4097, 65536])
@pytest.mark.parametrize("full", [False, True])
def test_exact_indices(device, rows, columns, full):
    logits, lengths = masked_scores(device, rows, columns, full)
    assert torch.equal(
        try_decode_topk(logits, lengths, 512), reference(logits, lengths)
    )


@pytest.mark.parametrize("columns", [63, 512, 513, 2048, 65536])
@pytest.mark.parametrize("full", [False, True])
def test_radix_ties_nonfinite(device, columns, full):
    logits, lengths = masked_scores(
        device, 8, columns, full, nonfinite=True, strided=True
    )
    assert torch.equal(
        try_decode_topk(logits, lengths, 512), reference(logits, lengths)
    )


def test_replay_lengths_and_mixed_verifier_rows(device):
    logits, lengths = masked_scores(device, 32, 65536, False)
    try_decode_topk(logits, lengths, 512)
    torch.xpu.synchronize(device)
    graph = torch.xpu.XPUGraph()
    stream = torch.xpu.Stream(device=device)
    with torch.xpu.graph(graph, stream=stream):
        result = try_decode_topk(logits, lengths, 512)
    for n in [0, 1, 511, 512, 513, 2048, 2049, 4096, 65536, None]:
        if n is None:
            lengths.copy_(
                torch.tensor([65536] * 4 + [280] * 28, dtype=torch.int32, device=device)
            )
        else:
            lengths.fill_(n)
        values = torch.randn_like(logits)
        values.masked_fill_(
            torch.arange(65536, device=device)[None, :] >= lengths[:, None], -torch.inf
        )
        logits.copy_(values)
        graph.replay()
        assert torch.equal(result, reference(logits, lengths))


def test_cpu_fallback(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_QSA_BATCHED_TOPK", "1")
    monkeypatch.setenv("SGLANG_XPU_QSA_NATIVE_TOPK", "1")
    assert (
        try_decode_topk(torch.zeros(8, 64), torch.ones(8, dtype=torch.int32), 512)
        is None
    )


def test_default_and_unsupported_keep_torch(device, monkeypatch):
    logits, lengths = masked_scores(device, 8, 513, False)
    monkeypatch.setenv("SGLANG_XPU_QSA_NATIVE_TOPK", "0")
    monkeypatch.setenv("SGLANG_XPU_QSA_TOPK_LIB", "/missing/unused.so")
    assert torch.equal(
        try_decode_topk(logits, lengths, 512), reference(logits, lengths)
    )
    monkeypatch.setenv("SGLANG_XPU_QSA_NATIVE_TOPK", "1")
    # 65 rows and noncontiguous columns both fall back without loading the DLL.
    logits, lengths = masked_scores(device, 65, 513, False)
    assert torch.equal(
        try_decode_topk(logits, lengths, 512), reference(logits, lengths)
    )
    logits, lengths = masked_scores(device, 8, 1026, False)
    logits = logits[:, ::2]
    lengths = torch.clamp((lengths + 1) // 2, max=513)
    assert torch.equal(
        try_decode_topk(logits, lengths, 512), reference(logits, lengths)
    )


def test_explicit_enable_requires_library(device, monkeypatch):
    logits, lengths = masked_scores(device, 8, 513, False)
    monkeypatch.delenv("SGLANG_XPU_QSA_TOPK_LIB", raising=False)
    monkeypatch.delenv("SGLANG_XPU_QSA_SYCL_LIB", raising=False)
    with pytest.raises(RuntimeError, match="QSA_TOPK_LIB"):
        try_decode_topk(logits, lengths, 512)
