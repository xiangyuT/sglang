# SPDX-License-Identifier: Apache-2.0
"""The eager XPU boundary transfer must preserve the reference relative IDs."""
import pytest
import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


@pytest.mark.parametrize('device_index', [0, 1])
@pytest.mark.parametrize('rows,width', [(0, 768), (1, 768), (128, 1536), (1024, 768)])
@pytest.mark.parametrize('mode', ['random', 'ties', 'nonfinite'])
def test_prefill_bounds_exact(monkeypatch, device_index, rows, width, mode):
    if not torch.xpu.is_available() or torch.xpu.device_count() <= device_index:
        pytest.skip('requires the selected XPU device')
    device = torch.device('xpu', device_index)
    if torch.xpu.get_device_properties(device).device_id != 0xE223:
        pytest.skip('validated on B70')
    generator = torch.Generator().manual_seed(1280)
    logits = torch.randn(rows, width, generator=generator)
    if mode == 'ties':
        logits = logits.round()
    elif mode == 'nonfinite':
        logits[:, ::101] = -torch.inf
        logits[:, ::257] = torch.inf
        logits[:, ::509] = torch.nan
    dtype = torch.int32 if mode == 'random' else torch.int64
    starts = torch.arange(rows, dtype=dtype) % 17
    lengths = torch.arange(rows, dtype=dtype) * 41 % (width - 17)
    lengths[::7] = 0
    logits, starts, ends = (value.to(device) for value in (logits, starts, starts + lengths))
    monkeypatch.delenv('SGLANG_XPU_QSA_PREFILL_BATCHED_BOUNDS', raising=False)
    expected = qsa_fast_topk(logits, starts, ends, 512)
    monkeypatch.setenv('SGLANG_XPU_QSA_PREFILL_BATCHED_BOUNDS', '1')
    observed = qsa_fast_topk(logits, starts, ends, 512)
    assert observed.dtype == torch.int32
    assert observed.shape == (rows, 512)
    assert torch.equal(observed, expected)


def test_prefill_bounds_cpu_unchanged(monkeypatch):
    logits = torch.arange(2 * 600, dtype=torch.float32).reshape(2, 600)
    starts = torch.tensor([13, 31], dtype=torch.int32)
    ends = torch.tensor([599, 31], dtype=torch.int32)
    monkeypatch.delenv('SGLANG_XPU_QSA_PREFILL_BATCHED_BOUNDS', raising=False)
    expected = qsa_fast_topk(logits, starts, ends, 512)
    monkeypatch.setenv('SGLANG_XPU_QSA_PREFILL_BATCHED_BOUNDS', '1')
    assert torch.equal(qsa_fast_topk(logits, starts, ends, 512), expected)
