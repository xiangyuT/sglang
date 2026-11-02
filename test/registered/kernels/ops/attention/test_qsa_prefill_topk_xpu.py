# SPDX-License-Identifier: Apache-2.0
"""Exact ragged top-k order, including equal scores and valid -inf rows."""
import pytest
import torch
from sglang.kernels.ops.attention.xpu_qsa_sycl import try_prefill_topk
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


@pytest.mark.parametrize('width,rows,topk', [(64,32,512),(256,1024,512),(512,512,512),(256,64,2048)])
@pytest.mark.parametrize('mode', ['random','ties','inf'])
def test_batched_prefill_matches_ragged_reference(monkeypatch,width,rows,topk,mode):
    if not torch.xpu.is_available() or torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip('requires B70')
    torch.manual_seed(42)
    logits=torch.randn(rows,width,device='xpu')
    if mode=='ties': logits=logits.round()
    if mode=='inf': logits[:,::7]=-torch.inf
    start=torch.arange(rows,device='xpu',dtype=torch.int32)%8
    end=torch.maximum(start,torch.arange(rows,device='xpu',dtype=torch.int32)%width)
    monkeypatch.delenv('SGLANG_XPU_QSA_PREFILL_BATCHED_TOPK',raising=False)
    reference=qsa_fast_topk(logits,start,end,topk)
    monkeypatch.setenv('SGLANG_XPU_QSA_PREFILL_BATCHED_TOPK','1')
    result=qsa_fast_topk(logits,start,end,topk)
    assert torch.equal(result.cpu(),reference.cpu())


def test_cpu_selector_retains_reference(monkeypatch):
    monkeypatch.setenv('SGLANG_XPU_QSA_PREFILL_BATCHED_TOPK','1')
    assert try_prefill_topk(torch.zeros(8,64),torch.zeros(8,dtype=torch.int32),
                            torch.ones(8,dtype=torch.int32),512) is None
