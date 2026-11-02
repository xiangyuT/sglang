"""Replicated MTP KV must not be packed into narrower target host rows."""
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("target_heads,draft_heads,expected", [(1,2,"sidecar"),(1,1,"packed"),(2,2,"packed")])
def test_nextn_host_plan_preserves_different_kv_rows(monkeypatch,target_heads,draft_heads,expected):
    from sglang.srt.speculative import base_spec_worker as module
    monkeypatch.setattr(module,"get_memory",lambda:NS(enable_hierarchical_cache=True,enable_unified_cache_external_linker=False))
    shared_state=object()
    def pool(heads):return NS(head_num=heads,head_dim=256,v_head_dim=256,store_dtype=torch.bfloat16)
    draft=NS(token_to_kv_pool=pool(draft_heads),req_to_token_pool=NS(mamba_pool=shared_state),
             model_config=NS(num_nextn_predict_layers=1))
    target=NS(token_to_kv_pool=NS(full_kv_pool=pool(target_heads)),req_to_token_pool=NS(mamba_pool=shared_state),
              spec_algorithm=NS(is_eagle=lambda:True,is_eagle3=lambda:False,is_dspark=lambda:False))
    worker=NS(target_worker=NS(model_runner=target),_draft_model_runners=lambda:(draft,))
    plan=module.BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode.value==expected
    assert plan.device_pools==(draft.token_to_kv_pool,)
    assert bool(target.mtp_draft_device_pools)==(expected=="packed")
