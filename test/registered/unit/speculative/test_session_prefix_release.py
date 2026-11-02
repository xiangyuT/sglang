"""A turn shorter than the next Mamba checkpoint still owns its cached prefix."""
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("aborted,checkpointed,expected",[(False,True,[17]),(True,True,[]),(False,False,[])])
def test_register_existing_prefix_when_finished_turn_inserts_no_new_state(aborted,checkpointed,expected):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    registered=[]
    cache=NS(enable_session_radix_cache=True,_components_tuple=(),
             session_refs=NS(register_session_ref=lambda req,leaf:registered.append(leaf)))
    req=NS(last_node=17,finished_reason=FINISH_ABORT() if aborted else object())
    UnifiedRadixCache.on_release(cache,req,checkpointed=checkpointed)
    assert registered==expected
