"""Short cached suffixes keep prefill semantics; verify remains speculative."""
from types import SimpleNamespace

import pytest

from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch


@pytest.mark.parametrize("mode,expected", [
    (ForwardMode.EXTEND, True),
    (ForwardMode.MIXED, True),
    (ForwardMode.SPLIT_PREFILL, True),
    (ForwardMode.DLLM_EXTEND, True),
    (ForwardMode.DECODE, False),
    (ForwardMode.IDLE, False),
    (ForwardMode.TARGET_VERIFY, False),
    (ForwardMode.DRAFT_EXTEND_V2, False),
])
def test_execution_mode_selects_prefill_without_row_count(mode, expected):
    with set_forward_batch(SimpleNamespace(forward_mode=mode)):
        assert is_gguf_prefill() is expected


def test_component_call_without_forward_batch_keeps_legacy_dispatch():
    assert is_gguf_prefill() is False
