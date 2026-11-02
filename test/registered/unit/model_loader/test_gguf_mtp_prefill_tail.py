# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest

from sglang.kernels.ops.quantization import gguf_mtp_prefill as tail
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


@pytest.fixture
def setup_tail(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_GGUF_MTP_PREFILL_LAST_ONLY", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_PREFILL_F16", "1")
    monkeypatch.setattr(tail.torch.xpu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(tail, "get_parallel", lambda: NS(pp_size=1, attn_cp_size=1))
    method = object.__new__(CompressedGGUFMoEMethod)
    method.cache_slots = method.num_experts = 512
    method._gpu_cache = NS(_closed=False, device=NS(type="xpu"))
    model = NS(config=NS(num_hidden_layers=1, layer_types=["full_attention"]),
               model=NS(has_ple=False, modules=lambda: [NS(quant_method=method)]))
    mode = NS(is_extend=lambda: True, is_draft_extend_v2=lambda: False,
              is_target_verify=lambda: False)
    batch = NS(input_ids=NS(device=NS(type="xpu")), batch_size=1, capture_hidden_mode=CaptureHiddenMode.LAST,
               return_logprob=False, forward_mode=mode)
    return model, batch, method


def test_enables_only_inside_scope(setup_tail):
    model, batch, method = setup_tail
    with tail.mtp_prefill_tail(model, batch):
        assert method._mtp_prefill_last_only
    assert not method._mtp_prefill_last_only


def test_exception_restores_scope(setup_tail):
    model, batch, method = setup_tail
    with pytest.raises(RuntimeError):
        with tail.mtp_prefill_tail(model, batch):
            assert method._mtp_prefill_last_only
            raise RuntimeError("injected failure")
    assert not method._mtp_prefill_last_only


@pytest.mark.parametrize("case", ["batch2", "full_hidden", "logprob", "multilayer", "ple", "paged", "decode", "draft_extend"])
def test_unsupported_consumers_keep_full_prefix(setup_tail, case):
    model, batch, method = setup_tail
    if case == "batch2": batch.batch_size = 2
    elif case == "full_hidden": batch.capture_hidden_mode = CaptureHiddenMode.FULL
    elif case == "logprob": batch.return_logprob = True
    elif case == "multilayer": model.config.num_hidden_layers = 2
    elif case == "ple": model.model.has_ple = True
    elif case == "paged": method.cache_slots = 128
    elif case == "decode": batch.forward_mode.is_extend = lambda: False
    elif case == "draft_extend": batch.forward_mode.is_draft_extend_v2 = lambda: True
    with tail.mtp_prefill_tail(model, batch):
        assert not getattr(method, "_mtp_prefill_last_only", False)


def test_capture_keeps_full_prefix(setup_tail, monkeypatch):
    model, batch, method = setup_tail
    monkeypatch.setattr(tail.torch.xpu, "is_current_stream_capturing", lambda: True)
    with tail.mtp_prefill_tail(model, batch):
        assert not getattr(method, "_mtp_prefill_last_only", False)


def test_cpu_batch_does_not_enter_xpu_runtime(setup_tail, monkeypatch):
    model, batch, method = setup_tail
    batch.input_ids.device.type = "cpu"
    def forbidden():
        raise AssertionError("CPU batch must not query XPU capture")
    monkeypatch.setattr(tail.torch.xpu, "is_current_stream_capturing", forbidden)
    with tail.mtp_prefill_tail(model, batch):
        assert not getattr(method, "_mtp_prefill_last_only", False)
