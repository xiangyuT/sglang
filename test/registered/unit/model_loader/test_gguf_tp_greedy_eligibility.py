from types import SimpleNamespace
import pytest
from sglang.srt.layers.ar449_head import eligibility
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def request():
    return SimpleNamespace(batch_size=1, forward_mode=ForwardMode.TARGET_VERIFY,
        actual_forward_mode=None, spec_info=object(), return_logprob=False,
        sampling_info=SimpleNamespace(is_all_greedy=True))


def test_target_verify_opt_in_keeps_plain_decode_guard():
    batch=request()
    assert eligibility(batch,target_verify=True)==(True,"plain-greedy")
    assert eligibility(batch)[0] is False
    batch.forward_mode=ForwardMode.DECODE;batch.spec_info=None
    assert eligibility(batch)[0] is True
    assert eligibility(batch,target_verify=True)[0] is False


@pytest.mark.parametrize("field", ["logit_bias","acc_additive_penalties","acc_scaling_penalties","grammar_mask","custom_logit_processor","watermark_keys"])
def test_verify_score_consumers_fall_back(field):
    batch=request();setattr(batch.sampling_info,field,object())
    assert eligibility(batch,target_verify=True)[0] is False


def test_verify_logprobs_and_sampling_fall_back():
    batch=request();batch.return_logprob=True
    assert eligibility(batch,target_verify=True)[0] is False
    batch.return_logprob=False;batch.sampling_info.is_all_greedy=False
    assert eligibility(batch,target_verify=True)[0] is False


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("unsupported", ["logprob", "sampling", "grammar", "penalty", None])
@pytest.mark.parametrize("batch_size", range(1,9))
def test_real_graph_admission_checks_both_padded_and_exact_shapes(padding, unsupported, batch_size, monkeypatch):
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    monkeypatch.setenv("SGLANG_XPU_GGUF_TP_GREEDY_GATHER", "1")
    runner=object.__new__(DecodeCudaGraphRunner)
    runner.backend=SimpleNamespace(_native_gguf=True,can_run=lambda *args:True)
    runner.disable_padding=not padding;runner.max_bs=8;runner.ragged_verify_mode=False
    runner.captured_req_width=4;runner.require_mlp_tp_gather=False
    runner.enable_pdmux=False;runner._make_graph_key=lambda *args,**kwargs:None
    runner._resolve_lora_variant=lambda *args:None;runner._resolve_attention_variant=lambda *args:None;runner.require_mlp_sync=False
    runner.is_encoder_decoder=False;runner.enable_two_batch_overlap=False
    runner.model_runner=SimpleNamespace(spec_algorithm=SimpleNamespace(is_ngram=lambda:False))
    batch=request();batch.replace_embeds=None
    batch.batch_size=batch_size
    batch.spec_info=SimpleNamespace(num_tokens_per_req=4)
    if unsupported=="logprob":batch.return_logprob=True
    if unsupported=="sampling":batch.sampling_info.is_all_greedy=False
    if unsupported=="grammar":batch.sampling_info.grammar_mask=object()
    if unsupported=="penalty":batch.sampling_info.acc_additive_penalties=object()
    assert runner.can_run_graph(batch) is (unsupported is None)


def test_bounded_single_row_head_remains_single_request():
    batch=request();batch.batch_size=2
    assert eligibility(batch,target_verify=True)[0] is False
    assert eligibility(batch,target_verify=True,max_batch=8)[0] is True
