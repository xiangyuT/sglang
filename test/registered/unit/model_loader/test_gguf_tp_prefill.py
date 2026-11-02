# SPDX-License-Identifier: Apache-2.0
"""Contract checks for the opt-in GGUF long-row precision transition."""
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.layers.quantization import gguf_tp_prefill as precision


@pytest.mark.parametrize('rows', [0, 1, 4])
def test_decode_keeps_parent_call_and_packed_storage(rows):
    packed = torch.zeros((16, 136), dtype=torch.uint8)
    x = torch.zeros((rows, 256), dtype=torch.bfloat16)
    bias = torch.tensor([3.0])
    result = object()
    calls = []
    class Parent:
        width = 256
        def apply(self, layer, actual, actual_bias):
            calls.append((layer, actual, actual_bias))
            return result
        def reconstruct_weight(self, layer):
            raise AssertionError('decode must not reconstruct packed weights')
    layer = SimpleNamespace(weight=packed)
    method = precision.PrefillFP32RowMethod(Parent())
    assert method.width == 256
    assert method.apply(layer, x, bias) is result
    assert calls == [(layer, x, bias)]
    assert layer.weight is packed


def test_prefill_checks_reconstructed_matrix_instead_of_packed_dtype(monkeypatch):
    x = torch.zeros((128, 256), dtype=torch.bfloat16)
    weight = torch.ones((2560, 256), dtype=torch.bfloat16)
    partial = torch.empty((128, 2560), dtype=torch.float32)
    class Parent:
        def reconstruct_weight(self, layer):
            return weight
        def apply(self, *args):
            raise AssertionError('prefill must use FP32 local output')
    def mm(actual, matrix, *, out_dtype):
        assert actual is x and torch.equal(matrix, weight.t())
        assert out_dtype == torch.float32
        return partial
    monkeypatch.setattr(torch, 'mm', mm)
    layer = SimpleNamespace(weight=torch.zeros((2560, 136), dtype=torch.uint8))
    assert precision.PrefillFP32RowMethod(Parent()).apply(layer, x) is partial


def test_bias_is_rejected_before_silent_precision_change():
    layer = SimpleNamespace(weight=torch.zeros((2560, 256), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='bias-free BF16'):
        precision.PrefillFP32RowMethod(object()).apply(
            layer, torch.zeros((128, 256), dtype=torch.bfloat16), torch.zeros(2560))


def test_cpu_partial_cannot_enter_xpu_collective():
    with pytest.raises(ValueError, match='prefill partial requires H2560'):
        precision.reduce_prefill_partial(torch.zeros((128, 2560)), SimpleNamespace(world_size=2))


def test_precision_flag_is_explicit(monkeypatch):
    monkeypatch.delenv('SGLANG_XPU_GGUF_TP_PREFILL_FP32', raising=False)
    assert not precision.enabled()
    monkeypatch.setenv('SGLANG_XPU_GGUF_TP_PREFILL_FP32', '1')
    assert precision.enabled()


@pytest.mark.parametrize('group_name', ['ATTN_TP', 'TP', 'MOE_OUTPUT'])
def test_pending_residual_read_owns_fp32_to_bf16_transition(monkeypatch, group_name):
    from sglang.srt.layers.layer_boundary import prepare
    from sglang.srt.layers.layer_boundary.layout import SumGroup
    monkeypatch.setenv('SGLANG_XPU_GGUF_TP_PREFILL_FP32', '1')
    partial = SimpleNamespace(dtype=torch.float32, device=SimpleNamespace(type='xpu'))
    complete = torch.zeros((128, 2560), dtype=torch.bfloat16)
    group = object()
    calls = []
    def reduce(actual, selected):
        assert actual is partial and selected is group
        calls.append('native_fp32')
        return complete
    def wrong_reduce(*args, **kwargs):
        raise AssertionError('FP32 partial must not reach ordinary BF16/XCCL sum')
    monkeypatch.setattr(precision, 'reduce_prefill_partial', reduce)
    monkeypatch.setattr(prepare, '_sum_group', lambda _: group)
    monkeypatch.setattr(prepare, 'attn_tp_all_reduce', wrong_reduce)
    monkeypatch.setattr(prepare, 'tensor_model_parallel_all_reduce', wrong_reduce)
    class Read:
        def update_and_read(self, update, actual, residual, norm, **kwargs):
            assert actual is complete
            return actual, residual
    residual = object()
    out = prepare._reduce_update_read(
        partial, residual, None, None, gathers_residual=False,
        group=getattr(SumGroup, group_name), fusions=(), read_fusions=(), read=Read())
    assert out[0] is complete and out[1] is residual and calls == ['native_fp32']



def test_separate_row_preserves_bf16_boundary_sum(monkeypatch):
    x = torch.zeros((128, 256), dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=torch.zeros((2560, 256), dtype=torch.bfloat16))
    group = SimpleNamespace(world_size=4)
    partial = torch.zeros((128, 2560), dtype=torch.float32)
    complete = torch.full((128, 2560), 8, dtype=torch.bfloat16)
    monkeypatch.setattr(torch, 'mm', lambda *args, **kwargs: partial)
    def reduce(value, actual_group):
        assert value is partial and actual_group is group
        return complete
    monkeypatch.setattr(precision, 'reduce_prefill_partial', reduce)
    output = precision.PrefillFP32RowMethod(object(), group).apply(layer, x)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output * 4, complete)


@pytest.mark.parametrize('fp32,separate', [(False, False), (True, False), (True, True)])
def test_resident_ep_prefill_dtype_and_separate_sum(monkeypatch, fp32, separate):
    from sglang.srt.layers.quantization.gguf_ep import ResidentEPGGUFMoEMethod
    from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
    import sglang.srt.runtime_context as runtime
    x = torch.zeros((128, 2560), dtype=torch.bfloat16)
    group = SimpleNamespace(world_size=4)
    monkeypatch.setattr(runtime, 'get_parallel', lambda: SimpleNamespace(moe_ep_group=group))
    def parent(self, *args, output_dtype=None):
        assert output_dtype == (torch.float32 if fp32 else None)
        return torch.ones_like(x, dtype=output_dtype or x.dtype)
    monkeypatch.setattr(CompressedGGUFMoEMethod, '_apply_gpu_prefill', parent)
    def reduce(value, actual_group):
        assert value.dtype == torch.float32 and actual_group is group
        return (value * 4).to(torch.bfloat16)
    monkeypatch.setattr(precision, 'reduce_prefill_partial', reduce)
    method = ResidentEPGGUFMoEMethod.__new__(ResidentEPGGUFMoEMethod)
    method.prefill_fp32, method.prefill_separate, method.ep_size = fp32, separate, 4
    result = method._apply_gpu_prefill(x, None, None, None)
    assert result.dtype == (torch.float32 if fp32 and not separate else torch.bfloat16)
    assert torch.equal(result, torch.ones_like(result))
