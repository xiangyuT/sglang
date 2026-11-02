# SPDX-License-Identifier: Apache-2.0
"""Native grouped prefill: independent reference, ragged rows and store bounds."""
import os
import pytest
import torch

from sglang.kernels.ops.quantization.gguf_prefill import grouped_f16


@pytest.mark.parametrize('counts', [(1,), (0, 7, 8, 9, 0, 17), (1, 2, 3, 4, 5, 6, 7, 8) * 4])
@pytest.mark.parametrize('K,N', [(32, 64), (640, 2560), (2560, 1280)])
def test_ragged_groups_against_independent_float_reference(counts, K, N):
    if not torch.xpu.is_available() or not os.getenv('SGLANG_XPU_GGUF_PREFILL_SYCL_LIB'):
        pytest.skip('requires the built B70 grouped prefill library')
    torch.manual_seed(42)
    x = (torch.randn(sum(counts), K) * .1).half()
    weight = (torch.randn(len(counts), N, K) * .1).half()
    reference = torch.cat([part.float() @ weight[e].float().t()
                           for e, part in enumerate(x.split(counts))])
    storage = torch.full((sum(counts) * N + 128,), float('nan'), device='xpu')
    output = storage[:-128].view(sum(counts), N)
    result = grouped_f16(x.xpu(), weight.xpu(), counts, out=output)
    torch.xpu.synchronize()
    assert result is output
    torch.testing.assert_close(result.cpu(), reference, rtol=2e-5, atol=1e-5)
    assert torch.isnan(storage[-128:]).all()


def test_cpu_inputs_cannot_enter_native_queue():
    with pytest.raises(ValueError, match='bounded B70'):
        grouped_f16(torch.zeros(1, 32, dtype=torch.float16),
                    torch.zeros(1, 64, 32, dtype=torch.float16), [1])


@pytest.mark.parametrize('copy_engine,host_pack', [('0', '0'), ('1', '0'), ('1', '1')])
@pytest.mark.parametrize('tokens,topk', [(1, 3), (2, 3), (3, 3), (4, 3), (8, 3), (128, 10)])
def test_q2_native_layer_preserves_classic_arithmetic_and_owners(
    monkeypatch, copy_engine, host_pack, tokens, topk
):
    if (not torch.xpu.is_available()
            or not os.getenv('SGLANG_XPU_GGUF_PREFILL_NATIVE_F16_LIB')
            or not os.getenv('SGLANG_XPU_GGUF_SYCL_LIB')
            or not os.getenv('SGLANG_XPU_GGUF_PREFILL_SYCL_LIB')):
        pytest.skip('requires B70 native and grouped prefill libraries')
    from types import SimpleNamespace
    import numpy as np
    from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache
    from sglang.kernels.ops.quantization.gguf_prefill import apply_prefill
    from sglang.kernels.ops.quantization.gguf_prefill_f16_layer import apply_layer

    torch.manual_seed(721)
    # Q2_0: one finite FP16 scale and 16 packed bytes per 64 values.
    raw = []
    for _ in range(3):
        value = torch.randint(0, 256, (11, 460800), dtype=torch.uint8,
                              pin_memory=True)
        blocks = value.view(-1, 18)
        scale = torch.tensor([0.003], dtype=torch.float16).view(torch.uint8)
        blocks[:, :2] = scale
        value[-1].zero_()
        raw.append(value)
    cache = GPUExpertCache(
        42, range(4), [v[:4].xpu() for v in raw], range(4, 11),
        [torch.empty(v[4:].shape, dtype=torch.uint8, pin_memory=True).copy_(v[4:])
         for v in raw], layout='original', down_kind=42,
    )
    method = SimpleNamespace(
        _gpu_cache=cache, hidden_size=2560, intermediate=640, num_experts=11,
        sources=[SimpleNamespace(kind=42)] * 3, _paged_zero_expert=10,
    )
    try:
        cache.prepare(torch.tensor([[5, 7, 9]], dtype=torch.int32, device='xpu'))
        owners = [v.cpu().clone() for v in (cache.expert_slots, cache.slot_addresses,
                                           cache.mirror_addresses, cache.counters)]
        rng = np.random.default_rng(721)
        ids = np.stack([rng.choice(11, topk, replace=False)
                        for _ in range(tokens)]).astype(np.int32)
        x = (torch.randn(tokens, 2560) * .05).bfloat16().xpu()
        weights = torch.softmax(torch.randn(tokens, topk), -1).xpu()
        experts = np.unique(ids).tolist()
        monkeypatch.setenv('SGLANG_XPU_GGUF_PREFILL_NATIVE_MTP', '0')
        monkeypatch.setenv('SGLANG_XPU_GGUF_PREFILL_F16_GROUP', '8')
        monkeypatch.setenv('SGLANG_XPU_GGUF_PREFILL_COPY_ENGINE', copy_engine)
        monkeypatch.setenv('SGLANG_XPU_GGUF_PREFILL_F16_HOST_PACK', host_pack)
        reference = apply_prefill(method, x, weights, ids, experts,
                                  output_dtype=torch.float32)
        output = apply_layer(method, x, weights, ids, experts,
                             output_dtype=torch.float32)
        assert torch.isfinite(output).all()
        assert torch.equal(reference.view(torch.int32), output.view(torch.int32))
        for before, after in zip(owners, (cache.expert_slots, cache.slot_addresses,
                                         cache.mirror_addresses, cache.counters)):
            assert torch.equal(before, after.cpu())
    finally:
        cache.close()
