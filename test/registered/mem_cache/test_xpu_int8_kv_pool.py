"""Encoded KV transfer geometry and fail-closed recipe checks (CPU metadata)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_int8 import _record_ok
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)
from sglang.srt.mem_cache.int8_kv_pool import Int8QSAKVCacheMethod, Int8QSATokenToKVPool
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype


@pytest.mark.parametrize("heads", [1, 2])
def test_encoded_rows_move_and_register_scales_once(heads):
    pool = Int8QSATokenToKVPool(
        size=128,
        page_size=64,
        dtype=torch.int8,
        head_num=heads,
        head_dim=256,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    method = pool.quant_method
    for k, v in zip(method.raw_k, method.raw_v):
        k.copy_(torch.randint(-128, 128, k.shape, dtype=torch.int8))
        v.copy_(-k)
    info = pool.get_device_pool_infos()[0].buffer_info
    info.validate()
    assert info.scales is None and info.head_dim == info.v_head_dim == 264
    assert pool.is_quantized_kv_cache and method.kv_storage_dtype() == torch.int8
    assert all(
        a.kind == KVCacheAttentionAccessKind.NATIVE_INT8
        for a in method.attention_accesses()
    )
    assert (
        not method.needs_dequant_workspace()
        and not method.needs_plain_kv_dequant_read()
    )
    assert _record_ok(pool.get_key_buffer(0))
    ptrs, lens, items = pool.get_contiguous_buf_infos()
    assert ptrs == [x.data_ptr() for x in method.raw_k + method.raw_v]
    assert lens == [192 * heads * 264] * 4 and items == [64 * heads * 264] * 4
    src = torch.tensor([3, 4, 5])
    dst = torch.tensor([4, 5, 3])
    expected = [raw[src].clone() for raw in method.raw_k + method.raw_v]
    pool.move_kv_cache(dst, src)
    for raw, e in zip(method.raw_k + method.raw_v, expected):
        assert torch.equal(raw[dst], e)
    copy = pool.get_cpu_copy(dst)
    for raw in method.raw_k + method.raw_v:
        raw[dst] = 0
    pool.load_cpu_copy(copy, dst)
    for raw, e in zip(method.raw_k + method.raw_v, expected):
        assert torch.equal(raw[dst], e)
    assert sum(pool.get_kv_size_bytes()) == 2 * 2 * 192 * heads * 264
    assert method.compute_cell_size(heads, 256, 2, 1) == 2 * 2 * heads * 264 + 4


def test_truncated_inline_scale_storage_is_rejected():
    data = torch.zeros(256, dtype=torch.int8)
    view = data.as_strided((1, 1, 256), (264, 264, 1))
    assert not _record_ok(view)
    assert not _record_ok(torch.zeros(1, 1, 256, dtype=torch.int8))


@pytest.mark.parametrize("heads,dim", [(3, 256), (1, 128), (2, 512)])
def test_unsupported_geometry_rejected(heads, dim):
    with pytest.raises(ValueError):
        Int8QSAKVCacheMethod().create_buffers(64, heads, dim, 1, "cpu")


@pytest.mark.parametrize(
    "xpu,activation,expected",
    [
        (True, torch.bfloat16, torch.int8),
        (True, torch.float16, None),
        (False, torch.bfloat16, None),
    ],
)
def test_int8_dtype_requires_xpu_bf16(xpu, activation, expected):
    with patch(
        "sglang.srt.mem_cache.kv_cache_dtype.current_platform",
        SimpleNamespace(is_xpu=lambda: xpu),
    ):
        args = dict(
            server_args_kv_cache_dtype="int8",
            model=None,
            model_dtype=activation,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="intel_xpu",
        )
        if expected is None:
            with pytest.raises(ValueError):
                configure_kv_cache_dtype(**args)
        else:
            assert configure_kv_cache_dtype(**args) == ("int8", expected)


@pytest.mark.parametrize('unsupported', [None, 'non_qsa', 'no_hybrid', 'cp', 'dcp', 'pp', 'mla', 'backend', 'unified', 'page_major', 'hisparse', 'sharding'])
def test_configurator_rejects_unsupported_int8_routes(monkeypatch, unsupported):
    import sglang.srt.mem_cache.kv_cache_configurator as module
    profile=SimpleNamespace(indexer_n_heads=4,indexer_kv_heads=1,indexer_head_dim=128,indexer_budget=2048,indexer_compress_ratio=4)
    config=SimpleNamespace(kv_cache_dtype=torch.int8, model_config=SimpleNamespace(hf_config=profile),
                           mambaish_config=object(),pp_size=1,use_mla_backend=False)
    parallel=SimpleNamespace(attn_dcp_size=1,attn_cp_size=1,enable_kv_cache_sharding=False)
    memory=SimpleNamespace(enable_unified_memory=False,enable_page_major_kv_layout=False,enable_hisparse=False)
    backends=('intel_xpu',)
    if unsupported=='non_qsa':config.model_config.hf_config=SimpleNamespace()
    if unsupported=='no_hybrid':config.mambaish_config=None
    if unsupported=='cp':parallel.attn_cp_size=2
    if unsupported=='dcp':parallel.attn_dcp_size=2
    if unsupported=='pp':config.pp_size=2
    if unsupported=='mla':config.use_mla_backend=True
    if unsupported=='backend':backends=('triton',)
    if unsupported=='unified':memory.enable_unified_memory=True
    if unsupported=='page_major':memory.enable_page_major_kv_layout=True
    if unsupported=='hisparse':memory.enable_hisparse=True
    if unsupported=='sharding':parallel.enable_kv_cache_sharding=True
    monkeypatch.setattr(module,'current_platform',SimpleNamespace(is_xpu=lambda:True))
    monkeypatch.setattr(module,'get_parallel',lambda:parallel)
    monkeypatch.setattr(module,'get_memory',lambda:memory)
    monkeypatch.setattr(module,'attention_backends',lambda:backends)
    monkeypatch.setenv('SGLANG_USE_HND_KVCACHE','false')
    if unsupported is None:
        assert isinstance(module.KVCacheConfigurator._build_mha_quant_method(config,num_layers=12),Int8QSAKVCacheMethod)
    else:
        with pytest.raises(ValueError):module.KVCacheConfigurator._build_mha_quant_method(config,num_layers=12)
        config._build_mha_quant_method = lambda **kwargs: module.KVCacheConfigurator._build_mha_quant_method(config, **kwargs)
        # Alternate factories must fail before sizing/allocation, too.
        with pytest.raises(ValueError):module.KVCacheConfigurator.configure(config, pre_model_load_memory=0)
