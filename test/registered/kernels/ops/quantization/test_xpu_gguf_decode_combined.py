"""Optional B70 decode integrations: numerical, graph and pool ownership checks."""
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="Intel XPU required")


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    if torch.xpu.get_device_properties(0).device_id != 0xE223:
        pytest.skip("B70 geometry")
    if not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"):
        pytest.skip("Build the optional GGUF SYCL library")
    monkeypatch.setenv("SGLANG_XPU_GGUF_DENSE_BF16", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_MTP_TOP1", "1")


def random(shape, seed=75, scale=1, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * scale).to(dtype).to("xpu")


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("width", [2560, 6144])
def test_projection_output_dtype_and_workgroups(tokens, width, monkeypatch):
    from sglang.kernels.ops.quantization import gguf_dense_bf16
    from sglang.kernels.ops.quantization.gguf_dense_q8 import dense_q8_projection

    fixture = runpy.run_path(str(Path(__file__).with_name("test_xpu_gguf_dense_q8.py")))["fixture"]
    parts = [(8, fixture(8, width, 3)[0].to("xpu")), (23, fixture(23, width, 19)[0].to("xpu"))]
    x = random((tokens, width), scale=.1)
    permutation = "gdn_16x3x128" if width == 6144 else None
    with torch.no_grad():
        gold = dense_q8_projection(x, parts, input_permutation=permutation).bfloat16().cpu()
        for wg8 in (False, True):
            monkeypatch.setenv("SGLANG_XPU_GGUF_IQ4_WG8", "1" if wg8 else "0")
            out = gguf_dense_bf16.project(x, parts, packed_parts=(False, False), input_permutation=permutation).cpu()
            if not wg8:
                assert torch.equal(out.view(torch.int16), gold.view(torch.int16))
            torch.testing.assert_close(out.float(), gold.float(), rtol=.015625, atol=3e-4)
        assert not gguf_dense_bf16.supports(x.float(), parts, (False, False))


@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("stride", [4, 324])
def test_hc_combined_norm_retains_rounding_and_score_stride(tokens, stride):
    from sglang.kernels.ops.gemm.gguf_hc_xpu import combine, combine_and_normalize, normalize

    residual, block = random((tokens, 10240)), random((tokens, 2560), seed=76)
    storage = random((tokens, stride), seed=77)
    scores = storage[:, -4:]
    weight = random((10240,), seed=78, scale=.1)
    with torch.no_grad():
        gold = combine(block, residual, scores)
        expected_norm = normalize(gold, weight, 1e-6)
        actual, normed = combine_and_normalize(block, residual, scores, weight, 1e-6)
        assert torch.equal(actual.cpu().view(torch.int16), gold.cpu().view(torch.int16))
        assert torch.equal(normed.cpu().view(torch.int16), expected_norm.cpu().view(torch.int16))


def test_hc_paired_down_native_matches_projection_bound(monkeypatch):
    from sglang.kernels.ops.gemm.gguf_hc_xpu import mix

    x = random((4, 10240), scale=.1)
    down = random((324, 10240), seed=79, scale=.01)
    up = random((10240, 320), seed=80, scale=.01)
    with torch.no_grad():
        monkeypatch.setenv("SGLANG_XPU_GGUF_HC_DOWN_DPAS", "0")
        gold, gold_scores = mix(x, down, up)
        monkeypatch.setenv("SGLANG_XPU_GGUF_HC_DOWN_DPAS", "1")
        out, scores = mix(x, down, up)
        torch.testing.assert_close(out.float(), gold.float(), rtol=.015625, atol=1e-4)
        torch.testing.assert_close(scores.float(), gold_scores.float(), rtol=.015625, atol=1e-4)


@pytest.mark.parametrize("tokens", [1, 4])
def test_router_normalization_preserves_selected_fp32_values(tokens, monkeypatch):
    from sglang.srt.layers.moe.topk import fused_topk_torch_native

    hidden = random((tokens, 2560))
    logits = random((tokens, 512), seed=81, dtype=torch.float32)
    with torch.no_grad():
        monkeypatch.setenv("SGLANG_XPU_GGUF_TOPK", "0")
        gold, gold_ids = fused_topk_torch_native(hidden, logits, 10, True)
        monkeypatch.setenv("SGLANG_XPU_GGUF_TOPK", "1")
        monkeypatch.setenv("SGLANG_XPU_GGUF_TOPK_NORM", "1")
        actual, ids = fused_topk_torch_native(hidden, logits, 10, True)
        assert torch.equal(actual.cpu().view(torch.int32), gold.cpu().view(torch.int32))
        assert torch.equal(ids.cpu(), gold_ids.cpu())


def test_mtp_confidence_dynamic_graph_keeps_argmax():
    from sglang.kernels.ops.speculative.gguf_mtp_top1 import try_top1

    logits = random((1, 106299), seed=82, dtype=torch.float32)
    with torch.no_grad():
        try_top1(logits)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            probability, index = try_top1(logits)
        for seed in (83, 89, 97):
            cpu = torch.randn(logits.shape, generator=torch.Generator().manual_seed(seed))
            logits.copy_(cpu)
            graph.replay()
            assert torch.equal(index.cpu(), cpu.argmax(dim=-1, keepdim=True))
            expected = torch.softmax(cpu.double(), dim=-1).amax(dim=-1, keepdim=True)
            torch.testing.assert_close(probability.cpu().double(), expected, rtol=3e-6, atol=3e-7)


def test_gdn_pool_layer_views_and_accepted_prefix_commit():
    if not os.environ.get("SGLANG_XPU_GGUF_GDN_SYCL_LIB"):
        pytest.skip("Build the separate precise GDN library")
    from sglang.kernels.ops.attention.gguf_gdn_sycl import GGUFGdnScratch, commit_gdn, verify_gdn
    from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
    from sglang.srt.mem_cache.memory_pool import MambaPool

    source = random((2, 3, 48, 128, 128), seed=98, scale=.1, dtype=torch.float32)
    scratch = GGUFGdnScratch.create(
        source,
        torch.empty((2, 82688), dtype=torch.float32, device="xpu"),
        torch.empty((2, 1), dtype=torch.int32, device="xpu"),
        torch.empty((2, 1, 4, 48, 128), dtype=torch.bfloat16, device="xpu"),
    )
    checkpoints = torch.empty((2, 1, 4, 48, 128, 128), dtype=torch.float32, device="xpu")
    cache = MambaPool.SpeculativeState(
        conv=[], temporal=source, intermediate_ssm=None,
        intermediate_conv_window=[], gguf_gdn_sycl=scratch,
    )
    expected_bytes = source.numel() * 4 + sum(t.numel() * t.element_size() for t in scratch.buffers)
    assert cache.mem_usage_bytes() == expected_bytes
    index = torch.tensor([1], dtype=torch.int32, device="xpu")
    cidx = torch.tensor([0], dtype=torch.int32, device="xpu")
    destination = torch.tensor([2], dtype=torch.int32, device="xpu")
    last = torch.tensor([2], dtype=torch.int32, device="xpu")
    cu = torch.tensor([0, 4], dtype=torch.int32, device="xpu")
    for layer in range(2):
        base = random((4, 10240), seed=100 + layer)
        q, k, v = base[:, :2048].view(1, 4, 16, 128), base[:, 2048:4096].view(1, 4, 16, 128), base[:, 4096:].view(1, 4, 48, 128)
        a, b = random((4, 48), seed=102), random((4, 48), seed=103)
        alog, bias = random((48,), seed=104, scale=.2, dtype=torch.float32), random((48,), seed=105)
        layer_cache = cache.at_layer_idx(layer)
        assert layer_cache.gguf_gdn_sycl.matches_source(source[layer])
        with torch.no_grad():
            gold = fused_sigmoid_gating_delta_rule_update(
                A_log=alog, a=a, dt_bias=bias, softplus_beta=1., softplus_threshold=20., q=q, k=k, v=v, b=b,
                initial_state_source=source[layer], initial_state_indices=index, use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu, disable_state_update=True, intermediate_states_buffer=checkpoints[layer],
                intermediate_state_indices=cidx, cache_steps=4, round_gdn_gate_to_bf16=True,
            )
            out = verify_gdn(
                layer_cache, q=q, k=k, v=v, a=a, b=b, A_log=alog, dt_bias=bias, indices=index,
                cu_seqlens=cu, steps=4, native_bf16_gdn=True, retrieve_parent_token=None, stable_rows=False,
            )
            torch.testing.assert_close(out.float(), gold.float(), rtol=.015625, atol=1e-5)
    expected = checkpoints[:, 0, 2].clone()
    commit_gdn(cache, destination, last, None, None, None)
    torch.testing.assert_close(source[:, 2], expected, rtol=2e-5, atol=2e-5)

    # Delta records are in-flight scratch, not slot-indexed RDMA state.
    pool = MambaPool.__new__(MambaPool)
    pool.mamba_cache = cache
    pool.mamba_layer_ids = [0, 1]
    pool.conv_slice_axis = 0
    pool._slot_siblings = []
    entries = list(pool._iter_transfer_state_entries())
    assert len(entries) == 2 and all(row[0] == "temporal" for row in entries)


def test_gdn_runtime_routing_selects_views_only_for_linear_verify(monkeypatch):
    from sglang.srt.layers.attention.linear import gdn_backend

    monkeypatch.setattr(gdn_backend, "pp_spec_stable_rows_enabled", lambda: False)
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=SimpleNamespace(gguf_gdn_sycl=object()))
        ),
        forward_metadata=SimpleNamespace(retrieve_parent_token=None),
        _use_strided_target_verify_qkv=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        spec_info=SimpleNamespace(draft_token_num=4),
    )
    gdn_backend.GDNAttnBackend._init_target_verify_qkv_routing(owner, batch)
    assert owner._use_strided_target_verify_qkv
    batch.spec_info.draft_token_num = 2
    with pytest.raises(ValueError):
        gdn_backend.GDNAttnBackend._init_target_verify_qkv_routing(owner, batch)
    batch.forward_mode.is_target_verify = lambda: False
    gdn_backend.GDNAttnBackend._init_target_verify_qkv_routing(owner, batch)
    assert not owner._use_strided_target_verify_qkv
