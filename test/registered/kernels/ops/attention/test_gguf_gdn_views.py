"""Verify projection views preserve convolution and every rollback prefix."""
import pytest
import torch

from sglang.kernels.ops.attention.gguf_gdn_views import try_verify_views
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="B70 required")


@torch.inference_mode()
@pytest.mark.parametrize("tp", (1, 2, 4))
def test_strided_verify_convolution_and_rollback(monkeypatch, tp):
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 bounded GDN views")
    monkeypatch.setenv("SGLANG_XPU_GGUF_GDN_VERIFY_VIEWS", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_GDN_SYCL", "1")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    qkvz = torch.randn(4, (16384 // tp)).bfloat16().to("xpu")
    ba = torch.randn(4, (96 // tp)).bfloat16().to("xpu")
    initial = torch.randn(2, (10240 // tp), 3).bfloat16().to("xpu")
    weight = torch.randn((10240 // tp), 4).bfloat16().to("xpu")
    indices = torch.ones(1, device="xpu", dtype=torch.int32)
    metadata = dict(is_verify=True, key_heads=16 // tp, value_heads=(48 // tp), head_dim=128)
    expected = None
    for use_views in (False, True):
        state = initial.clone()
        middle = torch.empty(2, 4, (10240 // tp), 3, dtype=torch.bfloat16, device="xpu")
        def run():
            if use_views:
                mixed, z, b, a = try_verify_views(qkvz, ba, **metadata)
                assert mixed.data_ptr() == qkvz.data_ptr()
                assert b.stride(0) == a.stride(0) == (96 // tp)
            else:
                q, k, v, z = qkvz.split([(2048 // tp), (2048 // tp), (6144 // tp), (6144 // tp)], dim=-1)
                mixed = torch.cat((q, k, v), dim=-1)
            return causal_conv1d_update(
                mixed.view(1, 4, (10240 // tp)).transpose(1, 2), state, weight,
                activation="silu", conv_state_indices=indices,
                intermediate_conv_window=middle, intermediate_state_indices=indices,
                fp32_products=True,
            )
        run()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            output = run()
        graph.replay()
        current = [output.cpu(), state.cpu(), middle[1].cpu()]
        if expected is None:
            expected = current
        else:
            for a, b in zip(current, expected):
                assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert try_verify_views(qkvz, ba, **dict(metadata, is_verify=False)) is None
    assert try_verify_views(qkvz[:1], ba[:1], **metadata) is None
    monkeypatch.delenv("SGLANG_XPU_GGUF_GDN_VERIFY_VIEWS")
    assert try_verify_views(qkvz, ba, **metadata) is None


@torch.inference_mode()
@pytest.mark.parametrize("tp", (1, 2, 4))
def test_gated_norm_reads_three_dimensional_z_without_copy(tp):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 bounded GDN views")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    source = torch.randn(4, (16384 // tp)).bfloat16().to("xpu")
    z = source[:, (10240 // tp):].view(4, (48 // tp), 128)
    x = torch.randn((192 // tp), 128).bfloat16().to("xpu")
    norm = RMSNorm(128, eps=1e-6, device="xpu", dtype=torch.bfloat16)
    norm.requires_grad_(False)
    norm(x, z)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = norm(x, z)
    for seed in (3, 7, 11):
        torch.manual_seed(seed)
        source.normal_()
        x.normal_()
        graph.replay()
        expected = norm(x, z.reshape((192 // tp), 128))
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
