"""Batched GGUF GDN must match independent four-token request execution."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_GDN_SYCL_LIB"),
    reason="B70 GGUF GDN library required",
)


@pytest.mark.parametrize("batch_kernel", (False, True))
@pytest.mark.parametrize("tracking", (False, True))
@pytest.mark.parametrize("batch", range(1, 9))
@torch.inference_mode()
def test_batched_verify_commit_live_indices_and_accept_lengths(monkeypatch, batch, tracking, batch_kernel):
    from sglang.srt import runtime_context as runtime
    from sglang.kernels.ops.attention.gguf_gdn_sycl import (
        GGUFGdnScratch, allocate_scratch, checked, commit_gdn, load_library, verify_gdn,
    )

    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70")
    monkeypatch.setenv("SGLANG_XPU_GGUF_TP_EP", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_GDN_BATCH", "1" if batch_kernel else "0")
    monkeypatch.setattr(runtime, "get_spec", lambda: NS(
        speculative_algorithm="EAGLE", speculative_adaptive=False,
        speculative_eagle_topk=1,
    ))
    monkeypatch.setattr(runtime, "get_model", lambda: NS(quantization="gguf"))
    monkeypatch.setattr(runtime, "get_parallel", lambda: NS(tp_size=2, moe_ep_size=2, pp_size=1))
    monkeypatch.setattr(runtime, "get_schedule", lambda: NS(max_running_requests=batch))
    monkeypatch.setattr(runtime, "get_memory", lambda: NS(disable_radix_cache=True))
    torch.manual_seed(420 + batch)
    heads, key_heads = 24, 8
    width = 5 * key_heads * 128
    source = torch.zeros((36, 2 * batch + 2, heads, 128, 128), device="xpu")
    source[:2].normal_(std=0.1)
    scratch = allocate_scratch(source, 4, is_kda=False, replayssm=False, envelope=False)
    # Two real layer strides exercise the fused commit without running 36 copies
    # of identical arithmetic. The remaining pool layers must stay untouched.
    source = source[:2]
    scratch = GGUFGdnScratch.create(
        source, scratch.records[:2], scratch.source_index[:2], scratch.output[:2]
    )
    cache = NS(temporal=source, gguf_gdn_sycl=scratch)
    mixed = torch.randn((1, batch * 4, width), device="xpu").bfloat16() * 0.2
    q, k, v = mixed.split((key_heads * 128, key_heads * 128, heads * 128), -1)
    q, k = (x.view(1, batch * 4, key_heads, 128) for x in (q, k))
    v = v.view(1, batch * 4, heads, 128)
    gates = torch.randn((batch * 4, heads * 2), device="xpu").bfloat16()
    a, b = gates[:, heads:], gates[:, :heads]
    alog = torch.rand(heads, device="xpu") * 4 - 2
    bias = torch.rand(heads, device="xpu").bfloat16()
    indices = torch.arange(1, batch + 1, device="xpu", dtype=torch.int32)
    accepted = torch.arange(batch, device="xpu", dtype=torch.int32) % 4
    cu = torch.arange(batch + 1, device="xpu", dtype=torch.int32) * 4
    track = torch.arange(batch + 1, 2 * batch + 1, device="xpu", dtype=torch.int32) if tracking else None
    track_steps = accepted.clone() if tracking else None
    initial = source.clone()

    def run():
        outputs = []
        for layer in range(2):
            layer_cache = NS(temporal=source[layer], gguf_gdn_sycl=scratch.at_layer_idx(layer))
            outputs.append(verify_gdn(
                layer_cache, q=q, k=k, v=v, a=a, b=b, A_log=alog, dt_bias=bias,
                indices=indices, cu_seqlens=cu, steps=4, native_bf16_gdn=True,
                retrieve_parent_token=None, stable_rows=False,
            ))
        commit_gdn(cache, indices, accepted, track, track_steps, None)
        return outputs

    run()
    source.copy_(initial)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = run()

    lib = load_library()
    for replay in range(4):
        source.copy_(initial)
        indices.copy_(torch.roll(torch.arange(1, batch + 1, device="xpu", dtype=torch.int32), replay))
        accepted.copy_((torch.arange(batch, device="xpu", dtype=torch.int32) + replay) % 4)
        if tracking:
            track_steps.copy_((accepted - 1).clamp(min=-1))
        if replay == 3:
            indices[-1] = -1
            accepted[-1] = -1
        mixed.add_(0.0078125)
        gates.mul_(0.95)
        expected_state = initial.clone()
        expected_outputs = torch.empty((2, batch, 4, heads, 128), dtype=torch.bfloat16, device="xpu")
        # Independent single-request ABI calls are the maintained reference.
        records = torch.empty((2, batch, 41344), device="xpu")
        saved = torch.empty((2, batch), device="xpu", dtype=torch.int32)
        for layer in range(2):
            for request in range(batch):
                first = request * 4
                tensors = [q[:, first:first + 4], k[:, first:first + 4], v[:, first:first + 4],
                           a[first:first + 4], b[first:first + 4], alog, bias,
                           expected_state[layer], indices[request:request + 1], None, None,
                           expected_outputs[layer, request], records[layer, request],
                           saved[layer, request:request + 1]]
                checked(lib.nextn_gdn24_verify(
                    torch.xpu.current_stream().sycl_queue,
                    *[x.data_ptr() if x is not None else None for x in tensors],
                    4, 2, 0, expected_state.stride(1), 0, a.stride(0), b.stride(0),
                ), heads)
                if tracking:
                    tensors = [expected_state[layer], saved[layer, request:request + 1],
                               records[layer, request], expected_state[layer],
                               track[request:request + 1], track_steps[request:request + 1]]
                    checked(lib.nextn_gdn24_commit(
                        torch.xpu.current_stream().sycl_queue, *[x.data_ptr() for x in tensors],
                        4, 2, 1, 0, expected_state.stride(1), expected_state.stride(1), 0, 0, 0,
                    ), heads)
                tensors = [expected_state[layer], saved[layer, request:request + 1],
                           records[layer, request], expected_state[layer],
                           indices[request:request + 1], accepted[request:request + 1]]
                checked(lib.nextn_gdn24_commit(
                    torch.xpu.current_stream().sycl_queue, *[x.data_ptr() for x in tensors],
                    4, 2, 1, 0, expected_state.stride(1), expected_state.stride(1), 0, 0, 0,
                ), heads)
        graph.replay()
        for layer in range(2):
            assert torch.equal(actual[layer].view(torch.int16), expected_outputs[layer].view(1, batch * 4, heads, 128).view(torch.int16))
        assert torch.equal(source.view(torch.int32), expected_state.view(torch.int32))
