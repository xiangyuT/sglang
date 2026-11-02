# SPDX-License-Identifier: Apache-2.0
"""FutureMap byte parity, dynamic indices and publish-fenced cross-stream use."""
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.gguf_relay_sycl import try_gather, try_stash
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_RELAY_LIB"),
    reason="Requires B70 and the built GGUF relay library",
)


def owner(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_GGUF_RELAY_SYCL", "1")
    pool = SimpleNamespace(req_to_token=torch.zeros((2, 64), dtype=torch.int32, device="xpu"))
    result = FutureMap(torch.device("xpu"), SpeculativeAlgorithm.EAGLE, pool)
    result.need_topk = result.need_hidden_states = True
    result.topk_p_buf = torch.full((2, 1), -9., dtype=torch.float32, device="xpu")
    result.topk_index_buf = torch.full((2, 1), -9, dtype=torch.int64, device="xpu")
    result.output_tokens_buf.fill_(-9)
    result.new_seq_lens_buf.fill_(-1)
    result.hidden_states_buf = torch.full((2, 10240), -9., dtype=torch.bfloat16, device="xpu")
    return result


def payload(dtype, seed):
    return RelayPayload(
        bonus_tokens=torch.tensor([1234 + seed], dtype=dtype, device="xpu"),
        topk_p=torch.tensor([[.25 + seed / 64]], dtype=torch.float32, device="xpu"),
        topk_index=torch.tensor([[4321 + seed]], dtype=dtype, device="xpu"),
        hidden_states=torch.randn((1, 10240), generator=torch.Generator().manual_seed(seed)).bfloat16().to("xpu"),
    )


def equal_bits(actual, expected):
    dtype = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.int64: torch.int64}[actual.dtype]
    assert torch.equal(actual.detach().cpu().contiguous().view(dtype), expected.detach().cpu().contiguous().view(dtype))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_stash_gather_and_other_slot_preservation(monkeypatch, dtype, index_dtype):
    relay = owner(monkeypatch)
    for seed, slot in [(3, 1), (7, 0), (11, -1)]:
        values = payload(dtype, seed)
        index = torch.tensor([slot], dtype=index_dtype, device="xpu")
        fields = [relay.output_tokens_buf, relay.topk_p_buf, relay.topk_index_buf, relay.hidden_states_buf]
        before = [x.clone() for x in fields]
        assert try_stash(relay, index, values)
        gathered = try_gather(relay, index)
        for actual, expected in zip(gathered, [values.topk_p, values.topk_index.long(), values.bonus_tokens.long(), values.hidden_states]):
            equal_bits(actual, expected)
        other = 1 - (slot % 2)
        for actual, expected in zip(fields, before):
            equal_bits(actual[other], expected[other])


def test_graph_replay_reads_current_sources_and_index(monkeypatch):
    relay = owner(monkeypatch)
    values = payload(torch.int64, 17)
    index = torch.ones(1, dtype=torch.int64, device="xpu")
    assert try_stash(relay, index, values)
    try_gather(relay, index)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        assert try_stash(relay, index, values)
        outputs = try_gather(relay, index)
    for step, slot in enumerate((0, 1, 0, 1)):
        index.fill_(slot)
        values.bonus_tokens.add_(1)
        values.topk_index.add_(3)
        values.topk_p.add_(.125)
        values.hidden_states.add_(.25)
        graph.replay()
        for actual, expected in zip(outputs, [values.topk_p, values.topk_index, values.bonus_tokens, values.hidden_states]):
            equal_bits(actual, expected)


def test_private_length_copy_and_future_map_cross_streams(monkeypatch):
    relay = owner(monkeypatch)
    assert relay.new_seq_lens_cpu_pinned.is_pinned()
    assert relay.fwd_prepare_d2h_stream is not None
    producer, consumer = torch.xpu.Stream(), torch.xpu.Stream()
    producer.wait_stream(torch.xpu.current_stream())
    for seed, slot in [(19, 1), (23, 0), (29, 1), (31, 0)]:
        values = payload(torch.int64, seed)
        index = torch.tensor([slot], dtype=torch.int64, device="xpu")
        length = torch.tensor([1024 + seed], dtype=torch.int64, device="xpu")
        producer.wait_stream(torch.xpu.current_stream())
        done = torch.xpu.Event()
        with torch.xpu.stream(producer):
            relay.publish(index, length)
            relay.stash(index, values)
            done.record()
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=index, future_dsa_topk_indices_available=False),
            req_pool_indices_cpu=torch.tensor([slot], dtype=torch.int64),
        )
        with torch.xpu.stream(consumer):
            relay.resolve_seq_lens_cpu(batch)
            assert batch.seq_lens_cpu.tolist() == [1024 + seed]
            assert batch.seq_lens_sum == 1024 + seed
            consumer.wait_event(done)
            relay._resolve_spec_extras(batch)
            outputs = [batch.spec_info.topk_p, batch.spec_info.topk_index,
                       batch.spec_info.bonus_tokens, batch.spec_info.hidden_states]
        consumer.synchronize()
        for actual, expected in zip(outputs, [values.topk_p, values.topk_index, values.bonus_tokens, values.hidden_states]):
            equal_bits(actual, expected)


def test_fallback_keeps_partial_staging_and_optional_buffers(monkeypatch):
    relay = owner(monkeypatch)
    index = torch.ones(1, dtype=torch.int64, device="xpu")
    values = payload(torch.int64, 37)
    values.hidden_states = None
    assert not try_stash(relay, index, values)
    relay.draft_probs_buf = torch.empty((2, 4), device="xpu")
    assert try_gather(relay, index) is None
    relay.draft_probs_buf = None
    assert try_gather(relay, index.expand(2)) is None
    monkeypatch.setenv("SGLANG_XPU_GGUF_RELAY_SYCL", "0")
    assert try_gather(relay, index) is None
