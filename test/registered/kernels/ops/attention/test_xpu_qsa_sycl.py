"""Physical-slot, numerical and graph checks for the BF16 SYCL QSA adapter."""

import os

import pytest
import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import available, sparse_attention

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_QSA_SYCL_LIB"),
    reason="requires the separately built B70 QSA SYCL library",
)


@pytest.mark.parametrize("queries,width", [(4, 129), (37, 129), (1024, 2112)])
@pytest.mark.parametrize("batch", [8, 32, 64])
@pytest.mark.parametrize("skip_empty", ["0", "1"])
def test_bounded_prefill_queries_preserve_all_bits(monkeypatch, queries, width, batch, skip_empty):
    torch.manual_seed(73)
    q = torch.randn(queries, 24, 256, device="xpu", dtype=torch.bfloat16) * .1
    # Padded physical rows exercise different K and V leading strides.
    k = (torch.randn(269, 3, 256, device="xpu", dtype=torch.bfloat16) * .1)[:, :2]
    v = (torch.randn(269, 4, 256, device="xpu", dtype=torch.bfloat16) * .1)[:, :2]
    assert k.stride(0) == 768 and v.stride(0) == 1024
    slots = torch.randint(-3, 273, (queries, width), device="xpu", dtype=torch.int32)
    slots[0].fill_(-1)
    slots[-1].fill_(-1)
    slots[-1, -1] = 7
    monkeypatch.setenv("SGLANG_XPU_QSA_PREFILL_QUERY_BATCH", "0")
    monkeypatch.setenv("SGLANG_XPU_QSA_SKIP_EMPTY_CHUNKS", "0")
    reference = sparse_attention(q, k, v, slots)
    monkeypatch.setenv("SGLANG_XPU_QSA_PREFILL_QUERY_BATCH", str(batch))
    monkeypatch.setenv("SGLANG_XPU_QSA_SKIP_EMPTY_CHUNKS", skip_empty)
    output = sparse_attention(q, k, v, slots)
    assert torch.equal(reference.view(torch.int16), output.view(torch.int16))
    if queries == 37:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            captured = sparse_attention(q, k, v, slots)
        q.add_(.03)
        slots[-1, -1] = 12
        expected = sparse_attention(q, k, v, slots)
        graph.replay()
        assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("queries,length", [(37, 37), (128, 65), (1024, 1001)])
def test_whole_padding_chunk_trim_preserves_merge_bits(monkeypatch, queries, length):
    from sglang.kernels.ops.attention.xpu_qsa_sycl import trim_compacted_prefill_indices

    torch.manual_seed(117)
    q = torch.randn(queries, 24, 256, device="xpu", dtype=torch.bfloat16) * .1
    k = torch.randn(269, 2, 256, device="xpu", dtype=torch.bfloat16) * .1
    v = torch.randn_like(k)
    slots = torch.full((queries, 2111), -1, dtype=torch.int32, device="xpu")
    for row in range(queries):
        count = row * length // max(1, queries - 1)
        slots[row, :count] = torch.randint(0, 269, (count,), device="xpu", dtype=torch.int32)
    monkeypatch.setenv("SGLANG_XPU_QSA_PREFILL_QUERY_BATCH", "32")
    monkeypatch.setenv("SGLANG_XPU_QSA_SKIP_EMPTY_CHUNKS", "1")
    reference = sparse_attention(q, k, v, slots)
    trimmed = trim_compacted_prefill_indices(slots, length)
    output = sparse_attention(q, k, v, trimmed)
    assert torch.equal(reference.view(torch.int16), output.view(torch.int16))


@pytest.mark.parametrize("rows", [0, 1, 128])
@pytest.mark.parametrize("dtype,columns", [(torch.bfloat16, 128), (torch.int64, 3)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_index_cache_store_keeps_last_source_and_graph_updates(
    rows, dtype, columns, index_dtype
):
    from sglang.kernels.ops.attention.xpu_qsa_sycl import try_store_rows

    cpu = torch.arange(max(rows, 1) * columns).reshape(max(rows, 1), columns).to(dtype)
    # MRoPE positions arrive transposed, and one-dimensional positions expand
    # with a zero column stride. Both must copy values without materialization.
    if dtype == torch.int64 and rows == 1:
        source = cpu[:rows, :1].to("xpu").expand(-1, columns)
    else:
        source = cpu[:rows].T.contiguous().to("xpu").T
    locations = (torch.arange(rows) % 4 + 4).to(index_dtype)
    if rows:
        locations[-1] = -1
    loc = locations.to("xpu")
    backing = torch.full((18, columns), 11, dtype=dtype, device="xpu")
    output = backing[1:-1]

    def reference():
        expected = output.cpu()
        values, indices = source.cpu(), loc.cpu().tolist()
        for row, destination in enumerate(indices):
            expected[destination] = values[row]
        return expected

    expected = reference()
    assert try_store_rows(output, loc, source)
    assert torch.equal(output.cpu(), expected)
    if rows:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            assert try_store_rows(output, loc, source)
        for offset in (1, 7, -3):
            loc.copy_(((torch.arange(rows) + offset) % 16).to(index_dtype))
            expected = reference()
            graph.replay()
            assert torch.equal(output.cpu(), expected)
    assert bool((backing[0] == 11).all() and (backing[-1] == 11).all())
    assert not try_store_rows(output, loc[:1], output[:1])


def test_qsa_pool_setters_keep_duplicate_ring_and_position_semantics():
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool._transfer_full_attention_id = lambda layer: 0
    pool._wait_for_layer = lambda layer: None
    pool.qsa_key_state_buffer_pool = [
        torch.zeros((20, 1, 128), dtype=torch.bfloat16, device="xpu")
    ]
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros((20, 1, 128), dtype=torch.bfloat16, device="xpu")
    ]
    pool.qsa_rope_position_buffer = torch.zeros(
        (20, 3), dtype=torch.int64, device="xpu"
    )
    locations = torch.arange(128) % 4 + 4
    values = torch.arange(128 * 128).reshape(128, 1, 128).bfloat16()
    positions = torch.stack(
        (torch.arange(128), torch.arange(128) + 11, torch.arange(128) + 29)
    )
    loc, source, pos = [v.to("xpu") for v in (locations, values, positions)]
    pool.set_qsa_key_state_buffer(0, loc, source)
    pool.set_qsa_compressed_k_buffer(0, loc, source)
    pool.set_qsa_rope_position_buffer(loc, pos)
    expected = torch.zeros((20, 1, 128), dtype=torch.bfloat16)
    expected_pos = torch.zeros((20, 3), dtype=torch.int64)
    for row, target in enumerate(locations.tolist()):
        expected[target] = values[row]
        expected_pos[target] = positions[:, row]
    assert torch.equal(pool.qsa_key_state_buffer_pool[0].cpu(), expected)
    assert torch.equal(pool.qsa_compressed_k_buffer_pool[0].cpu(), expected)
    assert torch.equal(pool.qsa_rope_position_buffer.cpu(), expected_pos)


@pytest.mark.parametrize("ratio,tokens", [(2, 133), (4, 2048), (4, 8192), (8, 133)])
@pytest.mark.parametrize("rows", [0, 1, 4, 128])
def test_expansion_matches_stable_torch_compaction(ratio, tokens, rows):
    from sglang.kernels.ops.attention.xpu_qsa_sycl import can_expand_blocks
    from sglang.srt.layers.attention.qsa.kernel import (
        expand_qsa_block_indices,
        torch_expand_qsa_block_indices,
    )

    generator = torch.Generator().manual_seed(42)
    blocks = torch.randint(
        -3,
        1000,
        (rows, (tokens + ratio - 1) // ratio),
        generator=generator,
        dtype=torch.int32,
    )
    positions = torch.randint(-5, 4000, (rows,), generator=generator)
    lengths = torch.randint(0, 4001, (rows,), generator=generator)
    if rows:
        blocks[0].fill_(-1)
        lengths[0] = 0
        if rows > 1:
            blocks[1, 1] = 2147483647
    expected = torch_expand_qsa_block_indices(blocks, positions, lengths, ratio, tokens)
    bx, px, lx = [value.to("xpu") for value in (blocks, positions, lengths)]
    assert can_expand_blocks(bx, px, lx, ratio, tokens)
    actual = expand_qsa_block_indices(bx, px, lx, ratio, tokens)
    assert torch.equal(actual.cpu(), expected)
    if rows == 0:
        return
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = expand_qsa_block_indices(bx, px, lx, ratio, tokens)
    for shift in (1, -2, 17):
        changed_blocks = (blocks + shift).clamp_min(-1)
        changed_positions = positions + shift
        changed_lengths = (lengths + shift).clamp_min(0)
        bx.copy_(changed_blocks)
        px.copy_(changed_positions)
        lx.copy_(changed_lengths)
        graph.replay()
        expected = torch_expand_qsa_block_indices(
            changed_blocks, changed_positions, changed_lengths, ratio, tokens
        )
        assert torch.equal(actual.cpu(), expected)


def _reference(q, k, v, slots):
    result = []
    for row in range(q.shape[0]):
        selected = slots[row][slots[row] >= 0].long()
        if selected.numel() == 0:
            result.append(torch.zeros_like(q[row]))
            continue
        keys = k[selected].double().repeat_interleave(12, dim=1)
        values = v[selected].double().repeat_interleave(12, dim=1)
        score = torch.einsum("hd,khd->hk", q[row].double(), keys) * 0.0625
        result.append(torch.einsum("hk,khd->hd", score.softmax(-1), values).to(q.dtype))
    return torch.stack(result)


def _check(actual, expected):
    error = (actual.float() - expected.float()).abs()
    assert (error <= (expected.float().abs() * 0.008).clamp_min(1e-5)).all()
    assert error.norm() / expected.float().norm().clamp_min(1e-20) < 0.001


@pytest.mark.parametrize(
    "batch,width",
    [(1, 1), (1, 63), (1, 64), (1, 65), (1, 2048), (4, 129), (8, 513), (128, 65)],
)
def test_bf16_sparse_physical_slots(batch, width):
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(batch, 24, 256, generator=generator).bfloat16()
    k = torch.randn(2307, 2, 256, generator=generator).bfloat16()
    v = torch.randn(2307, 2, 256, generator=generator).bfloat16()
    slots = torch.randint(
        0, 2307, (batch, width), generator=generator, dtype=torch.int32
    )
    slots[:, 1::7] = -1
    if batch > 1:
        slots[-1].fill_(-1)
    expected = _reference(q, k, v, slots)
    # K/V have different padded token strides, as opposed to compact gathers.
    kx = torch.empty((2307, 3, 256), dtype=torch.bfloat16, device="xpu")[:, :2]
    vx = torch.empty((2307, 4, 256), dtype=torch.bfloat16, device="xpu")[:, :2]
    kx.copy_(k)
    vx.copy_(v)
    actual = sparse_attention(q.to("xpu"), kx, vx, slots.to("xpu")).cpu()
    _check(actual, expected)
    if batch > 1:
        assert not actual[-1].count_nonzero()


def test_bf16_qsa_graph_updates_every_input():
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, 24, 256, generator=generator).bfloat16()
    k = torch.randn(301, 2, 256, generator=generator).bfloat16()
    v = torch.randn(301, 2, 256, generator=generator).bfloat16()
    slots = torch.randint(0, 301, (1, 129), generator=generator, dtype=torch.int32)
    qx, kx, vx, sx = [t.to("xpu") for t in (q, k, v, slots)]
    for _ in range(3):
        sparse_attention(qx, kx, vx, sx)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = sparse_attention(qx, kx, vx, sx)
    for _ in range(3):
        for cpu, gpu in ((q, qx), (k, kx), (v, vx)):
            cpu.copy_(torch.randn(cpu.shape, generator=generator))
            gpu.copy_(cpu)
        slots.copy_(torch.randint(0, 301, slots.shape, generator=generator))
        slots[:, 1::3] = -1
        sx.copy_(slots)
        graph.replay()
        _check(output.cpu(), _reference(q, k, v, slots))
    assert available(qx, kx, vx, sx)
    assert not available(qx, kx, vx, sx.long())
    with pytest.raises(ValueError, match="scale"):
        sparse_attention(qx, kx, vx, sx, float("nan"))


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("width", [1, 63, 64, 65, 1024])
def test_local_tp_heads_match_full_native_graph(tp, width, monkeypatch):
    """Head partitions, replicated KV and dynamic masking preserve native bits."""
    monkeypatch.setenv("SGLANG_XPU_GGUF_TP_EP", "1")
    rows, pool, heads = 4, 1031, 24 // tp
    q = torch.zeros((rows, 24, 256), device="xpu", dtype=torch.bfloat16)
    k = torch.zeros((pool, 3, 256), device="xpu", dtype=torch.bfloat16)[:, :2]
    v = torch.zeros_like(k)
    slots = torch.zeros((rows, width), device="xpu", dtype=torch.int32)
    local = []
    for rank in range(tp):
        kv = rank * heads // 12
        ql = q[:, rank * heads:(rank + 1) * heads].contiguous()
        kl, vl = k[:, kv:kv + 1], v[:, kv:kv + 1]
        assert available(ql, kl, vl, slots)
        sparse_attention(ql, kl, vl, slots)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            output = sparse_attention(ql, kl, vl, slots)
        local.append((ql, graph, output))
    for seed in (3, 7, 11):
        torch.manual_seed(seed)
        q.normal_(); k.normal_(); v.normal_()
        values = torch.randint(-2, pool + 3, (rows, width), dtype=torch.int32)
        values[:, ::7] = -1
        values[-1].fill_(-1)
        slots.copy_(values)
        full = sparse_attention(q, k, v, slots).cpu().view(torch.int16)
        for rank, (ql, graph, output) in enumerate(local):
            ql.copy_(q[:, rank * heads:(rank + 1) * heads])
            graph.replay()
            assert torch.equal(output.cpu().view(torch.int16), full[:, rank * heads:(rank + 1) * heads])
