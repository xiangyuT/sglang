"""Original compressed output rows, independently partitioned fused segments."""
from types import SimpleNamespace
import pytest
import torch
from sglang.srt.layers.quantization.gguf_iq4_linear import partition_dense_rows

@pytest.mark.parametrize("tp", [2, 4])
def test_gdn_output_segments_reassemble_without_changing_source_bytes(tp):
    config = SimpleNamespace(linear_num_key_heads=16, linear_num_value_heads=48,
                             linear_key_head_dim=128, linear_value_head_dim=128)
    sizes = [2048, 2048, 6144, 6144]
    full = torch.arange(sum(sizes) * 3).reshape(-1, 3)
    parts = []
    for rank in range(tp):
        module = SimpleNamespace(tp_group=SimpleNamespace(rank_in_group=rank, world_size=tp))
        parts.append(partition_dense_rows(module, full, "merged_qkvz", config).split([n // tp for n in sizes]))
    restored = torch.cat([torch.cat([parts[rank][segment] for rank in range(tp)]) for segment in range(4)])
    assert torch.equal(restored, full)

@pytest.mark.parametrize("part", [1, 2])
def test_tp4_kv_heads_replicate_in_pairs(part):
    full = torch.arange(512 * 3).reshape(512, 3)
    shards = []
    for rank in range(4):
        module = SimpleNamespace(tp_group=SimpleNamespace(rank_in_group=rank, world_size=4),num_kv_head_replicas=2)
        shards.append(partition_dense_rows(module, full, "attention_qkv", None, part=part))
    assert torch.equal(shards[0], shards[1])
    assert torch.equal(shards[2], shards[3])
    assert torch.equal(torch.cat([shards[0], shards[2]]), full)
