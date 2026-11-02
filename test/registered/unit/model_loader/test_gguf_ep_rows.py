"""Partition ownership and zero dummy rows without copying other ranks' data."""
import pytest
import torch
from types import SimpleNamespace

from sglang.srt.layers.quantization import gguf_ep
from sglang.srt.layers.quantization.gguf_ep import (
    ExpertPartitionRows,
    PartitionedGGUFMoEMethod,
    ep_cache_capacity,
)
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Source:
    width, row_bytes, kind, rows = 64, 18, 42, 32
    data = torch.arange(32 * 18, dtype=torch.int64).remainder(256).to(torch.uint8).view(32, 18)

    def packed_row_range(self, start, stop, *, pin_memory=False):
        return self.data[start:stop].clone()


@pytest.mark.parametrize("start,count", [(0, 4), (4, 4), (0, 2), (2, 2), (4, 2), (6, 2)])
def test_partition_rows_and_zero_dummy(start, count):
    source = Source()
    part = ExpertPartitionRows(source, start=start, count=count, total=8)
    assert part.rows == (count + 1) * 4
    actual = part.packed_row_range(0, part.rows)
    assert torch.equal(actual[:count * 4], source.data[start * 4:(start + count) * 4])
    assert torch.count_nonzero(actual[count * 4:]) == 0
    assert torch.count_nonzero(part.packed_row_range(count * 4, part.rows)) == 0
    assert part.packed_row_range(part.rows, part.rows).shape == (0, 18)
    for first, last in [(-1, 0), (0, part.rows + 1), (3, 2)]:
        with pytest.raises(ValueError, match="bounds"):
            part.packed_row_range(first, last)


@pytest.mark.parametrize("start,count,total", [(0, 0, 8), (8, 1, 8), (0, 3, 7)])
def test_invalid_partition(start, count, total):
    with pytest.raises(ValueError, match="partition"):
        ExpertPartitionRows(Source(), start=start, count=count, total=total)


@pytest.mark.parametrize("world,budget", [(2, None), (2, "128"), (4, None), (4, "64")])
def test_partition_budget_reuses_compressed_pager(monkeypatch, world, budget):
    import sglang.srt.runtime_context as runtime

    monkeypatch.setattr(runtime, "get_parallel", lambda: SimpleNamespace(
        moe_ep_size=world, moe_ep_rank=world - 1, moe_tp_size=1
    ))
    if budget is None:
        monkeypatch.delenv("SGLANG_XPU_GGUF_EP_CACHE_SLOTS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_XPU_GGUF_EP_CACHE_SLOTS", budget)
    real = 512 // world
    method = PartitionedGGUFMoEMethod(288, decode_q8=False)
    method.create_weights(SimpleNamespace(), real, 2560, 640, torch.bfloat16)
    assert isinstance(method, CompressedGGUFMoEMethod)
    assert method.num_experts == real + 1
    assert method.cache_slots == (real + 1 if budget is None else int(budget))
    assert method.fully_resident == (budget is None)
    assert method.real_experts == real
    # Weight sources are attached during loading, after create_weights. Even a
    # full-capacity partition must reject growth before those sources exist.
    with pytest.raises(ValueError, match="bound sources"):
        method.grow_cache(None, real + 1)


@pytest.mark.parametrize("budget", ["39", "258", "garbage", "128.5"])
def test_invalid_ep_cache_budget_fails_closed(budget):
    with pytest.raises(ValueError, match="GGUF EP cache slots"):
        ep_cache_capacity(256, budget)


def test_paged_partition_maps_nonlocal_ids_to_the_dummy(monkeypatch):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    method = PartitionedGGUFMoEMethod(128)
    method.real_experts = 256
    method._resident_preparation = None
    ids = torch.tensor([[0, -1, 255, -7]], dtype=torch.int32)
    weights = torch.tensor([[.1, .2, .3, .4]])
    dispatch = StandardDispatchOutput(
        torch.zeros((1, 2560)), None, StandardTopKOutput(weights, ids, None)
    )
    monkeypatch.setattr(CompressedGGUFMoEMethod, "apply", lambda self, layer, x: x)
    actual = method.apply(None, dispatch)
    assert torch.equal(actual.topk_output.topk_ids, torch.tensor([[0, 256, 255, 256]]))
    assert actual.topk_output.topk_weights is weights
    assert torch.equal(ids, torch.tensor([[0, -1, 255, -7]], dtype=torch.int32))
    assert not hasattr(method, "_ep_zero_slot")


def test_full_resident_import_is_an_alias():
    assert gguf_ep.ResidentEPGGUFMoEMethod is PartitionedGGUFMoEMethod


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
