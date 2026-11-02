"""Cold mirrors shrink after capture without losing swapped expert bytes."""
import os
import weakref

import pytest
import torch

from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache, make_host_expert_aos

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.getenv("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="B70 native GGUF library required",
)


@pytest.mark.parametrize("layout", ["fields", "expert_aos"])
@torch.inference_mode()
def test_fragmented_mirrors_shrink_to_zero_after_captured_growth(layout):
    torch.manual_seed(347)
    sizes = (460800,)*3
    original = tuple(torch.randint(256, (12,n), dtype=torch.uint8) for n in sizes)
    mirrors = tuple(t[8:].pin_memory() for t in original)
    if layout == "expert_aos":
        mirrors = make_host_expert_aos(mirrors)
    state = GPUExpertCache(42,range(8),tuple(t[:8].to("xpu") for t in original),
                           range(8,12),mirrors,layout="original",host_layout=layout)
    del mirrors
    try:
        ids = torch.tensor([[10,8,3,1]],device="xpu",dtype=torch.int32)
        state.prepare(ids)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            routes = state.prepare(ids)
        table_pointer = state.mirror_addresses.data_ptr()
        for count in (2,2):
            cold = [e for e,s in enumerate(state.expert_slots.cpu().tolist()) if s<0]
            prior = [weakref.ref(t) for t in state.mirrors]
            state.grow(tuple(torch.empty((count,n),dtype=torch.uint8,device="xpu") for n in sizes),cold[:count])
            assert all(ref() is None for ref in prior)
            assert state.mirror_addresses.data_ptr() == table_pointer
            remaining = len(cold)-count
            assert all(t.shape[0]==remaining for t in state.mirrors)
            owned = sum({t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in state.mirrors}.values())
            assert owned == remaining*sum(sizes)
            assert state.compact_mirrors()["released_storage_bytes"] == 0
            for row in ([0,4,9,11],[2,5,7,10],[1,6,8,11]):
                ids.copy_(torch.tensor([row],device="xpu",dtype=torch.int32))
                graph.replay()
                assert routes.error.item()==0
                # All logical experts, including untouched cold rows, must retain
                # their original bytes after arbitrary post-compaction exchanges.
                staged = tuple(torch.empty_like(t,device="xpu") for t in original)
                state.stage_original(range(12),staged)
                assert all(torch.equal(a.cpu(),b) for a,b in zip(staged,original))
        del graph
    finally:
        state.close()


@torch.inference_mode()
def test_ep_growth_updates_device_capacity_and_preserves_captured_tables(monkeypatch):
    from types import SimpleNamespace
    from sglang.srt.layers.quantization.gguf_ep import PartitionedGGUFMoEMethod
    from sglang.srt.layers.quantization.gguf_cache_capacity import (
        grow_model_expert_caches,
    )

    monkeypatch.delenv("SGLANG_XPU_GGUF_EXPERT_PROFILE", raising=False)
    torch.manual_seed(328)
    sizes = (460800,) * 3
    source = tuple(torch.randint(256, (12, n), dtype=torch.uint8) for n in sizes)
    buffers = tuple(t[:8].to("xpu") for t in source)
    state = GPUExpertCache(
        42,
        range(8),
        buffers,
        range(8, 12),
        tuple(t[8:].pin_memory() for t in source),
        layout="original",
    )
    method = PartitionedGGUFMoEMethod(8)
    method.num_experts, method.real_experts = 12, 11
    method.sources = (SimpleNamespace(kind=42),) * 3
    method._gpu_cache = state
    method.fully_resident = False
    layer = SimpleNamespace(
        quant_method=method,
        gguf_gate=buffers[0],
        gguf_up=buffers[1],
        gguf_down=buffers[2],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(gguf_model_path="fixture.gguf"), modules=lambda: [layer]
    )
    ids = torch.tensor([[10, 8, 3, 1]], device="xpu", dtype=torch.int32)
    state.prepare(ids)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        routes = state.prepare(ids)
    pointers = [
        x.data_ptr()
        for x in (state.capacity, state.slot_addresses, state.mirror_addresses)
    ]
    try:
        for capacity in (10, 12):
            receipt = grow_model_expert_caches(model, capacity)
            assert receipt["added_bytes"] == 2 * sum(sizes)
            assert method.cache_slots == state.capacity.item() == capacity
            assert method.fully_resident == (capacity == 12)
            assert state.owned_resident_storage_bytes() == capacity * sum(sizes)
            assert pointers == [
                x.data_ptr()
                for x in (state.capacity, state.slot_addresses, state.mirror_addresses)
            ]
            graph.replay()
            assert routes.error.item() == 0
            staged = tuple(torch.empty_like(t, device="xpu") for t in source)
            state.stage_original(range(12), staged)
            assert all(torch.equal(x.cpu(), y) for x, y in zip(staged, source))
        with pytest.raises(ValueError):
            method.grow_cache(layer, 13)
        with pytest.raises(ValueError):
            method.grow_cache(layer, 11)
    finally:
        del graph
        state.close()


@torch.inference_mode()
def test_growth_receipt_rejects_silent_partition_noop(monkeypatch):
    from types import SimpleNamespace
    from sglang.srt.layers.quantization.gguf_ep import PartitionedGGUFMoEMethod
    from sglang.srt.layers.quantization.gguf_cache_capacity import (
        grow_model_expert_caches,
    )

    method = PartitionedGGUFMoEMethod(8)
    method.num_experts = 12
    method.sources = (object(),)
    method._gpu_cache = None
    layer = SimpleNamespace(quant_method=method)
    model = SimpleNamespace(
        config=SimpleNamespace(gguf_model_path="fixture.gguf"), modules=lambda: [layer]
    )
    monkeypatch.setattr(method, "grow_cache", lambda *_: 0)
    with pytest.raises(RuntimeError, match="requested 10 slots"):
        grow_model_expert_caches(model, 10)
    assert not hasattr(model, "gguf_cache_growth_receipt")
