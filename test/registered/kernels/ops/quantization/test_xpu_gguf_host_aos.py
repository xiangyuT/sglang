# SPDX-License-Identifier: Apache-2.0
"""Byte ownership across native swaps, growth and graph replay with host AoS."""
import os

import pytest
import torch

from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache, make_host_expert_aos, empty_expert_aos


def require_device():
    if not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"):
        pytest.skip("Requires native B70 GGUF cache")
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70-only primitive")


@pytest.mark.parametrize("resident_aos", [False, True])
def test_aos_byte_ownership_growth_graph(resident_aos):
    require_device()
    sizes = (704000, 704000, 921600)
    torch.manual_seed(42)
    raw = [torch.randint(256, (48, size), dtype=torch.uint8) for size in sizes]
    if resident_aos:
        resident = empty_expert_aos(40, sizes, "xpu")
        for field, original in zip(resident, raw):
            field.copy_(original[:40])
    else:
        resident = [x[:40].xpu() for x in raw]
    host = make_host_expert_aos([x[40:].contiguous() for x in raw])
    assert len({x.untyped_storage().data_ptr() for x in host}) == 1
    assert host[0].untyped_storage().nbytes() == 8 * sum(sizes)
    cache = GPUExpertCache(21, range(40), resident, range(40, 48), host,
                           layout="original", down_kind=20, host_layout="expert_aos",
                           resident_layout="expert_aos" if resident_aos else "fields")
    ids = torch.tensor([[40, 41, 42]], dtype=torch.int32, device="xpu")
    staging = [torch.empty((48, size), dtype=torch.uint8, device="xpu") for size in sizes]
    def check_all():
        cache.stage_original(range(48), staging)
        for expected, actual in zip(raw, staging):
            assert torch.equal(expected, actual.cpu())
        cache.residency_snapshot()
    try:
        cache.prepare(ids)
        check_all()
        missing = torch.nonzero(cache.expert_slots.cpu() < 0).flatten().tolist()[:4]
        extra = (empty_expert_aos(4, sizes, "xpu") if resident_aos else
                 [torch.empty((4, size), dtype=torch.uint8, device="xpu") for size in sizes])
        cache.grow(extra, missing)
        check_all()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            cache.prepare(ids)
        for values in ([43, 44, 45], [0, 1, 2], [46, 47, 3]):
            ids.copy_(torch.tensor([values], dtype=torch.int32, device="xpu"))
            graph.replay()
            check_all()
        del graph
    finally:
        cache.close()


def test_wrong_aos_views_rejected():
    require_device()
    sizes = (704000, 704000, 921600)
    resident = [torch.zeros((40, size), dtype=torch.uint8, device="xpu") for size in sizes]
    host = [torch.zeros((8, size), dtype=torch.uint8, pin_memory=True) for size in sizes]
    with pytest.raises(ValueError, match="AoS views"):
        GPUExpertCache(21, range(40), resident, range(40, 48), host,
                       layout="original", down_kind=20, host_layout="expert_aos")


@pytest.mark.parametrize("tokens", [1, 4])
def test_addressed_decode_graph_matches_field_layout(tokens):
    require_device()
    from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts

    torch.manual_seed(17)
    sizes = (704000, 704000, 921600)
    raw = [torch.randint(256, (48, size), dtype=torch.uint8) for size in sizes]
    for data, block in zip(raw, (110, 110, 18)):
        blocks = data.view(-1, block)
        scales = (torch.rand(len(blocks)) * .01 + .001).half()
        blocks[:, :2] = scales.view(torch.uint8).reshape(-1, 2)
    caches, buffers = [], []
    graphs = []
    workspaces = []
    try:
        for aos in (False, True):
            fields = empty_expert_aos(40, sizes, "xpu") if aos else [x[:40].xpu() for x in raw]
            if aos:
                for field, data in zip(fields, raw):
                    field.copy_(data[:40])
            host = make_host_expert_aos([x[40:].contiguous() for x in raw])
            caches.append(GPUExpertCache(21, range(40), fields, range(40, 48), host,
                layout="original", down_kind=20, host_layout="expert_aos",
                resident_layout="expert_aos" if aos else "fields"))
            buffers.append(fields)
        x = (torch.randn((tokens, 2560), device="xpu") * .05).bfloat16()
        weights = torch.softmax(torch.randn((tokens, 3), device="xpu"), dim=1)
        ids = torch.tensor([[40, 41, 42]], device="xpu", dtype=torch.int32).repeat(tokens, 1)
        outputs = []
        for cache, fields in zip(caches, buffers):
            workspace = {}
            workspaces.append(workspace)
            def forward():
                routes = cache.prepare(ids)
                return fused_q8_experts(x, *fields, 21, routes.slots, weights, workspace,
                    grouping=routes.grouping, slot_addresses=cache.slot_addresses, down_kind=20)
            forward()
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output = forward()
            graphs.append(graph)
            outputs.append(output)
        for values in ([43, 44, 45], [0, 1, 2], [46, 47, 3]):
            ids.copy_(torch.tensor([values], device="xpu", dtype=torch.int32).repeat(tokens, 1))
            for graph in graphs:
                graph.replay()
            assert torch.isfinite(outputs[0]).all()
            assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
        graphs.clear()
        del graph
    finally:
        for cache in caches:
            cache.close()
