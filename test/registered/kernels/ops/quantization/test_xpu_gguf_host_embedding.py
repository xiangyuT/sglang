# SPDX-License-Identifier: Apache-2.0
"""Original Q8 bytes, model dtype, shared MTP ownership and graph replay."""
import os

import pytest
import torch

from sglang.kernels.ops.quantization.gguf_host_embedding import gather_host_embedding, prepare_host_embedding
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFEmbeddingMethod


def fixture():
    if not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_HOST_EMBEDDING_LIB"):
        pytest.skip("Requires the B70 host-embedding SYCL library")
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70-only primitive")
    torch.manual_seed(42)
    raw = torch.randint(0, 256, (4096, 80, 34), dtype=torch.uint8)
    scales = torch.rand((4096, 80), dtype=torch.float16)
    raw[..., :2] = scales.view(torch.uint8).reshape(4096, 80, 2)
    raw = raw.view(4096, 2720)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(raw.xpu(), requires_grad=False)
    module.weight.loader_marker = "preserved"
    return raw, module


@pytest.mark.parametrize("T", [0, 1, 4, 8, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_host_embedding_exact(T, dtype):
    raw, module = fixture()
    method = CompressedGGUFEmbeddingMethod(8)
    method.params_dtype = dtype
    ids = torch.randint(4096, (T,), device="xpu")
    reference = method.embedding(module, ids)
    prepare_host_embedding(module)
    assert module.weight.device.type == "cpu"
    assert module.weight.loader_marker == "preserved"
    assert torch.equal(module.weight, raw)
    actual = method.embedding(module, ids)
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(gather_host_embedding(module.weight, ids), raw.xpu().index_select(0, ids))


def test_shared_graph_embedding():
    raw, target = fixture()
    prepare_host_embedding(target)
    draft = torch.nn.Module()
    draft.weight = target.weight
    method = CompressedGGUFEmbeddingMethod(8, allow_host=False)
    method.params_dtype = torch.bfloat16
    ids = torch.zeros((4,), device="xpu", dtype=torch.int32)
    method.embedding(draft, ids)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        out = method.embedding(draft, ids)
    gpu = torch.nn.Module()
    gpu.weight = torch.nn.Parameter(raw.xpu(), requires_grad=False)
    for seed in (7, 19, 31):
        torch.manual_seed(seed)
        ids.copy_(torch.randint(4096, (4,), device="xpu", dtype=torch.int32))
        graph.replay()
        expected = method.embedding(gpu, ids)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    del graph
