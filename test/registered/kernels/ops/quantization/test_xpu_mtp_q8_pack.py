# SPDX-License-Identifier: Apache-2.0
"""Actual runtime Q8 rows, merged metadata, linear interfaces and live graphs."""
import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_loader.qwen4_mtp_q8_pack import (
    PROJECTIONS,
    bind_projection,
    read_index,
)

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_TEST_PACK"),
    reason="Requires B70, native GGUF libraries and the existing local runtime pack",
)


def fixture():
    root = Path(os.environ["SGLANG_XPU_GGUF_MTP_Q8_TEST_PACK"])
    index = read_index(root)
    names = [name for parts in PROJECTIONS.values() for name in parts]
    assert len(names) == len(set(names)) == 10
    assert all(index[name]["kind"] == "q8_0" for name in names)
    row = index["fc_embedding.weight"]
    with (root / "dense.bin").open("rb") as f:
        f.seek(row["offset"])
        packed = torch.frombuffer(bytearray(f.read(32 * row["columns"] // 32 * 34)), dtype=torch.uint8)
    return packed.view(32, -1), row["columns"]


def expected_dot(x, packed, width):
    quant = _library(os.environ["SGLANG_XPU_GGUF_IQ4_SYCL_LIB"])
    scratch = torch.empty((x.shape[0], width // 32 * 36), dtype=torch.uint8, device="xpu")
    for start in range(0, x.shape[0], 4):
        rc = quant.xpu_iq4_quantize_layout(torch.xpu.current_stream().sycl_queue,
                                         x[start:].data_ptr(), scratch[start:].data_ptr(), width, min(4, x.shape[0]-start), 1, 0)
        assert rc == 0
    blocks = scratch.cpu().numpy().reshape(-1, 36)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float64)
    values = blocks[:, 4:].copy().view(np.int8).astype(np.float64) * d
    activation = torch.from_numpy(values.reshape(x.shape))
    weight = torch.from_numpy(gguf.dequantize(packed.numpy().reshape(-1),
                             gguf.GGMLQuantizationType.Q8_0).reshape(packed.shape[0], width)).double()
    return activation @ weight.T, activation.abs() @ weight.abs().T


@pytest.mark.parametrize("tuple_result", [False, True])
@torch.inference_mode()
def test_original_pack_bytes_interfaces_and_bounded_prefill(tuple_result):
    packed, width = fixture()
    with torch.device("xpu"):
        module = (ReplicatedLinear(width, 32, bias=False, params_dtype=torch.bfloat16)
                  if tuple_result else torch.nn.Linear(width, 32, bias=False, dtype=torch.bfloat16))
    receipt = bind_projection(module, packed, width=width)
    assert receipt["packed_bytes"] == packed.numel()
    restored = module.weight.cpu()
    if getattr(module.quant_method, "packed_soa", False):
        from sglang.kernels.ops.quantization.mtp_q8_soa import pack_rows
        restored = pack_rows(restored, inverse=True)
    assert torch.equal(restored, packed)
    for shape in [(1, width), (1, 4, width), (4, 4, width), (5, 4, width), (0, width)]:
        x = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
        actual = module(x)
        if tuple_result:
            assert actual[1] is None
            actual = actual[0]
        assert actual.shape == (*shape[:-1], 32)
        rows = x.numel() // width
        if rows > 4 and (not getattr(module.quant_method, "packed_soa", False) or rows not in (8, 16)):
            expected = F.linear(x, module.quant_method.reconstruct_weight(module))
            assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
        elif rows:
            expected, condition = expected_dot(x.reshape(-1, width), packed, width)
            error = (actual.cpu().double().reshape(expected.shape) - expected).abs()
            assert bool((error <= .004 * expected.abs() + 4e-6 * condition + 1e-5).all())
    restored = module.weight.cpu()
    if getattr(module.quant_method, "packed_soa", False):
        from sglang.kernels.ops.quantization.mtp_q8_soa import pack_rows
        restored = pack_rows(restored, inverse=True)
    assert torch.equal(restored, packed)


@torch.inference_mode()
def test_dynamic_graph_inputs_and_original_gemm_reference():
    packed, width = fixture()
    module = torch.nn.Linear(width, 32, bias=False, dtype=torch.bfloat16, device="xpu")
    bind_projection(module, packed, width=width)
    x = torch.randn(1, 4, width, dtype=torch.bfloat16, device="xpu")
    module(x)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        out = module(x)
    for scale in [-.5, 2, .1]:
        x.mul_(scale)
        graph.replay()
        expected, condition = expected_dot(x.reshape(4, width), packed, width)
        error = (out.cpu().double().view(4, 32) - expected).abs()
        assert bool((error <= .004 * expected.abs() + 4e-6 * condition + 1e-5).all())


@pytest.mark.skipif(not os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_SOA_LIB"), reason="SoA kernel not selected")
def test_extended_quantizer_bytes_match_retained_quantizer():
    from sglang.kernels.ops.quantization.mtp_q8_soa import library
    lib = library()
    retained = _library(os.environ["SGLANG_XPU_GGUF_IQ4_SYCL_LIB"])
    for width in (640, 2560, 6144):
        x = torch.randn(16, width, device="xpu", dtype=torch.bfloat16)
        x[0, :32] = 0
        x[1, :4] = torch.tensor([127., 63.5, -127., -63.5], device="xpu", dtype=torch.bfloat16)
        actual = torch.empty((16, width//32*36), device="xpu", dtype=torch.uint8)
        expected = torch.empty_like(actual)
        assert lib.xpu_mtp_q8_quant(torch.xpu.current_stream().sycl_queue, x.data_ptr(), actual.data_ptr(), width, 16) == 0
        for start in range(0, 16, 4):
            assert retained.xpu_iq4_quantize_layout(torch.xpu.current_stream().sycl_queue, x[start:].data_ptr(), expected[start:].data_ptr(), width, 4, 1, 0) == 0
        assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_SOA_LIB"), reason="SoA kernel not selected")
@pytest.mark.parametrize("rows", [2, 3, 5, 7, 12, 32, 128])
@torch.inference_mode()
def test_batched_draft_waves_and_graph(rows):
    from types import SimpleNamespace
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch

    packed, width = fixture()
    module = torch.nn.Linear(width, 32, bias=False, dtype=torch.bfloat16, device="xpu")
    bind_projection(module, packed, width=width)
    x = torch.randn(rows, width, device="xpu", dtype=torch.bfloat16)
    with set_forward_batch(SimpleNamespace(forward_mode=ForwardMode.DECODE)):
        module(x)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = module(x)
        for scale in (1, -.5, 2):
            x.mul_(scale)
            graph.replay()
            reference = torch.cat([module(x[i:i+1]) for i in range(rows)])
            assert torch.equal(actual.cpu().view(torch.int16), reference.cpu().view(torch.int16))
