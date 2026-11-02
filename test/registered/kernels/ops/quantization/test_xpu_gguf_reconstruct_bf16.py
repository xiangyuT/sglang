# SPDX-License-Identifier: Apache-2.0
"""Bitwise matrix contract against the existing FP32 decoder plus BF16 cast."""
import os

import pytest
import torch

from sglang.kernels.ops.quantization.gguf_dense_q8 import permute_q8_fields
from sglang.kernels.ops.quantization.gguf_reconstruct_bf16 import reconstruct_bf16
from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES


@pytest.fixture(autouse=True)
def native_b70():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70 required")
    if not all(os.environ.get(name) for name in (
        "SGLANG_XPU_GGUF_RECONSTRUCT_BF16_LIB", "SGLANG_XPU_GGUF_SYCL_LIB"
    )):
        pytest.skip("Explicit original and fused native libraries required")


def packed_matrix(kind, width, rows):
    block, size = BLOCK_GEOMETRY[GGML_TYPES[kind]]
    generator = torch.Generator().manual_seed(1000 + kind + width + rows)
    raw = torch.randint(0, 256, (rows, width // block, size),
                        dtype=torch.uint8, generator=generator)
    scales = (torch.randn((rows, width // block), generator=generator) * 0.003).half()
    scales[0, 0] = -0.0
    offset = 208 if kind == 14 else 0
    raw[..., offset:offset + 2] = scales.view(torch.uint8).reshape(rows, width // block, 2)
    return raw.reshape(rows, -1).to("xpu")


@pytest.mark.parametrize("width", [640, 2560, 6144])
@pytest.mark.parametrize("rows", [1, 16, 33])
def test_q8_original_ragged_rows(width, rows):
    packed = packed_matrix(8, width, rows)
    expected = dequantize_gguf(packed.reshape(-1), 8, dtype=torch.float32).view(rows, width).bfloat16()
    actual = reconstruct_bf16(packed, 8, width)
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("width", [640, 2560, 6144])
def test_q8_soa16_preserves_original_bf16(width):
    packed = packed_matrix(8, width, 32)
    expected = dequantize_gguf(packed.reshape(-1), 8, dtype=torch.float32).view(32, width).bfloat16()
    actual = reconstruct_bf16(permute_q8_fields(packed, width), 8, width, q8_soa=True)
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("kind,width", [(14, 2560), (20, 640), (21, 2560), (23, 2560), (42, 640)])
def test_other_original_dense_formats(kind, width):
    packed = packed_matrix(kind, width, 16)
    expected = dequantize_gguf(packed.reshape(-1), kind, dtype=torch.float32).view(16, width).bfloat16()
    actual = reconstruct_bf16(packed, kind, width)
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
