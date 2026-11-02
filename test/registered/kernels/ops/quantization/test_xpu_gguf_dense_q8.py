"""Original compressed dense dots, mixed strides and graph ownership on B70."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_dense_q8 import (
    dense_q8_projection,
    permute_iq3_fields,
    permute_q8_fields,
)
from sglang.srt.layers.quantization.gguf_dense_linear import (
    CompressedDenseProjectionMethod,
)
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.srt.model_loader.qwen4_gguf import convert_qwen4_dense_value
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=10, suite="xpu-unit-test")


def fixture(kind, k, n, seed=42):
    qk, size = BLOCK_GEOMETRY[GGML_TYPES[kind]]
    raw = np.random.default_rng(seed).integers(
        0, 256, (n * k // qk, size), dtype=np.uint8
    )
    offset = 208 if kind == 14 else 0
    raw[:, offset : offset + 2] = np.array([0.03125], dtype=np.float16).view(np.uint8)
    if kind == 42:
        codes = (raw[:, 2 + np.arange(64) // 4] >> ((np.arange(64) % 4) * 2)) & 3
        weights = (codes.astype(np.float32) - 1) * np.float32(0.03125)
    else:
        weights = gguf.dequantize(raw.reshape(-1), gguf.GGMLQuantizationType(kind))
    return torch.from_numpy(raw.reshape(n, -1)), torch.from_numpy(weights.reshape(n, k))


def quantized_reference(x):
    values = x.float().reshape(-1, 32).numpy()
    # Strata rounds codes using the saturated FP32 scale, then stores the
    # scale as FP16 in Q8_1. Do not round the divisor before coding.
    scales = np.minimum(np.abs(values).max(axis=1) / np.float32(127), 65504)
    ratios = np.divide(
        values, scales[:, None], out=np.zeros_like(values), where=scales[:, None] != 0
    )
    codes = np.clip(np.copysign(np.floor(np.abs(ratios) + 0.5), ratios), -127, 127)
    decoded = (
        codes.astype(np.float64) * scales.astype(np.float16).astype(np.float64)[:, None]
    )
    return torch.from_numpy(decoded.reshape(x.shape))


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestDenseQ8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 port")
        if not all(
            os.environ.get(name)
            for name in ("SGLANG_XPU_GGUF_SYCL_LIB", "SGLANG_XPU_GGUF_IQ4_SYCL_LIB")
        ):
            raise unittest.SkipTest("Build both native libraries")
        torch.use_deterministic_algorithms(True)

    def assert_reference(self, actual, x, weights):
        decoded = quantized_reference(x)
        expected = decoded @ weights.double().T
        condition = decoded.abs() @ weights.double().abs().T
        error = (actual.cpu().double() - expected).abs()
        self.assertTrue(
            bool((error <= condition * 2e-6 + 2e-5).all()),
            f"FP64 reference mismatch: max error {error.max().item()}",
        )

    def test_formats_widths_tails_and_token_counts(self):
        generator = torch.Generator().manual_seed(42)
        for kind, k in (
            (8, 640),
            (20, 640),
            (42, 640),
            (14, 2560),
            (21, 2560),
            (23, 2560),
            (14, 6144),
            (21, 6144),
        ):
            raw, weights = fixture(kind, k, 17)
            packed = raw.to("xpu")
            for tokens in (1, 2, 3, 4):
                with self.subTest(kind=kind, k=k, tokens=tokens):
                    x = torch.randn((tokens, k), generator=generator).bfloat16()
                    x[0, :32] = 0
                    actual = dense_q8_projection(x.to("xpu"), ((kind, packed),))
                    self.assert_reference(actual, x, weights)
                    self.assertTrue(torch.equal(packed.cpu(), raw))

    def test_batched_verify_uses_original_four_row_arithmetic(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch

        for kind in (8, 21, 23):
            raw, _ = fixture(kind, 2560, 32)
            layer = SimpleNamespace(weight=raw.to("xpu"))
            method = CompressedDenseProjectionMethod((kind,), 2560, torch.bfloat16)
            method.process_weights_after_loading(layer)
            for rows in (5, 8, 32):
                with self.subTest(kind=kind, rows=rows), torch.inference_mode(), set_forward_batch(SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)):
                    x = torch.randn(rows, 2560, device="xpu", dtype=torch.bfloat16)
                    method.apply(layer, x)
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        actual = method.apply(layer, x)
                    for scale in (1, -.5, 2):
                        x.mul_(scale)
                        graph.replay()
                        expected = torch.cat([method.apply(layer, wave) for wave in x.split(4)])
                        self.assertTrue(torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)))

    def test_q8_eight_row_wave_preserves_four_row_bits(self):
        from sglang.kernels.ops.quantization.gguf_dense_bf16 import project, supports
        from sglang.kernels.ops.quantization.gguf_sycl import _library
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch

        lib = _library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"])
        self.assertTrue(hasattr(lib, "xpu_gguf_dense_q8_soa_wide"))
        flags = {"SGLANG_XPU_GGUF_DENSE_Q8_WAVE_ROWS": "8",
                 "SGLANG_XPU_GGUF_DENSE_Q8_SOA": "1",
                 "SGLANG_XPU_GGUF_DENSE_BF16": "1"}
        for width in (640, 2560, 6144):
            raw, _ = fixture(8, width, 32)
            layer = SimpleNamespace(weight=raw.to("xpu"))
            method = CompressedDenseProjectionMethod((8,), width, torch.bfloat16)
            with patch.dict(os.environ, flags), torch.inference_mode():
                method.process_weights_after_loading(layer)
                for rows in (5, 6, 7, 8, 9, 16, 32):
                    with self.subTest(width=width, rows=rows), set_forward_batch(
                        SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
                    ):
                        x = torch.randn(rows, width, device="xpu", dtype=torch.bfloat16)
                        method.apply(layer, x)
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            actual = method.apply(layer, x)
                        for factor in (0, -0.5, 2):
                            x.copy_(torch.randn_like(x) * factor)
                            graph.replay()
                            expected = torch.cat([method.apply(layer, v) for v in x.split(4)])
                            self.assertTrue(torch.equal(actual.cpu().view(torch.int16),
                                                        expected.cpu().view(torch.int16)))
                        if rows == 8:
                            parts = method.parts(layer)
                            self.assertFalse(supports(x, parts, (False,), (False,)))
                            with self.assertRaisesRegex(ValueError, "scratch"):
                                project(x, parts, packed_parts=(False,), q8_soa_parts=(True,),
                                        scratch=torch.empty(4 * (width // 32) * 36,
                                                            dtype=torch.uint8, device="xpu"))
                # Larger prefill still reconstructs the original BF16 weights.
                with set_forward_batch(SimpleNamespace(forward_mode=ForwardMode.EXTEND)):
                    x = torch.randn(8, width, device="xpu", dtype=torch.bfloat16)
                    self.assertTrue(torch.equal(method.apply(layer, x).view(torch.int16),
                                                F.linear(x, method.reconstruct_weight(layer)).view(torch.int16)))

    def test_q8_wide_fp32_bits_match_raw_rows_with_graph_updates(self):
        from sglang.kernels.ops.quantization.gguf_dense_bf16 import quantize_batch
        from sglang.kernels.ops.quantization.gguf_sycl import _library

        lib = _library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"])
        for width in (640, 2560, 6144):
            raw = fixture(8, width, 32)[0].to("xpu")
            packed = permute_q8_fields(raw, width)
            for rows in (5, 6, 7, 8):
                with self.subTest(width=width, rows=rows):
                    x = torch.randn(rows, width, device="xpu", dtype=torch.bfloat16)
                    encoded = torch.empty(rows * (width // 32) * 36,
                                          device="xpu", dtype=torch.uint8)
                    # A padded output detects writes outside the vocabulary rows.
                    output = torch.empty(rows, 39, device="xpu", dtype=torch.float32)
                    output.view(torch.int32).fill_(0x13579BDF)

                    def submit():
                        quantize_batch(x, encoded)
                        rc = lib.xpu_gguf_dense_q8_soa_wide(
                            torch.xpu.current_stream().sycl_queue,
                            packed.data_ptr(), encoded.data_ptr(), output.data_ptr(),
                            width, 32, rows, 39, 0,
                        )
                        self.assertEqual(rc, 0)

                    submit()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph, stream=torch.xpu.Stream(device=x.device)):
                        submit()
                    for factor in (0, -0.5, 32):
                        x.copy_(torch.randn_like(x) * factor)
                        graph.replay()
                        expected = torch.cat([
                            dense_q8_projection(chunk, ((8, raw),))
                            for chunk in x.split(4)
                        ])
                        self.assertTrue(torch.equal(
                            output[:, :32].view(torch.int32), expected.view(torch.int32)
                        ))
                        self.assertTrue(bool(
                            (output[:, 32:].view(torch.int32) == 0x13579BDF).all()
                        ))
                    del graph

    def test_batch_quantization_preserves_dynamic_decode_waves(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch

        flags = {"SGLANG_XPU_GGUF_DENSE_Q8_WAVE_ROWS": "8",
                 "SGLANG_XPU_GGUF_DENSE_Q8_SOA": "1",
                 "SGLANG_XPU_GGUF_DENSE_BF16": "1",
                 "SGLANG_XPU_GGUF_DENSE_BATCH_QUANT": "1"}
        for width in (640, 2560, 6144):
            raw, _ = fixture(8, width, 32)
            layer = SimpleNamespace(weight=raw.to("xpu"))
            config = SimpleNamespace(linear_num_key_heads=16,
                                     linear_num_value_heads=48,
                                     linear_value_head_dim=128) if width == 6144 else None
            method = CompressedDenseProjectionMethod((8,), width, torch.bfloat16,
                                                    gdn_config=config)
            with patch.dict(os.environ, flags), torch.inference_mode():
                method.process_weights_after_loading(layer)
                pointer = None
                for rows in (4, 5, 8, 12, 16, 20, 24, 28, 32):
                    with self.subTest(width=width, rows=rows), set_forward_batch(
                        SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
                    ):
                        # Exercise a strided input as well as the contiguous model case.
                        storage = torch.randn(rows, width * 2, device="xpu", dtype=torch.bfloat16)
                        x = storage[:, ::2]
                        method.apply(layer, x)
                        scratch = next(iter(method._scratch.values()))
                        self.assertEqual(scratch.numel(), 32 * (width // 32) * 36)
                        pointer = scratch.data_ptr() if pointer is None else pointer
                        self.assertEqual(scratch.data_ptr(), pointer)
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            actual = method.apply(layer, x)
                        for factor in (0, -.5, 2):
                            storage.copy_(torch.randn_like(storage) * factor)
                            graph.replay()
                            with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_BATCH_QUANT": "0"}):
                                expected = method.apply(layer, x)
                            self.assertTrue(torch.equal(actual.cpu().view(torch.int16),
                                                        expected.cpu().view(torch.int16)))
                with set_forward_batch(SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)):
                    x = torch.randn(8, width, device="xpu", dtype=torch.bfloat16)
                    self.assertTrue(torch.equal(method.apply(layer, x).view(torch.int16),
                                                F.linear(x, method.reconstruct_weight(layer)).view(torch.int16)))

    def test_batch_quantization_capacity_and_wave_guards(self):
        from sglang.kernels.ops.quantization.gguf_dense_bf16 import project, quantize_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch

        x = torch.randn(8, 640, device="xpu", dtype=torch.bfloat16)
        encoded = torch.empty(8 * 20 * 36, dtype=torch.uint8, device="xpu")
        quantize_batch(x, encoded)
        with self.assertRaisesRegex(ValueError, "scratch"):
            quantize_batch(x, encoded[:-1])
        with self.assertRaisesRegex(ValueError, "scratch"):
            quantize_batch(x[:4], encoded)
        with self.assertRaisesRegex(ValueError, "scratch"):
            quantize_batch(x, encoded, input_permutation="gdn_16x3x128")
        with self.assertRaisesRegex(ValueError, "scratch"):
            quantize_batch(x, x.view(torch.uint8).reshape(-1))
        raw, _ = fixture(8, 640, 32)
        layer = SimpleNamespace(weight=permute_q8_fields(raw.to("xpu"), 640),
                                gguf_dense_q8_soa_parts=(True,))
        flags = {"SGLANG_XPU_GGUF_DENSE_Q8_WAVE_ROWS": "8",
                 "SGLANG_XPU_GGUF_DENSE_BF16": "1",
                 "SGLANG_XPU_GGUF_DENSE_BATCH_QUANT": "0"}
        method = CompressedDenseProjectionMethod((8,), 640, torch.bfloat16)
        with patch.dict(os.environ, flags), torch.inference_mode(), set_forward_batch(
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        ):
            method.apply(layer, x[:4])
            with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_BATCH_QUANT": "1"}):
                with self.assertRaisesRegex(RuntimeError, "before model initialization"):
                    method.apply(layer, x)
            with self.assertRaisesRegex(ValueError, "pre-encoded"):
                project(x, ((8, layer.weight),), packed_parts=(False,),
                        q8_soa_parts=(True,), quantized_input=encoded[:-1])

    def test_mixed_parts_output_stride_dtypes_and_dynamic_graph(self):
        cpu_parts = [fixture(kind, 2560, n) for kind, n in ((23, 17), (21, 5), (14, 3))]
        parts = tuple(
            (kind, raw.to("xpu")) for kind, (raw, _) in zip((23, 21, 14), cpu_parts)
        )
        weights = torch.cat([w for _, w in cpu_parts])
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(
                    (4, 2560), generator=torch.Generator().manual_seed(71)
                ).to(dtype)
                xgpu = x.to("xpu")
                scratch = torch.empty(4 * 80 * 36, dtype=torch.uint8, device="xpu")
                first = dense_q8_projection(xgpu, parts, scratch=scratch)
                saved = first.clone()
                second = dense_q8_projection(xgpu * 2, parts, scratch=scratch)
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertTrue(
                    torch.equal(first.view(torch.int32), saved.view(torch.int32))
                )
                self.assert_reference(first, x, weights)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    actual = dense_q8_projection(xgpu, parts, scratch=scratch)
                for scale in (0.5, -2.0, 1.0):
                    changed = (x * scale).to(dtype)
                    xgpu.copy_(changed)
                    graph.replay()
                    self.assert_reference(actual, changed, weights)

    def test_gdn_reconstruction_prefill_and_input_permutation(self):
        config = SimpleNamespace(
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_value_head_dim=128,
        )
        for kind in (14, 21):
            with self.subTest(kind=kind):
                raw, weights = fixture(kind, 6144, 5)
                layer = torch.nn.Module()
                layer.register_buffer("weight", raw.to("xpu"))
                method = CompressedDenseProjectionMethod(
                    (kind,), 6144, torch.bfloat16, gdn_config=config
                )
                method.process_weights_after_loading(layer)
                restored = convert_qwen4_dense_value(
                    weights.bfloat16(), "gdn_value_columns", config
                ).contiguous()
                actual_weights = method.reconstruct_weight(layer).cpu()
                self.assertTrue(
                    torch.equal(
                        actual_weights.view(torch.int16), restored.view(torch.int16)
                    )
                )
                x = torch.randn(
                    (4, 6144), generator=torch.Generator().manual_seed(42)
                ).bfloat16()
                ordered = x.view(4, 16, 3, 128).transpose(1, 2).reshape(4, 6144)
                actual = dense_q8_projection(
                    x.to("xpu"),
                    ((kind, layer.weight),),
                    input_permutation="gdn_16x3x128",
                )
                self.assert_reference(actual, ordered, weights)
                prefill = x.repeat(2, 1).to("xpu")
                actual = method.apply(layer, prefill)
                expected = F.linear(prefill, restored.to("xpu"))
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )

    def test_iq3_xmx_exact_fp32_tree_mixed_strides_and_graph(self):
        for width in (2560, 6144):
            cpu_parts = [fixture(kind, width, n) for kind, n in ((8, 3), (21, 17), (23, 5))]
            parts = tuple(
                (kind, raw.to("xpu"))
                for kind, (raw, _) in zip((8, 21, 23), cpu_parts)
            )
            permutation = "gdn_16x3x128" if width == 6144 else None
            for tokens in (1, 2, 3, 4):
                with self.subTest(width=width, tokens=tokens):
                    x = torch.randn(
                        tokens, width, generator=torch.Generator().manual_seed(2557)
                    ).bfloat16().to("xpu")
                    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_IQ3_XMX": "0"}):
                        expected = dense_q8_projection(
                            x, parts, input_permutation=permutation
                        )
                    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_IQ3_XMX": "1"}):
                        actual = dense_q8_projection(
                            x, parts, input_permutation=permutation
                        )
                        self.assertTrue(torch.equal(
                            actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                        ))
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            captured = dense_q8_projection(
                                x, parts, input_permutation=permutation
                            )
                    for factor in (0.5, -2.0, 1.0):
                        x.mul_(factor)
                        with patch.dict(os.environ, {"SGLANG_XPU_GGUF_IQ3_XMX": "0"}):
                            expected = dense_q8_projection(
                                x, parts, input_permutation=permutation
                            )
                        graph.replay()
                        self.assertTrue(torch.equal(
                            captured.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                        ))
                    del graph

    def test_packed_iq3_layout_lifecycle_prefill_and_graph(self):
        for width in (2560, 6144):
            with self.subTest(width=width):
                kinds = (8, 21, 21, 23)
                cpu = [fixture(kind, width, n)[0]
                       for kind, n in zip(kinds, (3, 32, 17, 5))]
                raw_parts = tuple((kind, raw.to("xpu"))
                                  for kind, raw in zip(kinds, cpu))
                layer = torch.nn.Module()
                for i, (_, raw) in enumerate(raw_parts):
                    layer.register_buffer(
                        "weight" if i == 0 else f"gguf_dense_part_{i}", raw
                    )
                config = SimpleNamespace(
                    linear_num_key_heads=16, linear_num_value_heads=48,
                    linear_value_head_dim=128,
                ) if width == 6144 else None
                method = CompressedDenseProjectionMethod(
                    kinds, width, torch.bfloat16, gdn_config=config
                )
                reference_weight = method.reconstruct_weight(layer)
                original_bytes = sum(raw.numel() for _, raw in raw_parts)
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_IQ3_PACKED_S4": "1"}):
                    method.process_weights_after_loading(layer)
                self.assertEqual(method.iq3_layouts(layer), (False, True, False, False))
                pointer = layer.gguf_dense_part_1.data_ptr()
                with patch.dict(os.environ, {
                    "SGLANG_XPU_GGUF_IQ3_PACKED_S4": "0",
                    "SGLANG_XPU_GGUF_IQ3_XMX": "0",
                }):
                    method.process_weights_after_loading(layer)
                    self.assertEqual(pointer, layer.gguf_dense_part_1.data_ptr())
                    self.assertEqual(
                        original_bytes, sum(raw.numel() for _, raw in method.parts(layer))
                    )
                    restored = permute_iq3_fields(
                        layer.gguf_dense_part_1, width, inverse=True
                    )
                    self.assertTrue(torch.equal(restored.cpu(), cpu[1]))
                    self.assertTrue(torch.equal(
                        method.reconstruct_weight(layer).view(torch.int16),
                        reference_weight.view(torch.int16),
                    ))
                    permutation = "gdn_16x3x128" if config else None
                    for tokens in (1, 2, 3, 4):
                        x = torch.randn(tokens, width).bfloat16().to("xpu")
                        expected = dense_q8_projection(
                            x, raw_parts, input_permutation=permutation
                        )
                        actual = dense_q8_projection(
                            x, method.parts(layer), input_permutation=permutation,
                            iq3_packed_parts=method.iq3_layouts(layer),
                        )
                        self.assertTrue(torch.equal(
                            actual.view(torch.int32), expected.view(torch.int32)
                        ))
                        self.assertTrue(torch.equal(
                            method.apply(layer, x).view(torch.int16),
                            expected.bfloat16().view(torch.int16),
                        ))
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        captured = method.apply(layer, x)
                    for scale in (0.5, -2.0, 1.0):
                        x.mul_(scale)
                        graph.replay()
                        expected = dense_q8_projection(
                            x, raw_parts, input_permutation=permutation
                        ).bfloat16()
                        self.assertTrue(torch.equal(
                            captured.view(torch.int16), expected.view(torch.int16)
                        ))
                    prefill = x.repeat(2, 1)
                    self.assertTrue(torch.equal(
                        method.apply(layer, prefill).view(torch.int16),
                        F.linear(prefill, reference_weight).view(torch.int16),
                    ))
                    del graph

    def test_q8_soa_byte_lifecycle_prefill_and_mixed_graph(self):
        from sglang.kernels.ops.quantization.gguf_sycl import _library
        from sglang.kernels.ops.quantization.mtp_q8_soa import pack_rows

        if not hasattr(_library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"]),
                       "xpu_gguf_dense_q8_soa"):
            self.skipTest("Build the Q8 SoA16 dense library")
        for width in (640, 2560, 6144):
            with self.subTest(width=width):
                kinds = (8,) if width == 640 else (8, 21, 23)
                sizes = (16,) if width == 640 else (16, 32, 5)
                cpu = [fixture(kind, width, n)[0] for kind, n in zip(kinds, sizes)]
                raw_parts = tuple((kind, raw.to("xpu")) for kind, raw in zip(kinds, cpu))
                layer = torch.nn.Module()
                for index, (_, raw) in enumerate(raw_parts):
                    layer.register_buffer("weight" if index == 0 else f"gguf_dense_part_{index}", raw)
                method = CompressedDenseProjectionMethod(kinds, width, torch.bfloat16)
                original = method.reconstruct_weight(layer)
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_Q8_SOA": "1",
                                            "SGLANG_XPU_GGUF_IQ3_PACKED_S4": "1"}):
                    method.process_weights_after_loading(layer)
                self.assertEqual(method.q8_layouts(layer), (True,) + (False,) * (len(kinds)-1))
                self.assertTrue(torch.equal(layer.weight.cpu(), pack_rows(cpu[0])))
                self.assertTrue(torch.equal(permute_q8_fields(layer.weight, width, inverse=True).cpu(), cpu[0]))
                pointer = layer.weight.data_ptr()
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_Q8_SOA": "0"}):
                    method.process_weights_after_loading(layer)
                self.assertEqual(pointer, layer.weight.data_ptr())
                self.assertTrue(method.q8_layouts(layer)[0])
                self.assertTrue(torch.equal(method.reconstruct_weight(layer).view(torch.int16),
                                            original.view(torch.int16)))
                for count in (1, 2, 3, 4):
                    x = torch.randn(count, width).bfloat16().to("xpu")
                    expected = dense_q8_projection(x, raw_parts)
                    actual = dense_q8_projection(x, method.parts(layer),
                        iq3_packed_parts=method.iq3_layouts(layer), q8_soa_parts=method.q8_layouts(layer))
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
                    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_BF16": "1"}):
                        actual = method.apply(layer, x)
                        self.assertTrue(torch.equal(actual.view(torch.int16), expected.bfloat16().view(torch.int16)))
                        torch.xpu.synchronize()
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            captured = method.apply(layer, x)
                    for scale in (.5, -2.):
                        x.mul_(scale)
                        graph.replay()
                        reference = dense_q8_projection(x, raw_parts).bfloat16()
                        self.assertTrue(torch.equal(captured.view(torch.int16), reference.view(torch.int16)))
                    del graph
                prefill = x.repeat(3, 1)
                self.assertTrue(torch.equal(method.apply(layer, prefill).view(torch.int16),
                                            F.linear(prefill, original).view(torch.int16)))

    def test_q6_head_lifecycle_tail_mixed_and_graph(self):
        from sglang.kernels.ops.quantization.gguf_dense_q8 import permute_q6_fields
        from sglang.kernels.ops.quantization.gguf_sycl import _library
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFHeadMethod

        if not hasattr(_library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"]),
                       "xpu_gguf_dense_q6_soa"):
            self.skipTest("Build the Q6 SoA16 library")
        for rows in (17, 32, 35):
            with self.subTest(rows=rows):
                raw = fixture(14, 2560, rows, seed=5800 + rows)[0].to("xpu")
                packed = permute_q6_fields(raw)
                self.assertEqual(packed.numel(), raw.numel())
                self.assertTrue(torch.equal(permute_q6_fields(packed, inverse=True), raw))
                full = rows // 16 * 16
                self.assertTrue(torch.equal(packed[full:], raw[full:]))
                other = fixture(8, 2560, 3)[0].to("xpu")
                for count in (1, 2, 3, 4):
                    x = torch.randn(count, 2560).bfloat16().to("xpu")
                    actual = dense_q8_projection(x, ((14, packed), (8, other)),
                                                  q6_soa_parts=(True, False))
                    expected = dense_q8_projection(x, ((14, raw), (8, other)))
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
                # MTP selects/reorders a vocabulary after loader callbacks.
                method = CompressedGGUFHeadMethod(14)
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(raw.clone(), requires_grad=False)
                method.process_weights_after_loading(layer)
                ids = torch.arange(rows - 1, -1, -1, device="xpu")[:17]
                selected = raw.index_select(0, ids)
                layer.weight.data = layer.weight.index_select(0, ids)
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_HEAD_Q6_SOA": "1"}):
                    method.apply(layer, x)
                self.assertTrue(layer.gguf_head_q6_soa)
                self.assertTrue(torch.equal(permute_q6_fields(layer.weight, inverse=True), selected))
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_HEAD_Q6_SOA": "0"}):
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph, stream=torch.xpu.Stream(device=x.device)):
                        output = method.apply(layer, x)
                for scale in (.5, -2., 1.):
                    x.mul_(scale)
                    graph.replay()
                    expected = dense_q8_projection(x, ((14, selected),))
                    self.assertTrue(torch.equal(output.view(torch.int32), expected.view(torch.int32)))
                del graph
                prefill = x.repeat(2, 1)
                actual = method.apply(layer, prefill)
                expected = torch.cat([dense_q8_projection(chunk, ((14, selected),))
                                      for chunk in prefill.split(4)])
                self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
                layer.weight.data = selected.clone()
                with self.assertRaisesRegex(RuntimeError, "changed after preparation"):
                    method.apply(layer, x)

    def test_packed_iq3_parameter_replacement(self):
        raw, _ = fixture(21, 2560, 16)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(raw.to("xpu"), requires_grad=False)
        method = CompressedDenseProjectionMethod((21,), 2560, torch.bfloat16)
        with patch.dict(os.environ, {"SGLANG_XPU_GGUF_IQ3_PACKED_S4": "1"}):
            method.process_weights_after_loading(layer)
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertEqual(method.iq3_layouts(layer), (True,))
        self.assertTrue(torch.equal(
            permute_iq3_fields(layer.weight, 2560, inverse=True).cpu(), raw
        ))

    def test_invalid_geometry_rejected(self):
        raw, _ = fixture(20, 640, 3)
        packed = raw.to("xpu")
        for x, parts in (
            (torch.zeros(4, 640), ((20, packed),)),
            (torch.zeros(5, 640, device="xpu"), ((20, packed),)),
            (torch.zeros(1, 640, device="xpu"), ((255, packed),)),
            (torch.zeros(1, 640, device="xpu"), ((20, packed[:, :-1]),)),
        ):
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    dense_q8_projection(x, parts)

    def test_q6_vocabulary_exceeds_dense_projection_row_limit(self):
        # Vocabulary shards exceed the old N<=16384 projection-only bound.
        # Vary rows, exercise the final partial packet and change graph inputs.
        packed, _ = fixture(14, 2560, 16, seed=3501)
        packed = packed.repeat(1025, 1)[:16385].contiguous().to("xpu")
        for count in (1, 4):
            with self.subTest(tokens=count):
                x = torch.randn(
                    count, 2560, generator=torch.Generator().manual_seed(3502)
                ).bfloat16().to("xpu")
                with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_DP4A": "1"}):
                    dense_q8_projection(x, ((14, packed),))
                    torch.xpu.synchronize()
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        actual = dense_q8_projection(x, ((14, packed),))
                for scale in (0.5, -2.0, 1.0):
                    x.mul_(scale)
                    with patch.dict(os.environ, {"SGLANG_XPU_GGUF_DENSE_DP4A": "0"}):
                        expected = dense_q8_projection(x, ((14, packed),))
                    graph.replay()
                    self.assertTrue(torch.equal(
                        actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                    ))
                del graph

    def test_raw_dp4a_and_iq4_s4_preserve_fp32_mixed_stride_and_graph(self):
        flags = ("SGLANG_XPU_GGUF_DENSE_DP4A", "SGLANG_XPU_GGUF_IQ4_RAW_S4")
        for width in (2560, 6144):
            parts = tuple(
                (kind, fixture(kind, width, n)[0].to("xpu"))
                for kind, n in ((8, 3), (14, 17), (20, 5), (23, 19))
            )
            permutation = "gdn_16x3x128" if width == 6144 else None
            for tokens in (1, 2, 3, 4):
                with self.subTest(width=width, tokens=tokens):
                    x = torch.randn(
                        tokens, width, generator=torch.Generator().manual_seed(2701)
                    ).bfloat16().to("xpu")
                    with patch.dict(os.environ, dict.fromkeys(flags, "0")):
                        expected = dense_q8_projection(x, parts, input_permutation=permutation)
                    with patch.dict(os.environ, dict.fromkeys(flags, "1")):
                        actual = dense_q8_projection(x, parts, input_permutation=permutation)
                        self.assertTrue(torch.equal(
                            actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                        ))
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            captured = dense_q8_projection(x, parts, input_permutation=permutation)
                    for scale in (0.5, -2.0, 1.0):
                        x.mul_(scale)
                        with patch.dict(os.environ, dict.fromkeys(flags, "0")):
                            expected = dense_q8_projection(x, parts, input_permutation=permutation)
                        graph.replay()
                        self.assertTrue(torch.equal(
                            captured.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                        ))
                    del graph

    def test_q6_head_wide_exact_and_graph(self):
        from sglang.kernels.ops.quantization.gguf_q6_head_sycl import try_project
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFHeadMethod,
        )

        if not os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_LIB"):
            self.skipTest("Build the optional wide Q6 head library")
        # Each test must restore its caller's device even when a subtest fails.
        self.addCleanup(torch.xpu.set_device, torch.xpu.current_device())
        for rank in range(torch.xpu.device_count()):
            if torch.xpu.get_device_properties(rank).device_id != 0xE223:
                continue
            torch.xpu.set_device(rank)
            device = torch.device("xpu", rank)
            for rows in (13, 17, 32, 45):
                raw = fixture(14, 2560, rows, seed=1197 + rows)[0].to(device)
                for wave in ("8", "16"):
                    with patch.dict(
                        os.environ,
                        {
                            "SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS": wave,
                            "SGLANG_XPU_GGUF_HEAD_Q6_SOA": "1",
                        },
                    ):
                        method = CompressedGGUFHeadMethod(14)
                        layer = torch.nn.Module()
                        layer.weight = torch.nn.Parameter(
                            raw.clone(), requires_grad=False
                        )
                        method.process_weights_after_loading(layer)
                        method.apply(
                            layer,
                            torch.zeros(1, 2560, dtype=torch.bfloat16, device=device),
                        )
                        marker = layer.gguf_head_q6_soa

                        def reference(x):
                            return torch.cat(
                                [
                                    dense_q8_projection(
                                        chunk,
                                        ((14, layer.weight),),
                                        q6_soa_parts=(marker,),
                                    )
                                    for chunk in x.split(4)
                                ]
                            )

                        for count in (*range(1, 17), 20, 28, 32, 64):
                            x = torch.randn(
                                count, 2560, dtype=torch.bfloat16, device=device
                            )
                            self.assertTrue(
                                torch.equal(
                                    method.apply(layer, x).view(torch.int32),
                                    reference(x).view(torch.int32),
                                )
                            )
                        x = torch.randn(32, 2560, dtype=torch.bfloat16, device=device)
                        method.apply(layer, x)
                        torch.xpu.synchronize(device)
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(
                            graph, stream=torch.xpu.Stream(device=device)
                        ):
                            result = method.apply(layer, x)
                        for scale in (0.5, -2.0):
                            x.mul_(scale)
                            graph.replay()
                            self.assertTrue(
                                torch.equal(
                                    result.view(torch.int32),
                                    reference(x).view(torch.int32),
                                )
                            )
                        del graph, result
                        self.assertIsNone(try_project(x.float(), layer.weight))
                        with patch.dict(
                            os.environ, {"SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS": "4"}
                        ):
                            self.assertIsNone(try_project(x, layer.weight))
                            self.assertTrue(
                                torch.equal(
                                    method.apply(layer, x).view(torch.int32),
                                    reference(x).view(torch.int32),
                                )
                            )


if __name__ == "__main__":
    unittest.main()
