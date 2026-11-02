"""IQ4_XS/Q8_1 head port: CPU arithmetic, graph updates and compressed loading."""

import os
import unittest
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library, iq4_xs_mmvq
from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
    CompressedIQ4EmbeddingMethod,
    CompressedIQ4HeadMethod,
)
from sglang.srt.model_loader.qwen4_gguf_xpu import iter_dense_weights
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=8, suite="xpu-unit-test")


def fixture(k, n, seed):
    raw = np.random.default_rng(seed).integers(
        0, 256, (n * k // 256, 136), dtype=np.uint8
    )
    raw[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    weights = gguf.dequantize(
        raw.reshape(-1), gguf.GGMLQuantizationType.IQ4_XS
    ).reshape(n, k)
    return raw.reshape(n, -1), torch.from_numpy(weights).double()


def reference(x, weights):
    values = x.float().numpy().reshape(-1, 32)
    scales = np.minimum(np.abs(values).max(axis=1) / np.float32(127), 65504)
    ratios = np.divide(
        values, scales[:, None], out=np.zeros_like(values), where=scales[:, None] != 0
    )
    codes = np.clip(
        np.copysign(np.floor(np.abs(ratios) + np.float32(0.5)), ratios), -127, 127
    )
    decoded = codes.astype(np.float64) * scales.astype(np.float16)[:, None].astype(
        np.float64
    )
    decoded = torch.from_numpy(decoded.reshape(x.shape))
    return decoded @ weights.T, decoded.abs() @ weights.abs().T


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestIQ4MMVQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 port")
        if not os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"):
            raise unittest.SkipTest("Build and select the IQ4_XS MMVQ library")

    def assert_reference(self, actual, x, weights):
        expected, condition = reference(x, weights)
        self.assertEqual(actual.dtype, torch.float32)
        error = (actual.cpu().double() - expected).abs()
        self.assertTrue(bool((error <= 2e-6 * condition + 1e-5).all()))

    def test_supported_widths_token_groups_tails_and_dynamic_graph(self):
        generator = torch.Generator().manual_seed(42)
        for k in (2560, 6144):
            raw, weights = fixture(k, 17, 42)
            packed = torch.from_numpy(raw).to("xpu")
            for tokens in (0, 1, 2, 3, 4, 7):
                with self.subTest(k=k, tokens=tokens):
                    x = torch.randn((tokens, k), generator=generator).bfloat16()
                    if tokens:
                        x[0, :32] = 0
                    actual = iq4_xs_mmvq(x.to("xpu"), packed)
                    self.assertEqual(actual.shape, (tokens, 17))
                    self.assert_reference(actual, x, weights)
            xgpu = x.to("xpu")
            iq4_xs_mmvq(xgpu, packed)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = iq4_xs_mmvq(xgpu, packed)
            for scale in (0.5, -2.0, 1.0):
                changed = (x * scale).bfloat16()
                xgpu.copy_(changed)
                graph.replay()
                self.assert_reference(actual, changed, weights)

    def test_method_retains_original_bytes_and_fp32_output(self):
        raw, weights = fixture(2560, 64, 19)
        layer = torch.nn.Module()
        method = CompressedIQ4HeadMethod()
        with torch.device("xpu"):
            method.create_weights(layer, 2560, [64], 2560, 64, torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight.shape, raw.shape)
        self.assertEqual(layer.weight.output_dim, 0)
        layer.weight.copy_(torch.from_numpy(raw))
        method.process_weights_after_loading(layer)
        x = torch.randn((1, 2560), generator=torch.Generator().manual_seed(71))
        actual = method.apply(layer, x.to("xpu"))
        self.assert_reference(actual, x, weights)
        self.assertTrue(np.array_equal(layer.weight.cpu().numpy(), raw))

    def test_layout_quantization_preserves_original_q8_bytes(self):
        library = _library(os.environ["SGLANG_XPU_GGUF_IQ4_SYCL_LIB"])
        generator = torch.Generator().manual_seed(42)
        for k in (2560, 6144):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for gdn in (False, True) if k == 6144 else (False,):
                    with self.subTest(k=k, dtype=dtype, gdn=gdn):
                        x = torch.randn((3, k), generator=generator).to(dtype)
                        x[0, :32] = 0
                        xgpu = x.to("xpu")
                        ordered = x
                        if gdn:
                            ordered = (
                                x.view(3, 16, 3, 128).transpose(1, 2).reshape(3, k)
                            )
                        old_input = ordered.float().to("xpu")
                        expected = torch.empty(
                            (3, k // 32 * 36), dtype=torch.uint8, device="xpu"
                        )
                        actual = torch.empty_like(expected)
                        queue = torch.xpu.current_stream().sycl_queue
                        self.assertEqual(
                            library.xpu_iq4_quantize(
                                queue, old_input.data_ptr(), expected.data_ptr(), k, 3
                            ),
                            0,
                        )
                        self.assertEqual(
                            library.xpu_iq4_quantize_layout(
                                queue,
                                xgpu.data_ptr(),
                                actual.data_ptr(),
                                k,
                                3,
                                {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[
                                    dtype
                                ],
                                int(gdn),
                            ),
                            0,
                        )
                        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))

    def test_dense_projection_reconstruction_prefill_and_native_decode(self):
        from sglang.srt.layers.quantization.gguf_iq4_linear import (
            CompressedIQ4ProjectionMethod,
        )
        from sglang.srt.model_loader.qwen4_gguf import convert_qwen4_dense_value

        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        generator = torch.Generator().manual_seed(42)
        for k, gdn in ((2560, False), (6144, False), (6144, True)):
            with self.subTest(k=k, gdn=gdn):
                raw, weights = fixture(k, 17, 42)
                config = SimpleNamespace(
                    linear_num_key_heads=16,
                    linear_num_value_heads=48,
                    linear_value_head_dim=128,
                )
                method = CompressedIQ4ProjectionMethod(
                    torch.bfloat16, gdn_config=config if gdn else None
                )
                layer = torch.nn.Module()
                layer.register_buffer("weight", torch.from_numpy(raw).to("xpu"))
                method.process_weights_after_loading(layer)
                dense = weights.float()
                if gdn:
                    dense = convert_qwen4_dense_value(
                        dense, "gdn_value_columns", config
                    )
                # The GGUF loader copies into contiguous model Parameters;
                # keeping the conversion's transposed stride selects a different
                # GEMM layout and is not that model's prefill reference.
                dense = dense.bfloat16().contiguous().to("xpu")
                self.assertTrue(
                    torch.equal(
                        method.reconstruct_weight(layer).view(torch.int16),
                        dense.view(torch.int16),
                    )
                )
                for tokens in (0, 1, 4, 8, 128):
                    x = torch.randn((tokens, k), generator=generator).bfloat16()
                    xgpu = x.to("xpu")
                    actual = method.apply(layer, xgpu)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    if tokens > 4:
                        expected = torch.nn.functional.linear(xgpu, dense)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int16), expected.view(torch.int16)
                            )
                        )
                    else:
                        ordered = (
                            x.view(tokens, 16, 3, 128)
                            .transpose(1, 2)
                            .reshape(tokens, k)
                            if gdn
                            else x
                        )
                        expected, condition = reference(ordered, weights)
                        error = (actual.cpu().double() - expected).abs()
                        self.assertTrue(
                            bool(
                                (
                                    error
                                    <= condition * 2e-6 + expected.abs() * 0.004 + 8e-5
                                ).all()
                            )
                        )

    def test_loader_yields_compressed_head_without_dequantization(self):
        raw, _ = fixture(2560, 64, 19)
        tensor = SimpleNamespace(
            name="output.weight",
            type_id=23,
            shape=(2560, 64),
            expected_bytes=lambda: raw.nbytes,
        )
        shard = SimpleNamespace(
            tensors=[tensor], read_tensor_range=lambda *args: raw.tobytes()
        )
        config = SimpleNamespace(hidden_size=2560, vocab_size=64)
        loaded = list(
            iter_dense_weights([shard], config, "xpu", compressed_iq4_head=True)
        )
        self.assertEqual(len(loaded), 1)
        name, value = loaded[0]
        self.assertEqual(name, "lm_head.weight")
        self.assertEqual(value.dtype, torch.uint8)
        self.assertTrue(np.array_equal(value.cpu().numpy(), raw))
        tensor.type_id = 22
        with self.assertRaises(ValueError):
            list(iter_dense_weights([shard], config, "xpu", compressed_iq4_head=True))

    def test_embedding_selected_rows_match_expanded_model_bits(self):
        raw, weights = fixture(2560, 64, 42)
        layer = torch.nn.Module()
        method = CompressedIQ4EmbeddingMethod()
        with torch.device("xpu"):
            method.create_weights(layer, 2560, [64], 2560, 64, torch.bfloat16)
        layer.weight.copy_(torch.from_numpy(raw))
        ids = torch.tensor([63, 0, 19, 19], dtype=torch.int64, device="xpu")
        actual = method.embedding(layer, ids)
        expected = weights.bfloat16().index_select(0, ids.cpu())
        self.assertTrue(
            torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        )
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual = method.embedding(layer, ids)
        ids.copy_(torch.tensor([1, 2, 0, 31], dtype=torch.int64))
        graph.replay()
        expected = weights.bfloat16().index_select(0, ids.cpu())
        self.assertTrue(
            torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        )
        self.assertEqual(method.embedding(layer, ids[:0]).shape, (0, 2560))

    def test_unsupported_geometry_is_rejected(self):
        x = torch.empty((1, 2560), device="xpu")
        raw, _ = fixture(2560, 16, 42)
        packed = torch.from_numpy(raw).to("xpu")
        for bad in (packed.cpu(), packed[:, :-1], packed.view(torch.int16)):
            with self.assertRaises(ValueError):
                iq4_xs_mmvq(x, bad)
        with self.assertRaises(ValueError):
            iq4_xs_mmvq(x[:, :-1], packed)
        with self.assertRaises(ValueError):
            iq4_xs_mmvq(x.cpu(), packed.cpu())


if __name__ == "__main__":
    unittest.main()
