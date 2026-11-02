"""Mixed expert byte bounds and original compressed vocabulary loading."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.gguf_expert_geometry import expert_field_sizes
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFHeadMethod
from sglang.srt.model_loader.qwen4_gguf_xpu import iter_dense_weights
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=1, suite="xpu-unit-test")


class TestMixedExpertGeometry(unittest.TestCase):
    def test_checkpoint_field_sizes_include_each_formats_own_block_headers(self):
        self.assertEqual(expert_field_sizes(21, 20), (704000, 704000, 921600))
        self.assertEqual(expert_field_sizes(21, 8), (704000, 704000, 1740800))
        self.assertEqual(expert_field_sizes(23, 20), (870400, 870400, 921600))
        self.assertEqual(expert_field_sizes(23, 8), (870400, 870400, 1740800))
        self.assertEqual(expert_field_sizes(22), (524800, 524800, 460800))

    def test_unimplemented_packed_layout_combinations_are_rejected(self):
        for gu, down in ((18, 20), (21, 14), (22, 20), (16, 8)):
            with self.subTest(gu=gu, down=down), self.assertRaises(ValueError):
                expert_field_sizes(gu, down)

    def test_vocabulary_parameters_follow_q6_and_q8_row_bytes(self):
        for kind, row_bytes in ((14, 2100), (8, 2720)):
            with self.subTest(kind=kind):
                method = CompressedGGUFHeadMethod(kind)
                layer = torch.nn.Module()
                method.create_weights(layer, 2560, [96], 2560, 96, torch.bfloat16)
                self.assertEqual(layer.weight.shape, (96, row_bytes))
                self.assertEqual(layer.weight.dtype, torch.uint8)
                self.assertEqual(layer.weight.output_dim, 0)

    def test_loader_retains_compressed_bytes_without_eager_vocab_dequantization(self):
        config = SimpleNamespace(hidden_size=2560, vocab_size=96)
        for name, kind, row_bytes in (
            ("output.weight", 14, 2100),
            ("token_embd.weight", 8, 2720),
            ("output.weight", 23, 1360),
        ):
            with self.subTest(name=name, kind=kind):
                total = 96 * row_bytes
                raw = (bytes(range(256)) * ((total + 255) // 256))[:total]
                tensor = SimpleNamespace(
                    name=name, type_id=kind, shape=[2560, 96],
                    expected_bytes=lambda: len(raw),
                )
                shard = SimpleNamespace(
                    tensors=[tensor], read_tensor_range=lambda *args: raw,
                )
                with patch("sglang.srt.model_loader.qwen4_gguf_xpu.decode_tensor",
                           side_effect=AssertionError("Vocabulary was expanded")):
                    loaded = list(iter_dense_weights(
                        [shard], config, "cpu", compressed_iq4_head=True,
                    ))
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0][1].shape, (96, row_bytes))
                self.assertEqual(loaded[0][1].numpy().tobytes(), raw)


if __name__ == "__main__":
    unittest.main()
