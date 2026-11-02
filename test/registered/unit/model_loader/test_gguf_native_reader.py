"""GSQ/RCO headers must remain readable without an enum entry for Q2_0."""

import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sglang.srt.model_loader.gguf_native_reader import GGUFFile, open_gguf_shards
from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows
from sglang.srt.model_loader.qwen4_gguf import (
    convert_qwen4_dense_value,
    qwen4_checkpoint_target,
)
from sglang.srt.utils.hf_transformers.gguf_native import (
    build_gguf_tokenizer,
    read_gguf_architecture,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _string(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def _write(path, tensors, metadata=None):
    metadata = metadata or {}
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key)
        if isinstance(value, str):
            header += struct.pack("<I", 8) + _string(value)
        else:
            header += struct.pack("<II", 4, value)
    payload = b""
    for name, dims, kind, data in tensors:
        offset = (len(payload) + 31) // 32 * 32
        payload += b"\0" * (offset - len(payload)) + data
        header += _string(name) + struct.pack("<I", len(dims))
        header += struct.pack(f"<{len(dims)}QIQ", *dims, kind, offset)
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + payload)


class TestNativeGgufReader(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_q20_type_and_bounded_byte_reads(self):
        path = self.root / "model.gguf"
        payload = bytes(range(36))
        _write(path, [("blk.0.ffn_down_exps.weight", [64, 2], 42, payload)])
        reader = GGUFFile(path)
        tensor = reader.tensors[0]
        self.assertEqual(
            (tensor.type_id, tensor.type_name, tensor.expected_bytes()),
            (42, "Q2_0", 36),
        )
        self.assertEqual(reader.read_tensor_range(tensor, 15, 7), payload[15:22])
        for offset, length in [(-1, 1), (0, -1), (35, 2)]:
            with self.assertRaises(ValueError):
                reader.read_tensor_range(tensor, offset, length)
        with self.assertRaises(ValueError):
            reader.read_tensor_range(GGUFFile(path).tensors[0], 0, 1)

    def test_compressed_row_selection_keeps_ids_and_table_read_only(self):
        path = self.root / "rows.gguf"
        blocks = np.zeros((3, 18), dtype=np.uint8)
        blocks[:, :2] = np.array([0.25], np.float16).view(np.uint8)
        blocks[:, 2:] = np.array([0x00, 0x55, 0xAA], np.uint8)[:, None]
        _write(path, [("down", [64, 3], 42, blocks.tobytes())])
        before = path.read_bytes()
        shard = GGUFFile(path)
        rows = CompressedGGUFRows(shard, shard.tensors[0])
        ids = np.array([[2, 0, 2], [1, 0, 1]], dtype=np.int64)
        actual = rows.decode_rows(ids, device="cpu", dtype=torch.float32)
        expected = torch.tensor([[0.25, -0.25, 0.25], [0, -0.25, 0]])
        self.assertTrue(torch.equal(actual, expected[..., None].expand(2, 3, 64)))
        self.assertEqual(rows.packed_rows(ids).numel(), 6 * 18)
        for selection in (np.array(0), ids, np.array([], dtype=np.int64)):
            packed = rows.packed_rows(selection)
            self.assertTrue(packed.numpy().flags.writeable)
            self.assertFalse(np.shares_memory(packed.numpy(), rows._mapping))
            if packed.numel():
                packed.reshape(-1)[0] ^= 255
            self.assertEqual(path.read_bytes(), before)
        for invalid in [np.array([-1]), np.array([3]), np.array([0.5])]:
            with self.assertRaises(ValueError):
                rows.packed_rows(invalid)
        for start, stop in [(0, 0), (0, 3), (1, 3), (2, 3)]:
            packed = rows.packed_row_range(start, stop)
            self.assertTrue(torch.equal(packed, torch.from_numpy(blocks[start:stop])))
            self.assertFalse(np.shares_memory(packed.numpy(), rows._mapping))
            if packed.numel():
                packed.reshape(-1)[0] ^= 255
            self.assertEqual(path.read_bytes(), before)
        for start, stop in [(-1, 1), (0, 4), (2, 1), (0.5, 1)]:
            with self.assertRaises(ValueError):
                rows.packed_row_range(start, stop)
        self.assertEqual(
            rows.decode_rows(np.empty((0, 2), dtype=np.int64), device="cpu").shape,
            (0, 2, 64),
        )
        self.assertEqual(path.read_bytes(), before)

    def test_compressed_ple_uses_selected_rows_only(self):
        from sglang.srt.models.qwen4_exp_gguf_ple import Qwen4ExpGGUFEmbedding

        path = self.root / "ple.gguf"
        blocks = np.zeros((5, 18), dtype=np.uint8)
        blocks[:, :2] = np.array([0.125], np.float16).view(np.uint8)
        blocks[:, 2:] = np.arange(5, dtype=np.uint8)[:, None]
        _write(path, [("per_layer_token_embd.weight", [160, 1], 20, blocks.tobytes())])
        config = SimpleNamespace(
            gguf_model_path=str(path),
            ngram_size=2,
            heads_per_ngram=1,
            ple_embed_dim=160,
            gguf_ple_head_vocab_sizes=[1],
            gguf_ple_head_offsets=[0],
        )
        embedding = Qwen4ExpGGUFEmbedding(config)
        ids = torch.zeros((2, 1), dtype=torch.int64)
        output = embedding.allocate_output((2, 1, 160), "cpu")
        self.assertIs(embedding.gather(ids, out=output), output)
        self.assertTrue(torch.equal(output, embedding(ids)))
        self.assertEqual(sum(p.numel() for p in embedding.parameters()), 0)
        for invalid in [torch.ones(1, dtype=torch.long), torch.zeros(1)]:
            with self.assertRaises(ValueError):
                embedding(invalid)
        with self.assertRaises(ValueError):
            embedding.gather(ids, out=torch.empty(2, 1, 160))

    def test_mtp_fused_expert_sections_keep_original_bytes_at_boundaries(self):
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )
        from sglang.srt.model_loader.qwen4_mtp_gguf_xpu import mtp_expert_sources

        path = self.root / "mtp.gguf"
        fused = (
            np.arange(3 * 128 * 18, dtype=np.uint32)
            .astype(np.uint8)
            .reshape(3, 128, 18)
        )
        down = (fused[:, :64] ^ 91).copy()
        _write(
            path,
            [
                (
                    "mtp.layers.0.mlp.experts.gate_up_proj",
                    [64, 128, 3],
                    42,
                    fused.tobytes(),
                ),
                ("mtp.layers.0.mlp.experts.down_proj", [64, 64, 3], 42, down.tobytes()),
            ],
        )
        before = path.read_bytes()
        config = SimpleNamespace(
            hidden_size=64, moe_intermediate_size=64, num_experts=3
        )
        sources = mtp_expert_sources(GGUFFile(path), config)
        self.assertIs(sources[0].source, sources[1].source)
        for source, reference in zip(sources, [fused[:, :64], fused[:, 64:], down]):
            reference = torch.from_numpy(reference.copy().reshape(192, 18))
            ids = np.array([[191, 0, 64], [63, 127, 128]])
            self.assertTrue(torch.equal(source.packed_rows(ids), reference[ids]))
            for start, stop in [
                (0, 0),
                (192, 192),
                (0, 192),
                (63, 65),
                (127, 129),
                (64, 128),
            ]:
                actual = source.packed_row_range(start, stop)
                self.assertTrue(torch.equal(actual, reference[start:stop]))
                if actual.numel():
                    actual.reshape(-1)[0] ^= 255
                self.assertEqual(path.read_bytes(), before)
            for ids in [np.array([-1]), np.array([192]), np.array([0.5])]:
                with self.assertRaises(ValueError):
                    source.packed_rows(ids)
            for bounds in [(-1, 1), (191, 193), (3, 2), (0.5, 1)]:
                with self.assertRaises(ValueError):
                    source.packed_row_range(*bounds)
        method = CompressedGGUFMoEMethod(2)
        method.create_weights(None, 3, 64, 64, torch.bfloat16)
        layer = torch.nn.Module()
        method.bind_rows(layer, sources, "cpu")
        self.assertEqual(layer.gguf_gate.shape, (2, 64 * 18))
        with self.assertRaises(RuntimeError):
            method.bind_rows(layer, sources, "cpu")
        with self.assertRaises(ValueError):
            mtp_expert_sources(
                GGUFFile(path),
                SimpleNamespace(
                    hidden_size=64, moe_intermediate_size=64, num_experts=4
                ),
            )

    def test_raw_mtp_norm_parameters_are_not_runtime_pack_scales(self):
        from sglang.srt.model_loader.qwen4_mtp_gguf_xpu import iter_mtp_dense_weights

        path = self.root / "norms.gguf"
        norm = np.array([-0.5, 0.25, 2.0], dtype=np.float32)
        _write(path, [("mtp.pre_fc_norm_hidden.weight", [3], 0, norm.tobytes())])
        values = dict(iter_mtp_dense_weights(GGUFFile(path), "cpu"))
        self.assertTrue(
            torch.equal(values["mtp.pre_fc_norm_hidden.weight"], torch.from_numpy(norm))
        )

    def _split(self, duplicate=False):
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        _write(
            first,
            [("output.weight", [1], 0, b"\0" * 4)],
            {
                "general.architecture": "qwen4exp",
                "split.count": 2,
                "split.no": 0,
                "split.tensors.count": 2,
            },
        )
        _write(
            second,
            [
                (
                    "output.weight" if duplicate else "per_layer_token_embd.weight",
                    [32],
                    20,
                    b"\0" * 18,
                )
            ],
            {"split.count": 2, "split.no": 1},
        )
        return first, second

    def test_second_shard_resolves_first_header(self):
        first, second = self._split()
        shards = open_gguf_shards(second)
        self.assertEqual([s.path for s in shards], [first, second])
        self.assertEqual(shards[0].metadata["general.architecture"], "qwen4exp")
        self.assertEqual(sum(len(s.tensors) for s in shards), 2)
        self.assertEqual(read_gguf_architecture(second), "qwen4exp")

    def test_missing_shard_is_rejected(self):
        first = self.root / "model-00001-of-00002.gguf"
        _write(
            first,
            [("output.weight", [1], 0, b"\0" * 4)],
            {"split.count": 2, "split.no": 0},
        )
        with self.assertRaises(FileNotFoundError):
            open_gguf_shards(first)

    def test_duplicate_across_shards_is_rejected(self):
        first, _ = self._split(duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            open_gguf_shards(first)

    def test_truncated_payload_is_rejected(self):
        path = self.root / "bad.gguf"
        _write(path, [("weight", [64], 42, b"\0" * 17)])
        with self.assertRaisesRegex(ValueError, "range"):
            GGUFFile(path)

    def test_incomplete_tensor_count_is_rejected(self):
        first, _ = self._split()
        _write(
            first,
            [("output.weight", [1], 0, b"\0" * 4)],
            {
                "general.architecture": "qwen4exp",
                "split.count": 2,
                "split.no": 0,
                "split.tensors.count": 3,
            },
        )
        with self.assertRaises(ValueError):
            open_gguf_shards(first)

    def test_qwen35_embedded_controls_keep_checkpoint_ids(self):
        metadata = {
            "tokenizer.ggml.tokens": [
                "a",
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "b",
                "[PAD5]",
                "Ã©",
            ],
            "tokenizer.ggml.token_type": [1, 3, 3, 4, 1, 5, 1],
            "tokenizer.ggml.merges": [],
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
        }
        with patch(
            "sglang.srt.utils.hf_transformers.gguf_native._read_gguf_metadata",
            return_value=metadata,
        ):
            tokenizer = build_gguf_tokenizer("unused.gguf")
        ids = tokenizer.encode(
            "<|im_start|>a<think>b<|im_end|>", add_special_tokens=False
        )
        self.assertEqual(ids, [1, 0, 3, 4, 2])
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "a<think>b")
        self.assertEqual(tokenizer.encode("e\u0301", add_special_tokens=False), [6])


class TestQwen4GdnConversion(unittest.TestCase):
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=2,
        linear_value_head_dim=4,
    )

    def _encoded_rows(self, value, width):
        return (
            value.reshape(2, 3, width, *value.shape[1:])
            .transpose(0, 1)
            .reshape_as(value)
        )

    def test_value_head_and_matrix_orders_are_restored(self):
        original = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        encoded = self._encoded_rows(original, 1)
        self.assertTrue(
            torch.equal(
                convert_qwen4_dense_value(encoded, "gdn_value_heads", self.config),
                original,
            )
        )
        matrix = torch.arange(24 * 7, dtype=torch.float32).reshape(24, 7)
        encoded = self._encoded_rows(matrix, 4)
        self.assertTrue(
            torch.equal(
                convert_qwen4_dense_value(encoded, "gdn_value_rows", self.config),
                matrix,
            )
        )
        self.assertTrue(
            torch.equal(
                convert_qwen4_dense_value(encoded.T, "gdn_value_columns", self.config),
                matrix.T,
            )
        )

    def test_qkv_conv_keeps_qk_and_restores_only_value_heads(self):
        prefix = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        values = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4) + 100
        encoded = torch.cat((prefix, self._encoded_rows(values, 4)))
        expected = torch.cat((prefix, values))
        self.assertTrue(
            torch.equal(
                convert_qwen4_dense_value(encoded, "gdn_qkv_rows", self.config),
                expected,
            )
        )
        self.assertTrue(
            torch.equal(
                convert_qwen4_dense_value(encoded, "gdn_qkv_conv", self.config),
                expected.unsqueeze(1),
            )
        )

    def test_log_scale_and_model_conversion_actions(self):
        original = torch.arange(6, dtype=torch.float32) / 8 - 1
        encoded = -torch.exp(self._encoded_rows(original, 1))
        self.assertTrue(
            torch.allclose(
                convert_qwen4_dense_value(
                    encoded, "gdn_inverse_negative_exp", self.config
                ),
                original,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        with self.assertRaises(ValueError):
            convert_qwen4_dense_value(-encoded, "gdn_inverse_negative_exp", self.config)
        self.assertEqual(
            qwen4_checkpoint_target("blk.1.ple_norm_key.weight")[1],
            "undo_gemma_norm_offset",
        )
        self.assertEqual(
            qwen4_checkpoint_target("blk.3.indexer.q_proj.weight")[1], "indexer_q_part"
        )
        with self.assertRaises(KeyError):
            qwen4_checkpoint_target("blk.3.unmapped.weight")


if __name__ == "__main__":
    unittest.main()
