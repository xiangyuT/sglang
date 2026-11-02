# SPDX-License-Identifier: Apache-2.0
"""Compressed PLE rows for the single-device GGUF execution path."""

import torch
from sglang.srt.model_loader.gguf_native_reader import open_gguf_shards
from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows
from torch import nn


class Qwen4ExpGGUFEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        matches = [
            (shard, tensor)
            for shard in open_gguf_shards(config.gguf_model_path)
            for tensor in shard.tensors
            if tensor.name == "per_layer_token_embd.weight"
        ]
        if len(matches) != 1:
            raise ValueError("Expected exactly one compressed Qwen4 PLE table")
        self.source = CompressedGGUFRows(*matches[0])
        heads = (config.ngram_size - 1) * config.heads_per_ngram
        self.embedding_dim = config.ple_embed_dim // heads
        if self.source.width != self.embedding_dim or self.source.kind != 20:
            raise ValueError(
                "Expected an IQ4_NL PLE table with the configured row width"
            )
        required_rows = max(
            size + offset
            for size, offset in zip(
                config.gguf_ple_head_vocab_sizes, config.gguf_ple_head_offsets
            )
        )
        if self.source.rows < required_rows:
            raise ValueError(
                "Compressed PLE table is smaller than its hash address space"
            )
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16))
        from sglang.kernels.ops.memory.gguf_ple_sycl import create_pinned_ple

        self._gpu_gather = create_pinned_ple(self.source)

    def allocate_output(self, shape, device):
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(self, input_ids, out=None):
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE IDs must be INT32 or INT64")
        expected = (*input_ids.shape, self.embedding_dim)
        if out is not None and (
            tuple(out.shape) != expected
            or out.dtype != torch.bfloat16
            or out.device != input_ids.device
        ):
            raise ValueError("PLE output shape, dtype or device mismatch")
        if self._gpu_gather is not None and input_ids.device.type == "xpu" and input_ids.numel():
            return self._gpu_gather.gather(input_ids.contiguous(), out)
        result = self.source.decode_rows(
            input_ids.cpu().numpy(), device=input_ids.device, dtype=torch.bfloat16
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    def reduce(self, output):
        return output

    def forward(self, input_ids):
        return self.gather(input_ids)
