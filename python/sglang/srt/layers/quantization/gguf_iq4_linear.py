# SPDX-License-Identifier: Apache-2.0
"""Original IQ4_XS dense projections with bounded BF16 prefill reconstruction."""

import os

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf
from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library, iq4_xs_mmvq
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedIQ4HeadMethod
from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill
from sglang.srt.model_loader.qwen4_gguf import convert_qwen4_dense_value


class CompressedIQ4ProjectionMethod(CompressedIQ4HeadMethod):
    """Q8_1/MMVQ decode; reconstruct the original BF16 matrix for long rows.

    The packed matrix always contains original source blocks. The GDN output
    projection keeps GGUF column order and permutes activations during native
    quantization. Prefill restores the existing SGLang column order instead.
    """

    def __init__(self, params_dtype, *, gdn_config=None):
        self.params_dtype = params_dtype
        self.gdn_config = gdn_config
        if gdn_config is not None and (
            gdn_config.linear_num_key_heads,
            gdn_config.linear_num_value_heads,
            gdn_config.linear_value_head_dim,
        ) != (16, 48, 128):
            raise ValueError("Native IQ4 GDN projection requires 16/48/128 heads")

    def process_weights_after_loading(self, layer):
        library = _library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
        if self.gdn_config is not None and not hasattr(
            library, "xpu_iq4_quantize_layout"
        ):
            raise ValueError("Rebuild the IQ4 library for GDN projection inputs")

    def reconstruct_weight(self, layer):
        width = layer.weight.shape[1] // 136 * 256
        value = dequantize_gguf(layer.weight.reshape(-1), 23, dtype=torch.float32)
        value = value.view(layer.weight.shape[0], width)
        if self.gdn_config is not None:
            value = convert_qwen4_dense_value(
                value, "gdn_value_columns", self.gdn_config
            )
        return value.to(self.params_dtype).contiguous()

    def apply(self, layer, x, bias=None):
        if x.shape[0] == 0:
            return x.new_empty((0, layer.weight.shape[0]))
        if x.shape[0] > 4 or is_gguf_prefill():
            # Only this active matrix is expanded. It is released after GEMM;
            # no permanent BF16 copy competes with the routed-expert cache.
            return F.linear(x, self.reconstruct_weight(layer), bias)
        if bias is None:
            from sglang.kernels.ops.quantization import gguf_dense_bf16

            parts = ((23, layer.weight),)
            if gguf_dense_bf16.supports(x, parts, (False,)):
                return gguf_dense_bf16.project(
                    x, parts, packed_parts=(False,),
                    input_permutation="gdn_16x3x128" if self.gdn_config else None,
                )
        output = iq4_xs_mmvq(
            x,
            layer.weight,
            input_permutation="gdn_16x3x128" if self.gdn_config is not None else None,
        )
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype)


def packed_iq4_rows(shard, tensor):
    if tensor.type_id != 23 or len(tensor.shape) != 2 or tensor.shape[0] % 256:
        raise ValueError("Dense IQ4 installation requires whole original IQ4_XS rows")
    raw = bytearray(shard.read_tensor_range(tensor, 0, tensor.expected_bytes()))
    return torch.frombuffer(raw, dtype=torch.uint8).view(
        tensor.shape[1], tensor.shape[0] // 256 * 136
    )


def partition_dense_rows(module, packed, role, config, *, part=None):
    """Select original encoded output rows for the module's retained TP group.

    Row-parallel column slicing is a separate layout contract. Callers keep
    those original BF16 shards until that contract has native-kernel support.
    """
    from sglang.srt.distributed.utils import get_group_rank_size

    rank, size = get_group_rank_size(getattr(module, "tp_group", None))
    if size == 1:
        return packed
    if size not in (2, 4):
        raise ValueError("Compressed GGUF dense shards require TP2 or TP4")
    if role == "merged_qkvz":
        sizes = (config.linear_num_key_heads * config.linear_key_head_dim,) * 2 + (
            config.linear_num_value_heads * config.linear_value_head_dim,
        ) * 2
        if packed.shape[0] != sum(sizes):
            raise ValueError("Invalid global QKVZ encoded rows")
        return torch.cat([v.chunk(size, dim=0)[rank] for v in packed.split(sizes)])
    if role == "attention_qkv" and part in (1, 2):
        replicas = getattr(module, "num_kv_head_replicas", 1)
        owners = size // replicas
        if packed.shape[0] % owners:
            raise ValueError("KV head partition requires complete rows")
        return packed.chunk(owners, dim=0)[rank // replicas].contiguous()
    if packed.shape[0] % size:
        raise ValueError("Dense output partition requires whole rows")
    return packed.chunk(size, dim=0)[rank].contiguous()


def is_sharded_dense_input(module):
    from sglang.srt.distributed.utils import get_group_rank_size

    return get_group_rank_size(getattr(module, "tp_group", None))[1] > 1


def install_iq4_dense_projections(model, shards, config, device):
    """Install only eligible matrices, verifying every reconstructed weight bit.

    This runs after the ordinary loader has initialized all dense parameters.
    Its former BF16 matrix is the independent layout reference before replacing
    storage. Expert GPU caches can be bound afterwards to avoid a higher peak.
    """
    tensors = {
        tensor.name: (shard, tensor) for shard in shards for tensor in shard.tensors
    }
    records = []

    def install(module, packed, role, index, *, gdn=False):
        original = module.weight.data
        packed = partition_dense_rows(module, packed, role, config)
        if not original.is_contiguous():
            raise ValueError(
                f"IQ4 dense installation requires contiguous weights: {index} {role}"
            )
        method = CompressedIQ4ProjectionMethod(
            original.dtype, gdn_config=config if gdn else None
        )
        packed = packed.to(device)
        candidate = torch.nn.Module()
        candidate.register_buffer("weight", packed)
        restored = method.reconstruct_weight(candidate)
        if restored.shape != original.shape or not torch.equal(
            restored.view(torch.int16), original.view(torch.int16)
        ):
            raise ValueError(
                f"IQ4 dense reconstruction differs in layer {index} {role}"
            )
        method.process_weights_after_loading(candidate)
        before = original.numel() * original.element_size()
        module.weight.data = packed
        module.quant_method = method
        records.append(
            dict(
                layer=index,
                role=role,
                expanded_bytes=before,
                packed_bytes=packed.numel(),
                reconstructed_bits_exact=True,
            )
        )

    for index in range(config.num_hidden_layers):
        prefix = f"blk.{index}."
        layer = model.model.layers[index]
        qkv = tensors.get(prefix + "attn_qkv.weight")
        z = tensors.get(prefix + "attn_gate.weight")
        if qkv and z and qkv[1].type_id == z[1].type_id == 23:
            qkv_rows = convert_qwen4_dense_value(
                packed_iq4_rows(*qkv), "gdn_qkv_rows", config
            )
            z_rows = convert_qwen4_dense_value(
                packed_iq4_rows(*z), "gdn_value_rows", config
            )
            install(
                layer.linear_attn.in_proj_qkvz,
                torch.cat((qkv_rows, z_rows)),
                "merged_qkvz",
                index,
            )
        out = tensors.get(prefix + "ssm_out.weight")
        if out and out[1].type_id == 23 and not is_sharded_dense_input(layer.linear_attn.out_proj):
            install(
                layer.linear_attn.out_proj,
                packed_iq4_rows(*out),
                "ssm_out",
                index,
                gdn=True,
            )
        out = tensors.get(prefix + "attn_output.weight")
        if out and out[1].type_id == 23 and not is_sharded_dense_input(layer.o_proj):
            install(
                layer.o_proj,
                packed_iq4_rows(*out),
                "attention_out",
                index,
            )
    return records
