# SPDX-License-Identifier: Apache-2.0
"""Retain original mixed-format Qwen4 dense matrices instead of BF16 copies."""

import logging
import os

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_dense_q8 import (
    DENSE_TYPES,
    dense_q8_projection,
    permute_iq3_fields,
    permute_q8_fields,
)
from sglang.kernels.ops.quantization.gguf_sycl import _library, dequantize_gguf
from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library as _iq4_library
from sglang.srt.layers.quantization.gguf_iq4_linear import (
    CompressedIQ4ProjectionMethod, partition_dense_rows, is_sharded_dense_input,
)
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedIQ4HeadMethod
from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.srt.model_loader.qwen4_gguf import convert_qwen4_dense_value

logger = logging.getLogger(__name__)

class CompressedDenseProjectionMethod(CompressedIQ4HeadMethod):
    def __init__(self, kinds, width, params_dtype, *, gdn_config=None):
        self.kinds = tuple(kinds)
        self.width = width
        self.params_dtype = params_dtype
        self.gdn_config = gdn_config
        self._scratch = {}
        if gdn_config is not None and (
            gdn_config.linear_num_key_heads,
            gdn_config.linear_num_value_heads,
            gdn_config.linear_value_head_dim,
        ) != (16, 48, 128):
            raise ValueError("Native dense GDN input requires 16/48/128 heads")

    def parts(self, layer):
        return tuple(
            (kind, getattr(layer, "weight" if i == 0 else f"gguf_dense_part_{i}"))
            for i, kind in enumerate(self.kinds)
        )

    def process_weights_after_loading(self, layer):
        library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
        iq4 = _iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
        if not (
            hasattr(library, "xpu_gguf_dense_q8")
            and hasattr(iq4, "xpu_iq4_mmvq_output_stride")
        ):
            raise ValueError("Rebuild native GGUF/IQ4 libraries for mixed projections")
        # Layout is fixed before graph capture. Repeated loader callbacks or
        # later environment changes must never reinterpret packed bytes as raw.
        if hasattr(layer, "gguf_dense_iq3_packed_parts"):
            if not hasattr(layer, "gguf_dense_q8_soa_parts"):
                layer.gguf_dense_q8_soa_parts = (False,) * len(self.kinds)
            return
        replacements, layouts, q8_layouts = [], [], []
        enabled = os.environ.get("SGLANG_XPU_GGUF_IQ3_PACKED_S4") == "1"
        q8_enabled = os.environ.get("SGLANG_XPU_GGUF_DENSE_Q8_SOA") == "1"
        for kind, packed in self.parts(layer):
            eligible = (
                enabled and kind == 21 and self.width in (2560, 6144)
                and packed.shape[0] % 16 == 0
            )
            q8_eligible = (
                q8_enabled and kind == 8 and self.width in (640, 2560, 6144)
                and 16 <= packed.shape[0] <= 16384 and packed.shape[0] % 16 == 0
            )
            replacements.append(permute_iq3_fields(packed, self.width) if eligible
                                else permute_q8_fields(packed, self.width) if q8_eligible
                                else packed)
            layouts.append(eligible)
            q8_layouts.append(q8_eligible)
        for i, value in enumerate(replacements):
            name = "weight" if i == 0 else f"gguf_dense_part_{i}"
            previous = getattr(layer, name)
            if isinstance(previous, torch.nn.Parameter):
                previous.data = value
            else:
                setattr(layer, name, value)
        layer.gguf_dense_iq3_packed_parts = tuple(layouts)
        layer.gguf_dense_q8_soa_parts = tuple(q8_layouts)

    def iq3_layouts(self, layer):
        return getattr(layer, "gguf_dense_iq3_packed_parts", (False,) * len(self.kinds))

    def q8_layouts(self, layer):
        return getattr(layer, "gguf_dense_q8_soa_parts", (False,) * len(self.kinds))

    def reconstruct_weight(self, layer):
        fused = (os.environ.get("SGLANG_XPU_GGUF_RECONSTRUCT_BF16") == "1"
                 and self.params_dtype == torch.bfloat16
                 and not torch.xpu.is_current_stream_capturing())
        matrices = []
        for (kind, packed), field_packed, q8_soa in zip(
            self.parts(layer), self.iq3_layouts(layer), self.q8_layouts(layer)
        ):
            if fused:
                from sglang.kernels.ops.quantization.gguf_reconstruct_bf16 import reconstruct_bf16
                if field_packed:
                    packed = permute_iq3_fields(packed, self.width, inverse=True)
                matrices.append(reconstruct_bf16(packed, kind, self.width, q8_soa=q8_soa))
            else:
                if q8_soa:
                    packed = permute_q8_fields(packed, self.width, inverse=True)
                elif field_packed:
                    packed = permute_iq3_fields(packed, self.width, inverse=True)
                matrices.append(dequantize_gguf(packed.reshape(-1), kind, dtype=torch.float32)
                                .view(packed.shape[0], self.width).to(self.params_dtype))
        value = matrices[0] if len(matrices) == 1 else torch.cat(matrices)
        if self.gdn_config is not None:
            value = convert_qwen4_dense_value(
                value, "gdn_value_columns", self.gdn_config
            )
        return value.contiguous()

    def _input_scratch(self, x, rows):
        stream = torch.xpu.current_stream(x.device)
        key = (x.device, stream.sycl_queue)
        scratch = self._scratch.get(key)
        required = rows * (self.width // 32) * 36
        if scratch is None:
            scratch = torch.empty(required, dtype=torch.uint8, device=x.device)
            self._scratch[key] = scratch
        if scratch.numel() < required:
            raise RuntimeError("Set Q8 options before model initialization and graph capture")
        return scratch

    def _apply_batch_q8(self, layer, x, wave_rows):
        from sglang.kernels.ops.quantization import gguf_dense_bf16

        scratch = self._input_scratch(x, 32)
        permutation = "gdn_16x3x128" if self.gdn_config else None
        gguf_dense_bf16.quantize_batch(x, scratch, input_permutation=permutation)
        row_bytes = (self.width // 32) * 36
        parts, packed = self.parts(layer), self.iq3_layouts(layer)
        soa = self.q8_layouts(layer)
        outputs = []
        for row in range(0, x.shape[0], wave_rows):
            wave = x[row:row + wave_rows]
            encoded = scratch[row * row_bytes:(row + wave.shape[0]) * row_bytes]
            outputs.append(gguf_dense_bf16.project(
                wave, parts, packed_parts=packed, input_permutation=permutation,
                q8_soa_parts=soa, quantized_input=encoded,
            ))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def apply(self, layer, x, bias=None):
        if 0 < x.shape[0] <= 4 and is_gguf_prefill():
            # A short suffix after a radix hit is still prefill. It must not
            # acquire the decode path's additional Q8 activation quantization.
            return F.linear(x, self.reconstruct_weight(layer), bias)
        wide_q8 = (
            os.environ.get("SGLANG_XPU_GGUF_DENSE_Q8_WAVE_ROWS") == "8"
            and os.environ.get("SGLANG_XPU_GGUF_DENSE_BF16") == "1"
            and x.dtype == torch.bfloat16
            and bias is None
            and bool(self.kinds)
            and all(kind == 8 and soa for kind, soa in zip(self.kinds, self.q8_layouts(layer)))
        )
        wave_rows = 8 if wide_q8 else 4
        batch_quant = wide_q8 and os.environ.get("SGLANG_XPU_GGUF_DENSE_BATCH_QUANT") == "1"
        if x.shape[0] > 4:
            from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

            try:
                mode = get_forward_batch().forward_mode
            except RuntimeError:
                mode = None
            if mode is not None and (mode.is_decode() or mode.is_target_verify()) and x.shape[0] <= 64:
                # Token-row count does not distinguish prefill from batched
                # decode. Preserve the existing Q8 activation arithmetic and
                # compressed weights in bounded waves for the latter.
                if batch_quant and x.shape[0] <= 32:
                    return self._apply_batch_q8(layer, x, wave_rows)
                if x.shape[0] > wave_rows:
                    return torch.cat([self.apply(layer, wave.contiguous(), bias) for wave in x.split(wave_rows)])
            else:
                # Bounded to the active projection; never retain an expanded copy.
                return F.linear(x, self.reconstruct_weight(layer), bias)
        scratch = self._input_scratch(x, 32 if batch_quant else wave_rows)
        if bias is None:
            from sglang.kernels.ops.quantization import gguf_dense_bf16

            parts, packed_parts = self.parts(layer), self.iq3_layouts(layer)
            q8_parts = self.q8_layouts(layer)
            if gguf_dense_bf16.supports(x, parts, packed_parts, q8_parts):
                return gguf_dense_bf16.project(
                    x, parts, packed_parts=packed_parts,
                    input_permutation="gdn_16x3x128" if self.gdn_config else None,
                    scratch=scratch,
                    q8_soa_parts=q8_parts,
                )
        output = dense_q8_projection(
            x,
            self.parts(layer),
            input_permutation="gdn_16x3x128" if self.gdn_config is not None else None,
            scratch=scratch,
            iq3_packed_parts=self.iq3_layouts(layer),
            q8_soa_parts=self.q8_layouts(layer),
        )
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype)


def packed_dense_rows(shard, tensor):
    if tensor.type_id not in DENSE_TYPES or len(tensor.shape) != 2:
        raise ValueError("Unsupported original compressed dense source")
    qk, size = BLOCK_GEOMETRY[GGML_TYPES[tensor.type_id]]
    if tensor.shape[0] % qk:
        raise ValueError("Compressed dense source must contain whole rows")
    raw = bytearray(shard.read_tensor_range(tensor, 0, tensor.expected_bytes()))
    return torch.frombuffer(raw, dtype=torch.uint8).view(
        tensor.shape[1], tensor.shape[0] // qk * size
    )


def install_remaining_dense_projections(model, shards, config, device):
    """Verify former BF16 storage bit-for-bit before replacing it with raw rows."""
    tensors = {t.name: (s, t) for s in shards for t in s.tensors}
    records = []

    def install(module, sources, role, index, *, gdn=False):
        if isinstance(module.quant_method, CompressedIQ4ProjectionMethod):
            return
        original = module.weight.data
        if original.dtype != torch.bfloat16 or not original.is_contiguous():
            raise ValueError(
                f"Expected original contiguous BF16 matrix: {index} {role}"
            )
        if role in ("ssm_out", "attention_out", "shared_down") and is_sharded_dense_input(module):
            return
        parts = []
        for source_index, (source, action) in enumerate(sources):
            kind = source[1].type_id
            packed = convert_qwen4_dense_value(
                packed_dense_rows(*source), action, config
            ).contiguous()
            # QKVZ has two GGUF tensors but four independently sliced segments.
            if role == "merged_qkvz":
                sizes = ((config.linear_num_key_heads * config.linear_key_head_dim,) * 2
                         + (config.linear_num_value_heads * config.linear_value_head_dim,))
                if source_index == 1:
                    sizes = (config.linear_num_value_heads * config.linear_value_head_dim,)
                from sglang.srt.distributed.utils import get_group_rank_size
                rank, size = get_group_rank_size(getattr(module, "tp_group", None))
                packed = torch.cat([v.chunk(size, dim=0)[rank] for v in packed.split(sizes)])
            else:
                packed = partition_dense_rows(module, packed, role, config, part=source_index)
            if parts and parts[-1][0] == kind:
                parts[-1] = (kind, torch.cat((parts[-1][1], packed)))
            else:
                parts.append((kind, packed))
        method = CompressedDenseProjectionMethod(
            [kind for kind, _ in parts],
            original.shape[1],
            original.dtype,
            gdn_config=config if gdn else None,
        )
        candidate = torch.nn.Module()
        for i, (_, packed) in enumerate(parts):
            candidate.register_buffer(
                "weight" if i == 0 else f"gguf_dense_part_{i}", packed.to(device)
            )
        method.process_weights_after_loading(candidate)
        restored = method.reconstruct_weight(candidate)
        if restored.shape != original.shape or not torch.equal(
            restored.view(torch.int16), original.view(torch.int16)
        ):
            raise ValueError(f"Dense reconstruction differs in layer {index} {role}")
        module.weight.data = candidate.weight
        for i in range(1, len(parts)):
            module.register_buffer(
                f"gguf_dense_part_{i}", getattr(candidate, f"gguf_dense_part_{i}")
            )
        module.quant_method = method
        module.gguf_dense_iq3_packed_parts = candidate.gguf_dense_iq3_packed_parts
        module.gguf_dense_q8_soa_parts = candidate.gguf_dense_q8_soa_parts
        records.append(
            dict(
                layer=index,
                role=role,
                kinds=list(method.kinds),
                expanded_bytes=original.numel() * original.element_size(),
                packed_bytes=sum(p.numel() for _, p in method.parts(module)),
                reconstructed_bits_exact=True,
                q8_soa_parts=list(method.q8_layouts(module)),
            )
        )

    for index in range(config.num_hidden_layers):
        prefix = f"blk.{index}."
        layer = model.model.layers[index]

        def source(name):
            return tensors[prefix + name + ".weight"]

        if prefix + "attn_qkv.weight" in tensors:
            install(
                layer.linear_attn.in_proj_qkvz,
                [
                    (source("attn_qkv"), "gdn_qkv_rows"),
                    (source("attn_gate"), "gdn_value_rows"),
                ],
                "merged_qkvz",
                index,
            )
            install(
                layer.linear_attn.out_proj,
                [(source("ssm_out"), "identity")],
                "ssm_out",
                index,
                gdn=True,
            )
        else:
            install(
                layer.qkv_proj,
                [(source(name), "identity") for name in ("attn_q", "attn_k", "attn_v")],
                "attention_qkv",
                index,
            )
            install(
                layer.o_proj,
                [(source("attn_output"), "identity")],
                "attention_out",
                index,
            )
        shared = layer.mlp.shared_expert
        install(
            shared.gate_up_proj,
            [(source(name), "identity") for name in ("ffn_gate_shexp", "ffn_up_shexp")],
            "shared_gate_up",
            index,
        )
        install(
            shared.down_proj,
            [(source("ffn_down_shexp"), "identity")],
            "shared_down",
            index,
        )
    q8_parts = sum(sum(row["q8_soa_parts"]) for row in records)
    if q8_parts:
        logger.info("GGUF dense Q8 SoA16 enabled: %d parts; byte-preserving layout, "
                    "exact32 DPAS decode and verified original BF16 prefill reconstruction",
                    q8_parts)
    return records
