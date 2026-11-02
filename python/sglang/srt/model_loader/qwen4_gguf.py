# SPDX-License-Identifier: Apache-2.0
"""Configuration for text-only Qwen3.8-Flash-Next GSQ/RCO GGUF checkpoints."""

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.model_loader.gguf_native_reader import open_gguf_shards


def build_qwen4_gguf_config(path: str) -> Qwen4ExpConfig:
    shards = open_gguf_shards(path)
    metadata = shards[0].metadata
    if metadata.get("general.architecture") != "qwen4exp":
        raise ValueError("Expected the qwen4exp GGUF architecture")

    def required(suffix):
        return metadata[f"qwen4exp.{suffix}"]

    layers = int(required("block_count"))
    hidden = int(required("embedding_length"))
    interval = int(required("full_attention_interval"))
    head_dim = int(required("attention.key_length"))
    if interval < 1 or head_dim < 1:
        raise ValueError("Invalid Qwen4 GGUF attention dimensions")
    sections = list(required("rope.dimension_sections"))[:3]
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": int(required("attention.head_count")),
        "num_key_value_heads": int(required("attention.head_count_kv")),
        "head_dim": head_dim,
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "max_position_embeddings": int(required("context_length")),
        "num_experts": int(required("expert_count")),
        "num_experts_per_tok": int(required("expert_used_count")),
        "moe_intermediate_size": int(required("expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(
            required("expert_shared_feed_forward_length")
        ),
        "rms_norm_eps": float(required("attention.layer_norm_rms_epsilon")),
        "hc_count": int(required("hyper_connection.count")),
        "hc_lowrank": int(required("hyper_connection.low_rank")),
        "linear_conv_kernel_dim": int(required("ssm.conv_kernel")),
        "linear_key_head_dim": int(required("ssm.state_size")),
        "linear_value_head_dim": int(required("ssm.state_size")),
        "linear_num_key_heads": int(required("ssm.group_count")),
        "linear_num_value_heads": int(required("ssm.time_step_rank")),
        "full_attention_interval": interval,
        "layer_types": [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ],
        "indexer_budget": int(required("attention.indexer.top_k")),
        "indexer_head_dim": int(required("attention.indexer.key_length")),
        "indexer_n_heads": int(required("attention.indexer.head_count")),
        "indexer_kv_heads": 1,
        "indexer_compress_ratio": int(
            required("attention.compress_ratios")[interval - 1]
        ),
        # llama.cpp stores zero-based PLE layers; SGLang's config uses one-based IDs.
        "ple_layer_ids": [int(i) + 1 for i in required("ple.layers")],
        "ple_embed_dim": hidden,
        "ple_conv_kernel_size": int(required("ple.conv_kernel")),
        "ngram_size": int(required("ple.ngram_size")),
        "heads_per_ngram": int(required("ple.heads_per_ngram")),
        # Preserve the checkpoint's explicit hash metadata for the compressed PLE reader.
        "gguf_ple_head_vocab_sizes": list(required("ple.head_vocab_sizes")),
        "gguf_ple_head_offsets": list(required("ple.head_offsets")),
        "gguf_ple_layer_multipliers": list(required("ple.layer_multipliers")),
        "gguf_model_path": str(shards[0].path.resolve()),
        "ple_offload_embedding": True,
        "ple_offload_backend": "gguf",
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": int(required("ple.eos_token_id")),
        "tie_word_embeddings": False,
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 0,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(required("rope.freq_base")),
            "partial_rotary_factor": int(required("rope.dimension_count")) / head_dim,
            "mrope_section": sections,
            "mrope_interleaved": True,
        },
    }
    return Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        language_model_only=True,
        tie_word_embeddings=False,
    )


_GGUF_GLOBAL_TARGETS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
    "output_hc_down.weight": "model.hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight",
}

_GGUF_LAYER_TARGETS = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_exps.weight": "mlp.experts.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.down_proj.weight",
    "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "indexer.q_proj.weight": "self_attn.indexer.index_qk_proj.weight",
    "indexer.k_proj.weight": "self_attn.indexer.index_qk_proj.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
}


def qwen4_checkpoint_target(name: str) -> tuple[str, str]:
    """Return a HF checkpoint name plus its required GGUF conversion action.

    Routed expert tensors and the compressed PLE table remain separate resources;
    their actions deliberately do not present them as ordinary dense parameters.
    """
    import re

    if name == "per_layer_token_embd.weight":
        return "ple.ngram_embedding", "compressed_ple"
    if name in _GGUF_GLOBAL_TARGETS:
        target = _GGUF_GLOBAL_TARGETS[name]
    else:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if match is None or match[2] not in _GGUF_LAYER_TARGETS:
            raise KeyError(f"Unmapped Qwen4 GGUF tensor: {name}")
        target = f"model.layers.{match[1]}.{_GGUF_LAYER_TARGETS[match[2]]}"
    if "_exps.weight" in name:
        return target, "routed_expert"
    if name.endswith("indexer.q_proj.weight"):
        return target, "indexer_q_part"
    if name.endswith("indexer.k_proj.weight"):
        return target, "indexer_k_part"
    if name.endswith("ssm_a"):
        return target, "gdn_inverse_negative_exp"
    if name.endswith(("ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias")):
        return target, "gdn_value_heads"
    if name.endswith("attn_gate.weight"):
        return target, "gdn_value_rows"
    if name.endswith("attn_qkv.weight"):
        return target, "gdn_qkv_rows"
    if name.endswith("ssm_out.weight"):
        return target, "gdn_value_columns"
    if name.endswith("ssm_conv1d.weight"):
        return target, "gdn_qkv_conv"
    if name.endswith("ple_conv1d.weight"):
        return target, "conv_channel_axis"
    if name.endswith("ffn_gate_inp_shexp.weight"):
        return target, "singleton_gate_row"
    if (name.endswith("norm.weight") or ".ple_norm_" in name) and not name.endswith(
        "ssm_norm.weight"
    ):
        return target, "undo_gemma_norm_offset"
    return target, "identity"


def _restore_gdn_value_heads(value, config, *, axis=0, width=1):
    keys, values = config.linear_num_key_heads, config.linear_num_value_heads
    if values % keys or value.shape[axis] != values * width:
        raise ValueError("GGUF GDN head layout disagrees with the model dimensions")
    moved = value.movedim(axis, 0)
    return (
        moved.reshape(values // keys, keys, width, *moved.shape[1:])
        .transpose(0, 1)
        .reshape_as(moved)
        .movedim(0, axis)
    )


def convert_qwen4_dense_value(value, action: str, config=None):
    """Undo model conversion steps after dequantization; preserve main dtype."""
    if action == "identity":
        return value
    if action == "undo_gemma_norm_offset":
        return value - 1.0
    if action == "conv_channel_axis":
        return value.unsqueeze(1)
    if action == "singleton_gate_row":
        return value.unsqueeze(0)
    if action.startswith("gdn_"):
        import torch

        if config is None:
            raise ValueError("GDN GGUF conversion requires the model head dimensions")
        if action == "gdn_inverse_negative_exp":
            if not torch.isfinite(value).all() or not (value < 0).all():
                raise ValueError("Qwen4 GGUF ssm_a must contain finite -exp(A_log)")
            return _restore_gdn_value_heads(torch.log(-value), config)
        if action == "gdn_value_heads":
            return _restore_gdn_value_heads(value, config)
        if action == "gdn_value_rows":
            return _restore_gdn_value_heads(
                value, config, width=config.linear_value_head_dim
            )
        if action == "gdn_value_columns":
            return _restore_gdn_value_heads(
                value, config, axis=1, width=config.linear_value_head_dim
            )
        if action in ("gdn_qkv_rows", "gdn_qkv_conv"):
            prefix = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            restored = torch.cat(
                (
                    value[:prefix],
                    _restore_gdn_value_heads(
                        value[prefix:], config, width=config.linear_value_head_dim
                    ),
                )
            )
            return restored.unsqueeze(1) if action == "gdn_qkv_conv" else restored
    raise ValueError(f"Qwen4 tensor action is not a dense conversion: {action}")
