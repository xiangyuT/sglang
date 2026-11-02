# SPDX-License-Identifier: Apache-2.0
"""Connect Qwen4 GGUF sources to the existing SGLang Qwen4 model."""

import logging
import os
import re

import torch
from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf
from sglang.srt.layers.quantization.qwen4_gguf_xpu import Qwen4GGUFXPUConfig
from sglang.srt.model_loader.gguf_native_reader import open_gguf_shards, BLOCK_GEOMETRY, GGML_TYPES
from sglang.srt.model_loader.qwen4_gguf import (
    convert_qwen4_dense_value,
    qwen4_checkpoint_target,
)

logger = logging.getLogger(__name__)


def configure_gguf_determinism():
    """Retain deterministic primitives with an opt-in empty-fill ablation."""
    torch.use_deterministic_algorithms(True)
    if os.environ.get("SGLANG_XPU_GGUF_SKIP_EMPTY_FILL") == "1":
        torch.utils.deterministic.fill_uninitialized_memory = False


def decode_tensor(shard, tensor, device):
    raw = bytearray(shard.read_tensor_range(tensor, 0, tensor.expected_bytes()))
    if tensor.type_id in (0, 1, 30):
        dtype = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}[tensor.type_id]
        value = torch.frombuffer(raw, dtype=dtype).to(device)
    else:
        packed = torch.frombuffer(raw, dtype=torch.uint8).to(device)
        value = dequantize_gguf(packed, tensor.type_id, dtype=torch.float32)
    return value.view(tuple(reversed(tensor.shape)))


def iter_dense_weights(shards, config, device, *, compressed_iq4_head=False):
    indexer = {}
    for shard in shards:
        for tensor in shard.tensors:
            target, action = qwen4_checkpoint_target(tensor.name)
            if action in ("routed_expert", "compressed_ple"):
                continue
            if compressed_iq4_head and target in (
                "lm_head.weight",
                "model.embed_tokens.weight",
            ):
                if tensor.type_id not in (8, 14, 23) or tuple(tensor.shape) != (
                    config.hidden_size,
                    config.vocab_size,
                ):
                    raise ValueError(
                        "Compressed token embedding/head requires supported original GGUF rows"
                    )
                raw = bytearray(
                    shard.read_tensor_range(tensor, 0, tensor.expected_bytes())
                )
                yield (
                    target,
                    torch.frombuffer(raw, dtype=torch.uint8)
                    .to(device)
                    .view(config.vocab_size,
                          config.hidden_size // BLOCK_GEOMETRY[GGML_TYPES[tensor.type_id]][0]
                          * BLOCK_GEOMETRY[GGML_TYPES[tensor.type_id]][1]),
                )
                continue
            value = decode_tensor(shard, tensor, device)
            if action in ("indexer_q_part", "indexer_k_part"):
                parts = indexer.setdefault(target, {})
                parts[action] = value
                if len(parts) == 2:
                    yield (
                        target,
                        torch.cat((parts["indexer_q_part"], parts["indexer_k_part"])),
                    )
                    del indexer[target]
            else:
                yield target, convert_qwen4_dense_value(value, action, config)
    if indexer:
        raise ValueError(f"Incomplete GGUF indexer projection pairs: {list(indexer)}")


def load_qwen4_gguf_xpu(loader, model_config, device):
    from sglang.srt.model_loader.loader import _initialize_model
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    multi_ep = (
        os.environ.get("SGLANG_XPU_GGUF_TP_EP") == "1"
        and parallel.tp_size == parallel.moe_ep_size
        and parallel.tp_size in (2, 4)
        and parallel.pp_size == parallel.attn_dp_size == parallel.attn_cp_size == 1
    )
    if (parallel.tp_size != 1 or parallel.moe_ep_size != 1) and not multi_ep:
        raise ValueError(
            "Qwen4 compressed GGUF execution currently requires one device"
        )
    if model_config.is_draft_model:
        from sglang.srt.model_loader.qwen4_mtp_gguf_xpu import load_qwen4_mtp_gguf_xpu

        return load_qwen4_mtp_gguf_xpu(loader, model_config, device)
    config = model_config.hf_config.text_config
    if not getattr(config, "gguf_model_path", None):
        raise ValueError("Qwen4 native GGUF config is required")
    shards = open_gguf_shards(config.gguf_model_path)
    cache_slots = int(os.environ.get("SGLANG_XPU_GGUF_EXPERT_CACHE_SLOTS", "128"))
    compressed_iq4_head = bool(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
    decode_q8 = os.environ.get("SGLANG_XPU_GGUF_EXPERT_Q8", "0") == "1"
    compressed_iq4_dense = os.environ.get("SGLANG_XPU_GGUF_IQ4_PROJECTIONS", "0") == "1"
    compressed_all_dense = os.environ.get("SGLANG_XPU_GGUF_ALL_PROJECTIONS", "0") == "1"
    if multi_ep:
        logger.info("GGUF TP%d/EP%d: compressed column/shared dense projections; BF16 row shards; budgeted expert partitions",
                    parallel.tp_size, parallel.moe_ep_size)
    if compressed_all_dense and not compressed_iq4_dense:
        raise ValueError("Mixed projections require SGLANG_XPU_GGUF_IQ4_PROJECTIONS=1")
    if compressed_iq4_dense and not compressed_iq4_head:
        raise ValueError("IQ4 projections require SGLANG_XPU_GGUF_IQ4_SYCL_LIB")
    vocab_kinds = {t.name: t.type_id for shard in shards for t in shard.tensors
                   if t.name in ("output.weight", "token_embd.weight")}
    quant_config = Qwen4GGUFXPUConfig(
        cache_slots, compressed_iq4_head=compressed_iq4_head, decode_q8=decode_q8,
        head_kind=vocab_kinds["output.weight"],
        embedding_kind=vocab_kinds["token_embd.weight"],
    )
    # Both FP32 expert fallbacks and BF16 hyper-connection GEMMs can vary in
    # their final bits on B70. Small differences amplify through the recurrent
    # model. Select deterministic Torch primitives before building/compiling it.
    configure_gguf_determinism()
    logger.info(
        "Qwen4 GGUF XPU: deterministic Torch algorithms enabled before model build; "
        "fill_uninitialized_memory=%s",
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    with set_default_torch_dtype(model_config.dtype):
        with device:
            model = _initialize_model(model_config, loader.load_config, quant_config)
        experts = {}
        for shard in shards:
            for tensor in shard.tensors:
                match = re.fullmatch(
                    r"blk\.(\d+)\.ffn_(gate|up|down)_exps.weight", tensor.name
                )
                if match:
                    experts.setdefault(int(match[1]), {})[match[2]] = (shard, tensor)

        def bind_expert_caches():
            for index in range(config.num_hidden_layers):
                sources = experts.get(index, {})
                if set(sources) != {"gate", "up", "down"}:
                    raise ValueError(f"Incomplete GGUF expert sources in layer {index}")
                layer = model.model.layers[index].mlp.experts
                layer.quant_method.bind(
                    layer, [sources[k] for k in ("gate", "up", "down")], device
                )

        if not compressed_iq4_dense:
            bind_expert_caches()
        loaded = model.load_weights(
            iter_dense_weights(
                shards, config, device, compressed_iq4_head=compressed_iq4_head
            )
        )
        missing = set(dict(model.named_parameters())) - loaded
        if missing:
            raise ValueError(
                f"GGUF did not initialize all model parameters: {sorted(missing)}"
            )
        if compressed_iq4_dense:
            from sglang.srt.layers.quantization.gguf_iq4_linear import (
                install_iq4_dense_projections,
            )

            records = install_iq4_dense_projections(model, shards, config, device)
            saved = sum(r["expanded_bytes"] - r["packed_bytes"] for r in records)
            logger.info(
                "Qwen4 GGUF dense IQ4: %d projections, all reconstructed BF16 matrices bit-exact, %.3f GiB released; Q8_1 decode / reconstructed BF16 prefill",
                len(records),
                saved / 2**30,
            )
            if compressed_all_dense:
                from sglang.srt.layers.quantization.gguf_dense_linear import (
                    install_remaining_dense_projections,
                )

                torch.xpu.synchronize()
                torch.xpu.empty_cache()
                before_free = torch.xpu.mem_get_info()[0]
                before_allocated = torch.xpu.memory_allocated()
                mixed = install_remaining_dense_projections(
                    model, shards, config, device
                )
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
                after_free = torch.xpu.mem_get_info()[0]
                after_allocated = torch.xpu.memory_allocated()
                model.gguf_dense_storage_receipt = records + mixed
                model.gguf_dense_memory_receipt = dict(
                    before_driver_free=before_free,
                    after_driver_free=after_free,
                    before_allocated=before_allocated,
                    after_allocated=after_allocated,
                )
                logger.info(
                    "Qwen4 GGUF mixed dense: %d additional projections, original BF16 reconstruction bit-exact, %.3f GiB storage reduction; original compressed weights / Q8_1 decode",
                    len(mixed),
                    sum(r["expanded_bytes"] - r["packed_bytes"] for r in mixed) / 2**30,
                )
                logger.info(
                    "Qwen4 GGUF mixed dense actual memory: driver free %.3f -> %.3f GiB; Torch allocated %.3f -> %.3f GiB",
                    before_free / 2**30,
                    after_free / 2**30,
                    before_allocated / 2**30,
                    after_allocated / 2**30,
                )
            torch.xpu.empty_cache()
            bind_expert_caches()
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
    if os.environ.get("SGLANG_XPU_GGUF_ROUTER_FP32") == "1":
        from sglang.srt.layers.quantization.gguf_router_fp32 import install_router_fp32

        model.gguf_router_fp32_receipt = install_router_fp32(model)
        logger.info(
            "Qwen4 GGUF target router: %d gates, BF16 weights/inputs, FP32 logits "
            "in prefill/decode/verify; draft loader unchanged",
            len(model.gguf_router_fp32_receipt),
        )
    if os.environ.get("SGLANG_XPU_GGUF_TP_PREFILL_FP32") in ("1", "separate"):
        from sglang.srt.layers.quantization.gguf_tp_prefill import install_prefill_fp32

        install_prefill_fp32(model)
    if os.environ.get("SGLANG_XPU_GGUF_HC_SYCL") == "1":
        from sglang.kernels.ops.gemm.gguf_hc_xpu import install

        records = install(model)
        logger.info(
            "Qwen4 GGUF native HC: %d modules, %d paired down/inject layouts",
            len(records),
            sum(r["paired"] for r in records),
        )
    if os.environ.get("SGLANG_XPU_GGUF_IQ4_ROW_BF16_LIB"):
        from sglang.kernels.ops.quantization.gguf_iq4_bf16_row import (
            install_iq4_bf16_rows,
        )

        records = install_iq4_bf16_rows(model, shards, config, device)
        logger.info(
            "Qwen4 GGUF IQ4/BF16 row cache: %d TP2 projections prepared, "
            "%.3f GiB additional cache; original prefill weights retained",
            len(records),
            sum(r["cache_bytes"] for r in records) / 2**30,
        )
    logger.info(
        "Qwen4 GGUF XPU loaded: %d dense parameter names, %d expert-cache slots/layer",
        len(loaded),
        model.model.layers[0].mlp.experts.quant_method.cache_slots,
    )
    if compressed_iq4_head:
        logger.info(
            "Qwen4 GGUF token embedding/head: original %s/%s weights; head uses Q8_1 activations and FP32 logits",
            GGML_TYPES[vocab_kinds["token_embd.weight"]],
            GGML_TYPES[vocab_kinds["output.weight"]],
        )
    if decode_q8:
        logger.info(
            "Qwen4 GGUF single-row expert path: Strata finite Q8_1 activations and lane8 DP4A; prefill=%s",
            ("grouped INT8 with FP16 fallback" if (os.environ.get("SGLANG_XPU_GGUF_PREFILL_F16") == "1" and os.environ.get("SGLANG_XPU_GGUF_PREFILL_I8") == "1")
             else "grouped FP16" if os.environ.get("SGLANG_XPU_GGUF_PREFILL_F16") == "1"
             else "FP32 route reference")
        )
    return model
