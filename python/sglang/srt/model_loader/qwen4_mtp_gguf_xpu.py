# SPDX-License-Identifier: Apache-2.0
"""Load the original Q2_0 MTP GGUF into SGLang's Qwen4 NEXTN model."""

import logging
import os

import torch

from sglang.srt.layers.quantization.qwen4_gguf_xpu import Qwen4GGUFXPUConfig
from sglang.srt.model_loader.gguf_native_reader import GGUFFile, open_gguf_shards, BLOCK_GEOMETRY, GGML_TYPES
from sglang.srt.model_loader.gguf_rows import (
    CompressedGGUFExpertSlice,
    CompressedGGUFRows,
)
from sglang.srt.model_loader.qwen4_gguf_xpu import (
    configure_gguf_determinism,
    decode_tensor,
)

logger = logging.getLogger(__name__)

_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
_DOWN = "mtp.layers.0.mlp.experts.down_proj"


def mtp_expert_sources(shard, config):
    tensors = {tensor.name: tensor for tensor in shard.tensors}
    expected = {
        _GATE_UP: (
            config.hidden_size,
            2 * config.moe_intermediate_size,
            config.num_experts,
        ),
        _DOWN: (config.moe_intermediate_size, config.hidden_size, config.num_experts),
    }
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if tensor is None or tensor.type_id != 42 or tuple(tensor.shape) != shape:
            raise ValueError(f"MTP requires original Q2_0 expert geometry: {name}")
    fused = CompressedGGUFRows(shard, tensors[_GATE_UP])
    intermediate = config.moe_intermediate_size
    return [
        CompressedGGUFExpertSlice(fused, 0, intermediate),
        CompressedGGUFExpertSlice(fused, intermediate, 2 * intermediate),
        CompressedGGUFRows(shard, tensors[_DOWN]),
    ]


def iter_mtp_dense_weights(shard, device):
    for tensor in shard.tensors:
        if tensor.name in (_GATE_UP, _DOWN):
            continue
        if not tensor.name.startswith("mtp.") or tensor.type_id not in (0, 30):
            raise ValueError(f"Unexpected raw MTP dense tensor: {tensor.name}")
        # The raw GGUF retains HF Gemma parameters. Only Strata's separate
        # runtime pack adds 1; no norm offset conversion belongs here.
        yield tensor.name, decode_tensor(shard, tensor, device)


def load_qwen4_mtp_gguf_xpu(loader, model_config, device):
    from sglang.srt.model_loader.loader import _initialize_model
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    path = os.environ.get("SGLANG_XPU_GGUF_MTP_PATH")
    if not path:
        raise ValueError("GGUF NEXTN requires SGLANG_XPU_GGUF_MTP_PATH")
    if not os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"):
        raise ValueError("GGUF NEXTN requires the compressed IQ4 vocabulary library")
    if model_config.hf_config.architectures != ["Qwen4ExpForCausalLMMTP"]:
        raise ValueError("GGUF NEXTN requires Qwen4ExpForCausalLMMTP")
    config = model_config.hf_config.text_config
    shard = GGUFFile(path)
    if shard.metadata.get("general.architecture") != "qwen4exp-mtp":
        raise ValueError("Expected the raw qwen4exp-mtp GGUF")
    sources = mtp_expert_sources(shard, config)
    target_shards = open_gguf_shards(config.gguf_model_path)
    vocab_kinds = {t.name: t.type_id for target in target_shards for t in target.tensors
                   if t.name in ("output.weight", "token_embd.weight")}
    quant_config = Qwen4GGUFXPUConfig(
        int(os.environ.get("SGLANG_XPU_GGUF_MTP_CACHE_SLOTS", "512")),
        compressed_iq4_head=True,
        decode_q8=os.environ.get("SGLANG_XPU_GGUF_EXPERT_Q8", "0") == "1",
        head_kind=vocab_kinds["output.weight"],
        embedding_kind=vocab_kinds["token_embd.weight"],
    )
    configure_gguf_determinism()
    with set_default_torch_dtype(model_config.dtype):
        with device:
            model = _initialize_model(model_config, loader.load_config, quant_config)
        loaded = model.load_weights(iter_mtp_dense_weights(shard, device))
        # Initialize both modules before the worker shares target Parameters.
        # Retain the target GGUF vocabulary formats and matching module methods.
        wanted = {
            "token_embd.weight": (
                "model.embed_tokens.weight",
                model.model.embed_tokens,
            ),
            "output.weight": ("lm_head.weight", model.lm_head),
        }
        for target in target_shards:
            for tensor in target.tensors:
                if tensor.name not in wanted:
                    continue
                name, module = wanted.pop(tensor.name)
                if tensor.type_id not in (8, 14, 23) or tuple(tensor.shape) != (
                    config.hidden_size,
                    config.vocab_size,
                ):
                    raise ValueError(
                        "MTP shared vocabulary requires supported original GGUF rows"
                    )
                raw = bytearray(
                    target.read_tensor_range(tensor, 0, tensor.expected_bytes())
                )
                packed = torch.frombuffer(raw, dtype=torch.uint8).view(
                    config.vocab_size,
                    config.hidden_size // BLOCK_GEOMETRY[GGML_TYPES[tensor.type_id]][0]
                    * BLOCK_GEOMETRY[GGML_TYPES[tensor.type_id]][1],
                )
                weight_loader = getattr(
                    module.weight, "weight_loader", default_weight_loader
                )
                # The loader can copy CPU rows directly into its allocated XPU
                # parameter. A full temporary device vocabulary raises the
                # initialization peak without changing any stored weight byte.
                weight_loader(module.weight, packed)
                loaded.add(name)
        if wanted:
            raise ValueError(f"Missing MTP shared vocabulary tensors: {sorted(wanted)}")
        missing = set(dict(model.named_parameters())) - loaded
        if missing:
            raise ValueError(
                f"MTP GGUF did not initialize parameters: {sorted(missing)}"
            )
        experts = model.model.layers[0].mlp.experts
        experts.quant_method.bind_rows(experts, sources, device)
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
    if os.environ.get("SGLANG_XPU_GGUF_HC_SYCL") == "1":
        from sglang.kernels.ops.gemm.gguf_hc_xpu import install

        records = install(model)
        logger.info(
            "Qwen4 GGUF MTP native HC: %d modules, %d paired down/inject layouts",
            len(records),
            sum(r["paired"] for r in records),
        )
    logger.info(
        "Qwen4 GGUF MTP loaded: %d initialized dense parameter names, one QSA layer, "
        "%d Q2_0 expert-cache slots; original fused gate/up row views and GGUF vocabulary",
        len(loaded),
        experts.quant_method.cache_slots,
    )
    if pack := os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_PACK"):
        from sglang.srt.model_loader.qwen4_mtp_q8_pack import install

        model.gguf_mtp_q8_receipt = install(model, pack)
        logger.info(
            "Qwen4 GGUF MTP: 10 runtime Q8_0 entries installed in 7 projections; "
            "raw GGUF norms, HC, router and experts retained"
        )
    return model
