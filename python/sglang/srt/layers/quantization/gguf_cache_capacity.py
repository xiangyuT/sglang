# SPDX-License-Identifier: Apache-2.0
"""Grow compressed target expert residency after speculative graph capture."""

import logging

import torch

from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod

logger = logging.getLogger(__name__)


def grow_model_expert_caches(model, slots):
    config = getattr(model.config, "text_config", model.config)
    if not getattr(config, "gguf_model_path", None):
        raise ValueError("Post-capture expert growth requires the native GGUF model")
    layers = [
        module
        for module in model.modules()
        if isinstance(getattr(module, "quant_method", None), CompressedGGUFMoEMethod)
    ]
    if not layers or any(
        layer.quant_method.sources is None
        or not layer.quant_method.cache_slots <= slots <= layer.quant_method.num_experts
        for layer in layers
    ):
        raise ValueError("Invalid post-capture GGUF expert-cache capacity")
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    before = torch.xpu.mem_get_info()[0]
    added = 0
    host_before = 0
    host_after = 0
    for layer in layers:
        cache = layer.quant_method._gpu_cache
        if cache is not None:
            host_before += sum(
                {
                    v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                    for v in cache.mirrors
                }.values()
            )
        added += layer.quant_method.grow_cache(layer, slots)
        actual = layer.quant_method.cache_slots
        device_actual = int(cache.capacity.item()) if cache is not None else actual
        if actual != slots or device_actual != slots:
            raise RuntimeError(
                f"GGUF cache growth requested {slots} slots, but method={actual}, device={device_actual}"
            )
        if cache is not None:
            # Also repair an already-grown idle cache without changing K.
            cache.compact_mirrors()
            host_after += sum(
                {
                    v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                    for v in cache.mirrors
                }.values()
            )
        # Release each old layer before allocating the next. This also drains
        # pending native H2D and packed-layout work from graph warmups.
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    after = torch.xpu.mem_get_info()[0]
    receipt = dict(
        layers=len(layers),
        slots_per_layer=slots,
        added_bytes=added,
        before_driver_free=before,
        after_driver_free=after,
        before_host_expert_storage_bytes=host_before,
        after_host_expert_storage_bytes=host_after,
        released_host_expert_storage_bytes=host_before - host_after,
    )
    logger.info(
        "GGUF cold-expert mirrors compacted after growth: %.3f -> %.3f GiB owned host storage; "
        "%.3f GiB released (encoded expert buffers, not process RSS)",
        host_before / 2**30, host_after / 2**30, (host_before - host_after) / 2**30,
    )
    model.gguf_cache_growth_receipt = receipt
    logger.info(
        "Qwen4 GGUF post-capture expert growth: %d layers, %d slots/layer, +%.3f GiB; driver free %.3f -> %.3f GiB",
        len(layers),
        slots,
        added / 2**30,
        before / 2**30,
        after / 2**30,
    )
    return receipt
