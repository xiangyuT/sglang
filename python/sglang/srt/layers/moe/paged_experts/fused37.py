"""Scoped model prototype: explicit planned TopK output, never lazy IDs."""
import ctypes
import logging
import os
from functools import lru_cache

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)


class PlannedTopKOutput(StandardTopKOutput):
    """Valid logical IDs/weights plus the owner of already-prepared device state."""


@lru_cache(maxsize=1)
def load():
    assert not torch.xpu.is_current_stream_capturing(), 'warm before capture'
    library = ctypes.CDLL('/evidence/ar-round37/native/fused01.so')
    library.ar_fused37_abi.restype = ctypes.c_int
    assert library.ar_fused37_abi() == 1
    library.ar_fused37_error.restype = ctypes.c_char_p
    library.ar_fused37_run.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 2
    library.ar_fused37_run.restype = ctypes.c_int
    return library


def try_route(block, hidden, logits, defer_finalize):
    from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod
    from sglang.srt.layers.moe.paged_experts.xpu import XpuDeviceExecutor
    from sglang.srt.runtime_context import get_spec

    cpu_hybrid = os.environ.get('CODEX_AR_CPU40') == '1'
    if (os.environ.get('CODEX_AR_FUSED37') != '1' or defer_finalize
            or (cpu_hybrid and os.environ.get('CODEX_AR_CPU118_FUSED') != '1')):
        return None
    method = block.experts.quant_method
    if not isinstance(method, PagedExpertsMoEMethod):
        return None
    executor = method.device_executor
    config = block.topk.topk_config
    if not (
        isinstance(executor, XpuDeviceExecutor)
        and executor.use_scaled_fp8
        and method.num_resident == 192 and method.num_experts == 256
        and not block.enable_shared_expert_fusion
        and block.tp_size == block.experts.moe_ep_size == 1
        and get_spec().speculative_algorithm is None
        and hidden.shape == (1, 2048) and hidden.dtype == torch.bfloat16
        and logits.shape == (1, 256) and logits.dtype == torch.bfloat16
        and hidden.device == logits.device and logits.device.type == 'xpu'
        and logits.is_contiguous() and hidden.is_contiguous()
        and config.top_k == 8 and config.scoring_func == 'softmax'
        and not config.use_grouped_topk and config.custom_routing_function is None
        and config.correction_bias is None and config.num_shared_topk_slots == 0
        and not config.apply_routed_scaling_factor_on_output
        and torch.xpu.get_device_properties(logits.device).device_id == 0xe223
    ):
        return None
    if cpu_hybrid and os.environ.get('CODEX_AR362_PUBLISH') == '1':
        from sglang.srt.layers.moe.paged_experts.cpu_publish362 import try_route as publish_route
        published = publish_route(executor, hidden, logits, config.renormalize)
        if published is not None:
            return published
    library = load()
    weights = torch.empty((1, 8), device=logits.device, dtype=torch.float32)
    ids = torch.empty((1, 8), device=logits.device, dtype=torch.int32)
    r = executor.residency
    args = [logits, weights, ids, r.step, r.slot_expert, r.expert_slot, r.slot_lastuse,
            r.src, r.dst, r.count, executor.mapped_ids, executor.mapped_weights,
            executor.expert_hosts, executor.expert_gpus]
    stream = torch.xpu.current_stream(logits.device)
    # CPU mode6 owns all four refill planes. Planning must not copy scales
    # before hybrid_route_publish waits for the previous channel generation.
    rc = library.ar_fused37_run(stream.sycl_queue, *[v.data_ptr() for v in args], int(config.renormalize), int(not cpu_hybrid))
    if rc:
        raise RuntimeError(library.ar_fused37_error().decode())
    for tensor in args:
        tensor.record_stream(stream)
    if block.layer_id == 0 and not getattr(executor, '_fused37_logged', False):
        logger.info('AR37 prototype: SYCL top8/LRU fusion; CPU hybrid=%s; inline scale copy=%s', cpu_hybrid, not cpu_hybrid)
        executor._fused37_logged = True
    out = PlannedTopKOutput(weights, ids, logits)
    out.executor = executor
    return out
