"""B70 TP MoE: retain native prepare/GU/SiLU, fuse down and route reduction."""
import ctypes
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_use_tp_expert_down(layer, config, x, ids, weights):
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return False
    if x.ndim != 2 or x.shape != (1, 2048):
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    if not (
        parallel.tp_size in (2, 4) and parallel.pp_size == 1
        and (parallel.moe_ep_size == 1 or (parallel.tp_size == 2 and parallel.moe_ep_size == 2))
        and parallel.attn_cp_size == 1
        and not parallel.attn_dp_enabled
        and get_spec().speculative_algorithm is None
        and config.activation == "silu" and config.is_gated
        and not config.no_combine and not config.apply_router_weight_on_input
        and config.routed_scaling_factor in (None, 1.0)
        and config.gemm1_alpha is None and config.gemm1_clamp_limit is None
        and config.swiglu_limit is None
        and getattr(layer, "w13_weight_bias", None) is None
        and getattr(layer, "w2_weight_bias", None) is None
    ):
        return False
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    experts = 256 // parallel.moe_ep_size
    if parallel.moe_ep_size == 2 and envs.SGLANG_XPU_EP_REPLICA_BALANCE.get():
        experts = 192
    intermediate = 512 // (parallel.tp_size // parallel.moe_ep_size)
    tensors = (x, ids, weights, w13, w2, s13, s2)
    return (
        x.device.type == "xpu" and all(t.device == x.device for t in tensors)
        and x.dtype == torch.bfloat16
        and ids.dtype == torch.int32 and ids.shape == (1, 8)
        and weights.dtype == torch.float32 and weights.shape == (1, 8)
        and w13.dtype == w2.dtype == torch.float8_e4m3fn
        and w13.shape == (experts, 2 * intermediate, 2048)
        and w2.shape == (experts, 2048, intermediate)
        and s13.dtype == s2.dtype == torch.float32
        and s13.shape == (experts, 2 * intermediate // 128, 16)
        and s2.shape == (experts, 16, intermediate // 128)
        and all(t.is_contiguous() for t in tensors)
        and all(t.data_ptr() % 4 == 0 for t in tensors)
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


@lru_cache(maxsize=1)
def _library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_TP_EXPERT_DOWN_SYCL_LIB to the built library")
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm TP expert down before graph capture")
    library = ctypes.CDLL(path)
    library.tp_expert_down_run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int]
    library.tp_expert_down_run.restype = ctypes.c_int
    library.tp_expert_down_error.restype = ctypes.c_char_p
    logger.info(
        "B70 TP expert down SYCL enabled: packed-word down/route reduction; path=%s sha256=%s",
        path, hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
    return library


def tp_expert_down(layer, x, ids, weights):
    library = _library(envs.SGLANG_XPU_TP_EXPERT_DOWN_SYCL_LIB.get())
    intermediate = layer.w2_weight.shape[-1]
    scratch = getattr(layer, "_xpu_tp_down_workspace", None)
    if scratch is None:
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Warm TP expert scratch before graph capture")
        scratch = (
            torch.empty(layer.w13_weight.shape[0], device=x.device, dtype=torch.int32),
            torch.empty(8, device=x.device, dtype=torch.int32),
            torch.empty((8, 2048), device=x.device, dtype=torch.bfloat16),
            torch.empty((8, 2 * intermediate), device=x.device, dtype=torch.bfloat16),
            torch.empty((8, intermediate), device=x.device, dtype=torch.bfloat16),
        )
        layer._xpu_tp_down_workspace = scratch
    offsets, mapping, shuffled, raw, activated = scratch
    # The native permutation maps original route index to the sorted GEMM row.
    prepare_ids = ids
    if layer.w13_weight.shape[0] in (128, 192):
        # Native prepare requires valid local IDs. Keep the original negative
        # IDs for the SYCL down kernel so it can skip nonlocal routes.
        from sglang.kernels.ops.moe.xpu_ep_routes import mask_nonlocal_routes
        prepare_ids, _ = mask_nonlocal_routes(ids, weights)
    torch.ops.sgl_kernel.prepare_moe_input_small(x, prepare_ids, offsets, mapping, shuffled)
    torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w8a16(
        raw, shuffled, layer.w13_weight, layer.w13_weight_scale_inv, None,
        offsets, layer.w13_weight.shape[0],
    )
    torch.ops.sgl_kernel.silu_and_mul(activated, raw)
    output = torch.empty_like(x)
    stream = torch.xpu.current_stream(x.device)
    tensors = (
        activated, layer.w2_weight, layer.w2_weight_scale_inv, ids, mapping,
        weights, output,
    )
    code = library.tp_expert_down_run(
        stream.sycl_queue, *[tensor.data_ptr() for tensor in tensors], intermediate,
    )
    if code:
        raise RuntimeError((code, library.tp_expert_down_error().decode()))
    for tensor in tensors + scratch + (x, layer.w13_weight, layer.w13_weight_scale_inv):
        tensor.record_stream(stream)
    return output
