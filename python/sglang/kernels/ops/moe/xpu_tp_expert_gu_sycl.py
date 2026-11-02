"""Packed-word TP/EP gate/up, optional I512 SiLU fusion, and validated down."""
import ctypes
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.kernels.ops.moe.xpu_tp_expert_down_sycl import _library as down_library

logger=logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError('Warm the TP gate/up SYCL library before capture')
    lib=ctypes.CDLL(path)
    lib.tp_expert_gu_run.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]
    lib.tp_expert_gu_run.restype=ctypes.c_int
    lib.tp_expert_gu_error.restype=ctypes.c_char_p
    if hasattr(lib, 'tp_expert_gu_silu_run'):
        lib.tp_expert_gu_silu_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int]
        lib.tp_expert_gu_silu_run.restype = ctypes.c_int
    if hasattr(lib, 'tp_expert_map_run'):
        lib.tp_expert_map_run.argtypes = [ctypes.c_void_p] * 4
        lib.tp_expert_map_run.restype = ctypes.c_int
    if hasattr(lib, 'tp_expert_replica_map_run'):
        lib.tp_expert_replica_map_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
        lib.tp_expert_replica_map_run.restype = ctypes.c_int
    if hasattr(lib, 'tp_expert_replica_gu_silu_run'):
        lib.tp_expert_replica_gu_silu_run.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int]
        lib.tp_expert_replica_gu_silu_run.restype = ctypes.c_int
    logger.info('B70 TP expert gate/up SYCL library enabled: library=%s',
                hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return lib


def tp_expert_gu(layer,x,ids,weights,replica_routing=None):
    lib=_library(envs.SGLANG_XPU_TP_EXPERT_GU_SYCL_LIB.get())
    down=down_library(envs.SGLANG_XPU_TP_EXPERT_DOWN_SYCL_LIB.get())
    intermediate=layer.w2_weight.shape[-1]
    _announce_shape(layer.w13_weight.shape[0], intermediate)
    scratch=getattr(layer,'_xpu_tp_gu_workspace',None)
    if scratch is None:
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Warm TP gate/up scratch before capture')
        scratch=(torch.empty((8,2*intermediate),device=x.device,dtype=torch.bfloat16),
                 torch.empty((8,intermediate),device=x.device,dtype=torch.bfloat16),
                 torch.arange(8,device=x.device,dtype=torch.int32))
        layer._xpu_tp_gu_workspace=scratch
    raw,activated,mapping=scratch
    stream=torch.xpu.current_stream(x.device)
    fused_silu = intermediate == 512 and hasattr(lib, 'tp_expert_gu_silu_run')
    extra_tensors = ()
    if replica_routing is not None:
        _announce_fused_replica()
        table, rank = replica_routing
        if not (
            intermediate == 512 and layer.w13_weight.shape[0] == 192
            and ids.shape == (1, 8) and ids.dtype == torch.int32
            and table.shape == (256, 2) and table.dtype == torch.int32
            and table.device == ids.device == x.device
            and table.is_contiguous() and ids.is_contiguous()
            and rank in (0, 1) and fused_silu
            and hasattr(lib, 'tp_expert_replica_gu_silu_run')
        ):
            raise ValueError('Unsupported fused replica GU input')
        local_ids = getattr(layer, '_xpu_replica_gu_local_ids', None)
        if local_ids is None:
            if torch.xpu.is_current_stream_capturing():
                raise RuntimeError('Warm replica GU scratch before capture')
            local_ids = torch.empty_like(ids)
            layer._xpu_replica_gu_local_ids = local_ids
        code = lib.tp_expert_replica_gu_silu_run(
            stream.sycl_queue, *[t.data_ptr() for t in
                (x, layer.w13_weight, layer.w13_weight_scale_inv, ids, table,
                 raw, activated, local_ids)], rank,
        )
        extra_tensors = (ids, table)
        ids = local_ids
    elif fused_silu:
        _announce_fused_silu()
        code = lib.tp_expert_gu_silu_run(
            stream.sycl_queue, *[t.data_ptr() for t in
                (x, layer.w13_weight, layer.w13_weight_scale_inv, ids, raw, activated)],
            intermediate,
        )
    else:
        code=lib.tp_expert_gu_run(stream.sycl_queue,*[t.data_ptr()for t in
            (x,layer.w13_weight,layer.w13_weight_scale_inv,ids,raw)],intermediate)
    if code:raise RuntimeError((code,lib.tp_expert_gu_error().decode()))
    if not fused_silu:
        torch.ops.sgl_kernel.silu_and_mul(activated,raw)
    output=torch.empty_like(x)
    tensors=(activated,layer.w2_weight,layer.w2_weight_scale_inv,ids,mapping,weights,output)
    code=down.tp_expert_down_run(stream.sycl_queue,*[t.data_ptr()for t in tensors],intermediate)
    if code:raise RuntimeError((code,down.tp_expert_down_error().decode()))
    for t in tensors+scratch+extra_tensors+(x,layer.w13_weight,layer.w13_weight_scale_inv):t.record_stream(stream)
    return output


@lru_cache(maxsize=3)
def _announce_shape(experts, intermediate):
    logger.info("B70 TP/EP expert SYCL: E=%d I=%d; negative local IDs skip weights", experts, intermediate)


def try_ep_local_ids(ids, mapping):
    """Fuse the standard eight-route map, including its negative drop sentinel."""
    if not envs.SGLANG_XPU_TP_EXPERT_GU_SYCL.get():
        return None
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return None
    if not (
        ids.shape == (1, 8) and mapping.shape == (256,)
        and ids.device.type == "xpu" and mapping.device == ids.device
        and ids.dtype == mapping.dtype == torch.int32
        and ids.is_contiguous() and mapping.is_contiguous()
        and torch.xpu.get_device_properties(ids.device).device_id == 0xE223
    ):
        return None
    lib = _library(envs.SGLANG_XPU_TP_EXPERT_GU_SYCL_LIB.get())
    if not hasattr(lib, "tp_expert_map_run"):
        return None
    _announce_map()
    out = torch.empty_like(ids)
    stream = torch.xpu.current_stream(ids.device)
    code = lib.tp_expert_map_run(
        stream.sycl_queue, ids.data_ptr(), mapping.data_ptr(), out.data_ptr()
    )
    if code:
        raise RuntimeError((code, lib.tp_expert_gu_error().decode()))
    for tensor in (ids, mapping, out):
        tensor.record_stream(stream)
    return out


@lru_cache(maxsize=1)
def _announce_map():
    logger.info("B70 EP local-ID SYCL mapping enabled: shape=(1, 8), table=256 entries")


@lru_cache(maxsize=1)
def _announce_fused_silu():
    logger.info("B70 EP expert GU+SiLU fusion enabled: I=512, BF16 gate/up and SiLU intermediate, FP32 multiply")


@lru_cache(maxsize=1)
def _announce_fused_replica():
    logger.info("B70 EP replica map+GU+SiLU enabled: logical top8, one rank per route; local IDs produced for down on the same stream")
