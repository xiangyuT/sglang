"""Opt-in pure-AR B70 gated RMSNorm128 -> block128 FP8 producer.

M1 reduction and runtime-limit quantization match the retained Triton recipe.
No activation/weight cache or per-token observer. Warm before full-graph capture.
"""
import ctypes
from functools import lru_cache
import logging
import math
import os

import torch

logger = logging.getLogger(__name__)
ENABLED = os.environ.get('CODEX_AR482_NORM') == '1'
LIBRARY = '/evidence/tpot-iterations-20261023/native/norm478/norm.so'


@lru_cache(maxsize=1)
def library():
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('AR482 norm/quant must warm before graph capture')
    lib = ctypes.CDLL(LIBRARY)
    lib.ar478_norm_quant_abi.argtypes = []
    lib.ar478_norm_quant_abi.restype = ctypes.c_int
    assert lib.ar478_norm_quant_abi() == 1
    lib.ar478_norm_quant_run.argtypes = [ctypes.c_void_p] * 8 + [
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float
    ]
    lib.ar478_norm_quant_run.restype = ctypes.c_int
    lib.ar478_norm_quant_error.restype = ctypes.c_char_p
    logger.info('AR482 pure-AR SYCL gated norm/block128 FP8 producer enabled: M1, SG32, runtime-limit quant; library=%s', LIBRARY)
    return lib


def can_use(x, z, norm, projection, batch):
    if not ENABLED or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return False
    if not (batch.batch_size == 1 and batch.forward_mode.is_decode()
            and getattr(batch, 'spec_info', None) is None):
        return False
    if not (x.device.type == 'xpu' and x.shape == z.shape == (32, 128)
            and x.dtype == z.dtype == torch.bfloat16 and x.device == z.device
            and x.is_contiguous() and z.is_contiguous()):
        return False
    weight = getattr(norm, 'weight', None)
    if not (isinstance(weight, torch.Tensor) and weight.shape == (128,)
            and weight.dtype == torch.bfloat16 and weight.device == x.device
            and weight.is_contiguous() and norm.norm_before_gate
            and norm.activation in ('swish', 'silu')
            and getattr(norm, 'group_size', None) in (None, 128)
            and getattr(norm, 'bias', None) is None
            and isinstance(norm.eps, (int, float)) and math.isfinite(norm.eps) and norm.eps > 0):
        return False
    method = getattr(projection, 'quant_method', None)
    pw = getattr(projection, 'weight', None)
    if not (type(method).__name__ == 'Fp8LinearMethod'
            and getattr(method, 'block_quant', False)
            and tuple(method.weight_block_size) == (128, 128)
            and not any(getattr(method, name, False) for name in
                        ['use_mxfp8', 'block_fp8_as_mxfp8', 'use_marlin'])
            and isinstance(pw, torch.Tensor) and pw.shape == (2048, 4096)
            and pw.dtype == torch.float8_e4m3fn and pw.device == x.device
            and getattr(projection, 'bias', None) is None):
        return False
    from sglang.srt.runtime_context import get_parallel, get_exec
    return (get_parallel().tp_size == 1 and get_exec().moe.enable_paged_experts
            and torch.xpu.get_device_properties(x.device).device_id == 0xe223)


def try_run(x, z, norm, projection, batch):
    if not can_use(x, z, norm, projection, batch):
        if ENABLED and not getattr(norm, '_ar482_skip_reported', False):
            norm._ar482_skip_reported = True
            method = getattr(projection, 'quant_method', None)
            logger.info('AR482 startup dispatch skipped: x=%s/%s/%s contiguous=%s; z=%s/%s contiguous=%s; norm=%s/%s before_gate=%s activation=%s group=%s eps=%s; projection=%s/%s blocks=%s flags=%s; grad=%s mode=%s spec=%s',
                tuple(x.shape), x.dtype, x.device, x.is_contiguous(), tuple(z.shape), z.dtype, z.is_contiguous(),
                tuple(norm.weight.shape), norm.weight.dtype, norm.norm_before_gate, norm.activation, getattr(norm, 'group_size', None), norm.eps,
                type(method).__name__, tuple(projection.weight.shape), getattr(method, 'weight_block_size', None),
                {k:getattr(method,k,None)for k in ['block_quant','use_mxfp8','block_fp8_as_mxfp8','use_marlin']},
                torch.is_grad_enabled(), batch.forward_mode, getattr(batch,'spec_info',None))
        return None
    lib = library()
    q = torch.empty((1, 4096), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((1, 32), device=x.device, dtype=torch.float32)
    stream = torch.xpu.current_stream(x.device)
    rc = lib.ar478_norm_quant_run(stream.sycl_queue,
        x.data_ptr(), z.data_ptr(), norm.weight.data_ptr(), q.data_ptr(), s.data_ptr(),
        None, None, 32, norm.eps, 0, 448.)
    if rc:
        raise RuntimeError(f'AR482 native status {rc}: {lib.ar478_norm_quant_error().decode()}')
    for t in [x, z, norm.weight, q, s]:
        t.record_stream(stream)
    return q, s
