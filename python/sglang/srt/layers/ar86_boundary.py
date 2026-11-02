"""Opt-in AR norm/FP8 producer and explicit shared-projection transport.

R84 mode5, M1/H2048/B70 only. BF16 remains available to every non-FP8 reader.
Census attributes hold diagnostics, never route data between layers.
"""
import ctypes
import logging
import os
from functools import lru_cache

import torch

_SINGLE_ENABLED = os.environ.get('CODEX_AR86_BOUNDARY') == '1'
_TP_ENABLED = os.environ.get('SGLANG_XPU_TP_NORM_QUANT') == '1'
ENABLED = _SINGLE_ENABLED or _TP_ENABLED
CENSUS = os.environ.get('CODEX_AR86_CENSUS') == '1'
LIB_PATH = '/evidence/ar-round84/native/conversion01/boundary.so'
logger = logging.getLogger(__name__)


def supports_projection(projection):
    if not ENABLED or projection is None:
        return False
    method = getattr(projection, 'quant_method', None)
    weight = getattr(projection, 'weight', None)
    if not (type(method).__name__ == 'Fp8LinearMethod' and getattr(method, 'block_quant', False)
            and list(getattr(method, 'weight_block_size', [])) == [128, 128]
            and not getattr(method, 'use_mxfp8', False)
            and not getattr(method, 'block_fp8_as_mxfp8', False)
            and not getattr(method, 'use_marlin', False)
            and weight is not None and weight.ndim == 2):
        return False
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
    parallel = get_parallel()
    if parallel.tp_size == 1:
        return (_SINGLE_ENABLED and get_exec().moe.enable_paged_experts
                and tuple(weight.shape) in [(12288, 2048), (9216, 2048), (1024, 2048)])
    # Norm runs after the existing all-reduce. Its H2048 residual stream is
    # replicated; only the downstream projection's output rows are sharded.
    shapes = {2: (6144, 4608, 512), 4: (3072, 2560, 256)}
    return (_TP_ENABLED and parallel.tp_size in shapes
            and (parallel.moe_ep_size == 1 or (parallel.tp_size == 2 and parallel.moe_ep_size == 2))
            and parallel.attn_cp_size == 1
            and parallel.pp_group.world_size == 1 and not parallel.attn_dp_enabled
            and get_spec().speculative_algorithm is None
            and weight.shape[1] == 2048 and weight.shape[0] in shapes[parallel.tp_size])


def shared_readout(block):
    from sglang.srt.layers.layer_boundary.residual.add_norm import NORM_READOUT, NormQuantReadout, Fp8Input
    shared = getattr(block, 'shared_expert', None)
    if (shared is not None and getattr(block, 'alt_stream', None) is None
            and not getattr(block, 'enable_shared_expert_fusion', False)
            and not getattr(block, '_use_mega_moe', False)
            and supports_projection(getattr(shared, 'gate_up_proj', None))):
        return NormQuantReadout(fp8_input=Fp8Input.TUPLE_AND_BF16, ar_bmg=True)
    return NORM_READOUT


@lru_cache(maxsize=1)
def library():
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('AR86 boundary library must warm before capture')
    lib = ctypes.CDLL(LIB_PATH)
    lib.ar_boundary_error.restype = ctypes.c_char_p
    lib.ar_boundary_run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_float] * 2 + [ctypes.c_int] * 2
    logger.info('AR86 native norm/FP8 producer enabled: R84 mode5; explicit tuple consumers; census=%s', CENSUS)
    return lib


def _counter(owner, attribute, tensor):
    count = getattr(owner, attribute, None)
    if count is None:
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('AR86 census buffers must warm before capture')
        count = torch.zeros(1, dtype=torch.int64, device=tensor.device)
        setattr(owner, attribute, count)
    return count


def try_run(module, x, residual, fp8_input):
    if not ENABLED or fp8_input is None or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != (1, 2048):
        return None
    if not (type(module).__name__ == 'GemmaRMSNorm' and x.device.type == 'xpu'
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and tuple(module.weight.shape) == (2048,) and module.weight.dtype == torch.bfloat16
            and module.weight.is_contiguous() and module.weight.device == x.device
            and (residual is None or (residual.shape == x.shape and residual.dtype == x.dtype
                 and residual.device == x.device and residual.is_contiguous()))):
        return None
    if torch.xpu.get_device_properties(x.device).device_id != 0xe223:
        return None
    lib = library()
    # Match the original alias contract: fused-add updates x and residual;
    # no-residual norm leaves x as the residual stream and allocates its output.
    out = x if residual is not None else torch.empty_like(x)
    res = x if residual is None else residual
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 16), dtype=torch.float32, device=x.device)
    count = _counter(module, '_ar86_producer_count', x) if CENSUS else None
    stream = torch.xpu.current_stream(x.device)
    args = [stream.sycl_queue] + [t.data_ptr() for t in [x, res, module.weight, out, res, q, scale]]
    rc = lib.ar_boundary_run(*args, module.variance_epsilon, 448., int(residual is not None), 5)
    if rc:
        raise RuntimeError(f'AR86 native status {rc}: {lib.ar_boundary_error().decode()}')
    for tensor in [x, res, module.weight, out, q, scale]:
        tensor.record_stream(stream)
    if count is not None:
        count.add_(1)
        module._ar86_return_kind = fp8_input.name
    payload = (q, scale) if fp8_input.name == 'TUPLE' else (out, q, scale)
    return payload, res


def shared_projection_input(block, value):
    if not (ENABLED and isinstance(value, tuple) and len(value) == 3):
        raise ValueError('AR86 shared input requires an explicit BF16/FP8/scale triple')
    bf, q, scale = value
    shared = getattr(block, 'shared_expert', None)
    if not (shared is not None and supports_projection(shared.gate_up_proj)
            and block.alt_stream is None and not block.enable_shared_expert_fusion
            and not block._use_mega_moe and tuple(bf.shape) == (1, 2048)
            and bf.dtype == torch.bfloat16 and tuple(q.shape) == tuple(bf.shape)
            and q.dtype == torch.float8_e4m3fn and tuple(scale.shape) == (1, 16)
            and scale.dtype == torch.float32 and bf.device == q.device == scale.device
            and bf.device.type == 'xpu' and all(t.is_contiguous() for t in value)):
        raise ValueError('Unsupported AR86 shared-projection input or executor')
    return bf, (q, scale)


def record_consumer(layer, value):
    if not (ENABLED and CENSUS and supports_projection(layer)):
        return
    q, scale = value
    if q.shape != (1, 2048) or q.dtype != torch.float8_e4m3fn or scale.shape != (1, 16):
        raise ValueError('AR86 census saw an incompatible quantized consumer')
    _counter(layer, '_ar86_consumer_count', q).add_(1)


def readback(model):
    producers = []
    consumers = []
    for name, module in model.named_modules():
        count = getattr(module, '_ar86_producer_count', None)
        if count is not None:
            producers.append((name, module, count))
        count = getattr(module, '_ar86_consumer_count', None)
        if count is not None:
            consumers.append((name, module, count))
    values = torch.cat([c for _, _, c in producers + consumers]).cpu().tolist() if producers or consumers else []
    prows = [dict(name=name, role=name.rsplit('.', 1)[-1], return_kind=module._ar86_return_kind, calls=value)
             for (name, module, _), value in zip(producers, values[:len(producers)])]
    crows = [dict(name=name, shape=list(module.weight.shape), calls=value)
             for (name, module, _), value in zip(consumers, values[len(producers):])]
    return dict(enabled=ENABLED, census=CENSUS, producers=prows, consumers=crows, library=LIB_PATH,
                input_contract='explicit BF16/Q/scale triple for shared and GDN; Q/scale pair for full attention',
                final_head_norm_fused=False, torch_profiler_used=False, normal_TPS_evidence=False)
