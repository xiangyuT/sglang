"""Opt-in current B70 AR projection/conv fusion with explicit local outputs."""
import ctypes
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import torch

ENABLED=os.environ.get('CODEX_AR1776_INPUTCONV')=='1'
LIBRARY='/evidence/tpot-iterations-20261023/native/inputconv1820/inputconv.so'
EAGER_CALLS=CAPTURE_CALLS=0

@lru_cache(maxsize=1)
def library():
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('Warm AR1776 input/conv library before capture')
    lib=ctypes.CDLL(LIBRARY)
    lib.inputconv1820_run.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int]*9
    lib.inputconv1820_error.restype=ctypes.c_char_p
    logging.getLogger(__name__).info('AR1776 QKVZ/conv fusion enabled: input639 R16/GRF128; BF16 boundary; conv1220 FMA; recurrence unchanged')
    return lib

def try_run(model,hidden,batch):
    global EAGER_CALLS,CAPTURE_CALLS
    if not ENABLED or torch.is_grad_enabled() or torch.compiler.is_compiling():return None
    if not isinstance(hidden,tuple) or len(hidden)!=3 or batch is None:return None
    from sglang.srt.models.qwen3_5 import _select_fused_ar_input_for_linear
    from sglang.srt.layers.ar646_input import can_use
    quant=_select_fused_ar_input_for_linear(hidden,model.in_proj_qkvz)
    if not can_use(model.in_proj_qkvz,quant,batch):return None
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.environ import envs
    backend=get_attn_backend();backend=getattr(backend,'linear_attn_backend',backend)
    layer=model.attn
    if type(backend).__name__!='XpuGDNAttnBackend' or not backend.supports_fused_gdn(layer,batch):return None
    if not envs.SGLANG_XPU_GDN_DECODE.get():return None
    if (layer.num_k_heads,layer.num_v_heads,layer.head_k_dim,layer.head_v_dim)!=(16,32,128,128):return None
    if layer.activation not in ('silu','swish') or layer.bias is not None:return None
    cache=backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    state=cache.conv[0];ssm=cache.temporal;indices=backend.forward_metadata.mamba_cache_indices
    weight=layer.conv_weights;q,xs=quant;w=model.in_proj_qkvz.weight;ws=model.in_proj_qkvz.weight_scale_inv
    if not(state.device==q.device==weight.device==indices.device==ssm.device):return None
    if state.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or ssm.dtype!=torch.float32:return None
    if state.ndim!=3 or state.shape[1:]!=(8192,3) or weight.shape!=(8192,4):return None
    if not all(t.is_contiguous()for t in [state,weight,indices,ssm]):return None
    if indices.shape!=(1,) or indices.dtype not in [torch.int32,torch.int64]:return None
    if ssm.shape!=(state.shape[0],32,128,128):return None
    hs=hidden[0]
    if not isinstance(hs,torch.Tensor) or hs.dtype!=torch.bfloat16 or hs.shape!=(1,2048) or hs.device!=q.device:return None
    # Every guard precedes the indexed state mutation. A submitted fusion never
    # falls back and repeats the convolution update.
    projected=torch.empty((1,12288),device=q.device,dtype=torch.bfloat16)
    mixed=torch.empty((1,8192),device=q.device,dtype=torch.bfloat16)
    z=torch.empty((1,32,128),device=q.device,dtype=torch.bfloat16)
    lib=library();stream=torch.xpu.current_stream(q.device)
    vals=[q,w,xs,ws,projected,state,weight,indices,mixed,z]
    rc=lib.inputconv1820_run(stream.sycl_queue,*[v.data_ptr()for v in vals],12288,*ws.stride(),state.shape[0],*state.stride(),int(indices.dtype==torch.int64),1)
    if rc:raise RuntimeError((rc,lib.inputconv1820_error().decode()))
    for v in vals:v.record_stream(stream)
    ba=model.in_proj_ba(hs)[0]
    if torch.xpu.is_current_stream_capturing():CAPTURE_CALLS+=1
    else:EAGER_CALLS+=1
    return backend,projected,ba,(mixed,z,state.data_ptr(),indices.data_ptr(),layer.layer_id)

def readback():
    from sglang.srt.model_executor.cuda_graph_buffer_registry import _CODEX_AR1817_FILL
    return dict(enabled=ENABLED,eager_calls=EAGER_CALLS,capture_calls=CAPTURE_CALLS,
        library='native/inputconv1820/inputconv.so',library_sha256=hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest(),
        projection_rows=16,GRF=128,mode=1,fused_body_unchanged=True,BF16_boundary=True,conv_FMA=True,BA_projection_unchanged=True,
        recurrence_unchanged=True,explicit_local_outputs=True,GPU_census=False,graph_fill_unpadded_skip=_CODEX_AR1817_FILL)
