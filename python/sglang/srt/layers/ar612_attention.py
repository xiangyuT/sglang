"""Opt-in B70 pure AR head-batch layout using installed SYCL-TLA attention.

The actual KV cache stays contiguous/read-only. Queries and live metadata are
reordered by the tested attention277 kernels, then output heads are restored.
"""
import ctypes
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import torch

ENABLED=os.environ.get('CODEX_AR612_ATTENTION')=='1'
logger=logging.getLogger(__name__)
EAGER_CALLS=CAPTURE_CALLS=0
LIBRARY='/evidence/tpot-iterations-20261023/native/attention277/attention.so'
CONTEXT=None

def can_use(q,k,v,lens,pages):
    tensors=[q,k,v,lens,pages]
    if not all(isinstance(t,torch.Tensor) for t in tensors):return False
    return (q.device.type=='xpu' and all(t.device==q.device for t in tensors)
        and q.dtype==k.dtype==v.dtype==torch.bfloat16
        and lens.dtype==pages.dtype==torch.int32
        and tuple(q.shape)==(1,16,256) and tuple(lens.shape)==(1,)
        and tuple(pages.shape)==(1,32) and k.ndim==v.ndim==4
        and k.shape==v.shape and tuple(k.shape[1:])==(128,2,256) and k.shape[0]>=32
        and all(t.is_contiguous() for t in tensors)
        and q.data_ptr()%8==0 and lens.data_ptr()%4==pages.data_ptr()%4==0
        and torch.xpu.get_device_properties(q.device).device_id==0xe223)

class Context:
    def __init__(self,device):
        if torch.xpu.is_current_stream_capturing():raise RuntimeError('Warm AR612 resources before graph capture')
        self.lib=ctypes.CDLL(LIBRARY)
        self.lib.attention277_pack.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]
        self.lib.attention277_unpack.argtypes=[ctypes.c_void_p]*3
        self.lib.attention277_error.restype=ctypes.c_char_p
        self.cu=torch.arange(9,dtype=torch.int32,device=device)
        self.sha256=hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest()
    def run(self,q,k,v,lens,pages,post_request=None):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        q8=torch.empty((8,2,256),device=q.device,dtype=q.dtype)
        l8=torch.empty(8,device=q.device,dtype=torch.int32)
        p8=torch.empty((8,32),device=q.device,dtype=torch.int32)
        y8=torch.empty_like(q8)
        from sglang.srt.layers import ar2015_attention_post as post
        fused=post.can_use(y8,post_request)
        y=None if fused else torch.empty_like(q)
        stream=torch.xpu.current_stream(q.device)
        tensors=[q,lens,pages,q8,l8,p8]
        rc=self.lib.attention277_pack(stream.sycl_queue,*[t.data_ptr() for t in tensors],32)
        if rc:raise RuntimeError((rc,self.lib.attention277_error().decode()))
        flash_attn_with_kvcache(q8,k,v,page_table=p8,cache_seqlens=l8,cu_seqlens_q=self.cu,
            max_seqlen_q=1,softmax_scale=.0625,causal=True,num_splits=1,pack_gqa=False,out=y8)
        for t in tensors+[k,v,y8,self.cu]:t.record_stream(stream)
        if fused:return post.run(y8,post_request)
        rc=self.lib.attention277_unpack(stream.sycl_queue,y8.data_ptr(),y.data_ptr())
        if rc:raise RuntimeError((rc,self.lib.attention277_error().decode()))
        y.record_stream(stream)
        return y

@lru_cache(maxsize=1)
def context(device):
    global CONTEXT
    CONTEXT=Context(device)
    logger.info('AR612 pure-AR head-batch attention enabled: Hq16/Hkv2/D256/page128; contiguous KV, no KV copy, one KV partition')
    return CONTEXT

def try_run(owner,fb,q,k,v,pages,lens,*,scale,causal,window_size,softcap,k_descale,v_descale,sinks,cascade,max_q,post_request=None):
    global EAGER_CALLS,CAPTURE_CALLS
    if (not ENABLED or torch.is_grad_enabled() or torch.compiler.is_compiling()
        or fb is None or not fb.forward_mode.is_decode() or fb.spec_info is not None
        or fb.batch_size!=1 or cascade or sinks is not None or not causal
        or window_size!=(-1,-1) or softcap!=0 or scale!=.0625 or max_q!=1
        or k_descale is not None or v_descale is not None
        or owner.use_mla or owner.is_encoder_decoder or owner.dcp_size!=1
        or owner.page_size!=128 or owner.max_context_len>4096
        or owner.num_splits!=0 or owner.ar_decode_splits!=0):return None
    if not can_use(q,k,v,lens,pages):return None
    from sglang.srt.runtime_context import get_parallel,get_spec
    if get_parallel().tp_size!=1 or get_spec().speculative_algorithm is not None:return None
    from sglang.srt.layers.moe.paged_experts import cpu40
    if cpu40.MANAGER is None or len(cpu40.MANAGER.layers)!=40:return None
    out=context(q.device).run(q,k,v,lens,pages,post_request=post_request)
    if torch.xpu.is_current_stream_capturing():CAPTURE_CALLS+=1
    else:EAGER_CALLS+=1
    return out

def readback():
    from sglang.srt.layers import ar2015_attention_post as post
    return dict(post2015=post.readback(),enabled=ENABLED,eager_calls=EAGER_CALLS,capture_calls=CAPTURE_CALLS,
        library='native/attention277/attention.so',library_sha256=None if CONTEXT is None else CONTEXT.sha256,
        KV_copied=False,query_layout='(1,16,256) -> contiguous (8,2,256) -> restored',
        KV_partitions=1,ar_only=True,GPU_census=False)
