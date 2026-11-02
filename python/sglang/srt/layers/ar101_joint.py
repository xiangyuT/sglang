"""Opt-in explicit shared-GU transport through the standard MoE call chain."""
import ctypes,logging,os
from dataclasses import dataclass
from functools import lru_cache
import torch

ENABLED=os.environ.get('CODEX_AR101_JOINT')=='1'
LIB='/evidence/ar-round99/native/joint02/joint.so'

@dataclass
class Joint:
    q:torch.Tensor
    scale:torch.Tensor
    weight:torch.Tensor
    weight_scale:torch.Tensor
    gu:torch.Tensor
    issued:bool=False

@lru_cache(maxsize=1)
def resources():
    if torch.xpu.is_current_stream_capturing():raise RuntimeError('AR101 library must warm before capture')
    lib=ctypes.CDLL(LIB);lib.ar99_error.restype=ctypes.c_char_p
    lib.ar99_run.argtypes=[ctypes.c_void_p]*15+[ctypes.c_int]*3
    logging.getLogger(__name__).info('AR101 shared-first GU serving enabled; explicit per-call context; original down/combine; no GPU census')
    return lib

def prepare(block,x,shared_input,defer_finalize):
    if not ENABLED or defer_finalize or torch.is_grad_enabled()or torch.compiler.is_compiling():return None
    if os.environ.get('CODEX_AR_CPU40')=='1' or block.alt_stream is not None or block.enable_shared_expert_fusion:return None
    if not(isinstance(shared_input,tuple)and len(shared_input)==2 and type(block.experts).__name__=='FusedMoE'):return None
    shared=block.shared_expert
    if shared is None or shared._forward_hooks or shared._forward_pre_hooks or shared._enable_silu_fp4_quant_fusion:return None
    method=block.experts.quant_method;ex=getattr(method,'device_executor',None)
    if not(type(method).__name__=='PagedExpertsMoEMethod'and type(ex).__name__=='XpuDeviceExecutor'and ex.use_scaled_fp8 and ex.residency.slot_expert.numel()==192):return None
    if getattr(block.experts,'_dwdp_bound',False):return None
    from sglang.srt.runtime_context import get_spec,get_parallel
    if get_spec().speculative_algorithm is not None or get_parallel().tp_size!=1 or get_parallel().moe_ep_size!=1:return None
    q,s=shared_input;w=shared.gate_up_proj.weight;ws=shared.gate_up_proj.weight_scale_inv
    if not(x.device.type=='xpu'and x.shape==(1,2048)and x.dtype==torch.bfloat16 and x.is_contiguous()and
        q.shape==(1,2048)and q.dtype==torch.float8_e4m3fn and s.shape==(1,16)and s.dtype==torch.float32 and
        w.shape==(1024,2048)and w.dtype==torch.float8_e4m3fn and ws.shape==(8,16)and ws.dtype==torch.float32 and
        all(t.device==x.device for t in [q,s,w,ws])and all(t.is_contiguous()for t in [q,s,w])and all(0<v<2**31 for v in ws.stride())):return None
    if torch.xpu.get_device_properties(x.device).device_id!=0xe223:return None
    resources()
    key='_ar101_capture_calls'if torch.xpu.is_current_stream_capturing()else'_ar101_eager_calls'
    setattr(block,key,getattr(block,key,0)+1)
    block._ar101_scale_stride=tuple(ws.stride())
    return Joint(q,s,w,ws,torch.empty((1,1024),device=x.device,dtype=torch.bfloat16))

def run(ctx,x,w,s,ids,a,hp,paging):
    assert not ctx.issued,'AR101 context must be issued exactly once'
    assert x.shape==(1,2048)and w.shape==(192,1024,2048)and s.shape==(192,8,16)and ids.shape==(1,8)and a.shape==(8,512)
    assert all(t.is_contiguous()for t in [x,w,s,ids,a])
    lib=resources();stream=torch.xpu.current_stream(x.device)
    tensors=[x,w,s,ids,a,hp,paging.dst,paging.src,paging.count,ctx.q,ctx.weight,ctx.scale,ctx.weight_scale,ctx.gu]
    rc=lib.ar99_run(stream.sycl_queue,*[t.data_ptr()for t in tensors],*ctx.weight_scale.stride(),0)
    if rc:raise RuntimeError((rc,lib.ar99_error().decode()))
    for t in tensors:t.record_stream(stream)
    ctx.issued=True

def readback(model):
    rows=[]
    for name,m in model.named_modules():
        if hasattr(m,'_ar101_eager_calls')or hasattr(m,'_ar101_capture_calls'):
            rows.append(dict(name=name,eager_python_calls=getattr(m,'_ar101_eager_calls',0),capture_python_calls=getattr(m,'_ar101_capture_calls',0),shared_scale_stride=list(m._ar101_scale_stride)))
    return dict(enabled=ENABLED,observed_modules=len(rows),rows=rows,library=LIB,transport='explicit Joint per-call through FusedMoE/PagedExperts/device executor',
                consumes_joint_GU_and_A=True,GPU_census=False,torch_profiler_used=False,normal_TPS_evidence=False)
