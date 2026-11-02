"""Opt-in AR shared SiLU/FP8 producer; explicit tuple transport to down_proj.

One immutable BF16 SiLU table per selected process/device, warmed before graph.
No diagnostic GPU operations, data-dependent host readback or tensor registry.
"""
import os,ctypes,logging
from functools import lru_cache
import torch
ENABLED=os.environ.get('CODEX_AR91_SILU')=='1'
LIB_PATH='/evidence/ar-round89/native/lut06/silu.so'

@lru_cache(maxsize=1)
def resources():
    if torch.xpu.is_current_stream_capturing():raise RuntimeError('AR91 library/table must warm before graph capture')
    from sgl_kernel import silu_and_mul
    gates=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).to('xpu')
    lut=silu_and_mul(torch.cat([gates,torch.ones_like(gates)]).reshape(1,-1)).reshape(-1)
    lib=ctypes.CDLL(LIB_PATH);lib.ar_silu_error.restype=ctypes.c_char_p;lib.ar_silu_lut_run.argtypes=[ctypes.c_void_p]*6+[ctypes.c_float,ctypes.c_int]
    logging.getLogger(__name__).info('AR91 shared SiLU/FP8 producer enabled: R89 exact LUT WG16; explicit down tuple; no GPU census')
    return lib,lut

def try_run(module,x):
    if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling()or not isinstance(x,torch.Tensor):return None
    method=getattr(module.down_proj,'quant_method',None);weight=getattr(module.down_proj,'weight',None)
    if not(x.device.type=='xpu'and x.dtype==torch.bfloat16 and x.shape==(1,1024)and x.is_contiguous()and type(method).__name__=='Fp8LinearMethod'and getattr(method,'block_quant',False)and list(getattr(method,'weight_block_size',[]))==[128,128]and not getattr(method,'use_mxfp8',False)and not getattr(method,'block_fp8_as_mxfp8',False)and not getattr(method,'use_marlin',False)and weight is not None and weight.shape==(2048,512)and weight.dtype==torch.float8_e4m3fn and weight.device==x.device and weight.is_contiguous()):return None
    from sglang.srt.runtime_context import get_spec,get_parallel,get_exec
    if get_spec().speculative_algorithm is not None or get_parallel().tp_size!=1 or not get_exec().moe.enable_paged_experts:return None
    if torch.xpu.get_device_properties(x.device).device_id!=0xe223:return None
    lib,lut=resources();y=torch.empty((1,512),device=x.device,dtype=torch.bfloat16);q=torch.empty((1,512),device=x.device,dtype=torch.float8_e4m3fn);s=torch.empty((1,4),device=x.device,dtype=torch.float32)
    stream=torch.xpu.current_stream(x.device)
    rc=lib.ar_silu_lut_run(stream.sycl_queue,x.data_ptr(),y.data_ptr(),q.data_ptr(),s.data_ptr(),lut.data_ptr(),448.,16)
    if rc:raise RuntimeError((rc,lib.ar_silu_error().decode()))
    for t in [x,y,q,s,lut]:t.record_stream(stream)
    # Host-only metadata describes which code branches warmed/captured, never routes tensors.
    key='_ar91_capture_calls'if torch.xpu.is_current_stream_capturing()else'_ar91_eager_calls';setattr(module,key,getattr(module,key,0)+1)
    return q,s

def readback(model):
    rows=[]
    for name,m in model.named_modules():
        if hasattr(m,'_ar91_capture_calls')or hasattr(m,'_ar91_eager_calls'):
            rows.append(dict(name=name,down_shape=list(m.down_proj.weight.shape),quant_method=type(m.down_proj.quant_method).__name__,eager_python_calls=getattr(m,'_ar91_eager_calls',0),capture_python_calls=getattr(m,'_ar91_capture_calls',0)))
    return dict(enabled=ENABLED,rows=rows,observed_modules=len(rows),library=LIB_PATH,LUT_bytes=131072,transport='explicit(q,scale)toexistingblockFP8down',GPU_census=False,torch_profiler_used=False,no_host_readback_inside_graph=True)
