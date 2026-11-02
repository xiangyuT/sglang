"""Opt-in B70 sharedSiLU/FP8+Down; private SLM, original quant/dot bits."""
import ctypes,hashlib,logging,os
from pathlib import Path
from functools import lru_cache
import torch
ENABLED=os.environ.get('CODEX_AR1444_SHARED_DOWN')=='1'
LIBRARY=Path('/evidence/tpot-iterations-20261023/native/silu1440/fused.so')
SHA='7eee72be3a557d0af5157e327a9ef2e14efdcb8b4deb26217ddfd2227232beb9'
logger=logging.getLogger(__name__)
@lru_cache(maxsize=1)
def resources():
 if torch.xpu.is_current_stream_capturing():raise RuntimeError('Warm AR1444 shared kernel before capture')
 assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==SHA
 from sglang.srt.layers.ar91_silu import resources as original_resources
 _,lut=original_resources()
 lib=ctypes.CDLL(str(LIBRARY));lib.silu1440_run.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*4+[ctypes.c_float]
 lib.silu1440_run.restype=ctypes.c_int;lib.silu1440_error.restype=ctypes.c_char_p
 logger.info('AR1444 sharedSiLU/FP8+Down enabled: WG128/SG32/GRF256,4subgroups,Rows2,SLM2048,mode0; originalLUT/FP8/FP16/dot tree')
 return lib,lut

def try_run(module,x):
 if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling()or not isinstance(x,torch.Tensor):return None
 down=getattr(module,'down_proj',None);method=getattr(down,'quant_method',None)
 w=getattr(down,'weight',None);ws=getattr(down,'weight_scale_inv',None)
 if not(type(down).__name__=='RowParallelLinear'and not down._forward_hooks and not down._forward_pre_hooks
     and getattr(down,'bias',None)is None and type(method).__name__=='Fp8LinearMethod'
     and getattr(method,'block_quant',False)and list(getattr(method,'weight_block_size',[]))==[128,128]
     and not any(getattr(method,k,False)for k in ['use_mxfp8','block_fp8_as_mxfp8','use_marlin'])
     and isinstance(w,torch.Tensor)and isinstance(ws,torch.Tensor)
     and x.device.type=='xpu'and x.shape==(1,1024)and x.dtype==torch.bfloat16 and x.is_contiguous()
     and w.device==ws.device==x.device and w.shape==(2048,512)and w.dtype==torch.float8_e4m3fn and w.is_contiguous()
     and ws.shape==(16,4)and ws.dtype==torch.float32 and ws.stride()in [(4,1),(1,16)]
     and x.data_ptr()%8==w.data_ptr()%16==0 and ws.data_ptr()%4==0):return None
 from sglang.srt.runtime_context import get_parallel,get_spec,get_exec
 if get_parallel().tp_size!=1 or get_spec().speculative_algorithm is not None or not get_exec().moe.enable_paged_experts:return None
 from sglang.srt.layers.moe.paged_experts import cpu40
 manager=cpu40.MANAGER
 # CPU40.register establishes K192 for every registered layer.
 if manager is None or len(manager.layers)!=40 or manager.layers[0]['executor'].residency.slot_expert.numel()!=192:return None
 if torch.xpu.get_device_properties(x.device).device_id!=0xe223:return None
 lib,lut=resources();y=torch.empty((1,2048),device=x.device,dtype=torch.bfloat16);stream=torch.xpu.current_stream(x.device)
 rc=lib.silu1440_run(stream.sycl_queue,x.data_ptr(),w.data_ptr(),ws.data_ptr(),lut.data_ptr(),y.data_ptr(),None,None,2048,*ws.stride(),0,448.)
 if rc:raise RuntimeError((rc,lib.silu1440_error().decode()))
 for v in [x,w,ws,lut,y]:v.record_stream(stream)
 key='_ar1444_capture_calls'if torch.xpu.is_current_stream_capturing()else'_ar1444_eager_calls'
 setattr(module,key,getattr(module,key,0)+1)
 return y

def readback(model):
 rows=[dict(name=name,capture_calls=getattr(m,'_ar1444_capture_calls',0),eager_calls=getattr(m,'_ar1444_eager_calls',0))for name,m in model.named_modules()if hasattr(m,'_ar1444_capture_calls')or hasattr(m,'_ar1444_eager_calls')]
 return dict(enabled=ENABLED,rows=rows,observed_modules=len(rows),capture_calls=sum(r['capture_calls']for r in rows),eager_calls=sum(r['eager_calls']for r in rows),library='native/silu1440/fused.so',library_sha256=SHA,mode=0,rows_per_subgroup=2,subgroups=4,workgroup=128,SLM_bytes=2048,unchanged_LUT=True,GPU_census=False)
