"""Explicit request/result for exact B1 attention restore+gate+block128 FP8.

The backend returns a named quantized result only to a requesting model caller.
It never exposes the packed head layout as a logical attention Tensor.
"""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib,logging,os
from pathlib import Path
import torch
ENABLED=os.environ.get('CODEX_AR2015_ATTENTION_POST')=='1'
LIBRARY='/evidence/tpot-iterations-20261023/native/attention2012/epilogue.so'
PIN='38942bc501e408878cd2988596ea47593e17cbf3fb7244eb479c31485c26e151'
EAGER_CALLS=CAPTURE_CALLS=0
LIBRARY_SHA=None
logger=logging.getLogger(__name__)

@dataclass(frozen=True)
class GateRequest:
 gate:torch.Tensor
 head_stride:int

@dataclass(frozen=True)
class GatedFp8Output:
 quant:torch.Tensor
 scale:torch.Tensor

def _gate_stride(gate):
 if not isinstance(gate,torch.Tensor)or gate.device.type!='xpu'or gate.dtype!=torch.bfloat16 or gate.data_ptr()%8:return None
 if tuple(gate.shape)==(1,16,256)and gate.stride(-1)==1 and gate.stride(1)in(256,512):return gate.stride(1)
 if tuple(gate.shape)==(1,4096)and gate.is_contiguous():return 256
 return None

def make_request(attention,projection,gate,batch):
 if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling():return None
 from sglang.srt.layers import ar612_attention
 if not ar612_attention.ENABLED:return None
 target=getattr(projection,'_ar1532_target',None)
 if target is None or target.projection is not projection or not target.scope(batch):return None
 if attention._forward_hooks or attention._forward_pre_hooks or projection._forward_hooks or projection._forward_pre_hooks:return None
 method=getattr(projection,'quant_method',None)
 if not(type(projection).__name__=='RowParallelLinear'and projection.bias is None and not projection.reduce_results
  and type(method).__name__=='Fp8LinearMethod'and method.block_quant and list(method.weight_block_size)==[128,128]
  and not any(getattr(method,k,False)for k in ['use_marlin','use_mxfp8','block_fp8_as_mxfp8'])):return None
 stride=_gate_stride(gate)
 if stride is None:return None
 from sglang.srt.model_executor.forward_context import get_attn_backend
 backend=get_attn_backend();full=getattr(backend,'full_attn_backend',backend)
 if type(full).__name__!='XPUAttentionBackend':return None
 return GateRequest(gate,stride)

def can_use(packed,request):
 return(ENABLED and isinstance(request,GateRequest)and _gate_stride(request.gate)==request.head_stride
  and isinstance(packed,torch.Tensor)and packed.device.type=='xpu'and packed.device==request.gate.device
  and packed.dtype==torch.bfloat16 and packed.shape==(8,2,256)and packed.is_contiguous()and packed.data_ptr()%8==0
  and torch.xpu.get_device_properties(packed.device).device_id==0xe223)

@lru_cache(maxsize=1)
def library():
 global LIBRARY_SHA
 assert not torch.xpu.is_current_stream_capturing(),'Warm attention2015 before capture'
 LIBRARY_SHA=hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest();assert LIBRARY_SHA==PIN
 lib=ctypes.CDLL(LIBRARY);lib.attention2012_run.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*3+[ctypes.c_float]
 lib.attention2012_run.restype=ctypes.c_int;lib.attention2012_error.restype=ctypes.c_char_p
 logger.info('AR2015 exact attention restore/gate/block128 FP8 enabled: direct sigmoid; BF16 boundary; maxnum/minnum; explicit result; no lookup table')
 return lib

def run(packed,request):
 global EAGER_CALLS,CAPTURE_CALLS
 assert can_use(packed,request)
 lib=library();quant=torch.empty((1,4096),dtype=torch.float8_e4m3fn,device=packed.device);scale=torch.empty((1,32),dtype=torch.float32,device=packed.device)
 stream=torch.xpu.current_stream(packed.device)
 rc=lib.attention2012_run(stream.sycl_queue,packed.data_ptr(),request.gate.data_ptr(),None,quant.data_ptr(),scale.data_ptr(),None,request.head_stride,0,0,448.)
 if rc:raise RuntimeError((rc,lib.attention2012_error().decode()))
 for tensor in[packed,request.gate,quant,scale]:tensor.record_stream(stream)
 if torch.xpu.is_current_stream_capturing():CAPTURE_CALLS+=1
 else:EAGER_CALLS+=1
 return GatedFp8Output(quant,scale)

def readback():
 return dict(enabled=ENABLED,eager_calls=EAGER_CALLS,capture_calls=CAPTURE_CALLS,library='native/attention2012/epilogue.so',library_sha256=LIBRARY_SHA,direct_sigmoid=True,lookup_table=False,BF16_boundary=True,block_size=128,explicit_quantized_result=True,packed_tensor_exposed=False,GPU_census=False)
