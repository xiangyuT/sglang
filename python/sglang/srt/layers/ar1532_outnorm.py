"""B1 TP1 output-projection/read handoff through the existing boundary API.

TP1 attention uses the declared EXIT_SCOPED identity sum. A producer-owned
DeferredFinalize then reaches the FFN read, which consumes it once or completes
the original projection. No boundary implementation or tensor attributes change.
"""
import ctypes,hashlib,logging,os
from dataclasses import dataclass,replace
from functools import lru_cache,partial
from pathlib import Path
import torch
from sglang.srt.layers.layer_boundary.output import DeferredFinalize
from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion,ProducerReduction,BatchVariant
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.residual.add_norm import NormQuantReadout,Fp8Input
from sglang.srt.model_executor.forward_batch_info import ForwardMode
ENABLED=os.environ.get('CODEX_AR1532_OUTNORM')=='1'
CONSUMER=os.environ.get('CODEX_AR1532_CONSUMER','1')=='1'
LIBRARY='/evidence/tpot-iterations-20261023/native/outnorm1853/fused.so'
PIN='9621ecea6139217a08829f60fda135c92aa01f5463bb1d67289c6c2d10fbe290'
logger=logging.getLogger(__name__)

def _parallel():
 from sglang.srt.runtime_context import get_parallel,get_spec
 p=get_parallel()
 return p.tp_size==p.attn_tp_size==p.moe_ep_size==p.attn_cp_size==1 and p.pp_group.world_size==1 and not p.attn_dp_enabled and get_spec().speculative_algorithm is None

@dataclass(frozen=True)
class ProjectionReadout(NormQuantReadout):
 @property
 def completing_fusions(self):
  return (ReadoutFusion(SumGroup.ATTN_TP,partial(consume,self),reads=True,carried=True),)

def declarations(config,attn,ffn):
 if not(ENABLED and config.hidden_size==2048 and config.model_type=='qwen3_5_moe_text' and _parallel()
        and os.environ.get('CODEX_AR671_OUTPUT')=='1'and os.environ.get('CODEX_AR86_BOUNDARY')=='1'
        and isinstance(ffn.read,NormQuantReadout)and ffn.read.ar_bmg and ffn.read.fp8_input is not None):return attn,ffn
 read=ProjectionReadout(fp8_input=ffn.read.fp8_input,ar_bmg=True,reads_before_dp_gather=ffn.read.reads_before_dp_gather)
 return replace(attn,reduction=ProducerReduction.EXIT_SCOPED),replace(ffn,read=read)

@lru_cache(maxsize=1)
def library():
 assert not torch.xpu.is_current_stream_capturing(),'warm AR1532 before capture'
 assert hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest()==PIN
 lib=ctypes.CDLL(LIBRARY);lib.outnorm1524_run.argtypes=[ctypes.c_void_p]*12+[ctypes.c_float]*2+[ctypes.c_int]*3
 lib.outnorm1524_error.restype=ctypes.c_char_p
 logger.info('AR1532 projection/postnorm enabled: last128WG arrival; SG32 projection/virtualSG16 norm; runtime448; no grid polling')
 return lib

def _bf(t,shape):
 return isinstance(t,torch.Tensor)and t.device.type=='xpu'and t.dtype==torch.bfloat16 and tuple(t.shape)==shape and t.is_contiguous()

class Target:
 def __init__(self,layer,projection,read):
  self.layer=layer;self.projection=projection;self.read=read;self.norm=layer.ffn_boundary.norm;self.counter=None;self.counts={}
 def note(self,kind):
  key=kind+('_capture'if torch.xpu.is_current_stream_capturing()else'_eager');self.counts[key]=self.counts.get(key,0)+1
 def scope(self,batch):
  if not(ENABLED and batch is not None and batch.forward_mode is ForwardMode.DECODE and batch.input_ids.numel()==1
         and not torch.is_grad_enabled()and not torch.compiler.is_compiling()and _parallel()):return False
  if not(self.layer.attn_boundary.plan.variant_for(batch)is BatchVariant.ORDINARY
         and os.environ.get('CODEX_AR_CPU40')=='1'and os.environ.get('CODEX_AR671_OUTPUT')=='1'
         and os.environ.get('CODEX_AR86_BOUNDARY')=='1'):return False
  from sglang.srt.layers.moe.paged_experts import cpu40
  m=cpu40.MANAGER
  return m is not None and len(m.layers)==40 and all(t['executor'].residency.slot_expert.numel()==192 for t in m.layers)

@dataclass
class Lease:
 target:object
 consumed:bool=False
 def claim(self):
  if self.consumed:raise RuntimeError('AR1532 projection handoff already consumed')
  self.consumed=True

class ProjectionHandoff(DeferredFinalize,frozen=True):
 projection:object
 q:torch.Tensor
 scale:torch.Tensor
 lease:object
 @property
 def completion_group(self):return SumGroup.ATTN_TP
 def complete(self):
  self.lease.claim();self.lease.target.note('fallback')
  # Original module owns all dispatch and its ordinary TP1 identity sum.
  return self.projection((self.q,self.scale))[0]

def try_defer(projection,value,batch):
 target=getattr(projection,'_ar1532_target',None)
 if target is None or not target.scope(batch):return None
 method=getattr(projection,'quant_method',None);w=getattr(projection,'weight',None);ws=getattr(projection,'weight_scale_inv',None)
 norm=target.norm;residual=getattr(getattr(batch,'residual_stream',None),'residual',None)
 if not(type(projection).__name__=='RowParallelLinear'and getattr(projection,'bias',None)is None and not projection.reduce_results
        and not projection._forward_hooks and not projection._forward_pre_hooks
        and type(method).__name__=='Fp8LinearMethod'and method.block_quant and list(method.weight_block_size)==[128,128]
        and not getattr(method,'use_marlin',False)and not getattr(method,'use_mxfp8',False)and not getattr(method,'block_fp8_as_mxfp8',False)
        and type(norm).__name__=='GemmaRMSNorm'and _bf(norm.weight,(2048,))and norm.variance_epsilon>0
        and _bf(residual,(1,2048))and w is not None and ws is not None
        and w.device==ws.device==norm.weight.device==residual.device and w.dtype==torch.float8_e4m3fn
        and w.shape==(2048,4096)and w.is_contiguous()and w.data_ptr()%16==0
        and ws.dtype==torch.float32 and ws.shape==(16,32)and ws.stride()in[(32,1),(1,16)]
        and torch.xpu.get_device_properties(w.device).device_id==0xe223):return None
 if isinstance(value,tuple):
  if len(value)!=2:return None
  q,s=value
  if not(isinstance(q,torch.Tensor)and isinstance(s,torch.Tensor)and q.shape==(1,4096)and q.dtype==torch.float8_e4m3fn
         and s.shape==(1,32)and s.dtype==torch.float32 and q.device==s.device==w.device and q.is_contiguous()and s.is_contiguous()):return None
 else:
  if not _bf(value,(1,4096))or value.device!=w.device:return None
  # Reuse the exact quantization dispatch from torch_w8a8_block_fp8_linear.
  from sglang.srt.environ import envs
  use_sycl=False
  if envs.SGLANG_XPU_PAGED_DFLASH.get()and envs.SGLANG_XPU_DFLASH_QUANT_SYCL.get():
   from sglang.kernels.ops.quantization.dflash_quant_sycl import can_use_dflash_quant,dflash_quant
   use_sycl=can_use_dflash_quant(value)
  if use_sycl:q,s=dflash_quant(value)
  else:
   from sglang.srt.layers.quantization.fp8_utils import per_token_group_quant_fp8
   q,s=per_token_group_quant_fp8(value,128)
 lib=library()
 if target.counter is None:
  assert not torch.xpu.is_current_stream_capturing(),'warm AR1532 counter before capture'
  target.counter=torch.zeros((1,),device=q.device,dtype=torch.int32)
 target.note('producer')
 return ProjectionHandoff(projection,q,s,Lease(target))

def consume(read,owed,residual,batch,norm):
 if not(CONSUMER and isinstance(owed,ProjectionHandoff)):return None
 target=owed.lease.target
 if not(target.read is read and target.norm is norm and target.scope(batch)and _bf(residual,(1,2048))and _bf(norm.weight,(2048,))):return None
 q,s=owed.q,owed.scale;w=owed.projection.weight;ws=owed.projection.weight_scale_inv
 if not(q.device==s.device==w.device==ws.device==norm.weight.device==residual.device):return None
 lib=library();owed.lease.claim();y=torch.empty_like(residual);quant=torch.empty_like(y,dtype=torch.float8_e4m3fn);scale=torch.empty((1,16),dtype=torch.float32,device=y.device)
 stream=torch.xpu.current_stream(y.device)
 rc=lib.outnorm1524_run(stream.sycl_queue,q.data_ptr(),w.data_ptr(),s.data_ptr(),ws.data_ptr(),y.data_ptr(),residual.data_ptr(),norm.weight.data_ptr(),quant.data_ptr(),scale.data_ptr(),target.counter.data_ptr(),None,norm.variance_epsilon,448.,*ws.stride(),0)
 if rc:raise RuntimeError((rc,lib.outnorm1524_error().decode()))
 for t in[q,w,s,ws,y,residual,norm.weight,quant,scale,target.counter]:t.record_stream(stream)
 target.note('consumer')
 payload=(quant,scale)if read.fp8_input is Fp8Input.TUPLE else(y,quant,scale)
 return payload,residual

def install(layers):
 if not ENABLED:return
 installed=[]
 for layer in layers:
  read=next(iter(layer.ffn_boundary.plan.edges.values())).incoming.need.read
  if not isinstance(read,ProjectionReadout):continue
  projection=layer.linear_attn.out_proj if hasattr(layer,'linear_attn')else layer.o_proj
  projection._ar1532_target=Target(layer,projection,read);installed.append(layer.layer_id)
 logger.info('AR1532 projection handoff installed: layers=%s;TP1 EXIT_SCOPED identity;explicit read/fallback',installed)

def readback(model):
 rows=[]
 for m in model.modules():
  target=getattr(m,'_ar1532_target',None)
  if isinstance(target,Target):rows.append(dict(layer=target.layer.layer_id,counter=None if target.counter is None else int(target.counter.cpu()[0]),**target.counts))
 return dict(enabled=ENABLED,consumer_enabled=CONSUMER,rows=rows,library=LIBRARY,library_sha256=PIN,projection_WGs=128,projection_SG=32,norm_virtual_SG=16,runtime_limit=448.,explicit_finalize=True,single_use=True,TP1_identity_sum=True,GPU_census=False)
