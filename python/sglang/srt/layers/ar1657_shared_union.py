"""Optional nonterminal shared-down/resident-GU union; explicit single-use output."""
import ctypes,hashlib,logging,os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import torch
ENABLED=os.environ.get('CODEX_AR1657_SHARED')=='1'
LIBRARY=Path('/evidence/tpot-iterations-20261023/native/fusion1654/fused.so')
SHA='a0d4aa506f319a0c5ac3b53f66ea80761aeb39729b5cb85e28a55bc6a561c1b6'
logger=logging.getLogger(__name__)
@dataclass
class Output:
 key:tuple
 value:torch.Tensor
 retained:tuple
 consumed:bool=False
@lru_cache(maxsize=1)
def resources():
 assert not torch.xpu.is_current_stream_capturing(),'Warm AR1657 before capture'
 assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest()==SHA
 from sglang.srt.layers.ar1444_shared import resources as old_resources
 _,lut=old_resources()
 lib=ctypes.CDLL(str(LIBRARY));lib.fusion1654_run.argtypes=[ctypes.c_void_p]*16+[ctypes.c_int]*3+[ctypes.c_float];lib.fusion1654_error.restype=ctypes.c_char_p
 logger.info('AR1657 resident/shared down union enabled: mode0,WG128/SG32/GRF128,2305WGs; runtime FP32limit448; original router/CPU/refill preserved')
 return lib,lut

def _bf(x,shape,device):
 return isinstance(x,torch.Tensor)and x.device==device and x.dtype==torch.bfloat16 and x.shape==shape and x.is_contiguous()
def _key(ctx,gu,stream):
 down=ctx.block.shared_expert.down_proj
 return(id(ctx),id(ctx.block),ctx.hidden.data_ptr(),gu.data_ptr(),down.weight.data_ptr(),down.weight_scale_inv.data_ptr(),tuple(down.weight_scale_inv.stride()),stream.sycl_queue)
def _note(shared,kind):
 key='_ar1657_'+kind+('_capture'if torch.xpu.is_current_stream_capturing()else'_eager')
 setattr(shared,key,getattr(shared,key,0)+1)

def try_issue(manager,ex,x,ctx,a,stream):
 if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling():return False
 from sglang.srt.layers.moe.paged_experts.router_shared1014 import Receipt,key
 receipt=getattr(ctx,'_router1014',None)
 if not isinstance(receipt,Receipt)or receipt.consumed:return False
 if not getattr(ctx,'deferred_gate',False)or ctx.issued or ctx.apply_gate or ctx.gate is not None:return False
 if ctx.hidden is not x or ex._cpu40_channel>=39:return False
 if receipt.key!=key(manager,ex,x,ctx,stream):raise RuntimeError('AR1657 stale route receipt')
 gu=receipt.gu;shared=ctx.block.shared_expert;down=getattr(shared,'down_proj',None)
 method=getattr(down,'quant_method',None);w=getattr(down,'weight',None);ws=getattr(down,'weight_scale_inv',None)
 gw=ctx.block.shared_expert_gate.weight;device=x.device
 if not(_bf(x,(1,2048),device)and _bf(gu,(1,1024),device)and _bf(a,(8,512),device)and _bf(gw,(1,2048),device)
  and type(down).__name__=='RowParallelLinear'and not down._forward_hooks and not down._forward_pre_hooks
  and not getattr(shared,'_enable_silu_fp4_quant_fusion',False)and getattr(down,'bias',None)is None
  and type(method).__name__=='Fp8LinearMethod'and getattr(method,'block_quant',False)
  and list(getattr(method,'weight_block_size',[]))==[128,128]
  and not any(getattr(method,k,False)for k in ['use_mxfp8','block_fp8_as_mxfp8','use_marlin'])
  and isinstance(w,torch.Tensor)and isinstance(ws,torch.Tensor)and w.device==ws.device==device
  and w.shape==(2048,512)and w.dtype==torch.float8_e4m3fn and w.is_contiguous()
  and ws.shape==(16,4)and ws.dtype==torch.float32 and ws.stride()in[(4,1),(1,16)]
  and gu.data_ptr()%8==w.data_ptr()%16==0 and ws.data_ptr()%4==0
  and torch.xpu.get_device_properties(device).device_id==0xe223):return False
 assert not hasattr(ctx,'_ar1657_output'),'AR1657 output already issued'
 t=manager.layers[ex._cpu40_channel];r=ex.residency
 lib,lut=resources();gate=torch.empty((1,),device=device,dtype=torch.float32);y=torch.empty((1,2048),device=device,dtype=torch.bfloat16)
 tensors=[x,t['w1'],t['s1'],ex.mapped_ids,t['map'],r.count,a,manager.stats,gw,gate,gu,w,ws,lut,y]
 rc=lib.fusion1654_run(stream.sycl_queue,*[v.data_ptr()for v in tensors],*ws.stride(),0,448.)
 if rc:raise RuntimeError((rc,lib.fusion1654_error().decode()))
 for v in tensors:v.record_stream(stream)
 ctx.gate=gate;ctx._ar1657_output=Output(_key(ctx,gu,stream),y,tuple(tensors))
 ex._ar1657_resident_active=True;_note(shared,'producer')
 return True

def consume(ctx,gu):
 output=getattr(ctx,'_ar1657_output',None)
 if output is None:return None
 stream=torch.xpu.current_stream(gu.device)
 if output.consumed or output.key!=_key(ctx,gu,stream):raise RuntimeError('AR1657 stale/mismatched/already consumed output')
 assert ctx._router1014.consumed,'AR1657 route must complete before shared consume'
 output.consumed=True;_note(ctx.block.shared_expert,'consumer')
 return output.value

def readback(model):
 rows=[]
 for name,m in model.named_modules():
  if hasattr(m,'_ar1657_producer_capture')or hasattr(m,'_ar1657_producer_eager'):
   row=dict(name=name)
   for kind in ['producer','consumer']:
    for phase in ['capture','eager']:row[kind+'_'+phase]=getattr(m,'_ar1657_'+kind+'_'+phase,0)
   rows.append(row)
 return dict(enabled=ENABLED,rows=rows,library='native/fusion1654/fused.so',library_sha256=SHA,mode=0,groups=2305,GRF=128,WG=128,SG=32,SLM_bytes=2112,single_use=True,terminal_fallback=True,runtime_limit=448.,GPU_census=False)
