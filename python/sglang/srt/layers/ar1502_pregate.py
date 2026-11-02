"""Opt-in B1 CPU-hybrid shared gate/next-norm handoff; original router retained.

The resident GU command computes the scalar in one disjoint WG. The existing
DeferredFinalize boundary owns the next read, with the original combine as its
unfused completion. No tensor attributes route per-invocation payloads.
"""
import ctypes
from dataclasses import dataclass
from functools import lru_cache,partial
import hashlib
import logging
import os
from pathlib import Path

import torch
from sglang.srt.layers.layer_boundary.output import DeferredFinalize
from sglang.srt.layers.layer_boundary.residual.add_norm import Fp8Input,NormQuantReadout
from sglang.srt.model_executor.forward_batch_info import ForwardMode

ENABLED=os.environ.get('CODEX_AR1502_PREGATE')=='1'
CONSUMER=os.environ.get('CODEX_AR1502_CONSUMER','1')=='1'
R=Path('/evidence/tpot-iterations-20261023')
PINS={'native/pregate1496/resident.so':'b164caf2df3fa9ef6ae4c62100f8a87483a02220afe8f64d7b81abf61a6ec004',
      'native/pregate1492/norm.so':'9fb4370ad67c619cd2b62839e6bf835d63b6c55f18379ac0504b95e9d1b98568'}
logger=logging.getLogger(__name__)

@lru_cache(maxsize=1)
def libraries():
    assert not torch.xpu.is_current_stream_capturing(),'warm pregate libraries before capture'
    libs=[]
    for name,pin in PINS.items():
        path=R/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
        libs.append(ctypes.CDLL(str(path)))
    resident,norm=libs
    resident.resident1496_run.argtypes=[ctypes.c_void_p]*11
    resident.pregate1496_error.restype=ctypes.c_char_p
    norm.gatenorm1492_run.argtypes=[ctypes.c_void_p]*13+[ctypes.c_float]*2+[ctypes.c_int]
    norm.gatenorm1492_error.restype=ctypes.c_char_p
    return resident,norm

def _bf(t,shape):
    return (isinstance(t,torch.Tensor)and tuple(t.shape)==shape and t.device.type=='xpu'
            and t.dtype==torch.bfloat16 and t.is_contiguous())

def _decode(batch):
    return batch is not None and batch.forward_mode is ForwardMode.DECODE and batch.input_ids.numel()==1

def _parallel():
    from sglang.srt.runtime_context import get_parallel,get_spec
    p=get_parallel()
    return (p.tp_size==p.moe_ep_size==p.attn_cp_size==1
            and p.pp_group.world_size==1 and not p.attn_dp_enabled
            and get_spec().speculative_algorithm is None)

def requested(block,batch,hidden):
    fusion=getattr(block,'_ar1502_fusion',None)
    return bool(fusion is not None and fusion.producer and fusion._scope(batch)
                and _bf(hidden,(1,2048)))

@dataclass
class Lease:
    owner:object
    consumed:bool=False
    def claim(self):
        if self.consumed:raise RuntimeError('AR1502 finalize handoff already consumed')
        self.consumed=True

class GateHandoff(DeferredFinalize,frozen=True):
    hidden:torch.Tensor
    gate_weight:torch.Tensor
    shared:torch.Tensor
    routed:torch.Tensor
    gate:object
    lease:object
    def complete(self):
        from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
        self.lease.claim()
        fused_gate_sigmoid_mul_add(self.hidden,self.gate_weight.squeeze(),self.shared,self.routed)
        self.lease.owner.note('fallback')
        return self.routed

def handoff(block,ctx,hidden,shared,routed):
    fusion=block._ar1502_fusion
    assert shared is not None and not block.reduce_results
    out=GateHandoff(hidden,block.shared_expert_gate.weight,shared,routed,
                    getattr(ctx,'gate',None),Lease(fusion))
    fusion.note('handoff')
    return out

def issue_resident(manager,ex,x,ctx,a,stream):
    if not ENABLED or not getattr(ctx,'deferred_gate',False):return False
    assert ctx.gate is None and not ctx.apply_gate,'AR1502 requires one ungated shared branch'
    block=ctx.block;gw=block.shared_expert_gate.weight
    if not (_bf(x,(1,2048))and _bf(a,(8,512))and _bf(gw,(1,2048))):return False
    from sglang.srt.layers.ar1657_shared_union import try_issue
    if try_issue(manager,ex,x,ctx,a,stream):
        block._ar1502_fusion.note('producer');ex._ar1502_resident_active=True
        return True
    t=manager.layers[ex._cpu40_channel];residency=ex.residency
    tensors=[x,t['w1'],t['s1'],ex.mapped_ids,t['map'],residency.count,a,manager.stats]
    resident,_=libraries();gate=torch.empty((1,),device=x.device,dtype=torch.float32)
    rc=resident.resident1496_run(stream.sycl_queue,*[v.data_ptr()for v in tensors],gw.data_ptr(),gate.data_ptr())
    if rc:raise RuntimeError((rc,resident.pregate1496_error().decode()))
    for v in tensors+[gw,gate]:v.record_stream(stream)
    ctx.gate=gate;block._ar1502_fusion.note('producer')
    ex._ar1502_resident_active=True
    return True

class GateFusion:
    def __init__(self):
        self.block=None;self.producer=False;self.counts={}
    def note(self,kind):
        key=kind+('_capture'if torch.xpu.is_current_stream_capturing()else'_eager')
        self.counts[key]=self.counts.get(key,0)+1
    def _scope(self,batch):
        block=self.block
        if not (ENABLED and block is not None and _decode(batch)and not torch.is_grad_enabled()
                and not torch.compiler.is_compiling() and _parallel()):return False
        if not (os.environ.get('CODEX_AR_CPU40')=='1'and os.environ.get('CODEX_AR_CPU119_OVERLAP')=='1'
                and os.environ.get('CODEX_AR124_JOINT')=='1'and os.environ.get('CODEX_AR124_MODE','1')=='1'
                and os.environ.get('CODEX_AR548_WORD')=='1'and os.environ.get('CODEX_AR1014_ROUTER_SHARED')=='1'
                and os.environ.get('CODEX_AR86_BOUNDARY')=='1'):return False
        method=getattr(block.experts,'quant_method',None);ex=getattr(method,'device_executor',None)
        return (type(method).__name__=='PagedExpertsMoEMethod'and type(ex).__name__=='XpuDeviceExecutor'
                and ex.use_scaled_fp8 and hasattr(ex,'_cpu40_channel')and ex.residency.slot_expert.numel()==192
                and block.alt_stream is None and not block.enable_shared_expert_fusion
                and not block._use_mega_moe and not block.reduce_results
                and block.shared_expert is not None and block.shared_expert_gate is not None)
    def can_defer_finalize(self,plan,batch):
        return self.producer and not plan.terminal and self._scope(batch)
    def can_defer_all_reduce(self,plan,batch):return False
    def ffn_input_fusions(self,plan):return ()
    def attn_input_fusions(self,plan):return (partial(self.consume,plan),)
    def consume(self,plan,owed,residual,batch,post_residual_addition):
        if not (CONSUMER and isinstance(owed,GateHandoff)and owed.gate is not None
                and post_residual_addition is None and self._scope(batch)):
            return None
        norm=plan.norm;read=next(iter(plan.edges.values())).incoming.need.read
        if not (type(norm).__name__=='GemmaRMSNorm'and isinstance(read,NormQuantReadout)
                and read.ar_bmg and read.fp8_input is not None and _bf(norm.weight,(2048,))
                and _bf(residual,(1,2048))and _bf(owed.hidden,(1,2048))
                and _bf(owed.shared,(1,2048))and _bf(owed.routed,(1,2048))
                and _bf(owed.gate_weight,(1,2048))and owed.gate.shape==(1,)
                and owed.gate.dtype==torch.float32 and owed.gate.is_contiguous()
                and all(v.device==residual.device for v in [owed.hidden,owed.gate_weight,owed.shared,owed.routed,owed.gate,norm.weight])
                and float(norm.variance_epsilon)>0
                and torch.xpu.get_device_properties(residual.device).device_id==0xe223):return None
        _,lib=libraries();owed.lease.claim();x=owed.routed
        quant=torch.empty_like(x,dtype=torch.float8_e4m3fn)
        scale=torch.empty((1,16),device=x.device,dtype=torch.float32)
        stream=torch.xpu.current_stream(x.device)
        args=[stream.sycl_queue,owed.hidden.data_ptr(),owed.gate_weight.data_ptr(),owed.shared.data_ptr(),
              x.data_ptr(),residual.data_ptr(),norm.weight.data_ptr(),None,x.data_ptr(),residual.data_ptr(),
              quant.data_ptr(),scale.data_ptr(),owed.gate.data_ptr()]
        rc=lib.gatenorm1492_run(*args,norm.variance_epsilon,448.,2)
        if rc:raise RuntimeError((rc,lib.gatenorm1492_error().decode()))
        for v in [owed.hidden,owed.gate_weight,owed.shared,x,residual,norm.weight,quant,scale,owed.gate]:v.record_stream(stream)
        self.note('consumer')
        payload=(quant,scale)if read.fp8_input is Fp8Input.TUPLE else(x,quant,scale)
        return payload,residual

def install(layers):
    if not ENABLED:return
    installed=[]
    for layer in layers:
        fusion=layer.ffn_boundary.plan.fusions
        if not isinstance(fusion,GateFusion):continue
        fusion.block=layer.mlp
        fusion.producer=not layer.ffn_boundary.plan.terminal
        layer.mlp._ar1502_fusion=fusion
        installed.append(layer.layer_id)
    logger.info('AR1502 pregate handoff installed: layers=%s; nonterminal B1 decode only; router binary unchanged',installed)

def readback(model):
    rows=[]
    for module in model.modules():
        fusion=getattr(module,'_ar1502_fusion',None)
        if isinstance(fusion,GateFusion):rows.append(dict(layer=module.layer_id,producer=fusion.producer,**fusion.counts))
    return dict(enabled=ENABLED,consumer_enabled=CONSUMER,native=PINS,rows=rows,
                router_binary_unchanged=os.environ.get('CODEX_AR1690_SHARED_ROWS')!='1',router_math_unchanged=True,resident_groups=(2305 if os.environ.get('CODEX_AR1657_SHARED')=='1'else 2049),consumer_mode=2,
                explicit_finalize=True,single_use=True,terminal_fallback=True,GPU_census=False)
