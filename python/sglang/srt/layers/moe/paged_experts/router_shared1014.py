"""Scoped shared GU movement; original router publication protocol is retained."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import torch
from sglang.srt.layers.moe.paged_experts.router812 import Args

ROWS16=os.environ.get('CODEX_AR1690_SHARED_ROWS')=='1'
LIBRARY='native/router1677/router.so'if ROWS16 else'native/router1002/router.so'
LIB_SHA=('c9b16826bdd0b35561737010c8b9c47b9029a3a622c5d06262a0b04330494f15'if ROWS16 else'1bd40656826f804e6d98e438271026a0737ec1a96a93a3fa20c03010cf9a181f')
PREFIX='router1677'if ROWS16 else'router1002'
MODE=2 if ROWS16 else 0
logger=logging.getLogger(__name__)

class SharedArgs(ctypes.Structure):
    _fields_=[('route',Args)]+[(n,ctypes.c_void_p)for n in ['q','w','sx','ss','gu']]+[('ws0',ctypes.c_int),('ws1',ctypes.c_int)]

@lru_cache(maxsize=1)
def load():
    assert not torch.xpu.is_current_stream_capturing(),'warm router/shared native before capture'
    path=Path('/evidence/tpot-iterations-20261023')/LIBRARY
    assert hashlib.sha256(path.read_bytes()).hexdigest()==LIB_SHA
    lib=ctypes.CDLL(str(path))
    lib.route=getattr(lib,PREFIX+'_run');lib.route.argtypes=[ctypes.c_void_p,ctypes.POINTER(SharedArgs),ctypes.c_int]
    if ROWS16:
        original=Path('/evidence/tpot-iterations-20261023/native/router1002/router.so')
        assert hashlib.sha256(original.read_bytes()).hexdigest()=='1bd40656826f804e6d98e438271026a0737ec1a96a93a3fa20c03010cf9a181f'
        lib._resident_owner=ctypes.CDLL(str(original))
        lib.resident=lib._resident_owner.resident1002_run
        lib.resident_error=lib._resident_owner.router1002_error
    else:
        lib.resident=lib.resident1002_run;lib.resident_error=lib.router1002_error
    lib.resident.argtypes=[ctypes.c_void_p]*9;lib.resident_error.restype=ctypes.c_char_p
    lib.error=getattr(lib,PREFIX+'_error');lib.error.restype=ctypes.c_char_p
    size=getattr(lib,PREFIX+'_args_size');size.restype=ctypes.c_int
    assert size()==ctypes.sizeof(SharedArgs)==264
    return lib

@dataclass
class Receipt:
    key: tuple
    gu: object
    consumed: bool=False

def key(manager,ex,x,ctx,stream):
    q,sx=ctx.projection;shared=ctx.block.shared_expert
    w=shared.gate_up_proj.weight;ss=shared.gate_up_proj.weight_scale_inv
    return (id(manager),id(ex),ex._cpu40_channel,x.data_ptr(),q.data_ptr(),sx.data_ptr(),w.data_ptr(),ss.data_ptr(),tuple(ss.stride()),stream.sycl_queue)

def meta(ex):
    if not hasattr(ex,'_router1014_meta'):
        ex._router1014_meta=dict(route_eager=0,route_capture=0,resident_eager=0,resident_capture=0)
    return ex._router1014_meta

def try_issue(block,x,manager,ex,ctx,args,stream):
    if os.environ.get('CODEX_AR1014_ROUTER_SHARED')!='1':return False
    if not(os.environ.get('CODEX_AR124_JOINT')=='1'and os.environ.get('CODEX_AR124_MODE','1')=='1'and os.environ.get('CODEX_AR548_WORD')=='1'):return False
    from sglang.srt.layers.ar119_cpu_shared import CPUShared
    if not isinstance(ctx,CPUShared)or ctx.issued or ctx.block is not block or ctx.hidden.data_ptr()!=x.data_ptr():return False
    q,sx=ctx.projection;shared=block.shared_expert
    w=shared.gate_up_proj.weight;ss=shared.gate_up_proj.weight_scale_inv
    if not(q.shape==(1,2048)and q.dtype==torch.float8_e4m3fn and q.is_contiguous()
        and sx.shape==(1,16)and sx.dtype==torch.float32 and sx.is_contiguous()
        and w.shape==(1024,2048)and w.dtype==torch.float8_e4m3fn and w.is_contiguous()
        and ss.shape==(8,16)and ss.dtype==torch.float32 and all(0<v<2**31 for v in ss.stride())
        and all(v.device==x.device for v in [q,sx,w,ss])):return False
    assert not hasattr(ctx,'_router1014'),'shared route receipt must be unique'
    lib=load();gu=torch.empty((1,1024),device=x.device,dtype=torch.bfloat16)
    cfg=SharedArgs(args,*[v.data_ptr()for v in [q,w,sx,ss,gu]],*ss.stride())
    rc=lib.route(stream.sycl_queue,ctypes.byref(cfg),MODE)
    if rc:raise RuntimeError((rc,lib.error().decode()))
    for v in [q,w,sx,ss,gu]:v.record_stream(stream)
    ctx._router1014=Receipt(key(manager,ex,x,ctx,stream),gu)
    m=meta(ex);m['route_capture'if torch.xpu.is_current_stream_capturing()else'route_eager']+=1
    if ex._cpu40_channel==0 and not getattr(ex,'_router1014_logged',False):
        logger.info('AR1014 router/shared GU enabled: router64+shared%d Rows%d WG128 GRF128; 64 router arrivals unchanged; resident GU remains separate; library=%s',64 if ROWS16 else 256,16 if ROWS16 else 4,LIBRARY)
        ex._router1014_logged=True
    return True

def consume(manager,ex,x,ctx,a,stream):
    receipt=getattr(ctx,'_router1014',None)
    if receipt is None:return None
    assert not receipt.consumed and receipt.key==key(manager,ex,x,ctx,stream),'stale/mismatched shared route receipt'
    t=manager.layers[ex._cpu40_channel];r=ex.residency
    tensors=[x,t['w1'],t['s1'],ex.mapped_ids,t['map'],r.count,a,manager.stats]
    from sglang.srt.layers.ar1502_pregate import issue_resident
    if not issue_resident(manager,ex,x,ctx,a,stream):
        lib=load();rc=lib.resident(stream.sycl_queue,*[v.data_ptr()for v in tensors])
        if rc:raise RuntimeError((rc,lib.resident_error().decode()))
    for v in tensors+[receipt.gu]:v.record_stream(stream)
    receipt.consumed=True
    m=meta(ex);m['resident_capture'if torch.xpu.is_current_stream_capturing()else'resident_eager']+=1
    return receipt.gu

def readback(manager):
    rows=[]if manager is None else [dict(channel=i,**t['executor']._router1014_meta)for i,t in enumerate(manager.layers)if hasattr(t['executor'],'_router1014_meta')]
    return dict(enabled=os.environ.get('CODEX_AR1014_ROUTER_SHARED')=='1',active_channels=[v['channel']for v in rows],
        library=LIBRARY,library_sha256=LIB_SHA,mode=MODE,router_shared_groups=(128 if ROWS16 else 320),router_arrivals=64,shared_rows_per_group=(16 if ROWS16 else 4),resident_default_groups=2048,
        resident_groups_by_channel=[]if manager is None else[2305 if getattr(t['executor'],'_ar1657_resident_active',False)else(2049 if getattr(t['executor'],'_ar1502_resident_active',False)else 2048) for t in manager.layers],
        router_shared_GRF=128,resident_GRF=128,single_use_receipt=True,CPU_protocol_unchanged=True,
        GPU_census=False,rows=rows)
