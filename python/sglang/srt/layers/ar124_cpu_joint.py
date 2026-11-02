"""CPU-compatible shared/resident GU fusion; explicit call context, no GPU fill."""
import ctypes
import logging
import os
from functools import lru_cache
import torch

WORD_ENABLED=(os.environ.get('CODEX_AR548_WORD')=='1'
    and os.environ.get('CODEX_AR548_WORD_GU','1')=='1'
    and os.environ.get('CODEX_AR124_MODE','1')=='1')
LIB=('/evidence/tpot-iterations-20261023/native/decode540/joint.so' if WORD_ENABLED
     else '/evidence/tpot-iterations-20261023/native/cpu-joint124/joint.so')

@lru_cache(maxsize=1)
def resources():
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('AR124 library must warm before capture')
    lib=ctypes.CDLL(LIB)
    if WORD_ENABLED:
        lib.joint124_run=lib.joint540_run
        lib.joint124_error=lib.joint540_error
        logging.getLogger(__name__).info('AR548 word GU enabled: joint540; original guards and mode1 reduction')
    lib.joint124_error.restype=ctypes.c_char_p
    lib.joint124_run.argtypes=[ctypes.c_void_p]*14+[ctypes.c_int]*3
    return lib

def try_run(manager,executor,layer,dispatch,ctx,a):
    if os.environ.get('CODEX_AR124_JOINT')!='1' or ctx is None:
        return False
    from sglang.srt.layers.ar119_cpu_shared import CPUShared
    if not isinstance(ctx,CPUShared):
        return False
    assert not ctx.issued, 'AR124 context must issue exactly once'
    assert os.environ.get('CODEX_AR_CPU40')=='1'
    x=dispatch.hidden_states;q,sx=ctx.projection;shared=ctx.block.shared_expert
    w=shared.gate_up_proj.weight;ss=shared.gate_up_proj.weight_scale_inv
    t=manager.layers[executor._cpu40_channel];r=executor.residency
    tensors=[x,t['w1'],t['s1'],executor.mapped_ids,t['map'],r.count,a,manager.stats,q,w,sx,ss]
    if not(x.shape==(1,2048) and x.dtype==torch.bfloat16 and x.is_contiguous()
           and q.shape==x.shape and q.dtype==torch.float8_e4m3fn and q.is_contiguous()
           and sx.shape==(1,16) and sx.dtype==torch.float32 and sx.is_contiguous()
           and w.shape==(1024,2048) and w.dtype==torch.float8_e4m3fn and w.is_contiguous()
           and ss.shape==(8,16) and ss.dtype==torch.float32 and all(0<v<2**31 for v in ss.stride())
           and t['w1'].shape==(192,1024,2048) and t['w1'].is_contiguous()
           and t['s1'].shape==(192,8,16) and t['s1'].dtype==torch.float32 and t['s1'].is_contiguous()
           and a.shape==(8,512) and a.dtype==torch.bfloat16 and a.is_contiguous()
           and all(v.device==x.device for v in tensors)
           and torch.xpu.get_device_properties(x.device).device_id==0xe223):
        return False
    mode=int(os.environ.get('CODEX_AR124_MODE','1'));assert 0<=mode<=5
    stream=torch.xpu.current_stream(x.device)
    from sglang.srt.layers.moe.paged_experts.router_shared1014 import consume
    gu=consume(manager,executor,x,ctx,a,stream)
    shared_moved=gu is not None
    if not shared_moved:
        lib=resources();gu=torch.empty((1,1024),dtype=torch.bfloat16,device=x.device)
        tensors.append(gu)
        rc=lib.joint124_run(stream.sycl_queue,*[v.data_ptr()for v in tensors],*ss.stride(),mode)
        if rc:raise RuntimeError((rc,lib.joint124_error().decode()))
    else:
        tensors.append(gu)
    for v in tensors:v.record_stream(stream)
    from sglang.srt.layers.ar1657_shared_union import consume as consume_shared
    early=consume_shared(ctx,gu)
    ctx.output=shared.forward_from_gate_up(gu)if early is None else early
    if ctx.apply_gate:ctx.output=ctx.block._gate_shared_output_out_of_place(x,ctx.output)
    ctx.issued=True
    if t['index']==0 and not getattr(executor,'_ar124_logged',False):
        if shared_moved:logging.getLogger(__name__).info('AR124 CPU jointGU enabled mode=%d; shared GU supplied by router; separate resident GRF128; CPU miss masked; no GPU fill',mode)
        else:logging.getLogger(__name__).info('AR124 CPU jointGU enabled mode=%d; sharedFirstGRF128 mode1; CPU miss masked; no GPU fill',mode)
        executor._ar124_logged=True
    return True
