"""B70 K192 AR router+Top8/LRU+CPU publication research integration."""
import ctypes
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
import torch
from sglang.srt.layers.moe.paged_experts.cpu_publish362 import PublishedTopKOutput
logger=logging.getLogger(__name__)
class Args(ctypes.Structure):
    _fields_=[(n,ctypes.c_void_p)for n in ['logits','weights','ids','step','slots','inverse','age','src','dst','count','mapped','mapped_weights','hosts','gpus','hidden','map','hin','header','request','sequence','stats','issued','done','status','router','counter']]+[('channel',ctypes.c_int),('renorm',ctypes.c_int)]
assert ctypes.sizeof(Args)==216
@lru_cache(maxsize=1)
def load():
    assert not torch.xpu.is_current_stream_capturing(), 'warm router812 before capture'
    path=Path('/evidence/tpot-iterations-20261023/native/router807/router.so')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='46cda7b884361e56e41ff50e04048ef99d898e5aa41e18978b44ca58171b77a8'
    lib=ctypes.CDLL(str(path));lib.router807_run.argtypes=[ctypes.c_void_p,ctypes.POINTER(Args),ctypes.c_int];lib.router807_run.restype=ctypes.c_int;lib.router807_error.restype=ctypes.c_char_p
    return lib

def try_router(block,hidden,defer_finalize,ar_joint=None):
    if os.environ.get('CODEX_AR812_ROUTER')!='1' or defer_finalize:return None
    if not all(os.environ.get(k)=='1'for k in ['CODEX_AR_CPU40','CODEX_AR_FUSED37','CODEX_AR_CPU118_FUSED','CODEX_AR362_PUBLISH']):return None
    from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod
    from sglang.srt.layers.moe.paged_experts.xpu import XpuDeviceExecutor
    from sglang.srt.layers.moe.paged_experts import cpu40
    from sglang.srt.runtime_context import get_spec
    method=block.experts.quant_method
    if not isinstance(method,PagedExpertsMoEMethod):return None
    executor=method.device_executor;config=block.topk.topk_config;rw=block.gate.weight;manager=cpu40.MANAGER
    if not(isinstance(executor,XpuDeviceExecutor)and executor.use_scaled_fp8
        and method.num_resident==192 and method.num_experts==256
        and not block.enable_shared_expert_fusion and block.tp_size==block.experts.moe_ep_size==1
        and get_spec().speculative_algorithm is None
        and hidden.shape==(1,2048)and hidden.dtype==torch.bfloat16 and hidden.is_contiguous()
        and rw.shape==(256,2048)and rw.dtype==torch.bfloat16 and rw.is_contiguous()and rw.device==hidden.device
        and getattr(block.gate,'bias',None)is None and not block.gate._forward_hooks and not block.gate._forward_pre_hooks
        and hidden.device.type=='xpu'and torch.xpu.get_device_properties(hidden.device).device_id==0xe223
        and config.top_k==8 and config.scoring_func=='softmax'and not config.use_grouped_topk
        and config.custom_routing_function is None and config.correction_bias is None
        and config.num_shared_topk_slots==0 and not config.apply_routed_scaling_factor_on_output
        and manager is not None and hasattr(executor,'_cpu40_channel')):return None
    manager.start();channel=executor._cpu40_channel;t=manager.layers[channel];assert t['executor']is executor
    lib=load();r=executor.residency;stream=torch.xpu.current_stream(hidden.device)
    if not hasattr(executor,'_router812_counter'):
        assert not torch.xpu.is_current_stream_capturing()
        executor._router812_counter=torch.zeros(1,device=hidden.device,dtype=torch.int32)
    counter=executor._router812_counter
    logits=torch.empty((1,256),device=hidden.device,dtype=torch.bfloat16)
    weights=torch.empty((1,8),device=hidden.device,dtype=torch.float32);ids=torch.empty((1,8),device=hidden.device,dtype=torch.int32)
    tensors=[logits,weights,ids,r.step,r.slot_expert,r.expert_slot,r.slot_lastuse,r.src,r.dst,r.count,executor.mapped_ids,executor.mapped_weights,
        executor.expert_hosts,executor.expert_gpus,hidden,t['map'],manager.hin,manager.header,manager.request,manager.sequence,manager.stats,
        manager.issued,manager.refill_done,manager.cpu_status,rw,counter]
    args=Args(*[v.data_ptr()for v in tensors],channel,int(config.renormalize))
    from sglang.srt.layers.moe.paged_experts.router_shared1014 import try_issue
    shared_moved=try_issue(block,hidden,manager,executor,ar_joint,args,stream)
    if not shared_moved:
        rc=lib.router807_run(stream.sycl_queue,ctypes.byref(args),1)
        if rc:raise RuntimeError((rc,lib.router807_error().decode()))
    for v in tensors:
        if v.device.type=='xpu':v.record_stream(stream)
    out=PublishedTopKOutput(weights,ids,logits);out.executor=executor;out.publish_owner=manager;out.publish_channel=channel
    out.publish_input_pointer=hidden.data_ptr();out.publish_stream=stream.sycl_queue
    executor._ar362_published=True
    if not getattr(executor,'_router812_logged',False):
        if channel==0:
            if shared_moved:logger.info('AR812 router/Top8/LRU/CPU publish enabled: shared GU moved into router grid; original bounded host protocol')
            else:logger.info('AR812 router/Top8/LRU/CPU publish enabled: 64 WG128 SG32 GRF256; final-WG acquire-release election; original bounded host protocol')
        executor._router812_logged=True
    return out
