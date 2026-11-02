"""CPU-only exact topk/LRU+route-publish boundary fusion, scoped receipt."""
import ctypes
import logging
import os
from functools import lru_cache
import torch
from sglang.srt.layers.moe.paged_experts.fused37 import PlannedTopKOutput
logger=logging.getLogger(__name__)
class PublishedTopKOutput(PlannedTopKOutput):
    """Planning and publication belong to the explicit owner/input/stream."""
@lru_cache(maxsize=1)
def load():
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('Warm AR362 library before capture')
    lib=ctypes.CDLL('/evidence/tpot-iterations-20261023/native/route293/route.so')
    lib.route291_run.argtypes=[ctypes.c_void_p]*25+[ctypes.c_int]*2
    lib.route291_run.restype=ctypes.c_int
    lib.route291_error.restype=ctypes.c_char_p
    return lib

def try_route(executor,hidden,logits,renorm):
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager=cpu40.MANAGER
    if manager is None or not hasattr(executor,'_cpu40_channel'):
        return None
    manager.start()
    channel=executor._cpu40_channel;t=manager.layers[channel]
    assert t['executor']is executor and hidden.dtype==torch.bfloat16 and hidden.shape==(1,2048)
    r=executor.residency;stream=torch.xpu.current_stream(hidden.device)
    weights=torch.empty((1,8),device=logits.device,dtype=torch.float32)
    ids=torch.empty((1,8),device=logits.device,dtype=torch.int32)
    args=[logits,weights,ids,r.step,r.slot_expert,r.expert_slot,r.slot_lastuse,r.src,r.dst,r.count,
          executor.mapped_ids,executor.mapped_weights,executor.expert_hosts,executor.expert_gpus,hidden,t['map'],manager.hin,manager.header,
          manager.request,manager.sequence,manager.stats,manager.issued,manager.refill_done,manager.cpu_status]
    assert len(args)==24
    mode=int(os.environ.get('CODEX_AR362_MODE','3'));assert mode in [2,3]
    lib=load();rc=lib.route291_run(stream.sycl_queue,*[v.data_ptr()for v in args],channel,int(renorm))
    if rc:raise RuntimeError((rc,lib.route291_error().decode()))
    for v in args:
        if v.device.type=='xpu':v.record_stream(stream)
    out=PublishedTopKOutput(weights,ids,logits)
    out.executor=executor;out.publish_owner=manager;out.publish_channel=channel
    out.publish_input_pointer=hidden.data_ptr();out.publish_stream=stream.sycl_queue
    if channel==0 and not getattr(executor,'_ar362_logged',False):
        logger.info('AR362 top8/LRU+CPU publish enabled: GRF256; native mode6 owns four-plane refill; previous-generation fence retained')
        executor._ar362_logged=True
    executor._ar362_published=True
    return out

def was_published(topk,manager,executor,x,stream):
    if not isinstance(topk,PublishedTopKOutput):return False
    assert topk.executor is executor and topk.publish_owner is manager
    assert topk.publish_channel==executor._cpu40_channel
    assert topk.publish_input_pointer==x.data_ptr()and topk.publish_stream==stream.sycl_queue
    return True

def readback():
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager=cpu40.MANAGER
    from sglang.srt.layers.moe.paged_experts.router_shared1014 import readback as shared_readback
    shared=shared_readback(manager)
    return dict(shared_GU_router=shared,enabled=os.environ.get('CODEX_AR362_PUBLISH')=='1',
        route_channels=[] if manager is None else [i for i,t in enumerate(manager.layers)
            if getattr(t['executor'],'_ar362_published',False)],
        library=(shared['library'] if shared['active_channels'] else ('native/router807/router.so' if os.environ.get('CODEX_AR812_ROUTER')=='1' else 'native/route293/route.so')),
        router_fused=os.environ.get('CODEX_AR812_ROUTER')=='1',
        router_channels=[] if manager is None else [i for i,t in enumerate(manager.layers) if getattr(t['executor'],'_router812_logged',False)],
        router_counters=[] if manager is None else [int(t['executor']._router812_counter.cpu()[0]) for t in manager.layers if hasattr(t['executor'],'_router812_counter')],
        receipt='executor/manager/channel/input-pointer/queue',GPU_census=False)
