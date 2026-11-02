"""Opt-in AR shared-down K512 projection; same FP16 operands and F32 tree."""
import ctypes,hashlib,logging,os
from functools import lru_cache
from pathlib import Path
import torch
ENABLED=os.environ.get('CODEX_AR728_SHARED')=='1'
LIBRARY='/evidence/tpot-iterations-20261023/native/small722/small.so'
EXPECTED_SHA='0926f82077fdd806252f486cd6433e8f5f6946386fe5fefcb8ed8f2ae68d942e'
EAGER_CALLS=CAPTURE_CALLS=0
logger=logging.getLogger(__name__)

def can_use(q,w,xs,ws):
    if not ENABLED or torch.is_grad_enabled() or torch.compiler.is_compiling():return False
    ts=[q,w,xs,ws]
    if not all(isinstance(v,torch.Tensor)for v in ts):return False
    if not(q.device.type=='xpu' and all(v.device==q.device for v in ts)
        and q.dtype==w.dtype==torch.float8_e4m3fn and xs.dtype==ws.dtype==torch.float32
        and tuple(q.shape)==(1,512) and tuple(w.shape)==(2048,512)
        and tuple(xs.shape)==(1,4) and tuple(ws.shape)==(16,4)
        and all(v.is_contiguous()for v in [q,w,xs]) and ws.stride()in [(4,1),(1,16)]
        and q.data_ptr()%16==w.data_ptr()%16==0 and xs.data_ptr()%4==ws.data_ptr()%4==0):return False
    from sglang.srt.runtime_context import get_parallel,get_spec,get_exec
    if get_parallel().tp_size!=1 or get_spec().speculative_algorithm is not None or not get_exec().moe.enable_paged_experts:return False
    if torch.xpu.get_device_properties(q.device).device_id!=0xe223:return False
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager=cpu40.MANAGER
    return manager is not None and len(manager.layers)==40 and all(v['executor'].residency.slot_expert.numel()==192 for v in manager.layers)

@lru_cache(maxsize=1)
def library():
    if torch.xpu.is_current_stream_capturing():raise RuntimeError('Warm AR671 output before capture')
    assert hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest()==EXPECTED_SHA
    lib=ctypes.CDLL(LIBRARY);lib.small722_run.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*5
    lib.small722_run.restype=ctypes.c_int;lib.small722_error.restype=ctypes.c_char_p
    logger.info('AR728 shared down SYCL enabled: N2048/K512/R2/WARP1/GRF256; packed-word FP8 decoder')
    return lib

def try_run(q,w,xs,ws):
    global EAGER_CALLS,CAPTURE_CALLS
    if not can_use(q,w,xs,ws):return None
    y=torch.empty((1,2048),device=q.device,dtype=torch.bfloat16)
    lib=library();stream=torch.xpu.current_stream(q.device)
    rc=lib.small722_run(stream.sycl_queue,q.data_ptr(),w.data_ptr(),xs.data_ptr(),ws.data_ptr(),y.data_ptr(),2048,512,*ws.stride(),1)
    if rc:raise RuntimeError((rc,lib.small722_error().decode()))
    for v in [q,w,xs,ws,y]:v.record_stream(stream)
    if torch.xpu.is_current_stream_capturing():CAPTURE_CALLS+=1
    else:EAGER_CALLS+=1
    return y

def readback():
    return dict(enabled=ENABLED,capture_calls=CAPTURE_CALLS,eager_calls=EAGER_CALLS,
                library='native/small722/small.so',library_sha256=EXPECTED_SHA,
                mode=1,rows=2,warps=1,GRF=256,shape=[2048,512],pure_AR=True,GPU_census=False)
