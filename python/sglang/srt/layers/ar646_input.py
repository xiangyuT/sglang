"""Opt-in pure AR B70 GDN input projection; native input639 row16 only."""
import ctypes
from functools import lru_cache
import hashlib,logging,os
from pathlib import Path
import torch
ENABLED=os.environ.get('CODEX_AR646_INPUT')=='1'
LIBRARY='/evidence/tpot-iterations-20261023/native/input639/input.so'
EXPECTED_SHA='1b83ad1f45b8152934d2231df31e5a4d9130c55bd96d3e7a220c115fc2f65c42'
EAGER_CALLS=CAPTURE_CALLS=0
logger=logging.getLogger(__name__)

def can_use(projection,hidden,batch):
    if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling():return False
    if batch is None or batch.batch_size!=1 or not batch.forward_mode.is_decode()or getattr(batch,'spec_info',None)is not None:return False
    if not isinstance(hidden,tuple)or len(hidden)!=2:return False
    q,xs=hidden;w=getattr(projection,'weight',None);ws=getattr(projection,'weight_scale_inv',None)
    ts=[q,xs,w,ws]
    if not all(isinstance(v,torch.Tensor)for v in ts):return False
    if not(q.device.type=='xpu'and all(v.device==q.device for v in ts)
        and q.dtype==w.dtype==torch.float8_e4m3fn and xs.dtype==ws.dtype==torch.float32
        and tuple(q.shape)==(1,2048)and tuple(xs.shape)==(1,16)
        and tuple(w.shape)==(12288,2048)and tuple(ws.shape)==(96,16)
        and q.is_contiguous()and xs.is_contiguous()and w.is_contiguous()
        and ws.stride()in [(16,1),(1,96)]
        and q.data_ptr()%16==w.data_ptr()%16==0 and xs.data_ptr()%4==ws.data_ptr()%4==0
        and getattr(projection,'bias',None)is None):return False
    method=getattr(projection,'quant_method',None)
    if not(type(method).__name__=='Fp8LinearMethod'and getattr(method,'block_quant',False)
        and tuple(getattr(method,'weight_block_size',()))==(128,128)
        and not any(getattr(method,k,False)for k in ['use_mxfp8','block_fp8_as_mxfp8','use_marlin'])):return False
    from sglang.srt.runtime_context import get_parallel,get_spec,get_exec
    if get_parallel().tp_size!=1 or get_spec().speculative_algorithm is not None or not get_exec().moe.enable_paged_experts:return False
    if torch.xpu.get_device_properties(q.device).device_id!=0xe223:return False
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager=cpu40.MANAGER
    return(manager is not None and len(manager.layers)==40
        and all(v['executor'].residency.slot_expert.numel()==192 for v in manager.layers))

@lru_cache(maxsize=1)
def library():
    if torch.xpu.is_current_stream_capturing():raise RuntimeError('Warm AR646 input before capture')
    assert hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest()==EXPECTED_SHA
    lib=ctypes.CDLL(LIBRARY);lib.input639_run.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*4
    lib.input639_run.restype=ctypes.c_int;lib.input639_error.restype=ctypes.c_char_p
    logger.info('AR646 pure-AR GDN input SYCL enabled: N12288/K2048/R16/WARP4/GRF256; FP16 dequant, BF16 output')
    return lib

def try_run(projection,hidden,batch):
    global EAGER_CALLS,CAPTURE_CALLS
    if not can_use(projection,hidden,batch):return None
    q,xs=hidden;w,ws=projection.weight,projection.weight_scale_inv
    y=torch.empty((1,12288),device=q.device,dtype=torch.bfloat16)
    lib=library();stream=torch.xpu.current_stream(q.device)
    rc=lib.input639_run(stream.sycl_queue,q.data_ptr(),w.data_ptr(),xs.data_ptr(),ws.data_ptr(),y.data_ptr(),12288,*ws.stride(),2)
    if rc:raise RuntimeError((rc,lib.input639_error().decode()))
    for v in [q,w,xs,ws,y]:v.record_stream(stream)
    if torch.xpu.is_current_stream_capturing():CAPTURE_CALLS+=1
    else:EAGER_CALLS+=1
    return y

def readback():
    return dict(enabled=ENABLED,capture_calls=CAPTURE_CALLS,eager_calls=EAGER_CALLS,
        library='native/input639/input.so',library_sha256=EXPECTED_SHA,mode=2,rows=16,
        shape=[12288,2048],attention_QKV_selected=False,pure_AR=True,GPU_census=False)
