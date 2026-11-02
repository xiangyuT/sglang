import ctypes
from pathlib import Path
import torch
R=Path('/evidence/tpot-iterations-20261023')
class Argmax:
    def __init__(self):
        if torch.xpu.is_current_stream_capturing():raise RuntimeError('Warm argmax254 library before capture')
        self.lib=ctypes.CDLL(str(R/'native/argmax254/argmax.so'))
        self.lib.argmax254_run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]
        self.lib.argmax254_error.restype=ctypes.c_char_p
    def run(self,x,values=None,indices=None,out=None):
        n=x.numel();assert x.dtype==torch.float32 and x.device.type=='xpu' and x.is_contiguous() and 0<n<=1048576
        parts=(n+1023)//1024
        if values is None:values=torch.empty(parts,device=x.device,dtype=torch.float32)
        if indices is None:indices=torch.empty(parts,device=x.device,dtype=torch.int32)
        if out is None:out=torch.empty(1,device=x.device,dtype=torch.int64)
        stream=torch.xpu.current_stream(x.device)
        rc=self.lib.argmax254_run(stream.sycl_queue,x.data_ptr(),values.data_ptr(),indices.data_ptr(),out.data_ptr(),n)
        if rc:raise RuntimeError((rc,self.lib.argmax254_error().decode()))
        for t in [x,values,indices,out]:t.record_stream(stream)
        return out

# Native operation after the Sampler's standard logits preprocessing.
import logging,os
from functools import lru_cache
ENABLED=os.environ.get('CODEX_AR266_ARGMAX')=='1'
CALLS=0
HOST_TOKEN=os.environ.get('CODEX_AR919_HOST_TOKEN')=='1'
@lru_cache(maxsize=1)
def resource():
    if HOST_TOKEN:
        from sglang.srt.layers.ar919_host_token import HostArgmax
        op=HostArgmax()
        logging.getLogger(__name__).info('AR919 host token enabled: pool64/slot64B; system release/acquire; result-owned leases; eager sampler')
    else:op=Argmax()
    logging.getLogger(__name__).info('AR266 exact SYCL argmax enabled: F32 B1/vocab248320; firstindex/NaN semantics; after original preprocessing')
    return op
def try_run(logits):
    global CALLS
    if not ENABLED or torch.is_grad_enabled()or torch.compiler.is_compiling():return None
    if not(logits.device.type=='xpu'and logits.dtype==torch.float32 and tuple(logits.shape)==(1,248320)and logits.is_contiguous()):return None
    if torch.xpu.get_device_properties(logits.device).device_id!=0xe223:return None
    if HOST_TOKEN and torch.xpu.is_current_stream_capturing():return None
    value=resource().run(logits);CALLS+=1;return value
def readback():
    return dict(enabled=ENABLED,calls=CALLS,library='native/token911/token.so'if HOST_TOKEN else'native/argmax254/argmax.so',GPU_census=False,
                host_token=(__import__('sglang.srt.layers.ar919_host_token',fromlist=['readback']).readback(resource())if HOST_TOKEN else dict(enabled=False)),
                location='Sampler allgreedy after original preprocessing',input='contiguousF32 B1/vocab248320',indices='int64 firstindex withTorchNaN/tie semantics')
