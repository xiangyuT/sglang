# SPDX-License-Identifier: Apache-2.0
"""Caller-owned buffers for the ported Strata INT8 expert-prefill chain."""
import ctypes,os
from functools import lru_cache
import torch

@lru_cache(None)
def library():
    lib=ctypes.CDLL(os.environ['SGLANG_XPU_GGUF_PREFILL_I8_LIB'])
    lib.xpu_gguf_prefill_i8.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*5+[ctypes.c_void_p]*6
    lib.xpu_gguf_prefill_i8.restype=ctypes.c_int
    lib.xpu_gguf_prefill_i8_error.restype=ctypes.c_char_p
    return lib

def grouped_i8(x,gu_weight,down_weight,counts,*,return_gu=False):
    E=len(counts);H=x.shape[1];I=640;R=x.shape[0];P=sum((int(n)+7)//8*8 for n in counts)
    if not (1<=E<=16 and all(0<=int(n)<=32768 for n in counts)
            and x.device.type=='xpu' and x.dtype==torch.float16 and x.ndim==2
            and x.is_contiguous() and H==2560 and sum(counts)==R
            and gu_weight.device==down_weight.device==x.device
            and gu_weight.dtype==down_weight.dtype==torch.uint8
            and gu_weight.is_contiguous() and down_weight.is_contiguous()
            and torch.xpu.get_device_properties(x.device).device_id==0xE223):
        raise ValueError('Unsupported B70 INT8 prefill group')
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError('INT8 expert prefill requires eager execution')
    if gu_weight.shape!=(E,2*704000) or down_weight.shape!=(E,921600):
        raise ValueError('INT8 prefill requires original IQ3_S / IQ4_NL expert bytes')
    Xq=torch.empty(P*H,dtype=torch.int8,device=x.device);dx=torch.empty(P*H//32,dtype=torch.float32,device=x.device)
    Hq=torch.empty(P*I,dtype=torch.int8,device=x.device);dh=torch.empty(P*I//32,dtype=torch.float32,device=x.device)
    GU=torch.empty((R,2*I),dtype=torch.float32,device=x.device);out=torch.empty((R,H),dtype=torch.float32,device=x.device)
    cpu=(ctypes.c_int32*E)(*[int(n)for n in counts]);lib=library();stream=torch.xpu.current_stream(x.device)
    rc=lib.xpu_gguf_prefill_i8(stream.sycl_queue,x.data_ptr(),gu_weight.data_ptr(),down_weight.data_ptr(),ctypes.addressof(cpu),E,H,I,704000,921600,*[v.data_ptr()for v in [Xq,dx,Hq,dh,GU,out]])
    if rc:raise RuntimeError((rc,lib.xpu_gguf_prefill_i8_error().decode()))
    for v in [x,gu_weight,down_weight,Xq,dx,Hq,dh,GU,out]:v.record_stream(stream)
    return (out,GU)if return_gu else out
