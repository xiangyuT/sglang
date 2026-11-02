# SPDX-License-Identifier: Apache-2.0
"""One native submission per IQ3_S/IQ4_NL prefill layer, with bounded arenas."""
import ctypes
from functools import lru_cache
import os
import numpy as np
import torch
from sglang.kernels.ops.quantization.gguf_sycl import scatter_expert_routes

_workspaces={}
@lru_cache(None)
def library(path):
    lib=ctypes.CDLL(path)
    lib.xpu_gguf_prefill_i8_layer.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*6+[ctypes.c_void_p]*9
    lib.xpu_gguf_prefill_i8_layer.restype=ctypes.c_int
    if hasattr(lib,'xpu_gguf_prefill_i8_layer_dma'):
        lib.xpu_gguf_prefill_i8_layer_dma.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*6+[ctypes.c_void_p]*9
        lib.xpu_gguf_prefill_i8_layer_dma.restype=ctypes.c_int
    if hasattr(lib,'xpu_gguf_prefill_i8_layer_dma_packed'):
        lib.xpu_gguf_prefill_i8_layer_dma_packed.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*6+[ctypes.c_void_p]*9
        lib.xpu_gguf_prefill_i8_layer_dma_packed.restype=ctypes.c_int
    lib.xpu_gguf_prefill_i8_error.restype=ctypes.c_char_p
    return lib

def apply_layer(method,x,weights,ids_cpu,experts,*,output_dtype=None):
    cache=method._gpu_cache;T,H=x.shape;K=ids_cpu.shape[1]
    if not (x.device.type=='xpu' and x.dtype==torch.bfloat16 and x.is_contiguous()
            and T>4 and H==2560 and K<=10 and ids_cpu.shape==tuple(weights.shape)
            and cache.kind==21 and cache.down_kind==20 and tuple(cache.sizes)==(704000,704000,921600)
            and torch.xpu.get_device_properties(x.device).device_id==0xe223):
        raise ValueError('Unsupported B70 native IQ3_S/IQ4_NL prefill layer')
    if torch.xpu.is_current_stream_capturing() or cache._closed:
        raise RuntimeError('Native expert prefill requires an open eager cache')
    zero=getattr(method,'_paged_zero_expert',getattr(method,'_ep_zero_slot',None))
    selected_experts=[int(e)for e in experts if e!=zero]
    reduction_dtype=x.dtype if output_dtype is None else output_dtype
    fused_reduce=(os.environ.get('SGLANG_XPU_GGUF_PREFILL_FUSED_REDUCE')=='1'
                  and 5<=T<=4096 and 1<=K<=10
                  and reduction_dtype in (torch.bfloat16,torch.float32))
    parts=None if fused_reduce else (torch.zeros((ids_cpu.size,H),dtype=torch.float32,device=x.device)if zero is not None else torch.empty((ids_cpu.size,H),dtype=torch.float32,device=x.device))
    if not selected_experts:
        return torch.zeros((T,H),dtype=reduction_dtype,device=x.device)
    flat=ids_cpu.reshape(-1)
    resident_first=os.environ.get('SGLANG_XPU_GGUF_PREFILL_RESIDENT_FIRST')=='1'
    if resident_first:
        locations=cache.expert_slots.cpu().numpy()
        selected_experts.sort(key=lambda e:(locations[e]<0,e))
        ranks=np.full(method.num_experts,len(selected_experts),dtype=np.int32)
        ranks[selected_experts]=np.arange(len(selected_experts),dtype=np.int32)
        order=np.argsort(ranks[flat],kind='stable')
    else:
        order=np.argsort(flat,kind='stable')
    if zero is not None:order=order[flat[order]!=zero]
    counts=np.bincount(flat,minlength=method.num_experts)[selected_experts].astype(np.int32)
    caprows=max(int(counts[i:i+16].sum())for i in range(0,len(counts),16))
    cappadded=max(int(((counts[i:i+16]+7)//8*8).sum())for i in range(0,len(counts),16))
    banks=2 if os.environ.get('SGLANG_XPU_GGUF_PREFILL_COPY_ENGINE')=='1' else 1
    packed=(banks==2 and os.environ.get('SGLANG_XPU_GGUF_PREFILL_HOST_PACK')=='1')
    stream=torch.xpu.current_stream(x.device);key=(x.device,stream.sycl_queue,banks,packed)
    state=_workspaces.get(key)
    if state is None or state[0]<caprows or state[1]<cappadded:
        if state is not None:caprows=max(caprows,state[0]);cappadded=max(cappadded,state[1])
        state=(caprows,cappadded,
               torch.empty((1,)if packed else(16*banks,1408000),dtype=torch.uint8,device=x.device),
               torch.empty((1,)if packed else(16*banks,921600),dtype=torch.uint8,device=x.device),
               torch.empty((caprows,2560),dtype=torch.float16,device=x.device),
               torch.empty(cappadded*2560,dtype=torch.int8,device=x.device),
               torch.empty(cappadded*80,dtype=torch.float32,device=x.device),
               torch.empty(cappadded*640,dtype=torch.int8,device=x.device),
               torch.empty(cappadded*20,dtype=torch.float32,device=x.device),
               torch.empty((caprows,1280),dtype=torch.float32,device=x.device))
        _workspaces[key]=state
    gpu_ids=torch.tensor(selected_experts,dtype=torch.int32,device=x.device)
    if fused_reduce:
        from sglang.kernels.ops.quantization.gguf_prefill_reduce import route_metadata
        gpu_order,gpu_inverse=route_metadata(order,ids_cpu.size,x.device)
    else:
        gpu_order=torch.as_tensor(order.astype(np.int32),device=x.device)
    cpu_counts=(ctypes.c_int32*len(counts))(*[int(n)for n in counts])
    sorted_out=torch.empty((len(order),H),dtype=torch.float32,device=x.device)
    ptrs=[x,gpu_ids,cache.expert_slots,cache.slot_addresses,cache.mirror_addresses,gpu_order]
    lib=library(os.environ['SGLANG_XPU_GGUF_PREFILL_I8_LIB'])
    if getattr(cache,'down_soa',False) and not hasattr(lib,'xpu_gguf_prefill_i8_layer_soa_supported'):
        raise ValueError('Rebuild the native INT8 prefill library for IQ4 down field order')
    if getattr(cache,'gu_soa',False) and not hasattr(lib,'xpu_gguf_prefill_i8_layer_iq3_soa_supported'):
        raise ValueError('Rebuild native INT8 prefill for IQ3 gate/up field order')
    function=getattr(lib,'xpu_gguf_prefill_i8_layer_dma_packed' if packed else 'xpu_gguf_prefill_i8_layer_dma' if banks==2 else 'xpu_gguf_prefill_i8_layer')
    status=function(stream.sycl_queue,*[v.data_ptr()for v in ptrs],ctypes.addressof(cpu_counts),len(counts),method.num_experts,K,T,state[0],state[1],*[v.data_ptr()for v in state[2:]],sorted_out.data_ptr())
    if status:raise RuntimeError((status,lib.xpu_gguf_prefill_i8_error().decode()))
    for value in ptrs+list(state[2:])+[sorted_out]:value.record_stream(stream)
    for chunk in cache.resident_chunks:
        for value in chunk:value.record_stream(stream)
    # GPUExpertCache retains CPU mirrors in _live_states until close synchronizes.
    if fused_reduce:
        from sglang.kernels.ops.quantization.gguf_prefill_reduce import reduce_routes
        return reduce_routes(sorted_out,gpu_inverse,weights,reduction_dtype)
    scatter_expert_routes(sorted_out,gpu_order,parts)
    return (parts.view(T,K,H)*weights.float().unsqueeze(-1)).sum(dim=1).to(x.dtype if output_dtype is None else output_dtype)
