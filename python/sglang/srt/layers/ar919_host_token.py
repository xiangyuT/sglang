import ctypes,gc,torch
from pathlib import Path
R=Path('/evidence/tpot-iterations-20261023')
class Pool:
 def __init__(self,n=64):
  self.lib=ctypes.CDLL(str(R/'native/token911/token.so'))
  self.lib.token908_error.restype=ctypes.c_char_p
  specs={'token908_run':([ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p,ctypes.c_int],ctypes.c_int),'token908_pool_create':([ctypes.c_void_p,ctypes.c_int],ctypes.c_void_p),'token908_pool_claim':([ctypes.c_void_p],ctypes.c_int),'token908_pool_address':([ctypes.c_void_p,ctypes.c_int],ctypes.c_void_p),'token908_pool_release':([ctypes.c_void_p,ctypes.c_int],ctypes.c_int),'token908_pool_wait':([ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64],ctypes.c_int),'token908_pool_stats':([ctypes.c_void_p,ctypes.c_void_p],ctypes.c_int),'token908_pool_close':([ctypes.c_void_p],ctypes.c_int)}
  for name,(args,ret)in specs.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=ret
  self.handle=self.lib.token908_pool_create(torch.xpu.current_stream().sycl_queue,n)
  if not self.handle:raise RuntimeError(self.error())
 def error(self):return self.lib.token908_error().decode()
 def claim(self):
  idx=self.lib.token908_pool_claim(self.handle)
  if idx<0:raise RuntimeError(('no free host-token lease',idx,self.error()))
  return Receipt(self,idx,self.lib.token908_pool_address(self.handle,idx))
 def stats(self):
  a=(ctypes.c_uint64*4)();assert self.lib.token908_pool_stats(self.handle,ctypes.addressof(a))==0;return list(a)
 def close(self):
  gc.collect();rc=self.lib.token908_pool_close(self.handle)
  if rc:raise RuntimeError(('hosttoken pool close',rc,self.stats()))
  self.handle=None
class Receipt:
 def __init__(self,pool,index,address):self.pool,self.index,self.address=pool,index,address;self.released=False
 def wait(self,timeout_us=5000000):
  rc=self.pool.lib.token908_pool_wait(self.pool.handle,self.index,timeout_us)
  if rc:raise RuntimeError(('hosttoken readiness',rc))
 def value(self):self.wait();return ctypes.c_int64.from_address(self.address).value
 def tensor(self):
  self.wait();raw=(ctypes.c_int64*1).from_address(self.address)
  # frombuffer's storage keeps its buffer object alive across detach/views/
  # NumPy aliases. The lease belongs to that buffer, not only a Tensor attr.
  raw._codex_token908=self
  t=torch.frombuffer(raw,dtype=torch.int64);t._codex_token908=self;return t
 def release(self):
  if not self.released:
   rc=self.pool.lib.token908_pool_release(self.pool.handle,self.index)
   if rc:raise RuntimeError(('hosttoken lease release',rc))
   self.released=True
 def __del__(self):
  try:self.release()
  except Exception:pass
class HostArgmax:
 def __init__(self,pool=None):self.pool=pool or Pool()
 def run(self,x,values=None,indices=None,out=None):
  n=x.numel();parts=(n+1023)//1024
  assert x.device.type=='xpu'and x.dtype==torch.float32 and x.is_contiguous()and 0<n<=1048576
  if values is None:values=torch.empty(parts,device='xpu',dtype=torch.float32)
  if indices is None:indices=torch.empty(parts,device='xpu',dtype=torch.int32)
  if out is None:out=torch.empty(1,device='xpu',dtype=torch.int64)
  lease=self.pool.claim();stream=torch.xpu.current_stream()
  rc=self.pool.lib.token908_run(stream.sycl_queue,x.data_ptr(),values.data_ptr(),indices.data_ptr(),out.data_ptr(),n,self.pool.handle,lease.index)
  if rc:lease.release();raise RuntimeError((rc,self.pool.error()))
  for t in[x,values,indices,out]:t.record_stream(stream)
  out._codex_token908=lease
  return out

# Result-owned CPU views retain the native lease. The pool is fixed4KiB per
# scheduler process; a cancelled in-flight lease is retired until GPU-ready.
COPIES=0

def submit_if_eligible(worker,result,copy_fn):
    receipt=getattr(result.next_token_ids,"_codex_token908",None)
    if not isinstance(receipt,Receipt):return None
    for name in ["accept_lens","block_accept_lens","cap_lens","routed_experts_output","indexer_topk_output","expert_distribution_metrics"]:
        if getattr(result,name,None)is not None:return None
    logits=getattr(result,"logits_output",None)
    if logits is not None and any(getattr(logits,name,None)is not None for name in ["sampling_mask_output","auxiliary_device_output"]):return None
    from sglang.srt.managers.async_d2h_copy_worker import HostCopyDone
    # Install copy_done BEFORE queue publication. A ready GPU token allows an
    # immediate callback; it must never see an old CUDA event or None.
    done=HostCopyDone();result.copy_done=done
    def copy_ready_token():
        global COPIES
        result.next_token_ids=receipt.tensor()
        copy_fn()  # common.async_d2h returns an already-CPU tensor unchanged.
        COPIES+=1
    return worker.submit_host_token(copy_ready_token,receipt,done=done)

def readback(op):
    return dict(enabled=True,library="native/token911/token.so",pool_capacity=64,slot_bytes=64,
                system_release_GPU_ready=True,CPU_acquire=True,cancelled_lease_retired=True,
                eager_sampler_only=True,worker_copies=COPIES,pool=op.pool.stats(),
                unsupported_auxiliary_falls_back=True)
