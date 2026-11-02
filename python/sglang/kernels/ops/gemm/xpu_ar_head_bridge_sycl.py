"""Component binding for the dedicated head channel; strong owned buffers."""
from pathlib import Path
import ctypes,torch
R=Path('/evidence/tpot-iterations-20261023')
def library():
 lib=ctypes.CDLL(str(R/'native/head-bridge258/bridge.so'))
 lib.head_bridge258_error.restype=ctypes.c_char_p
 for name,nptr,nint,restype in [('service_create',9,3,ctypes.c_void_p),('service_stats',2,0,ctypes.c_int),('service_wait_idle',1,1,ctypes.c_int),('service_delay_next',1,1,ctypes.c_int),('service_destroy',1,0,ctypes.c_uint64),('publish',9,1,ctypes.c_int),('join',8,2,ctypes.c_int)]:
  fn=getattr(lib,'head_bridge258_'+name);fn.argtypes=[ctypes.c_void_p]*nptr+[ctypes.c_int]*nint;fn.restype=restype
 return lib
class Bridge:
 def __init__(self,fn,pool,packed,norm,rows=32768,safe=1,cpu=15,global_stats=None,cpu_status=None):
  self.lib=library();self.rows=rows;self.refs=(fn,pool,packed,norm)
  self.hin=torch.empty(2048,dtype=torch.bfloat16,pin_memory=True)
  self.hout=torch.empty(rows,dtype=torch.bfloat16,pin_memory=True)
  self.request=torch.zeros(16,dtype=torch.int32,pin_memory=True)
  self.ready=torch.zeros(16,dtype=torch.int32,pin_memory=True)
  self.status=torch.zeros(16,dtype=torch.int32,pin_memory=True)if cpu_status is None else cpu_status
  self.seq=torch.zeros(1,dtype=torch.int32,device='xpu')
  self.stats=torch.zeros(16,dtype=torch.int32,device='xpu')if global_stats is None else global_stats
  self.diag=torch.zeros(8,dtype=torch.int32,device='xpu')
  args=[ctypes.cast(fn,ctypes.c_void_p).value,pool,packed.data_ptr(),norm.data_ptr(),self.hin.data_ptr(),self.hout.data_ptr(),self.request.data_ptr(),self.ready.data_ptr(),self.status.data_ptr()]
  self.ptr=self.lib.head_bridge258_service_create(*args,rows,int(safe),cpu)
  assert self.ptr,self.lib.head_bridge258_error().decode()
 def publish(self,x,spin=0):
  tensors=[x,self.hin,self.request,self.ready,self.seq,self.status,self.stats,self.diag]
  rc=self.lib.head_bridge258_publish(torch.xpu.current_stream(x.device).sycl_queue,*[t.data_ptr()for t in tensors],spin)
  assert rc==0,(rc,self.lib.head_bridge258_error())
  for t in [x,self.seq,self.stats,self.diag]:t.record_stream(torch.xpu.current_stream(x.device))
 def join(self,y,spin=0):
  tensors=[self.hout,y,self.ready,self.seq,self.status,self.stats,self.diag]
  rc=self.lib.head_bridge258_join(torch.xpu.current_stream(y.device).sycl_queue,*[t.data_ptr()for t in tensors],self.rows,spin)
  assert rc==0,(rc,self.lib.head_bridge258_error())
  for t in [y,self.seq,self.stats,self.diag]:t.record_stream(torch.xpu.current_stream(y.device))
 def host_stats(self):
  value=(ctypes.c_uint64*8)();assert self.lib.head_bridge258_service_stats(self.ptr,value)==0;return list(value)
 def wait(self):assert self.lib.head_bridge258_service_wait_idle(self.ptr,5000)==0,self.host_stats()
 def delay(self,us):assert self.lib.head_bridge258_service_delay_next(self.ptr,us)==0
 def clear_errors(self):
  self.wait();self.status.zero_();self.stats.zero_();self.diag.zero_();torch.xpu.synchronize()
 def close(self):
  if self.ptr:
   torch.xpu.synchronize();self.wait();jobs=self.lib.head_bridge258_service_destroy(self.ptr);self.ptr=None;return jobs
