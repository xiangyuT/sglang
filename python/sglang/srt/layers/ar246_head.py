"""Pure-AR decode CPU prefix and original packed GPU tail; shared own pool."""
import ctypes,logging,os
import torch
from sglang.kernels.ops.gemm.xpu_ar_head_bridge_sycl import Bridge
logger=logging.getLogger(__name__)

class HeadContext:
    def __init__(self,manager,prepared):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Warm AR246 head bridge before graph capture')
        assert manager.head_enabled and manager.pool_pipeline and manager.service
        assert manager.cpu_library_path.endswith('/native/cpu-head243/expert.so')
        self.manager=manager;self.prepared=prepared;self.rows=32768
        from sglang.srt.runtime_context import get_device
        self.completion_watchdog=get_device().watchdog_timeout
        assert self.completion_watchdog is not None and self.completion_watchdog>0, 'Completion-wait head requires the existing scheduler watchdog'
        self.delay_test=os.environ.get('CODEX_AR260_DELAY_TEST')=='1';self.delay_id=0;self.injected_delays=[]
        raw=prepared.weight[:self.rows].cpu()
        old_affinity=os.sched_getaffinity(0);old_threads=torch.get_num_threads()
        try:
            os.sched_setaffinity(0,set(range(2,15)));torch.set_num_threads(8)
            self.packed=raw.reshape(-1,16,1024,2).permute(0,2,1,3).contiguous()
            self.norm=raw.double().square().sum(dim=1).sqrt()*(1+1e-8)
            bits=raw.view(torch.int16).to(torch.int32)&0x7fff
            safe=int(not bool(((bits>=0x7f80)|((bits>0)&(bits<0x80))).any()))
        finally:
            torch.set_num_threads(old_threads);os.sched_setaffinity(0,old_affinity)
        cpu=manager.cpu
        cpu.cpu_head_run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*4;cpu.cpu_head_run.restype=ctypes.c_int
        cpu.cpu_head_stats.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)];cpu.cpu_head_stats.restype=ctypes.c_int
        cpu.cpu_pool_busy.argtypes=[ctypes.c_void_p];cpu.cpu_pool_busy.restype=ctypes.c_int
        self.bridge=Bridge(cpu.cpu_head_run,manager.pool,self.packed,self.norm,rows=self.rows,safe=safe,cpu=15,global_stats=manager.stats,cpu_status=manager.cpu_status)
        logger.info('AR246 decode head bridge enabled: CPU32768/GPU215552; shared13-worker leased pool; CPUservice15; original packed GPU tail; completion/error wait with scheduler watchdog; single-WG join; prefill unchanged')
    def run(self,hidden):
        head=self.prepared;y=torch.empty((1,248320),dtype=torch.bfloat16,device=hidden.device)
        self.bridge.publish(hidden)
        offset=self.rows*2048;stream=torch.xpu.current_stream(hidden.device)
        rc=head.library.ar_static_head_gemv(stream.sycl_queue,hidden.data_ptr(),head.weight.data_ptr()+2*offset,
            head.low.data_ptr()+offset,head.codes.data_ptr()+offset//2,head.headers.data_ptr()+4*(offset//256),
            head.fallback.data_ptr(),y.data_ptr()+2*self.rows,248320-self.rows,8,1,0)
        if rc:raise RuntimeError(head.library.ar_static_head_error().decode())
        self.bridge.join(y)
        for t in [hidden,y,head.weight,*head.buffers]:t.record_stream(stream)
        return y
    def readback(self):
        self.bridge.wait();v=(ctypes.c_uint64*3)();assert self.manager.cpu.cpu_head_stats(self.manager.pool,v)==0
        service=self.bridge.host_stats();diag=self.bridge.diag.cpu().tolist()
        assert service[1]==0 and diag[0]==0 and self.manager.cpu.cpu_pool_busy(self.manager.pool)==0,(service,diag)
        if self.delay_test:
            import json
            from pathlib import Path
            control=Path(os.environ['CODEX_AR_CPU40_ROOT'])/'next-head-delay.json'
            if control.exists():
                value=json.loads(control.read_text())
                if value['id']>self.delay_id:
                    assert 0<=value['delay_us']<=250000
                    self.bridge.delay(value['delay_us']);self.delay_id=value['id'];self.injected_delays.append(value)
        return dict(completion_wait=True,polling_workgroups=1,watchdog_timeout=self.completion_watchdog,
            bridge_library='native/head-bridge258/bridge.so',injected_delays=self.injected_delays,
            enabled=True,CPU_prefix_rows=self.rows,GPU_tail_rows=248320-self.rows,shared_pool=True,service_cpu=15,
            service=service,head_guard=list(v),GPU_diag=diag,cpu_library_path=self.manager.cpu_library_path)
    def close(self):
        return self.bridge.close()

def try_run(prepared,hidden):
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager=cpu40.MANAGER
    if manager is None or not manager.head_enabled:return None
    manager.start()
    if manager.head_context is None:manager.head_context=HeadContext(manager,prepared)
    assert manager.head_context.prepared is prepared
    return manager.head_context.run(hidden)
