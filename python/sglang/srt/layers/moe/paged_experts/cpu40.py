"""AR67 ordinary CPU hybrid: no census, query trace or per-token timing observer."""
import atexit
import ctypes
import json
import logging
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0, '/evidence/ar-round40/notes')
from fixture01 import Fixture
from packing import pack_pairs as pack
logger = logging.getLogger(__name__)
MANAGER = None
MAIN_PINNED = False

class Manager(Fixture):
    def __init__(self):
        self.pool_pipeline = os.environ.get('CODEX_AR237_POOL') == '1'
        self.head_enabled = os.environ.get('CODEX_AR246_HEAD') == '1'
        self.head_context = None
        assert not self.head_enabled or self.pool_pipeline, 'AR246 requires the shared pipeline pool'
        self.cpu_library_path = ('/evidence/tpot-iterations-20261023/native/cpu-pipeline236/expert.so'
            if self.pool_pipeline else '/evidence/ar-round56/native/segment01/expert.so')
        if self.head_enabled:self.cpu_library_path='/evidence/tpot-iterations-20261023/native/cpu-head243/expert.so'
        self.guard865_enabled=os.environ.get('CODEX_AR865_GUARD')=='1'
        self.guard865_library_sha256=None
        if self.guard865_enabled:
            assert self.pool_pipeline and self.head_enabled, 'AR865 requires the existing guarded shared pipeline ABI'
            from hashlib import sha256
            self.cpu_library_path='/evidence/tpot-iterations-20261023/native/guard855/expert.so'
            self.guard865_library_sha256=sha256(Path(self.cpu_library_path).read_bytes()).hexdigest()
            assert self.guard865_library_sha256=='03ac7656e5639a9edcd18edf5bc58a6ed215800db2144a875a0580ad8a6a6e21'
            logger.info('AR865 CPU guard enabled: F64 certificate16rows; safe exact tile fallback>=4; original BF16/guard counts and pool protocol')
        self.cpu = ctypes.CDLL(self.cpu_library_path)
        self.cpu.cpu_expert_run = (self.cpu.cpu_expert_run_guard_pipeline
            if self.pool_pipeline else self.cpu.cpu_expert_run_guard)
        self.cpu.cpu_expert_register_guard.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*3
        self.cpu.cpu_expert_guard_stats.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
        self.cpu.cpu_expert_error.restype = ctypes.c_char_p
        self.cpu.cpu_expert_set_spin.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.cpu.cpu_expert_pool_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.cpu.cpu_expert_create.restype = ctypes.c_void_p
        self.cpu.cpu_expert_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.cpu.cpu_expert_destroy.argtypes = [ctypes.c_void_p]
        self.cpu.cpu_expert_destroy.restype = None
        self.cpu.cpu_expert_pool_stats.restype = None
        self.cpu.cpu_expert_guard_stats.restype = None
        self.cpu.cpu_expert_run.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 3
        self.defer_refill_enabled=os.environ.get('CODEX_AR1520_REFILL')=='1'
        self.service_library_path='/evidence/ar-round44/native/adaptive05/hybrid.so'
        self.service_library_sha256=None
        if self.defer_refill_enabled:
            assert self.guard865_enabled and self.pool_pipeline and self.head_enabled, 'AR1520 requires the validated guard855 CPU pipeline'
            from hashlib import sha256
            self.service_library_path='/evidence/tpot-iterations-20261023/native/refill1514/hybrid.so'
            self.service_library_sha256=sha256(Path(self.service_library_path).read_bytes()).hexdigest()
            assert self.service_library_sha256=='e888338139e564922743662773afe3d5d393262e78811075a3ec79437bf217de'
        self.lib = ctypes.CDLL(self.service_library_path)
        if self.defer_refill_enabled:
            self.lib.hybrid_service_defer_refill.argtypes=[ctypes.c_void_p,ctypes.c_int]
            self.lib.hybrid_service_defer_refill_get.argtypes=[ctypes.c_void_p]
        self.relaxed_refill = False
        self.current_scope = 0
        self.lib.hybrid_service_scope.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.hybrid_service_scope.restype = ctypes.c_int
        self.lib.hybrid_error.restype = ctypes.c_char_p
        self.lib.hybrid_service_delay_next.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.hybrid_service_wait_idle.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.hybrid_service_diagnostic.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.lib.hybrid_service_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.lib.hybrid_service_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.hybrid_publish.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int]
        self.lib.hybrid_gate.argtypes = [ctypes.c_void_p] * 9
        self.lib.hybrid_down.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int]
        self.lib.hybrid_merge.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int]
        self.lib.hybrid_down_merged.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 2
        self.lib.hybrid_route_publish.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] + [ctypes.c_void_p] * 2
        self.lib.hybrid_down_packed.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 2
        self.word_down_enabled=(os.environ.get('CODEX_AR548_WORD')=='1'
            and os.environ.get('CODEX_AR548_WORD_DOWN','1')=='1')
        self.word_down_library=None
        if self.word_down_enabled:
            word=ctypes.CDLL('/evidence/tpot-iterations-20261023/native/decode541/down.so')
            word.down541_run.argtypes=self.lib.hybrid_down_packed.argtypes
            word.down541_error.restype=ctypes.c_char_p
            legacy_error=self.lib.hybrid_error
            # Bind the original call slot once; no per-token dispatch/observer.
            self.lib.hybrid_down_packed=word.down541_run
            def combined_error(legacy=legacy_error,alternate=word.down541_error):
                return b'legacy hybrid: '+legacy()+b'; word541: '+alternate()
            self.lib.hybrid_error=combined_error
            self.word_down_library=word
            logger.info('AR548 word Down enabled: down541; original CPU readiness and reduction')
        self.fused = True
        self.lib.hybrid_service_create.restype = ctypes.c_void_p
        self.lib.hybrid_service_create.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p] * 9 + [ctypes.c_int]
        self.lib.hybrid_service_destroy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.hybrid_service_destroy.restype = ctypes.c_uint64
        self.hin = torch.empty((1, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.hout = torch.empty((8, 2048), dtype=torch.bfloat16, pin_memory=True)
        self.header = torch.zeros(32, dtype=torch.int32, pin_memory=True)
        self.request, self.ready = [torch.zeros(16, dtype=torch.int32, pin_memory=True) for _ in range(2)]
        self.sequence = torch.zeros(1, dtype=torch.int32, device='xpu')
        self.stats = torch.zeros(16, dtype=torch.int32, device='xpu')
        self.issued = torch.zeros(40, dtype=torch.int32, device='xpu')
        self.refill_done = torch.zeros(40, dtype=torch.int32, pin_memory=True)
        self.cpu_status = torch.zeros(16, dtype=torch.int32, pin_memory=True)

        self.flag760_enabled=os.environ.get('CODEX_AR760_FLAGS')=='1'
        self.flag760_storage=None
        self.flag760_library_sha256=None
        if self.flag760_enabled:
            # The native service owns only ready[0]/status[0]. Keep distinct views
            # in one pinned allocation so Down can issue one uncached uint64 read.
            assert self.word_down_enabled, 'AR760 requires original word541 math'
            self.flag760_storage=torch.zeros(64,dtype=torch.int32,pin_memory=True)
            self.ready=self.flag760_storage[0:1]
            self.cpu_status=self.flag760_storage[1:2]
            assert self.ready.data_ptr()%8==0 and self.cpu_status.data_ptr()==self.ready.data_ptr()+4
            from hashlib import sha256
            path=Path('/evidence/tpot-iterations-20261023/native/down755/down.so')
            self.flag760_library_sha256=sha256(path.read_bytes()).hexdigest()
            assert self.flag760_library_sha256=='40002b862193e7c528cc0359cebef1436c53d851075cd9d378a15346dc41c585'
            word=ctypes.CDLL(str(path))
            word.down755_run.argtypes=self.lib.hybrid_down_packed.argtypes
            word.down755_error.restype=ctypes.c_char_p
            previous_error=self.lib.hybrid_error
            self.lib.hybrid_down_packed=word.down755_run
            def flag760_error(previous=previous_error,current=word.down755_error):
                return previous()+b'; flag760: '+current()
            self.lib.hybrid_error=flag760_error
            self.word_down_library=word
            logger.info('AR760 adjacent CPU ready/status enabled: uncached64 poll; acquire/error/20000 bound retained')

        self.layers = []
        self.service = None
        self.pool = None
        self.channels = None
        self.copy_stream = None
        self.cached_bytes = 0
        self.norm_bytes = 0
        atexit.register(self.shutdown)

    def register(self, executor, layer):
        index = int(executor.base_method.runner.config.layer_id)
        assert index == len(self.layers) and index < 40
        assert executor.use_scaled_fp8 and executor.residency.slot_expert.numel() == 192
        h1, h2, hs1, hs2 = executor.expert_store
        assert h1.shape == (256, 1024, 2048) and h2.shape == (256, 2048, 512)
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        # Keep explicit allocation headroom; original FP8 host store remains alive.
        assert current + 2 * 2**30 + 8 * 2**30 < limit, (index, current, limit)
        if True:
            cached1 = torch.empty((256, 64, 1024, 16, 2), dtype=torch.bfloat16)
            cached2 = torch.empty((256, 128, 256, 16, 2), dtype=torch.bfloat16)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(8)
            norm1=torch.empty((256,1024),dtype=torch.float64)
            norm2=torch.empty((256,2048),dtype=torch.float64)
            safe1=safe2=True
            def normal_weights(t):
                bits=t.view(torch.int16).to(torch.int32)&0x7fff
                return not bool(((bits>=0x7f80)|((bits>0)&(bits<0x80))).any())
            for first in range(0, 256, 4):
                a1=pack(h1[first:first+4],hs1[first:first+4])
                a2=pack(h2[first:first+4],hs2[first:first+4])
                cached1[first:first+4].copy_(a1);cached2[first:first+4].copy_(a2)
                norm1[first:first+4].copy_((a1.double().square().sum(dim=(2,4)).sqrt()*(1+1e-8)).reshape(4,1024))
                norm2[first:first+4].copy_((a2.double().square().sum(dim=(2,4)).sqrt()*(1+1e-8)).reshape(4,2048))
                safe1=bool(safe1 and normal_weights(a1));safe2=bool(safe2 and normal_weights(a2))
            torch.set_num_threads(previous_threads)
        else:
            cached1 = cached2 = torch.empty(0, dtype=torch.bfloat16)
        t = dict(index=index, executor=executor, cached1=cached1, cached2=cached2,
                 norm1=norm1,norm2=norm2,safe1=int(safe1),safe2=int(safe2),
                 hs1=hs1, hs2=hs2, h1=h1, h2=h2,
                 w1=layer.w13_weight, w2=layer.w2_weight,
                 s1=layer.w13_weight_scale_inv, s2=layer.w2_weight_scale_inv,
                 map=torch.empty(8, dtype=torch.int32, device='xpu'))
        self.layers.append(t)
        self.cached_bytes += cached1.nbytes + cached2.nbytes
        self.norm_bytes += norm1.nbytes+norm2.nbytes
        executor._cpu40_channel = index
        logger.info('AR40 CPU cache layer=%d bytes=%d cgroup_current=%d', index, self.cached_bytes,
                    int(Path('/sys/fs/cgroup/memory.current').read_text()))

    def start(self):
        if self.service:
            return
        assert len(self.layers) == 40 and not torch.xpu.is_current_stream_capturing()
        self.channels = torch.tensor([[t[k].data_ptr() for k in ['cached1', 'cached2', 'hs1', 'hs2']]
                                     + [256] + [t[k].data_ptr() for k in ['h1', 'h2', 'w1', 'w2', 's1', 's2']] + [192]
                                     for t in self.layers], dtype=torch.uint64)
        self.copy_stream = torch.xpu.Stream()
        with torch.xpu.stream(self.copy_stream):
            self.copy_queue = torch.xpu.current_stream().sycl_queue
        self.lut = torch.load('/evidence/ar-round07/notes/silu05.pt', map_location='cpu', weights_only=True)
        cpus = (ctypes.c_int * 13)(*range(2, 15))
        self.pool = self.cpu.cpu_expert_create(13, cpus)
        assert self.pool, self.cpu.cpu_expert_error().decode()
        for t in self.layers:
            rc=self.cpu.cpu_expert_register_guard(self.pool,t['cached1'].data_ptr(),t['cached2'].data_ptr(),t['norm1'].data_ptr(),t['norm2'].data_ptr(),256,t['safe1'],t['safe2'])
            assert rc==0,(t['index'],rc)
        self.service = self.lib.hybrid_service_create(
            ctypes.cast(self.cpu.cpu_expert_run, ctypes.c_void_p), self.pool,
            self.channels.data_ptr(), 40, self.lut.data_ptr(), self.hin.data_ptr(),
            self.header.data_ptr(), self.hout.data_ptr(), self.request.data_ptr(), self.ready.data_ptr(),
            self.refill_done.data_ptr(), self.cpu_status.data_ptr(), self.copy_queue, 1)
        assert self.service, self.lib.hybrid_error().decode()
        self.setup_native_methods()
        self.configure(32, 6)
        if self.defer_refill_enabled:
            assert self.lib.hybrid_service_defer_refill(self.service,1)==0
            assert self.lib.hybrid_service_defer_refill_get(self.service)==1
            logger.info('AR1520 CPU result first enabled: mode6;40channels;pending/deadline registered before CPU;4plane copy submitted after CPU ready;prior-generation refill_done still required')
        logger.info('AR57 guarded segmented AMX and BF16 pair cache enabled: norm_bytes=%d',self.norm_bytes)
        if self.pool_pipeline:
            logger.info('AR237 CPU expert per-expert DAG enabled: one pool publication, original GU/down math; library=%s',self.cpu_library_path)
        logger.info('AR45 adaptive down enabled: single node; allhit original tree, mixed subgroup-packed')
        self.set_spin(200000)
        os.sched_setaffinity(0, {0})
        logger.info('AR40 full-model CPU expert enabled: 40 channels,13 AVX512 workers,mode6,merged down row1,FULL graph,cache_bytes=%d', self.cached_bytes)

    def run(self, executor, layer, dispatch, ar_shared=None):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        if ar_shared is not None:
            from sglang.srt.layers.ar119_cpu_shared import CPUShared
            assert isinstance(ar_shared, CPUShared), "CPU hybrid requires its explicit shared context"
        self.start()
        x = dispatch.hidden_states
        topk = dispatch.topk_output
        assert x.shape == (1, 2048) and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert topk.topk_ids.shape == (1, 8) and topk.topk_ids.dtype == torch.int32
        assert topk.topk_weights.dtype == torch.float32
        assert dispatch.hidden_states_scale is None and dispatch.hidden_states_pre_quant is None
        t = self.layers[executor._cpu40_channel]
        assert t['executor'] is executor
        r = executor.residency
        a = torch.empty((8, 512), dtype=x.dtype, device=x.device)
        partial = torch.empty((8, 2048), dtype=x.dtype, device=x.device)
        y = torch.empty_like(x)
        published = False
        if os.environ.get('CODEX_AR362_PUBLISH') == '1':
            from sglang.srt.layers.moe.paged_experts.cpu_publish362 import was_published
            published = was_published(topk, self, executor, x, torch.xpu.current_stream(x.device))
        if not published:
            self.call('hybrid_route_publish', x.data_ptr(), t['map'].data_ptr(), r.count.data_ptr(),
                      self.hin.data_ptr(), self.header.data_ptr(), self.request.data_ptr(), self.sequence.data_ptr(),
                      executor.mapped_ids.data_ptr(), self.stats.data_ptr(), self.issued.data_ptr(), self.refill_done.data_ptr(),
                      self.cpu_status.data_ptr(), t['index'], topk.topk_ids.data_ptr(), r.src.data_ptr())
        from sglang.srt.layers.ar124_cpu_joint import try_run as try_joint_gu
        joint_issued = try_joint_gu(self, executor, layer, dispatch, ar_shared, a)
        if not joint_issued:
            if ar_shared is not None:
                ar_shared.launch()
            self.call('hybrid_gate', x.data_ptr(), t['w1'].data_ptr(), t['s1'].data_ptr(),
                      executor.mapped_ids.data_ptr(), t['map'].data_ptr(), r.count.data_ptr(), a.data_ptr(), self.stats.data_ptr())
        self.call('hybrid_down_packed', a.data_ptr(), t['w2'].data_ptr(), t['s2'].data_ptr(),
                  executor.mapped_ids.data_ptr(), t['map'].data_ptr(), r.count.data_ptr(), executor.mapped_weights.data_ptr(),
                  partial.data_ptr(), y.data_ptr(), self.stats.data_ptr(), self.hout.data_ptr(), self.ready.data_ptr(),
                  self.sequence.data_ptr(), self.cpu_status.data_ptr(), 1, 20000)
        for v in [x, topk.topk_ids, topk.topk_weights, a, partial, y]:
            v.record_stream(torch.xpu.current_stream(x.device))
        return StandardCombineInput(hidden_states=executor.contract.finish(
            merged=y, hidden_states=x, routed_scaling_factor=executor.routed_scaling_factor))

    def snapshot(self):
        torch.xpu.synchronize()
        self.drain()
        head_state=self.head_context.readback() if self.head_context is not None else dict(enabled=False)
        scheduling = {}
        for task in Path('/proc/self/task').iterdir():
            try:
                scheduling[task.name] = dict(affinity=sorted(os.sched_getaffinity(int(task.name))), schedstat=list(map(int,(task/'schedstat').read_text().split())), comm=(task/'comm').read_text().strip())
            except (FileNotFoundError,ProcessLookupError):
                pass
        cpu_stat = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
        variant = (ctypes.c_uint64 * 3)()
        self.cpu.cpu_expert_guard_stats(self.pool,variant)
        result = dict(guard_stats=list(variant), norm_bytes=self.norm_bytes, scheduling=scheduling, cgroup_cpu=cpu_stat, stage=self.stage_stats(), pipeline=self.pipeline_stats(), pool=self.pool_stats(),
                      gpu=self.stats.cpu().tolist(), diagnostic=self.diagnostic(), cached_bytes=self.cached_bytes,
                      cgroup_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),
                      cgroup_peak=int(Path('/sys/fs/cgroup/memory.peak').read_text()),
                      issued=self.issued.cpu().tolist(), done=self.refill_done.tolist(), native=self.native_info(), head246=head_state)
        result['refill1520']=dict(enabled=self.defer_refill_enabled,
            library=self.service_library_path,library_sha256=self.service_library_sha256,
            actual_enabled=(bool(self.lib.hybrid_service_defer_refill_get(self.service)) if self.defer_refill_enabled else False),
            pending_registered_before_CPU=True,deadline_registered_before_CPU=True,
            ready_before_copy_submission=self.defer_refill_enabled,
            four_plane_refill=True,previous_generation_guard=True)
        root = Path(os.environ['CODEX_AR_CPU40_ROOT'])
        with (root / 'snapshot-cpu.jsonl').open('a') as handle:
            handle.write(json.dumps(result)+'\n')
        assert result['gpu'][0] == result['stage'][3] == result['pipeline'][2] == result['pipeline'][5] == result['pipeline'][9] == 0, result
        assert all(a == 0 or a == b for a, b in zip(result['issued'], result['done'])), result
        return result

    def close(self):
        if self.head_context is not None:
            self.head_destroyed_jobs=self.head_context.close()
            self.head_context=None
        return super().close()

    def shutdown(self):
        if not self.service:return
        result={}
        try:
            result=self.snapshot()
        except Exception:
            logger.exception('AR246 shutdown/drain diagnostic failed')
        finally:
            try:
                result['destroyed_jobs'],result['destroy_errors']=self.close()
                result['head_destroyed_jobs']=getattr(self,'head_destroyed_jobs',0)
            except Exception as exc:
                result['close_error']=repr(exc)
                logger.exception('AR246 native shutdown failed; retain until process exit')
            root=os.environ.get('CODEX_AR_CPU40_ROOT')
            if root:Path(root,'cpu-final.json').write_text(json.dumps(result,indent=2)+'\n')


def register(executor, layer):
    global MANAGER
    if os.environ.get('CODEX_AR_CPU40') != '1':
        return
    if MANAGER is None:
        MANAGER = Manager()
    MANAGER.register(executor, layer)


def before_apply(dispatch):
    if MANAGER and MANAGER.service and dispatch.hidden_states.shape != (1, 2048):
        assert not torch.xpu.is_current_stream_capturing()
        MANAGER.drain()


def run(executor, layer, dispatch, ar_shared=None):
    global MAIN_PINNED
    if dispatch.hidden_states.shape == (1, 2048) and not MAIN_PINNED:
        os.sched_setaffinity(0, {0})
        MAIN_PINNED = True
        logger.info('AR67 common scheduler main affinity=0; no census or timing observer')
    if os.environ.get('CODEX_AR_CPU40') != '1' or dispatch.hidden_states.shape != (1, 2048):
        return None
    assert MANAGER is not None
    return MANAGER.run(executor, layer, dispatch, ar_shared=ar_shared)



def graph_boundary(capture=False):
    """Preserve startup/capture drain; candidate uses bounded channel fences at replay."""
    if MANAGER and MANAGER.service and (capture or not MANAGER.relaxed_refill):
        torch.xpu.current_stream().synchronize()
        MANAGER.drain()


def readback():
    """Only explicit outside-interval readbacks can change the drained mode."""
    if MANAGER and MANAGER.service:
        result = MANAGER.snapshot()
        config = Path(os.environ['CODEX_AR_CPU40_ROOT']) / 'next-mode.json'
        if config.exists():
            setting = json.loads(config.read_text())
            assert set(setting) == {'mode'} and setting['mode'] in ['host-before', 'device-nodrain', 'host-after']
            relaxed = setting['mode'] == 'device-nodrain'
            rc = MANAGER.lib.hybrid_service_scope(MANAGER.service, int(relaxed))
            assert rc == 0, ('scope change not at drained boundary', rc)
            MANAGER.current_scope = int(relaxed)
            MANAGER.relaxed_refill = relaxed
            receipt = dict(mode=setting['mode'], scope=MANAGER.current_scope, skip_replay_host_drain=relaxed, setter_rc=rc,
                           pending=result['pipeline'][2], request_equals_ready=True, jobs=result['stage'][0])
            with Path(os.environ['CODEX_AR_CPU40_ROOT'], 'applied-modes.jsonl').open('a') as handle:
                handle.write(json.dumps(receipt)+'\n')
        result.update(pool_pipeline=MANAGER.pool_pipeline, cpu_library_path=MANAGER.cpu_library_path)
        result.update(word548_down=MANAGER.word_down_enabled)
        result['guard865']=dict(enabled=MANAGER.guard865_enabled,
            library='native/guard855/expert.so' if MANAGER.guard865_enabled else None,
            library_sha256=MANAGER.guard865_library_sha256,
            certificate='original-F64-bound-vector16',exact_tile_threshold=4,
            unsafe_path='original-scalar',BF16_dtype_unchanged=True,
            guard_factor_unchanged=True,pool_protocol_unchanged=True)
        result['flag760']=dict(enabled=MANAGER.flag760_enabled,
            adjacent=(MANAGER.cpu_status.data_ptr()==MANAGER.ready.data_ptr()+4),
            aligned8=(MANAGER.ready.data_ptr()%8==0),
            library='native/down755/down.so' if MANAGER.flag760_enabled else None,
            library_sha256=MANAGER.flag760_library_sha256,
            read_bits=64 if MANAGER.flag760_enabled else 32,wait_read_bound=20000,
            status_check=True,system_acquire=True,
            padding_intact=(bool((MANAGER.flag760_storage[2:]==0).all()) if MANAGER.flag760_enabled else None))

        result.update(enabled=True, census=False, query_trace=False, per_token_observer=False,
                      main_affinity=sorted(os.sched_getaffinity(0)), configured_scope=MANAGER.current_scope,
                      skip_replay_host_drain=MANAGER.relaxed_refill, capture_drain=True, prefill_drain=True, wait_read_bound=20000,UR_L0_USE_COPY_ENGINE=os.environ.get('UR_L0_USE_COPY_ENGINE'),UR_L0_DEBUG=os.environ.get('UR_L0_DEBUG'))
        return result
    return dict(enabled=False, census=False, query_trace=False, per_token_observer=False, main_affinity=sorted(os.sched_getaffinity(0)), cached_bytes=0, norm_bytes=0)
