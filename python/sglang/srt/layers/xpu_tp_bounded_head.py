"""TP-local Q4 bounds, original BF16 exact rows and compact greedy gathering.

Only a plain greedy breakable decode graph may consume partial logits.
The Q4 filter never supplies a final score: original BF16 rows do. A native
FP32 MAX combines conservative per-shard thresholds before candidate selection. Unsupported
requests use the original eager/full-logit path. One context per LM-head shard;
the startup static-head flag enforces the existing weight-update restriction.
"""
import ctypes as C
import hashlib
import logging
from typing import NamedTuple

import torch
import torch.distributed as dist
from pathlib import Path
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph

from sglang.srt.environ import envs
from sglang.srt.layers.ar449_head import Args, eligibility, pack_layout

logger = logging.getLogger(__name__)


class HeadOutput(NamedTuple):
    logits: torch.Tensor
    gathered: bool


def enabled():
    return envs.SGLANG_XPU_TP_BOUNDED_HEAD.get()


def graph_allowed(fb):
    return eligibility(fb)[0]


@eager_on_graph
def _global_threshold(threshold):
    from sglang.srt.distributed.parallel_state import get_tp_group
    dist.all_reduce(threshold, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)


class TpBoundedHead:
    def __init__(self, weight):
        if not (weight.device.type == 'xpu' and weight.dtype == torch.bfloat16
                and weight.ndim == 2 and weight.shape[1] == 2048
                and weight.shape[0] == 124160 and weight.is_contiguous()
                and weight.data_ptr() % 4 == 0
                and torch.xpu.get_device_properties(weight.device).device_id == 0xe223):
            raise ValueError('Unsupported TP bounded-head shard')
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Warm TP bounded head before capture')
        self.weight = weight
        self.n = weight.shape[0]
        self.pointer = weight.data_ptr()
        try:
            self.version = weight._version
        except RuntimeError:
            self.version = None
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(8)
            packed = pack_layout(weight.cpu(), bits=4)
            self.low, self.high, self.scale, self.rho = [x.to(weight.device) for x in packed]
        finally:
            torch.set_num_threads(previous)
        self.lower = torch.empty(self.n, device=weight.device, dtype=torch.float32)
        self.upper = torch.empty_like(self.lower)
        self.blockmax = torch.empty(self.n//8, device=weight.device, dtype=torch.float32)
        self.threshold = torch.empty(1, device=weight.device, dtype=torch.float32)
        self.norm = torch.empty_like(self.threshold)
        self.control = torch.zeros(18, device=weight.device, dtype=torch.int32)
        self.capacity = min(self.n, 32768)
        self.candidates = torch.empty(self.capacity, device=weight.device, dtype=torch.int32)
        self.eligible = torch.ones(1, device=weight.device, dtype=torch.int32)
        self.buffers = [self.low, self.high, self.scale, self.rho, self.lower,
                        self.upper, self.blockmax, self.threshold, self.norm,
                        self.control, self.candidates, self.eligible]
        path = Path(envs.SGLANG_XPU_TP_BOUNDED_HEAD_LIB.get())
        if not path.is_file():
            raise RuntimeError('Set SGLANG_XPU_TP_BOUNDED_HEAD_LIB to the built library')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.library_sha256 = sha
        self.lib = C.CDLL(str(path))
        self.lib.xpu_tp_bounded_head_run.argtypes = [C.c_void_p, C.POINTER(Args), C.c_int]
        self.lib.xpu_tp_bounded_head_error.restype = C.c_char_p
        assert self.lib.xpu_tp_bounded_head_abi_size() == C.sizeof(Args)
        assert self.lib.xpu_tp_bounded_head_control_words() == self.control.numel()
        self.collective = None
        if envs.SGLANG_XPU_TP_HEAD_GRAPH_COLLECTIVES.get():
            from sglang.srt.distributed.parallel_state import get_tp_group
            from sglang.srt.layers.xpu_tp_head_collective import HeadCollective

            group = get_tp_group()
            communicator = getattr(group, 'xpu_communicator', None)
            peer = getattr(communicator, 'peer_allreduce', None)
            self.collective = HeadCollective(peer)
        logger.info('B70 TP bounded head initialized: N=%d K=2048 Q4 bounds/BF16 exact, library=%s', self.n, sha)

    def run(self, hidden):
        if (hidden.shape != (1,2048) or hidden.dtype != torch.bfloat16
                or hidden.device != self.weight.device or not hidden.is_contiguous()
                or hidden.data_ptr() % 4 or self.weight.data_ptr() != self.pointer):
            raise ValueError('Unsupported/changed TP bounded-head input')
        try:
            version = self.weight._version
        except RuntimeError:
            version = None
        if version != self.version:
            raise RuntimeError('Static TP head changed; restart and recapture')
        out = torch.empty((1,self.n), device=hidden.device, dtype=torch.bfloat16)
        args = Args(hidden.data_ptr(), self.weight.data_ptr(), self.low.data_ptr(),
                    self.high.data_ptr(), self.scale.data_ptr(), self.rho.data_ptr(),
                    self.lower.data_ptr(), self.upper.data_ptr(), self.blockmax.data_ptr(),
                    self.threshold.data_ptr(), self.norm.data_ptr(), self.control.data_ptr(),
                    self.candidates.data_ptr(), out.data_ptr(), None,
                    self.eligible.data_ptr(), self.n, self.capacity, 256)
        stream = torch.xpu.current_stream(hidden.device)
        for phase in (0, 1):
            if phase == 1:
                if self.collective is None:
                    _global_threshold(self.threshold)
                else:
                    self.collective.maximum(self.threshold)
            rc = self.lib.xpu_tp_bounded_head_run(stream.sycl_queue, C.byref(args), phase)
            if rc:
                raise RuntimeError((rc, self.lib.xpu_tp_bounded_head_error().decode()))
        for x in [hidden, self.weight, out, *self.buffers]:
            x.record_stream(stream)
        return out


def try_run(processor, hidden, head, metadata, bias):
    if not enabled() or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    if envs.SGLANG_XPU_TP_HEAD_GRAPH_COLLECTIVES.get() and (
        envs.SGLANG_SANITIZE_NAN_LOGITS.get()
        or envs.SGLANG_ENABLE_NAN_LOGITS_CHECK.get()
        or envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
    ):
        # Sparse greedy scores cannot substitute for diagnostic/sanitized
        # full-logit behavior. Preserve the original complete path.
        return None
    from sglang.srt.runtime_context import get_parallel, get_spec
    parallel = get_parallel()
    if not (parallel.tp_size == 2 and parallel.moe_ep_size in (1, 2)
            and parallel.attn_cp_size == 1
            and parallel.pp_size == 1 and not parallel.attn_dp_enabled
            and get_spec().speculative_algorithm is None
            and metadata is not None and metadata.forward_mode.is_decode()
            and bias is None and hidden.shape == (1,2048)
            and hidden.dtype == torch.bfloat16 and hidden.device.type == 'xpu'
            and processor.logit_scale is None and not processor.final_logit_softcapping
            and not processor.use_fp32_lm_head and processor.rl_on_policy_target is None
            and processor.do_tensor_parallel_all_gather
            and not processor.do_tensor_parallel_all_gather_dp_attn
            and not hasattr(head, 'set_lora') and not hasattr(head, 'apply_lora')
            and type(getattr(head, 'quant_method', None)).__name__ == 'UnquantizedEmbeddingMethod'
            and getattr(head, 'weight', None) is not None
            and head.weight.shape == (248320//parallel.tp_size,2048)
            and head.weight.dtype == torch.bfloat16):
        return None
    if not envs.SGLANG_XPU_AR_STATIC_PACKED_HEAD_SYCL.get():
        raise RuntimeError('TP bounded head requires the static-head weight-update guard')
    state = getattr(head, '_xpu_tp_bounded_head', None)
    if state is None:
        state = TpBoundedHead(head.weight)
        head._xpu_tp_bounded_head = state
        # Warm native dispatch before capture; eager callers still get the
        # untouched original full-logit path below.
        warm = state.run(hidden)
        if state.collective is not None:
            state.collective.gather_greedy(warm)
    if state.weight is not head.weight:
        raise RuntimeError('Static TP head replaced; restart and recapture')
    if not torch.xpu.is_current_stream_capturing():
        return None
    if not is_in_breakable_cuda_graph():
        raise RuntimeError('TP bounded head requires the breakable decode graph backend')
    local = state.run(hidden)
    if state.collective is not None:
        return HeadOutput(state.collective.gather_greedy(local), True)
    return HeadOutput(local, False)


def readback(model):
    """Read existing GPU work counters through the engine's state API only."""
    if not enabled():
        return dict(enabled=False)
    from sglang.srt.runtime_context import get_parallel
    import json

    states = []
    for name, module in model.named_modules():
        state = getattr(module, '_xpu_tp_bounded_head', None)
        if state is None:
            continue
        values = state.control.cpu()
        work = values[4:16].view(torch.int64).tolist()
        seeds = int(values[16:18].view(torch.int64)[0])
        assert min(work) >= 0 and work[2] == work[3] + work[5]
        stats = dict(name=name, rank=get_parallel().tp_rank, n=state.n,
                     calls=work[0], exact_rows=work[1], full_fallbacks=work[2],
                     unsafe_or_ineligible=work[3], proposed_candidates=work[4],
                     overflows=work[5], seed_rows=seeds, capacity=state.capacity,
                     final_control=values[:4].tolist(),
                     library_sha256=state.library_sha256,
                     graph_collectives=state.collective is not None,
                     graph_codec_library_sha256=(state.collective.library_sha256
                                                 if state.collective is not None else None),
                     static_filter_bytes=sum(x.numel()*x.element_size() for x in
                                             (state.low, state.high, state.scale, state.rho)))
        states.append(stats)
        logger.info('TP bounded head work readback: %s', json.dumps(stats, sort_keys=True))
    return dict(enabled=True, states=states,
                evidence='GPU counters read outside timed requests; original BF16 exact-row tree, not a universal native-GEMM bitwise equivalence claim')
