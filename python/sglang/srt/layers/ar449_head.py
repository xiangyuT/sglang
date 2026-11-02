"""Q5 bounded greedy head, full BF16 fallback, runtime full-graph eligibility.

Single-row, static BF16 head. Original static-head weight-update restrictions
remain in force. All scratch storage belongs to one in-order graph workspace.
"""
import ctypes
import logging
import os
from pathlib import Path

import torch

R = Path('/evidence/tpot-iterations-20261023')
logger = logging.getLogger(__name__)
LAST = None
POINTERS = ['x', 'w', 'low', 'high', 'scale', 'rho', 'lower', 'upper', 'blockmax',
            'threshold', 'norm', 'control', 'candidates', 'y', 'approx', 'eligible']


class Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in POINTERS] + [
        (name, ctypes.c_int) for name in ['n', 'capacity', 'exact_groups']]


def enabled():
    return os.environ.get('CODEX_AR449_HEAD') == '1'


def eligibility(fb, *, target_verify=False, max_batch=1):
    """CPU metadata only; a missing/unsupported field disables pruning."""
    if fb is None or not 1 <= getattr(fb, 'batch_size', 0) <= max_batch:
        return False, 'batch'
    mode = getattr(fb, 'forward_mode', None)
    actual = getattr(fb, 'actual_forward_mode', None) or mode
    if mode is None:
        return False, 'mode'
    if target_verify:
        if not mode.is_target_verify() or not actual.is_target_verify():
            return False, 'mode'
    elif not mode.is_decode() or not actual.is_decode() or getattr(fb, 'spec_info', None) is not None:
        return False, 'mode'
    if getattr(fb, 'return_logprob', True):
        return False, 'logprobs'
    si = getattr(fb, 'sampling_info', None)
    if si is None or not getattr(si, 'is_all_greedy', False):
        return False, 'sampling'
    for name in ['logit_bias', 'acc_additive_penalties', 'acc_scaling_penalties', 'grammar_mask',
                 'sampling_mask_batch_indices', 'sampling_support_logprobs_capture_indices']:
        if getattr(si, name, None) is not None:
            return False, name
    penalizer = getattr(si, 'penalizer_orchestrator', None)
    if penalizer is not None and getattr(penalizer, 'is_required', True):
        return False, 'penalizer'
    if any(g is not None for g in (getattr(si, 'grammars', None) or [])):
        return False, 'grammar'
    if getattr(si, 'has_custom_logit_processor', False) or getattr(si, 'custom_logit_processor', None):
        return False, 'custom'
    if getattr(si, 'has_watermark_candidates', False):
        return False, 'watermark'
    for name in ['watermark_keys', 'watermark_context_windows', 'watermark_enabled']:
        if getattr(si, name, None) is not None:
            return False, name
    for name in ['return_sampling_masks', 'return_sampling_support_logprobs']:
        if any(getattr(si, name, None) or []):
            return False, name
    # Greedy requests have need_top_k_sampling=True because top_k=1. Sampler
    # chooses its greedy branch before those flags; they do not modify logits.
    return True, 'plain-greedy'


def pack_layout(raw, bits=5):
    assert bits in (4, 5)
    offset = (1 << (bits - 1)) - 1
    assert (raw.device.type == 'cpu' and raw.dtype == torch.bfloat16
            and raw.ndim == 2 and raw.shape[1] == 2048
            and 0 < raw.shape[0] <= 248320 and raw.shape[0] % 8 == 0)
    n = raw.shape[0]
    low = torch.empty((n, 1024), dtype=torch.uint8)
    high = torch.empty((n, 256), dtype=torch.uint8) if bits == 5 else torch.zeros(1, dtype=torch.uint8)
    scales = torch.empty((n, 16), dtype=torch.float32)
    rho = torch.empty(n, dtype=torch.float32)
    gamma = (4096 * 2**-24) / (1 - 4096 * 2**-24)
    for start in range(0, n, 1024):
        end = min(n, start + 1024); w = raw[start:end].float()
        if not torch.isfinite(w).all() or float(w.abs().max()) >= 2**16:
            raise ValueError('Q5 head preparation requires finite bounded static weights')
        blocks = w.view(-1, 16, 128)
        scale = blocks.abs().amax(-1) / offset
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quant = (blocks / scale[:, :, None]).round().clamp(-offset, offset).to(torch.int32)
        decoded = (quant.float() * scale[:, :, None]).view(-1, 2048)
        residual = (w.double() - decoded.double()).square().sum(1).sqrt() * (1 + 1e-8)
        wn = w.double().square().sum(1).sqrt() * (1 + 1e-8)
        qn = decoded.double().square().sum(1).sqrt() * (1 + 1e-8)
        rd = residual + gamma * (wn + qn)
        ru = torch.nextafter(rd.float() * (1 + 2**-16), torch.full_like(rd.float(), torch.inf))
        if not bool((ru.double() >= rd).all()) or not torch.isfinite(ru).all() or float(ru.max()) >= 2**16:
            raise ValueError('Q5 head residual bound is not finite/conservative')
        code = (quant + offset).view(-1, 2048)
        low[start:end] = ((code[:, 0::2] & 15) | ((code[:, 1::2] & 15) << 4)).to(torch.uint8)
        if bits == 5:
            c8 = code.view(-1, 256, 8); high_bits = torch.zeros(c8.shape[:2], dtype=torch.int32)
            for bit in range(8): high_bits |= ((c8[:, :, bit] >> 4) & 1) << bit
            high[start:end] = high_bits.to(torch.uint8)
        scales[start:end] = scale; rho[start:end] = ru
    return low, high, scales, rho


class HeadContext:
    def __init__(self, manager, prepared):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Warm Q5 head before graph capture')
        assert prepared.weight.shape == (248320, 2048) and prepared.weight.dtype == torch.bfloat16
        self.manager = manager; self.prepared = prepared; self.n = 248320; self.capacity = 4096
        self.seeded_q4 = os.environ.get("CODEX_AR1594_HEAD") == "1"
        if self.seeded_q4: self.capacity = 32768
        self.float_output = os.environ.get('CODEX_AR1609_OUTPUT') == '1'
        if self.float_output and not self.seeded_q4:
            raise ValueError('Direct FP32 output requires the seeded Q4 head')
        self.bound_output_calls = self.allocated_output_calls = self.bound_capture_calls = 0
        device = prepared.weight.device
        affinity, threads = os.sched_getaffinity(0), torch.get_num_threads()
        try:
            os.sched_setaffinity(0, set(range(2, 15))); torch.set_num_threads(8)
            cpu = pack_layout(prepared.weight.cpu(), bits=4 if self.seeded_q4 else 5)
            self.low, self.high, self.scale, self.rho = [v.to(device) for v in cpu]
        finally:
            torch.set_num_threads(threads); os.sched_setaffinity(0, affinity)
        self.lower = torch.empty(self.n, device=device, dtype=torch.float32)
        self.upper = torch.empty_like(self.lower)
        self.blockmax = torch.empty(self.n // 8, device=device, dtype=torch.float32)
        self.threshold = torch.empty(1, device=device, dtype=torch.float32)
        self.norm = torch.empty_like(self.threshold)
        self.control = torch.zeros(18 if self.seeded_q4 else 16, device=device, dtype=torch.int32)
        self.candidates = torch.empty(self.capacity, device=device, dtype=torch.int32)
        self.eligible = torch.zeros(1, device=device, dtype=torch.int32)
        self.buffers = [self.low, self.high, self.scale, self.rho, self.lower, self.upper,
                        self.blockmax, self.threshold, self.norm, self.control, self.candidates, self.eligible]
        self.packed_word_decode = os.environ.get('CODEX_AR660_HEAD') == '1'
        self.threshold_vector16 = os.environ.get('CODEX_AR977_THRESHOLD') == '1'
        if self.threshold_vector16 and not self.packed_word_decode:
            raise ValueError('Vector Threshold requires CODEX_AR660_HEAD=1')
        if self.seeded_q4 and not (self.threshold_vector16 and self.packed_word_decode):
            raise ValueError('Seeded Q4 head requires the existing packed-word/vector-threshold route')
        prefix = 'head1607' if self.float_output else 'head1592' if self.seeded_q4 else ('head938' if self.threshold_vector16 else ('head655' if self.packed_word_decode else 'head449'))
        self.library = f'native/{prefix}/head.so'
        if self.packed_word_decode:
            import hashlib
            self.library_sha256 = hashlib.sha256((R / self.library).read_bytes()).hexdigest()
            expected_sha = '5c75fd88da375081fb7cf4dc0eb51799d89c8152a316021827ebd44a9e10756b' if self.threshold_vector16 else 'd94b047c3a67dba221a233589d6cd197d7ff6f24237f5c9c6183337182f9dd17'
            if self.seeded_q4: expected_sha = '49a33f20f33fc6fc6ef68b814af6f881931c54576ac7a247f90493e89068e242'
            if self.float_output: expected_sha = '117a8172d54e0d85908db4931e063830763c4abd510a41e6d2996e53e737fca7'
            assert self.library_sha256 == expected_sha
        else:
            self.library_sha256 = None
        self.lib = ctypes.CDLL(str(R / self.library))
        abi = getattr(self.lib, prefix + '_abi_size'); abi.restype = ctypes.c_int
        words = getattr(self.lib, prefix + '_control_words'); words.restype = ctypes.c_int
        self.native_error = getattr(self.lib, prefix + '_error'); self.native_error.restype = ctypes.c_char_p
        self.native_run = getattr(self.lib, prefix + '_run')
        self.native_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(Args)] + ([ctypes.c_int] if self.threshold_vector16 else [])
        self.native_run.restype = ctypes.c_int
        assert abi() == ctypes.sizeof(Args) == 144
        assert words() == self.control.numel() == (18 if self.seeded_q4 else 16)
        self.replay_reasons = {}; self.replay_prepares = 0
        logger.info(('AR1594 bounded Q4 head enabled: five captured SYCL nodes, Approx GRF128, original BF16 exact rows; capacity32768, dynamic GPU eligibility, full fallback; explicit seed-row counter; no CPU head; decoder=%s library=%s threshold_mode=%d threshold_vector_width=%d' if self.seeded_q4 else 'AR449 bounded Q5 head enabled: five captured SYCL nodes, Approx GRF128, original BF16 exact rows; capacity4096, dynamic GPU eligibility, full fallback; no CPU head; decoder=%s library=%s threshold_mode=%d threshold_vector_width=%d'),
                    'packed-word' if self.packed_word_decode else 'scalar', self.library,
                    6 if self.threshold_vector16 else 0, 16 if self.threshold_vector16 else 1)

    def run(self, hidden, output=None):
        # Eager paths lack a live graph-replay batch hook. Keep full logits.
        if not torch.xpu.is_current_stream_capturing():
            self.eligible.zero_()
        capturing = torch.xpu.is_current_stream_capturing()
        bound = self.float_output and valid_output(output, hidden, self)
        if bound:
            out = output
            self.bound_output_calls += 1
            self.bound_capture_calls += int(capturing)
        else:
            out = torch.empty((1, self.n), device=hidden.device,
                              dtype=torch.float32 if self.float_output else torch.bfloat16)
            self.allocated_output_calls += 1
        args = Args(hidden.data_ptr(), self.prepared.weight.data_ptr(), self.low.data_ptr(), self.high.data_ptr(),
                    self.scale.data_ptr(), self.rho.data_ptr(), self.lower.data_ptr(), self.upper.data_ptr(),
                    self.blockmax.data_ptr(), self.threshold.data_ptr(), self.norm.data_ptr(), self.control.data_ptr(),
                    self.candidates.data_ptr(), out.data_ptr(), None, self.eligible.data_ptr(), self.n, self.capacity, 256)
        stream = torch.xpu.current_stream(hidden.device)
        if self.threshold_vector16:
            rc = self.native_run(stream.sycl_queue, ctypes.byref(args), 1 if self.seeded_q4 else 6)
        else:
            rc = self.native_run(stream.sycl_queue, ctypes.byref(args))
        if rc: raise RuntimeError((rc, self.native_error()))
        for v in [hidden, out, self.prepared.weight, *self.buffers]: v.record_stream(stream)
        return out

    def readback(self):
        values = self.control.cpu()
        work = values[4:16].view(torch.int64).tolist()
        seed_rows = int(values[16:].view(torch.int64)[0]) if self.seeded_q4 else 0
        assert 0 <= seed_rows <= work[0] - work[3]
        assert min(work) >= 0 and work[2] == work[3] + work[5] and work[2] <= work[0], work
        return dict(enabled=True, kind='q4-seeded-bounded-exact' if self.seeded_q4 else 'q5-bounded-exact', CPU_prefix_rows=0, GPU_tail_rows=self.n,
                    shared_pool=False, service_cpu=None, capacity=self.capacity,
                    float_output=self.float_output, bound_output_calls=self.bound_output_calls,
                    allocated_output_calls=self.allocated_output_calls, bound_capture_calls=self.bound_capture_calls,
                    output_rounding='original BF16 RNE then FP32 widening' if self.float_output else 'original BF16 RNE',
                    metrics=work, seed_rows=seed_rows, total_BF16_row_evaluations=work[1]+seed_rows, filter_bits=4 if self.seeded_q4 else 5, metric_names=['calls', 'exact_rows', 'full_fallbacks', 'unsafe_or_ineligible', 'proposed_candidates', 'overflows'],
                    final_control=values[:4].tolist(), replay_prepares=self.replay_prepares,
                    replay_reasons=dict(self.replay_reasons), library=self.library, library_sha256=self.library_sha256, packed_word_decode=self.packed_word_decode,
                    threshold_mode=6 if self.threshold_vector16 else 0,
                    threshold_vector_width=16 if self.threshold_vector16 else 1,
                    static_filter_bytes=sum(v.numel() * v.element_size() for v in [self.low, self.high, self.scale, self.rho]),
                    static_Q5_bytes=0 if self.seeded_q4 else sum(v.numel() * v.element_size() for v in [self.low, self.high, self.scale, self.rho]))

    def close(self):
        torch.xpu.synchronize()
        return int(self.control[4:16].view(torch.int64)[0].item())


def prepare_replay(fb):
    if not enabled() or LAST is None:
        return
    allowed, reason = eligibility(fb)
    LAST.eligible.fill_(int(allowed))
    LAST.replay_prepares += 1
    LAST.replay_reasons[reason] = LAST.replay_reasons.get(reason, 0) + 1


def try_run(prepared, hidden, output=None):
    global LAST
    if not enabled(): return None
    from sglang.srt.layers.moe.paged_experts import cpu40
    manager = cpu40.MANAGER
    if manager is None or not manager.head_enabled: return None
    manager.start()
    if manager.head_context is None: manager.head_context = HeadContext(manager, prepared)
    assert isinstance(manager.head_context, HeadContext) and manager.head_context.prepared is prepared
    LAST = manager.head_context
    return LAST.run(hidden, output=output)


def valid_output(output, hidden, context):
    """Accept only a separate contiguous graph-owned FP32 destination."""
    if not (isinstance(output, torch.Tensor) and output.dtype == torch.float32
            and output.device == hidden.device and output.shape == (1, context.n)
            and output.is_contiguous() and output.data_ptr() % 4 == 0):
        return False
    start = output.data_ptr()
    end = start + output.numel() * output.element_size()
    for tensor in [hidden, context.prepared.weight, *context.buffers]:
        other = tensor.data_ptr()
        if start < other + tensor.numel() * tensor.element_size() and other < end:
            return False
    return True
