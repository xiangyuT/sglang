"""Opt-in B1 ordinary FutureMap publication on the existing forward stream.

This fuses length increment/publication and token stash. Scheduler admission,
allocation, sampling, result processing and speculative relays keep their owners.
"""
import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from sglang.srt.layers import ar188_metadata as metadata

ENABLED = os.environ.get("CODEX_AR1564_RELAY") == "1"
FEED_ENABLED = os.environ.get("CODEX_AR1569_INPUT") == "1"
LIBRARY = "/evidence/tpot-iterations-20261023/native/relay1568/relay.so"
PIN = "35547686b0c6a127bdc467758264e2ea27a761f56f70fea213713d10fdd5b23d"

@lru_cache(maxsize=1)
def library():
    assert not torch.xpu.is_current_stream_capturing()
    assert hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest() == PIN
    lib = ctypes.CDLL(LIBRARY)
    lib.relay1568_run.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
    lib.relay1568_run.restype = ctypes.c_int
    lib.relay1568_error.restype = ctypes.c_char_p
    return lib


def _vector(tensor, size, dtypes, device=None):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "xpu"
        and tensor.shape == (size,)
        and tensor.dtype in dtypes
        and tensor.is_contiguous()
        and (device is None or tensor.device == device)
    )


class RelayContext:
    def __init__(self, future_map, scheduler):
        self.lengths = future_map.new_seq_lens_buf
        self.tokens = future_map.output_tokens_buf
        self.device = self.lengths.device
        self.capacity = self.lengths.numel()
        assert torch.xpu.get_device_properties(self.device).device_id == 0xE223
        self.lib = library()
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.calls = 0
        self.int32_calls = 0
        self.feed_calls = 0
        self.borrow_calls = 0
        self.feed_receipt = None
        self.feedback = None
        self.runner = None
        self.profiler = getattr(scheduler, "profiler_manager", None)
        args = getattr(scheduler, "server_args", None)
        self.consumer_safe = args is not None and not any(
            getattr(args, field, None) for field in (
                "debug_tensor_dump_output_folder", "forward_hooks",
                "enable_return_hidden_states", "enable_return_routed_experts",
                "enable_return_indexer_topk", "enable_torch_compile",
                "enable_torch_compile_debug_mode"))
        if FEED_ENABLED:
            runner = metadata.RUNNER
            candidate = None if runner is None else runner.buffers.input_ids
            if (metadata.MODE == "ingraph"
                    and _vector(candidate, 1, (torch.int64,), self.device)):
                self.runner = runner
                self.feedback = candidate

    def issue(self, indices, lengths, token_ids, batch):
        stream = torch.xpu.current_stream(self.device)
        tensors = (indices, lengths, token_ids, self.lengths, self.tokens, self.errors)
        feed = _can_feed(self, batch)
        rc = self.lib.relay1568_run(
            stream.sycl_queue,
            *(tensor.data_ptr() for tensor in tensors),
            self.feedback.data_ptr() if feed else None,
            self.capacity,
            int(token_ids.dtype == torch.int32),
            int(feed),
        )
        if rc:
            raise RuntimeError((rc, self.lib.relay1568_error().decode()))
        for tensor in tensors:
            tensor.record_stream(stream)
        self.calls += 1
        self.int32_calls += int(token_ids.dtype == torch.int32)
        self.feed_receipt = None
        if feed:
            self.feedback.record_stream(stream)
            self.feed_receipt = _receipt(batch)
            self.feed_calls += 1


def try_relay(scheduler, batch, result, indices):
    if not ENABLED:
        return False
    from sglang.srt.managers.overlap_utils import _DEBUG_ASSERT
    if (
        _DEBUG_ASSERT
        or not scheduler.enable_overlap
        or not batch.forward_mode.is_decode()
        or not batch.spec_algorithm.is_none()
        or len(batch.reqs) != 1
        or batch.beam_tail is not None
        or scheduler.beam_coordinator._num_live_groups != 0
        or result.delay_sample_func is not None
        or result.next_draft_input is not None
    ):
        return False
    future_map = scheduler.future_map
    if (
        not future_map.spec_algo.is_none()
        or future_map.needs_confidence_relay
        or future_map.need_topk
        or future_map.need_hidden_states
        or future_map.draft_probs_buf is not None
        or future_map.dsa_topk_indices_buf is not None
    ):
        return False
    lengths, token_ids = batch.seq_lens, result.next_token_ids
    if not (
        _vector(indices, 1, (torch.int64,))
        and _vector(lengths, 1, (torch.int64,), indices.device)
        and _vector(token_ids, 1, (torch.int32, torch.int64), indices.device)
        and _vector(future_map.new_seq_lens_buf, future_map.req_pool_size,
                    (torch.int64,), indices.device)
        and _vector(future_map.output_tokens_buf, future_map.req_pool_size,
                    (torch.int64,), indices.device)
    ):
        return False
    if result.ar1564_relayed:
        raise RuntimeError("AR1564 result relay already submitted")
    context = getattr(future_map, "_ar1564_context", None)
    if context is None:
        context = RelayContext(future_map, scheduler)
        future_map._ar1564_context = context
    if context.lengths is not future_map.new_seq_lens_buf or context.tokens is not future_map.output_tokens_buf:
        return False
    context.issue(indices, lengths, token_ids, batch)
    result.ar1564_relayed = True
    return True


def readback(future_map):
    context = getattr(future_map, "_ar1564_context", None)
    errors = 0 if context is None else int(context.errors.cpu()[0])
    assert errors == 0, ("AR1564 invalid FutureMap row", errors)
    return dict(enabled=ENABLED, calls=0 if context is None else context.calls,
                int32_calls=0 if context is None else context.int32_calls,
                errors=errors, library=LIBRARY, library_sha256=PIN,
                forward_stream=True, host_readback_in_hot_path=False,
                speculative_relay_unchanged=True,
                feedback_enabled=FEED_ENABLED,
                captured_input_bound=context is not None and context.feedback is not None,
                feed_calls=0 if context is None else context.feed_calls,
                borrow_calls=0 if context is None else context.borrow_calls)


def _receipt(batch):
    req = batch.reqs[0]
    return (id(req), req.cache_request_handle, req.kv.req_pool_idx, int(batch.forward_iter))


def _can_feed(context, batch):
    if not FEED_ENABLED or context.feedback is None or not context.consumer_safe:
        return False
    profiler = context.profiler
    return (
        metadata.RUNNER is context.runner
        and context.runner.buffers.input_ids is context.feedback
        and batch.forward_mode.is_decode()
        and batch.spec_algorithm.is_none()
        and len(batch.reqs) == 1
        and batch.reqs[0].kv.holds_kv
        and 0 <= batch.reqs[0].kv.req_pool_idx < context.capacity
        and not batch.return_logprob
        and not batch.return_hidden_states
        and not batch.has_grammar
        and batch.sampling_info is not None
        # copy_for_forward deliberately drops the CPU orchestrator after
        # materializing penalties. Inspect both representations of ownership.
        and (batch.sampling_info.penalizer_orchestrator is None
             or not batch.sampling_info.penalizer_orchestrator.is_required)
        and batch.sampling_info.acc_additive_penalties is None
        and batch.sampling_info.acc_scaling_penalties is None
        and batch.input_embeds is None
        and batch.replace_embeds is None
        and batch.replace_positions is None
        and batch.forward_iter > 0
        and (profiler is None or not profiler.profile_in_progress)
    )


def borrow_input(batch, future_map):
    """Consume one consecutive same-request feed on the forward stream.

    The preceding publication is queued after its sampled token and model.
    The next model reads this graph-owned buffer before its own publication
    can overwrite it. CPU result/diagnostic consumers and penalties decline.
    A new request, an intervening forward, recapture or a second consume falls
    back to the original FutureMap gather.
    """
    context = getattr(future_map, "_ar1564_context", None)
    if context is None or not _can_feed(context, batch):
        return None
    previous = context.feed_receipt
    current = _receipt(batch)
    if (previous is None or previous[:3] != current[:3]
            or previous[3] + 1 != current[3]
            or context.lengths is not future_map.new_seq_lens_buf
            or context.tokens is not future_map.output_tokens_buf):
        return None
    context.feed_receipt = None
    context.feedback.record_stream(torch.xpu.current_stream(context.device))
    context.borrow_calls += 1
    return context.feedback
