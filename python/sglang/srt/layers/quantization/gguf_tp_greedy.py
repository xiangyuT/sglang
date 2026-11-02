# SPDX-License-Identifier: Apache-2.0
"""Compact exact greedy target exchange for opted-in GGUF TP/EP verify graphs.

The head computes every original score. Exchange only per-rank maxima and IDs,
then supply a greedy-only score carrier. Graph replay eligibility is checked
against live CPU sampling metadata; other requests retain complete eager logits.
"""
import os
import ctypes
from functools import lru_cache
import torch
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel


@lru_cache(None)
def _native(path):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm GGUF greedy exchange before XPU graph capture")
    lib = ctypes.CDLL(path)
    lib.xpu_gguf_greedy_error.restype = ctypes.c_char_p
    lib.xpu_gguf_greedy_prepare.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    lib.xpu_gguf_greedy_finish.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    return lib


def _native_exchange(peer, logits, *, carrier):
    rows, width = logits.shape
    if not (logits.device.type == "xpu" and logits.dtype == torch.float32
            and logits.is_contiguous() and 1 <= rows <= 32
            and 1 <= width <= 248320 and peer.world in (2, 4)
            and 0 <= peer.rank < peer.world):
        raise ValueError("Unsupported GGUF greedy exchange geometry")
    lib = _native(os.environ["SGLANG_XPU_GGUF_TP_GREEDY_LIB"])
    stream = torch.xpu.current_stream(logits.device)
    partial = torch.empty((rows, (width + 1023) // 1024), dtype=torch.int64, device=logits.device)
    packet = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
    received = torch.empty((peer.world, rows, 2), dtype=torch.float32, device=logits.device)
    winners = torch.empty(rows, dtype=torch.int64, device=logits.device)
    output = torch.empty((rows, width * peer.world), dtype=logits.dtype, device=logits.device) if carrier else None
    rc = lib.xpu_gguf_greedy_prepare(stream.sycl_queue, logits.data_ptr(), partial.data_ptr(),
                                   packet.data_ptr(), rows, width, peer.rank, peer.world)
    if rc:
        raise RuntimeError(lib.xpu_gguf_greedy_error().decode())
    peer.all_gather(received, packet)
    rc = lib.xpu_gguf_greedy_finish(stream.sycl_queue, received.data_ptr(), winners.data_ptr(),
                                  output.data_ptr() if output is not None else None,
                                  rows, width * peer.world, peer.world)
    if rc:
        raise RuntimeError(lib.xpu_gguf_greedy_error().decode())
    for tensor in (logits, partial, packet, received, winners, output):
        if tensor is not None:
            tensor.record_stream(stream)
    return output if carrier else winners


def compact_argmax(peer, logits):
    if os.environ.get("SGLANG_XPU_GGUF_TP_GREEDY_LIB"):
        return _native_exchange(peer, logits, carrier=False)
    rows, width = logits.shape
    index = logits.argmax(dim=-1, keepdim=True)
    value = logits.gather(1, index)
    packet = torch.cat([value, (index + peer.rank * width).float()], dim=-1)
    received = torch.empty((peer.world, rows, 2), device=logits.device, dtype=torch.float32)
    peer.all_gather(received, packet)
    owner = received[:, :, 0].argmax(dim=0, keepdim=True)
    return received[:, :, 1].gather(0, owner).squeeze(0).long()


def try_gather(processor, logits, head, metadata, bias):
    if not (os.environ.get("SGLANG_XPU_GGUF_TP_EP") == "1"
            and logits.device.type == "xpu" and logits.dtype == torch.float32
            and logits.ndim == 2 and logits.shape[0] in range(4, 33, 4)
            and metadata is not None and metadata.forward_mode.is_target_verify()
            and processor.do_tensor_parallel_all_gather
            and not processor.use_attn_tp_group
            and not processor.do_tensor_parallel_all_gather_dp_attn
            and processor.logit_scale is None and not processor.final_logit_softcapping
            and not processor.use_fp32_lm_head and processor.rl_on_policy_target is None
            and bias is None
            and type(getattr(head, "quant_method", None)).__name__ in
                ("CompressedIQ4HeadMethod", "CompressedGGUFHeadMethod")
            and not hasattr(head, "set_lora") and not hasattr(head, "apply_lora")
            and not envs.SGLANG_SANITIZE_NAN_LOGITS.get()
            and not envs.SGLANG_ENABLE_NAN_LOGITS_CHECK.get()
            and not envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
            and torch.xpu.get_device_properties(logits.device).device_id == 0xE223):
        return None
    parallel = get_parallel()
    if not (parallel.tp_size == parallel.moe_ep_size and parallel.tp_size in (2, 4)
            and parallel.pp_size == parallel.attn_cp_size == parallel.attn_dp_size == 1
            and processor.vocab_size == logits.shape[1] * parallel.tp_size == 248320):
        return None
    group = parallel.tp_group
    communicator = getattr(group, "xpu_communicator", None)
    peer = getattr(communicator, "peer_allreduce", None)
    if peer is None or not peer.byte_allgather:
        return None
    native = os.environ.get("SGLANG_XPU_GGUF_TP_GREEDY_LIB")
    if native:
        _native(native)
    if not torch.xpu.is_current_stream_capturing():
        return None
    if native:
        return _native_exchange(peer, logits, carrier=True)
    winners = compact_argmax(peer, logits)
    result = torch.full((logits.shape[0], processor.vocab_size), -float("inf"), device=logits.device, dtype=logits.dtype)
    result.scatter_(1, winners[:, None], 1.0)
    return result
