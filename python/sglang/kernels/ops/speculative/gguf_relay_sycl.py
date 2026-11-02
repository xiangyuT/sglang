# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact FutureMap relay for a single B70 GGUF NEXTN request."""
import ctypes
import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


def announce_private_d2h():
    logger.info("GGUF B70 publish-fenced private D2H stream enabled for sequence lengths")


def private_d2h_eligible(device, algorithm, rows):
    device = torch.device(device)
    return (os.environ.get("SGLANG_XPU_GGUF_RELAY_SYCL") == "1"
            and device.type == "xpu" and rows == 2 and algorithm.is_eagle()
            and torch.xpu.get_device_properties(device).device_id == 0xE223)


@lru_cache(maxsize=1)
def _library():
    path = os.environ.get("SGLANG_XPU_GGUF_RELAY_LIB")
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Set and warm SGLANG_XPU_GGUF_RELAY_LIB before capture")
    library = ctypes.CDLL(path)
    library.xpu_gguf_relay.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4
    library.xpu_gguf_relay.restype = ctypes.c_int
    library.xpu_gguf_relay_error.restype = ctypes.c_char_p
    return library


def _tensor(value, shape, dtype, device):
    device = torch.device(device)
    if device.type == "xpu" and device.index is None:
        device = torch.device("xpu", torch.xpu.current_device())
    return (isinstance(value, torch.Tensor) and value.device == device
            and value.shape == shape and value.dtype == dtype and value.is_contiguous())


def _eligible(owner, indices):
    if not (private_d2h_eligible(owner.device, owner.spec_algo, owner.req_pool_size)
            and owner.need_topk and owner.need_hidden_states
            and owner.draft_probs_buf is None and owner.dsa_topk_indices_buf is None
            and _tensor(indices, (1,), indices.dtype, owner.device)
            and indices.dtype in (torch.int32, torch.int64)):
        return False
    return all(_tensor(t, shape, dtype, owner.device) for t, shape, dtype in (
        (owner.output_tokens_buf, (2,), torch.int64),
        (owner.topk_p_buf, (2, 1), torch.float32),
        (owner.topk_index_buf, (2, 1), torch.int64),
        (owner.hidden_states_buf, (2, 10240), torch.bfloat16),
    )) and owner.hidden_states_buf.data_ptr() % 16 == 0


@lru_cache(maxsize=2)
def _announce(gather):
    logger.info("GGUF B70 single-request SYCL relay %s enabled", "gather" if gather else "stash")


def _submit(owner, indices, token, probability, choice, hidden, gather):
    stream = torch.xpu.current_stream(owner.device)
    library = _library()
    values = (indices, owner.output_tokens_buf, token, owner.topk_p_buf, probability,
              owner.topk_index_buf, choice, owner.hidden_states_buf, hidden)
    flags = (int(token.dtype == torch.int64) | (int(choice.dtype == torch.int64) << 1)
             | (int(indices.dtype == torch.int64) << 2))
    rc = library.xpu_gguf_relay(stream.sycl_queue, *[v.data_ptr() for v in values],
                               2, 10240, int(gather), flags)
    if rc:
        raise RuntimeError((rc, library.xpu_gguf_relay_error().decode()))
    for value in values:
        value.record_stream(stream)
    _announce(gather)


def try_stash(owner, indices, payload):
    if not _eligible(owner, indices):
        return False
    token, probability, choice, hidden = (payload.bonus_tokens, payload.topk_p,
                                         payload.topk_index, payload.hidden_states)
    if not (payload.draft_probs is None and payload.dsa_topk_indices is None
            and isinstance(token, torch.Tensor) and token.dtype in (torch.int32, torch.int64)
            and isinstance(choice, torch.Tensor) and choice.dtype in (torch.int32, torch.int64)
            and _tensor(token, (1,), token.dtype, owner.device)
            and _tensor(probability, (1, 1), torch.float32, owner.device)
            and _tensor(choice, (1, 1), choice.dtype, owner.device)
            and _tensor(hidden, (1, 10240), torch.bfloat16, owner.device)
            and hidden.data_ptr() % 16 == 0):
        return False
    _submit(owner, indices, token, probability, choice, hidden, False)
    return True


def try_gather(owner, indices):
    if not _eligible(owner, indices):
        return None
    token = torch.empty((1,), dtype=torch.int64, device=owner.device)
    probability = torch.empty((1, 1), dtype=torch.float32, device=owner.device)
    choice = torch.empty((1, 1), dtype=torch.int64, device=owner.device)
    hidden = torch.empty((1, 10240), dtype=torch.bfloat16, device=owner.device)
    _submit(owner, indices, token, probability, choice, hidden, True)
    return probability, choice, token, hidden
