"""Opt-in B70 BF16 top64 rescore/top16 for serialized DFlash M3/M7 calls.

Create one state per candidate head before capture. Counters reset in the last
work-group, so replay needs no zeroing kernel. Concurrent uses of one state on
independent queues are unsupported. Restart/recapture after replacing weights.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_library(path):
    library = ctypes.CDLL(path)
    try:
        library.dflash_draft_rerank_abi.argtypes = []
        library.dflash_draft_rerank_abi.restype = ctypes.c_int
        if library.dflash_draft_rerank_abi() != 1:
            raise RuntimeError("Unsupported SYCL draft rerank ABI")
        library.dflash_draft_rerank_error.restype = ctypes.c_char_p
        library.dflash_draft_rerank_run.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int
        ]
        library.dflash_draft_rerank_run.restype = ctypes.c_int
    except AttributeError as exc:
        raise RuntimeError(
            "Rebuild SYCL draft rerank library: missing ABI symbols"
        ) from exc
    return library


def _valid_source(source):
    return (
        source.device.type == "xpu"
        and source.dtype == torch.bfloat16
        and tuple(source.shape) == (248320, 2048)
        and source.is_contiguous()
        and torch.xpu.get_device_properties(source.device).device_id == 0xE223
    )


def can_use_rerank(hidden, source, ids):
    return (
        not torch.compiler.is_compiling()
        and _valid_source(source)
        and hidden.device == ids.device == source.device
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[0] in (3, 7)
        and hidden.shape[1] == 2048
        and hidden.is_contiguous()
        and ids.dtype == torch.int64
        and tuple(ids.shape) == (hidden.shape[0], 64)
        and ids.is_contiguous()
        and not (torch.is_grad_enabled() and hidden.requires_grad)
    )


@lru_cache(maxsize=2)
def _announce(rows):
    logger.info(
        "Using SYCL B70 DFlash draft BF16 shortlist rerank M=%s/top64->16; "
        "one kernel, preserved reduction/tie order, persistent self-reset counters",
        rows,
    )


class DraftRerankState:
    def __init__(self, source):
        if not _valid_source(source):
            raise ValueError("SYCL draft rerank requires contiguous B70 BF16 head")
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Prepare SYCL draft rerank state before graph capture")
        path = envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_LIB.get()
        if not path:
            raise RuntimeError(
                "SYCL draft rerank requires SGLANG_XPU_DFLASH_DRAFT_RERANK_LIB; "
                "build with test/manual/kernels/build_xpu_dflash_draft_rerank.py"
            )
        self.library = _load_library(path)
        self.source_pointer = source.data_ptr()
        self.source_version = source._version
        device = source.device
        self.scores = torch.empty((7, 64), device=device, dtype=torch.bfloat16)
        self.values = torch.empty((7, 16), device=device, dtype=torch.bfloat16)
        self.ids = torch.empty((7, 16), device=device, dtype=torch.int64)
        # One cache line per token. Only its first word is used by the ticket.
        self.counters = torch.zeros((7, 16), device=device, dtype=torch.int32)

    def run(self, hidden, source, ids):
        if not (
            can_use_rerank(hidden, source, ids)
            and source.data_ptr() == self.source_pointer
            and source._version == self.source_version
        ):
            raise ValueError("SYCL draft rerank layout/source changed; prepare again")
        rows = hidden.shape[0]
        rc = self.library.dflash_draft_rerank_run(
            torch.xpu.current_stream().sycl_queue,
            hidden.data_ptr(),
            source.data_ptr(),
            ids.data_ptr(),
            self.scores.data_ptr(),
            self.values.data_ptr(),
            self.ids.data_ptr(),
            self.counters.data_ptr(),
            rows,
        )
        if rc:
            raise RuntimeError(self.library.dflash_draft_rerank_error().decode())
        _announce(rows)
        return self.values[:rows], self.ids[:rows]
