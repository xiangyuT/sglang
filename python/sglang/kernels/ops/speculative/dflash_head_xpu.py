"""Private FP8 proposal copy of the BF16 head for B70 DFlash2.

Build before graph capture. Target parameters remain BF16; experimental target
screening is controlled separately. Draft logits and acceptance can change.
Restart and recapture after changing source weights or feature flags.
"""

import logging
import weakref

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.environ import envs

from sglang.kernels.ops.speculative.dflash_topk_xpu import (
    _ordered,
    _value,
    xpu_dflash_topk,
)


@triton.jit
def _quantize_head(W, Q, S, N: tl.constexpr, K: tl.constexpr):
    r = tl.program_id(0) * 128 + tl.arange(0, 128)
    c = tl.program_id(1) * 128 + tl.arange(0, 128)
    v = tl.load(W + r[:, None] * K + c[None, :]).to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.max(tl.abs(v), 1), 0), 1e-12), 448.0)
    tl.store(S + tl.program_id(0) * (K // 128) + tl.program_id(1), scale)
    tl.store(Q + r[:, None] * K + c[None, :], tl.div_rn(v, scale).to(tl.float8e4nv))


@triton.jit
def _rescore_shortlist(H, W, Indices, R, K: tl.constexpr):
    token = tl.program_id(0)
    row = tl.program_id(1) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, K)
    ids = tl.load(Indices + token * 64 + row)
    w = tl.load(W + ids[:, None] * K + cols[None, :]).to(tl.float32)
    h = tl.load(H + token * K + cols).to(tl.float32)
    tl.store(R + token * 64 + row, tl.sum(w * h[None, :], 1).to(tl.bfloat16))


@triton.jit
def _select_shortlist(R, Indices, V, Out):
    token = tl.program_id(0)
    row = tl.arange(0, 64)
    score = tl.load(R + token * 64 + row)
    ids = tl.load(Indices + token * 64 + row)
    # Preserve the full-head tie rule using global vocabulary IDs, rather
    # than the position assigned by the approximate FP8 shortlist.
    key = (_ordered(score).to(tl.uint64) << 32) | (0xFFFFFFFF - ids).to(tl.uint64)
    best = tl.topk(key, 16)
    tl.store(V + token * 16 + tl.arange(0, 16), _value((best >> 32).to(tl.uint32)))
    tl.store(
        Out + token * 16 + tl.arange(0, 16),
        0xFFFFFFFF - (best & 0xFFFFFFFF).to(tl.int64),
    )


_shared_candidate_heads = weakref.WeakValueDictionary()
_logger = logging.getLogger(__name__)


def shared_candidate_head(weight):
    """Share one derived copy between the opt-in target and draft runtimes."""
    key = (id(weight), weight.data_ptr(), weight._version)
    candidate = _shared_candidate_heads.get(key)
    if candidate is None:
        candidate = XpuDflashCandidateHead(weight)
        _shared_candidate_heads[key] = candidate
    else:
        _logger.info("Reusing B70 DFlash FP8 candidate for the shared BF16 head")
    return candidate


class XpuDflashCandidateHead(nn.Module):
    """Unshared block-128 weight and scales; no target parameter mutation."""

    def __init__(self, weight):
        super().__init__()
        n, k = weight.shape
        self.source_pointer = weight.data_ptr()
        self.source_version = weight._version
        self._sycl_rerank = None
        self.register_buffer(
            "weight", torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer(
            "scale",
            torch.empty(
                (n // 128, k // 128), device=weight.device, dtype=torch.float32
            ),
        )
        _quantize_head[(n // 128, k // 128)](
            weight, self.weight, self.scale, N=n, K=k, num_warps=8
        )

    def matches(self, hidden, source):
        return (
            hidden.ndim == 2
            and 1 <= hidden.shape[0] <= 8
            and hidden.shape[1] == self.weight.shape[1]
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
            and hidden.device == self.weight.device
            and source.data_ptr() == self.source_pointer
            and source._version == self.source_version
        )

    def forward(self, hidden):
        from sglang.srt.layers.quantization.fp8_utils import torch_w8a8_block_fp8_linear

        return torch_w8a8_block_fp8_linear(hidden, self.weight, [128, 128], self.scale)

    def prepare_sycl_rerank(self, source):
        from sglang.kernels.ops.speculative.dflash_draft_rerank_sycl import (
            DraftRerankState,
        )

        self._sycl_rerank = DraftRerankState(source)

    def refined_top16(self, hidden, source):
        """FP8 top-64, then original BF16 scores. Recall remains approximate."""
        _, ids = xpu_dflash_topk(self(hidden), 64)
        if envs.SGLANG_XPU_DFLASH_DRAFT_RERANK_SYCL.get():
            from sglang.kernels.ops.speculative.dflash_draft_rerank_sycl import (
                can_use_rerank,
            )

            if can_use_rerank(hidden, source, ids):
                if self._sycl_rerank is None:
                    raise RuntimeError("Prepare SYCL draft rerank before capture")
                return self._sycl_rerank.run(hidden, source, ids)
        scores = torch.empty(ids.shape, device=hidden.device, dtype=torch.bfloat16)
        _rescore_shortlist[(hidden.shape[0], 16)](
            hidden,
            source,
            ids,
            scores,
            K=hidden.shape[1],
            num_warps=4,
            grf_mode="256",
        )
        vals = torch.empty(
            (hidden.shape[0], 16), device=hidden.device, dtype=torch.bfloat16
        )
        out = torch.empty(
            (hidden.shape[0], 16), device=hidden.device, dtype=torch.int64
        )
        _select_shortlist[(hidden.shape[0],)](scores, ids, vals, out, num_warps=4)
        return vals, out


def can_prepare_candidate_head(head, tp_size):
    weight = getattr(head, "weight", None)
    return (
        tp_size == 1
        and isinstance(weight, torch.Tensor)
        and weight.device.type == "xpu"
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (248320, 2048)
        and weight.is_contiguous()
        and int(getattr(head, "org_vocab_size", 0)) == 248320
        and torch.xpu.get_device_properties(weight.device).device_id == 0xE223
    )
