"""Experimental B70 target head with device-side BF16 completion.

The FP8 exclusion envelope assumes the tested native cast model; it is not an
arbitrary-input lossless proof. Complete BF16 rows use a different FP32 reduction
from Torch. Enable explicitly and restart/recapture after weight or flag changes.
Requests requiring original logits must finish through the DFlash worker guard.
"""

import ctypes
import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.speculative.dflash_head_xpu import (
    can_prepare_candidate_head,
    shared_candidate_head,
)
from sglang.kernels.ops.speculative.dflash_native import (
    native_window,
    require_native_window,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_target_head():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError("SYCL target head requires SGLANG_XPU_DFLASH_NATIVE_LIB")
    library = ctypes.CDLL(path)
    try:
        library.dflash_head_abi.argtypes = []
        library.dflash_head_abi.restype = ctypes.c_int
        if library.dflash_head_abi() != 1:
            raise RuntimeError("Unsupported SYCL target head ABI")
        for name, count in (
            ("dflash_head_rescore", 6),
            ("dflash_head_finish", 13),
            ("dflash_head_fill", 3),
            ("dflash_head_complete", 7),
        ):
            fn = getattr(library, name)
            fn.argtypes = [ctypes.c_void_p] * count
            fn.restype = ctypes.c_int
        for name in ("dflash_head_error", "dflash_head_rescore_error"):
            fn = getattr(library, name)
            fn.argtypes = []
            fn.restype = ctypes.c_char_p
    except AttributeError as exc:
        raise RuntimeError(
            "SYCL target head symbols missing; rebuild with "
            "test/manual/kernels/build_xpu_dflash_native.py"
        ) from exc
    return library


def native_call(name, *tensors):
    library = load_target_head()
    require_native_window(library, tensors[0].shape[0])
    rc = getattr(library, name)(
        torch.xpu.current_stream().sycl_queue, *[t.data_ptr() for t in tensors]
    )
    if rc:
        error = (
            library.dflash_head_rescore_error
            if name == "dflash_head_rescore"
            else library.dflash_head_error
        )
        raise RuntimeError(error().decode())


def can_use_target_head(processor, hidden, head, metadata, bias=None):
    if (
        not envs.SGLANG_XPU_PAGED_DFLASH.get()
        or not envs.SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL.get()
        or torch.compiler.is_compiling()
        or not metadata.forward_mode.is_target_verify()
        or hidden.device.type != "xpu"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or hidden.shape[0] not in (4, 8)
        or hidden.shape[1] != 2048
        or not hidden.is_contiguous()
        or processor.vocab_size != 248320
        or processor.do_tensor_parallel_all_gather
        or processor.do_tensor_parallel_all_gather_dp_attn
        or processor.use_fp32_lm_head
        or processor.logit_scale is not None
        or processor.final_logit_softcapping
        or processor.return_full_logits
        or processor.rl_on_policy_target is not None
        or bias is not None
        or hasattr(head, "set_lora")
    ):
        return False
    from sglang.srt.runtime_context import get_parallel, get_spec

    spec = get_spec()
    algorithm = getattr(spec.speculative_algorithm, "name", spec.speculative_algorithm)
    return (
        algorithm == "DFLASH"
        and spec.speculative_num_draft_tokens == hidden.shape[0]
        and get_parallel().tp_size == 1
        and can_prepare_candidate_head(head, 1)
        and hidden.device == head.weight.device
    )


def weight_bounds(candidate, weight):
    """Conditional FP32/FP16/BF16 scale/operand envelopes, built before capture."""
    norms, errors, approx_norms = [], [], []
    recipes = [
        (scale, operand)
        for scale in (torch.float32, torch.float16, torch.bfloat16)
        for operand in (torch.float32, torch.float16, torch.bfloat16)
    ]
    for start in range(0, weight.shape[0], 2048):
        end = min(start + 2048, weight.shape[0])
        original = weight[start:end].float()
        raw = candidate.weight[start:end].float()
        scale = (
            candidate.scale[start // 128 : end // 128]
            .repeat_interleave(128, 0)
            .repeat_interleave(128, 1)
        )
        error, approx = [], []
        for sdtype, odtype in recipes:
            value = (raw * scale.to(sdtype).float()).to(odtype).float()
            error.append((original - value).square().sum(1).sqrt())
            approx.append(value.square().sum(1).sqrt())
        norms.append(original.square().sum(1).sqrt())
        errors.append(torch.stack(error).amax(0))
        approx_norms.append(torch.stack(approx).amax(0))
    result = [torch.cat(x) * 1.001 + 1e-10 for x in (norms, errors, approx_norms)]
    if not all(bool(x.isfinite().all()) for x in result):
        raise RuntimeError("SYCL target head requires finite weight envelopes")
    return torch.stack(result).contiguous()


class TargetHeadPlan:
    def __init__(self, state, hidden, head, metadata, use_buffer):
        self.state, self.hidden, self.head = state, hidden, head
        self.metadata, self.use_buffer = metadata, use_buffer
        rows = hidden.shape[0]
        self.scores = torch.empty((rows, 64), device=hidden.device, dtype=torch.float32)
        self.absolute = torch.empty_like(self.scores)
        self.norms = torch.empty((rows, 4), device=hidden.device, dtype=torch.float32)
        self.partial = torch.empty(
            (rows, 970), device=hidden.device, dtype=torch.float32
        )
        self.predictions = torch.empty(rows, device=hidden.device, dtype=torch.int64)
        self.flags = torch.empty(rows, device=hidden.device, dtype=torch.int32)
        self.source_pointer = head.weight.data_ptr()
        self.source_version = head.weight._version
        self.output_pointer = None

    def source_matches(self):
        return (
            self.head.weight.data_ptr() == self.source_pointer
            and self.head.weight._version == self.source_version
        )

    def propose(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.kernels.ops.speculative.dflash_topk_xpu import xpu_dflash_topk
        from sglang.srt.layers.quantization.fp8_utils import (
            torch_w8a8_block_fp8_linear,
        )

        use_sycl_quant = False
        if (
            envs.SGLANG_XPU_PAGED_DFLASH.get()
            and envs.SGLANG_XPU_DFLASH_QUANT_SYCL.get()
        ):
            from sglang.kernels.ops.quantization.dflash_quant_sycl import (
                can_use_dflash_quant,
                dflash_quant,
            )

            use_sycl_quant = can_use_dflash_quant(self.hidden)
        if use_sycl_quant:
            q, xs = dflash_quant(self.hidden)
        else:
            q, xs = per_token_group_quant_fp8(self.hidden, 128)
        coarse = torch_w8a8_block_fp8_linear(
            q,
            self.state.candidate.weight,
            [128, 128],
            self.state.candidate.scale,
            input_scale=xs,
        )
        _, ids = xpu_dflash_topk(coarse, 64)
        native_call(
            "dflash_head_rescore",
            self.hidden,
            self.head.weight,
            ids,
            self.scores,
            self.absolute,
        )
        native_call(
            "dflash_head_finish",
            self.hidden,
            q,
            xs,
            coarse,
            ids,
            self.state.coeff,
            self.scores,
            self.absolute,
            self.norms,
            self.partial,
            self.predictions,
            self.flags,
        )
        return coarse

    def complete(self, output):
        native_call(
            "dflash_head_complete",
            self.hidden,
            self.head.weight,
            self.predictions,
            self.flags,
            output,
            self.state.native_counts,
        )


class TargetHeadState:
    def __init__(self, head):
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Prepare SYCL target head before XPU graph capture")
        load_target_head()
        self.source = head.weight  # Keep captured source storage alive.
        self.source_pointer = self.source.data_ptr()
        self.source_version = self.source._version
        self.candidate = shared_candidate_head(self.source)
        self.coeff = weight_bounds(self.candidate, self.source)
        self.native_counts = torch.zeros(
            3, device=self.source.device, dtype=torch.int32
        )
        self.graph_plans = {}
        self.eager_plan = None
        logger.info(
            "Using experimental B70 SYCL target head M=%d/top64 with GPU BF16 "
            "completion; FP32 accumulation may differ; no host confidence readback",
            native_window(load_target_head()),
        )

    def source_matches(self, head):
        return (
            head.weight.data_ptr() == self.source_pointer
            and head.weight._version == self.source_version
        )

    def bind(self, plan, output):
        plan.output_pointer = output.data_ptr()
        if torch.xpu.is_current_stream_capturing():
            self.graph_plans[plan.output_pointer] = plan
        else:
            self.eager_plan = plan

    def plan_for(self, output, can_run_graph):
        if output is None:
            return None
        if can_run_graph:
            return self.graph_plans.get(output.data_ptr())
        plan = self.eager_plan
        return (
            plan
            if plan is not None and plan.output_pointer == output.data_ptr()
            else None
        )
