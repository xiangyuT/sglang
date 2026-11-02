"""Scoped CPU-hybrid shared branch placement; no arithmetic or worker changes."""
import logging
import os
from dataclasses import dataclass
import torch

@dataclass
class CPUShared:
    block: object
    hidden: torch.Tensor
    projection: tuple
    apply_gate: bool
    issued: bool = False
    output: object = None
    deferred_gate: bool = False
    gate: object = None

    def launch(self):
        assert not self.issued, "CPU shared branch must be issued once"
        self.output = self.block._forward_shared_experts(
            self.hidden, apply_gate=self.apply_gate, projection_input=self.projection)
        assert self.output is not None
        self.issued = True


def prepare(block, x, projection, defer_finalize, use_fused_gate, *, deferred_gate=False):
    if (os.environ.get('CODEX_AR_CPU119_OVERLAP') != '1'
            or os.environ.get('CODEX_AR_CPU40') != '1' or defer_finalize
            or torch.is_grad_enabled() or torch.compiler.is_compiling()):
        return None
    if block.alt_stream is not None or block.enable_shared_expert_fusion:
        return None
    shared = block.shared_expert
    if (shared is None or shared._forward_hooks or shared._forward_pre_hooks
            or type(block.experts).__name__ != 'FusedMoE'
            or getattr(block.experts, '_dwdp_bound', False)):
        return None
    method = block.experts.quant_method
    ex = getattr(method, 'device_executor', None)
    if not (type(method).__name__ == 'PagedExpertsMoEMethod'
            and type(ex).__name__ == 'XpuDeviceExecutor' and ex.use_scaled_fp8
            and ex.residency.slot_expert.numel() == 192
            and hasattr(ex, '_cpu40_channel')):
        return None
    from sglang.srt.runtime_context import get_spec, get_parallel
    if (get_spec().speculative_algorithm is not None
            or get_parallel().tp_size != 1 or get_parallel().moe_ep_size != 1):
        return None
    if not (x.device.type == 'xpu' and x.shape == (1, 2048)
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and isinstance(projection, tuple) and len(projection) == 2):
        return None
    q, s = projection
    if not (q.shape == x.shape and q.dtype == torch.float8_e4m3fn
            and s.shape == (1,16) and s.dtype == torch.float32
            and all(t.device == x.device and t.is_contiguous() for t in [q,s])
            and torch.xpu.get_device_properties(x.device).device_id == 0xe223):
        return None
    if block.layer_id == 0 and not getattr(block, '_ar119_logged', False):
        logging.getLogger(__name__).info('AR119 CPU shared FFN placed after route publish; original arithmetic; no additional stream')
        block._ar119_logged = True
    return CPUShared(block, x, projection, not use_fused_gate, deferred_gate=deferred_gate)
