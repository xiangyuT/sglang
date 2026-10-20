"""Native XPU expert paging without per-layer host synchronization."""

import torch

from sglang.kernels.ops.moe.paged_experts_xpu import decide, gather


class XpuDeviceExecutor:
    def __init__(
        self, *, base_method, store, residency, contract, routed_scaling_factor, layer
    ):
        self.base_method = base_method
        self.store = store
        self.residency = residency
        self.contract = contract
        self.routed_scaling_factor = routed_scaling_factor
        device = residency.step.device
        slots = residency.slot_expert.numel()
        self.mapped_ids = torch.empty(slots, dtype=torch.int32, device=device)
        self.mapped_weights = torch.empty(slots, dtype=torch.float32, device=device)
        hosts, gpus, rows = [], [], []
        for name, host in store.host.items():
            gpu = getattr(layer, name).data
            row_bytes = host[0].numel() * host.element_size()
            if not (
                host.is_pinned()
                and host.is_contiguous()
                and gpu.is_contiguous()
                and row_bytes % 4 == 0
            ):
                raise RuntimeError(
                    f"XPU paged experts needs pinned contiguous rows of whole 4-byte words: {name}"
                )
            hosts.append(host.data_ptr())
            gpus.append(gpu.data_ptr())
            rows.append(row_bytes // 4)
        # Level Zero device addresses may set bit 63; keep pointer bits unsigned.
        self.hosts = torch.tensor(hosts, dtype=torch.uint64, device=device)
        self.gpus = torch.tensor(gpus, dtype=torch.uint64, device=device)
        self.rows = tuple(rows)

    def run(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk = dispatch_output.topk_output
        n = topk.topk_ids.numel()
        decide(
            topk.topk_ids,
            topk.topk_weights,
            self.residency,
            self.mapped_ids,
            self.mapped_weights,
        )
        gather(self.hosts, self.gpus, self.rows, self.residency)
        remapped = dispatch_output._replace(
            topk_output=topk._replace(
                topk_ids=self.mapped_ids[:n].view(topk.topk_ids.shape),
                topk_weights=self.mapped_weights[:n].view(topk.topk_weights.shape),
            )
        )
        partial = self.base_method.apply(
            layer=layer, dispatch_output=remapped
        ).hidden_states
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=partial,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )
