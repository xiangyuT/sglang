"""Deterministic top8 replica ownership for B70 TP2/EP2 without A2A.

The existing loader supplies 384 physical slots for 256 logical experts. This
module only chooses which stored copy evaluates a route; it never changes its
logical identity or coefficient. Both ranks count forced routes first, then
assign flexible routes to the less loaded rank with the same tie break.
"""

import logging
from functools import lru_cache

import torch

from sglang.kernels.ops.moe.xpu_tp_expert_gu_sycl import _library
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _local_ids_by_rank(physical_to_logical):
    """Build both ranks' local indices from the loader's CPU slot layout."""
    if physical_to_logical.device.type != "cpu" or physical_to_logical.shape != (384,):
        raise ValueError("Expected the CPU physical384 expert layout")
    table = torch.full((256, 2), -1, dtype=torch.int32)
    for physical, logical in enumerate(physical_to_logical.tolist()):
        if not 0 <= logical < 256:
            raise ValueError("Invalid logical expert in replica layout")
        rank, local = divmod(physical, 192)
        if table[logical, rank] != -1:
            raise ValueError("More than one copy per logical expert on the same rank")
        table[logical, rank] = local
    if bool((table < 0).all(dim=1).any()):
        raise ValueError("Replica layout omits a logical expert")
    return table


@lru_cache(maxsize=4)
def _device_table(physical_to_logical, device):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the XPU replica table before graph capture")
    table = _local_ids_by_rank(torch.tensor(physical_to_logical))
    return table.to(device=device)


def prepare_replica_table(num_experts, num_local_experts, num_shared_experts, device):
    """Validate the supported immutable layout once, before dispatch/capture."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    parallel, moe = get_parallel(), get_exec().moe
    device = torch.device(device)
    supported = (
        envs.SGLANG_XPU_EP_REPLICA_BALANCE.get()
        and envs.SGLANG_XPU_EP_SAFE_ROUTES.get()
        and envs.SGLANG_XPU_TP_EXPERT_GU_SYCL.get()
        and envs.SGLANG_XPU_TP_EXPERT_DOWN_SYCL.get()
        and device.type == "xpu"
        and torch.xpu.get_device_properties(device).device_id == 0xE223
        and parallel.tp_size == parallel.moe_ep_size == 2
        and parallel.pp_size == parallel.nnodes == parallel.attn_cp_size == 1
        and not parallel.attn_dp_enabled
        and get_spec().speculative_algorithm is None
        and moe.moe_a2a_backend == "none"
        and not moe.enable_eplb
        and moe.init_expert_location == "trivial"
        and moe.ep_num_redundant_experts == 128
        and (num_experts, num_local_experts, num_shared_experts) == (384, 192, 0)
    )
    if not supported:
        raise RuntimeError(
            "XPU replica balance requires B70 TP2/EP2, pure AR, no A2A/DP/EPLB, "
            "trivial 256+128 experts and enabled native-safe/GU/down paths"
        )
    metadata = get_global_expert_location_metadata()
    if metadata is None or metadata.ep_size != 2:
        raise RuntimeError("Missing EP2 expert-location metadata")
    layout = metadata.physical_to_logical_map_cpu
    expected = torch.arange(384) % 256
    if layout.device.type != "cpu" or layout.shape != (40, 384):
        raise RuntimeError("Unexpected replica loader layout")
    if not torch.equal(layout, expected.repeat(40, 1)):
        raise RuntimeError("XPU replica balance requires the verified trivial layout")
    _replica_library()
    logger.info(
        "B70 EP replica balance enabled: logical=256 physical=384 local=192; "
        "rank-invariant unique route ownership, forced routes before replicas"
    )
    return _device_table(tuple(expected.tolist()), device)


def _replica_library():
    library = _library(envs.SGLANG_XPU_TP_EXPERT_GU_SYCL_LIB.get())
    if not hasattr(library, "tp_expert_replica_map_run"):
        raise RuntimeError("Rebuild GU library with the optional replica-map entry")
    if envs.SGLANG_XPU_EP_REPLICA_FUSED_GU.get() and not hasattr(
        library, "tp_expert_replica_gu_silu_run"
    ):
        raise RuntimeError("Rebuild GU library with the optional replica-GU entry")
    return library


def replica_local_ids(ids, local_by_rank, rank):
    """Map every selected route to exactly one rank; preserve negative drops."""
    if not (
        not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and ids.ndim == 2
        and ids.shape[1] == 8
        and 0 < ids.shape[0] < 2**28
        and ids.device.type == "xpu"
        and ids.device == local_by_rank.device
        and ids.dtype == local_by_rank.dtype == torch.int32
        and local_by_rank.shape == (256, 2)
        and ids.is_contiguous()
        and local_by_rank.is_contiguous()
        and rank in (0, 1)
    ):
        raise ValueError("Unsupported XPU replica top8 input/table")
    library = _replica_library()
    output = torch.empty_like(ids)
    stream = torch.xpu.current_stream(ids.device)
    status = library.tp_expert_replica_map_run(
        stream.sycl_queue,
        ids.data_ptr(),
        local_by_rank.data_ptr(),
        output.data_ptr(),
        ids.shape[0],
        rank,
    )
    if status:
        raise RuntimeError(library.tp_expert_gu_error().decode())
    for tensor in (ids, local_by_rank, output):
        tensor.record_stream(stream)
    return output
