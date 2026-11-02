# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/xpu_communicator.py

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.utils import is_xpu

class XpuCommunicator:
    def __init__(self, group: ProcessGroup, cpu_group=None):
        if not is_xpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.peer_allreduce = None
        if os.environ.get("SGLANG_XPU_PEER_ALLREDUCE") == "1":
            if cpu_group is None:
                raise ValueError("Peer XPU all-reduce requires a local CPU group")
            from sglang.srt.distributed.device_communicators.xpu_peer_allreduce import (
                PeerAllReduce,
            )
            self.peer_allreduce = PeerAllReduce(
                cpu_group, os.environ["SGLANG_XPU_PEER_ALLREDUCE_LIB"]
            )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.peer_allreduce is not None and self.peer_allreduce.can_use(x):
            return self.peer_allreduce.all_reduce(x)
        return self._native_all_reduce(x)

    @eager_on_graph
    def _native_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def can_native_all_gather(self, output, input):
        return (os.environ.get("SGLANG_XPU_NATIVE_ALLGATHER") == "1"
                and self.peer_allreduce is not None
                and self.peer_allreduce.can_all_gather(output, input))

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        if self.can_native_all_gather(output, input):
            return self.peer_allreduce.all_gather(output, input)
        return self._native_all_gather_into_tensor(output, input)

    @eager_on_graph
    def _native_all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        all_gather_single(output, input, group=self.group)

    @eager_on_graph
    def gather(
        self, input_: torch.Tensor, rank_in_group: int, dst: int = 0, dim: int = -1
    ):
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        all_gather_single(output_tensor, input_, group=self.group)
        if rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor
