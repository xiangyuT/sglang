# SPDX-License-Identifier: Apache-2.0
"""Initial single-device GGUF execution with compressed expert residency.

Dense layers use decoded weights except for the optional IQ4_XS vocabulary
head. Routed experts retain their source encoding in a bounded GPU cache.
"""

from collections import OrderedDict
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sglang.kernels.ops.quantization.gguf_sycl import (
    expert_matvec_gguf,
    scatter_expert_routes,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
)
from sglang.srt.layers.quantization.gguf import GGUFConfig
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_loader.gguf_rows import CompressedGGUFRows
from sglang.srt.layers.quantization.gguf_forward_mode import is_gguf_prefill
from sglang.srt.utils import set_weight_attrs

_gpu_prefill_arenas = {}


def _capture_gguf_routes(method, layer, x, weights, ids):
    # Captured producer kernels have not executed: IDs are not meaningful yet.
    # Warmup initializes the native path; replay runs the real residency logic.
    return torch.zeros_like(x)


class Qwen4GGUFXPUConfig(GGUFConfig):
    def __init__(self, cache_slots, *, compressed_iq4_head=False, decode_q8=False,
                 head_kind=23, embedding_kind=23):
        super().__init__()
        if cache_slots < 1:
            raise ValueError("Compressed expert cache must have at least one slot")
        self.cache_slots = cache_slots
        self.compressed_iq4_head = compressed_iq4_head
        self.decode_q8 = decode_q8
        self.head_kind = head_kind
        self.embedding_kind = embedding_kind

    def get_quant_method(self, layer, prefix):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if isinstance(layer, FusedMoE):
            from sglang.srt.runtime_context import get_parallel

            if (os.environ.get("SGLANG_XPU_GGUF_TP_EP") == "1"
                    and get_parallel().moe_ep_size > 1):
                from sglang.srt.layers.quantization.gguf_ep import PartitionedGGUFMoEMethod

                return PartitionedGGUFMoEMethod(self.cache_slots, decode_q8=self.decode_q8)
            return CompressedGGUFMoEMethod(self.cache_slots, decode_q8=self.decode_q8)
        if self.compressed_iq4_head and isinstance(layer, ParallelLMHead):
            return (CompressedIQ4HeadMethod() if self.head_kind == 23
                    else CompressedGGUFHeadMethod(self.head_kind))
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            if self.compressed_iq4_head and prefix.split(".")[-1] == "embed_tokens":
                return (CompressedIQ4EmbeddingMethod() if self.embedding_kind == 23
                        else CompressedGGUFEmbeddingMethod(self.embedding_kind, allow_host=not prefix.startswith("mtp.")))
            return UnquantizedEmbeddingMethod()
        return None


class CompressedIQ4HeadMethod(LinearMethodBase):
    """Original IQ4_XS head bytes with Strata Q8_1 activation arithmetic."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        if input_size_per_partition != input_size or input_size not in (2560, 6144):
            raise ValueError(
                "IQ4_XS vocabulary head requires an unsharded supported width"
            )
        weight = torch.nn.Parameter(
            torch.empty(
                (sum(output_partition_sizes), input_size // 256 * 136),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        # The model loader supplies already compressed row-major bytes. Only
        # vocabulary rows are sharded/padded by VocabParallelEmbedding's loader.
        set_weight_attrs(weight, {"output_dim": 0})
        set_weight_attrs(weight, attrs)
        layer.register_parameter("weight", weight)
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer):
        from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library

        _library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))

    def apply(self, layer, x, bias=None):
        from sglang.kernels.ops.quantization.iq4_mmvq_sycl import iq4_xs_mmvq

        output = iq4_xs_mmvq(x, layer.weight)
        return output if bias is None else output + bias


class CompressedIQ4EmbeddingMethod(CompressedIQ4HeadMethod):
    """Decode only selected IQ4_XS token rows, retaining original model dtype."""

    def embedding(self, layer, ids):
        from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf

        packed = layer.weight.index_select(0, ids.reshape(-1)).contiguous()
        width = layer.weight.shape[1] // 136 * 256
        if ids.numel() == 0:
            return torch.empty(
                (*ids.shape, width), device=ids.device, dtype=self.params_dtype
            )
        return (
            dequantize_gguf(packed.reshape(-1), 23, dtype=torch.float32)
            .to(self.params_dtype)
            .view(*ids.shape, width)
        )


class CompressedGGUFHeadMethod(CompressedIQ4HeadMethod):
    """Original Q6_K/Q8_0 vocabulary rows and finite Q8_1 activations."""

    def __init__(self, kind):
        if kind not in (8, 14):
            raise ValueError("Compressed vocabulary requires Q6_K or Q8_0")
        self.kind = kind

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

        block, size = BLOCK_GEOMETRY[GGML_TYPES[self.kind]]
        if input_size_per_partition != input_size or input_size not in (2560, 6144):
            raise ValueError("Compressed vocabulary requires an unsharded supported width")
        weight = torch.nn.Parameter(torch.empty(
            (sum(output_partition_sizes), input_size // block * size), dtype=torch.uint8,
        ), requires_grad=False)
        set_weight_attrs(weight, {"output_dim": 0})
        set_weight_attrs(weight, attrs)
        layer.register_parameter("weight", weight)
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer):
        from sglang.kernels.ops.quantization.gguf_sycl import _library
        from sglang.kernels.ops.quantization.iq4_mmvq_sycl import _library as iq4_library

        _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
        iq4_library(os.environ.get("SGLANG_XPU_GGUF_IQ4_SYCL_LIB"))
        if (type(self) is CompressedGGUFHeadMethod and self.kind == 14
                and layer.weight.ndim == 2 and layer.weight.shape[1] == 2100
                and layer.weight.shape[0] >= 16
                and os.environ.get("SGLANG_XPU_GGUF_HEAD_Q6_SOA") == "1"
                and os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS") in ("8", "16")):
            from sglang.kernels.ops.quantization.gguf_q6_head_sycl import warm_from_env

            warm_from_env()

    def apply(self, layer, x, bias=None):
        from sglang.kernels.ops.quantization.gguf_dense_q8 import dense_q8_projection

        if x.shape[0] == 0:
            return torch.empty((0, layer.weight.shape[0]), device=x.device, dtype=torch.float32)
        marker = getattr(layer, "gguf_head_q6_soa", None)
        if marker is None:
            enabled = (type(self) is CompressedGGUFHeadMethod and self.kind == 14
                       and x.shape[1] == 2560 and layer.weight.shape[0] >= 16
                       and os.environ.get("SGLANG_XPU_GGUF_HEAD_Q6_SOA") == "1")
            if enabled:
                if torch.xpu.is_current_stream_capturing():
                    raise RuntimeError("Warm the Q6 vocabulary head after MTP token selection before capture")
                from sglang.kernels.ops.quantization.gguf_dense_q8 import permute_q6_fields

                layer.weight.data = permute_q6_fields(layer.weight)
                layer.gguf_head_q6_signature = (layer.weight.data_ptr(), tuple(layer.weight.shape))
                logging.getLogger(__name__).info(
                    "GGUF Q6 SoA16 head prepared after vocabulary selection: rows=%d, raw_tail=%d",
                    layer.weight.shape[0], layer.weight.shape[0] % 16,
                )
            layer.gguf_head_q6_soa = enabled
            marker = enabled
        if marker and layer.gguf_head_q6_signature != (layer.weight.data_ptr(), tuple(layer.weight.shape)):
            raise RuntimeError("Q6 packed vocabulary changed after preparation")
        output = None
        if marker and os.environ.get("SGLANG_XPU_GGUF_Q6_HEAD_WAVE_ROWS") in ("8", "16"):
            from sglang.kernels.ops.quantization.gguf_q6_head_sycl import try_project

            output = try_project(x, layer.weight)
        if output is None:
            outputs = [dense_q8_projection(chunk, ((self.kind, layer.weight),),
                                          q6_soa_parts=(marker,))
                       for chunk in x.split(4, dim=0)]
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return output if bias is None else output + bias.float()


class CompressedGGUFEmbeddingMethod(CompressedGGUFHeadMethod):
    """Decode selected original vocabulary rows to the model activation dtype."""

    def __init__(self, kind, *, allow_host=True):
        super().__init__(kind)
        self.allow_host = allow_host

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        if (self.allow_host and self.kind == 8
                and os.environ.get("SGLANG_XPU_GGUF_HOST_EMBEDDING") == "1"
                and layer.weight.device.type == "xpu"):
            from sglang.kernels.ops.quantization.gguf_host_embedding import prepare_host_embedding

            prepare_host_embedding(layer)

    def embedding(self, layer, ids):
        from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf
        from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

        block, size = BLOCK_GEOMETRY[GGML_TYPES[self.kind]]
        width = layer.weight.shape[1] // size * block
        if ids.numel() == 0:
            return torch.empty((*ids.shape, width), device=ids.device, dtype=self.params_dtype)
        if self.kind == 8 and layer.weight.device.type == "cpu":
            from sglang.kernels.ops.quantization.gguf_host_embedding import gather_host_embedding

            packed = gather_host_embedding(layer.weight, ids)
        else:
            packed = layer.weight.index_select(0, ids.reshape(-1)).contiguous()
        return dequantize_gguf(packed.reshape(-1), self.kind, dtype=torch.float32).view(
            *ids.shape, width
        ).to(self.params_dtype)


class CompressedGGUFMoEMethod(FusedMoEMethodBase):
    def __init__(self, cache_slots, *, decode_q8=False):
        self.cache_slots = cache_slots
        self.decode_q8 = decode_q8
        self.sources = None
        self.cache = OrderedDict()
        self.hits = self.misses = self.transferred_bytes = 0
        self._decode_q8_workspaces = {}
        self._fused_q8_workspaces = {}
        self.fused_q8 = (
            decode_q8 and os.environ.get("SGLANG_XPU_GGUF_MOE_Q8_FUSED") == "1"
        )
        self.iq2s_pvr = os.environ.get("SGLANG_XPU_GGUF_IQ2S_PVR") == "1"
        self._iq2s_packed_slots = set()
        self._native_staging = None
        self._gpu_cache = None
        self.gpu_cache_enabled = os.environ.get("SGLANG_XPU_GGUF_GPU_CACHE") == "1"

    def _decode_q8_workspace(self, device, tokens, width):
        # The current supported model has one live request. Keep ownership per
        # stream as well so separate streams never share an in-flight Q8 image.
        key = (device, torch.xpu.current_stream(device).sycl_queue, tokens, width)
        if key not in self._decode_q8_workspaces:
            self._decode_q8_workspaces[key] = torch.empty(
                (tokens, width // 32 * 36), dtype=torch.uint8, device=device
            )
        return self._decode_q8_workspaces[key]

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        if attrs.get("with_bias"):
            raise ValueError("GGUF experts with bias are unsupported")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate = intermediate_size_per_partition
        self.cache_slots = min(self.cache_slots, num_experts)
        # Source headers determine the compressed byte dimensions at bind time.
        # There are no expanded expert Parameters to allocate or load.

    def create_moe_runner(self, layer, moe_runner_config):
        if moe_runner_config.activation != "silu":
            raise ValueError("Compressed GGUF experts require SiLU")
        self.moe_runner_config = moe_runner_config

    def bind(self, layer, sources, device):
        expected = [(self.hidden_size, self.intermediate, self.num_experts)] * 2 + [
            (self.intermediate, self.hidden_size, self.num_experts)
        ]
        if [tuple(t.shape) for _, t in sources] != expected:
            raise ValueError("Compressed expert source shapes disagree with the layer")
        self.bind_rows(layer, [CompressedGGUFRows(s, t) for s, t in sources], device)

    def bind_rows(self, layer, sources, device):
        if self.sources is not None:
            raise RuntimeError("Compressed GGUF expert sources were already bound")
        expected = [(self.hidden_size, self.intermediate * self.num_experts)] * 2 + [
            (self.intermediate, self.hidden_size * self.num_experts)
        ]
        if [(source.width, source.rows) for source in sources] != expected:
            raise ValueError("Compressed expert row geometry disagrees with the layer")
        if sources[0].kind != sources[1].kind:
            raise ValueError("Fused gate/up execution requires matching quantization")
        self.sources = sources
        sizes = tuple(source.rows // self.num_experts * source.row_bytes for source in sources)
        resident_aos = (self.gpu_cache_enabled and sources[0].kind in (21, 23)
                        and os.environ.get("SGLANG_XPU_GGUF_RESIDENT_EXPERT_AOS") == "1")
        if resident_aos:
            from sglang.kernels.ops.quantization.gguf_gpu_cache import empty_expert_aos

            fields = empty_expert_aos(self.cache_slots, sizes, device)
        else:
            fields = tuple(torch.empty((self.cache_slots, size), dtype=torch.uint8, device=device) for size in sizes)
        for name, field in zip(("gate", "up", "down"), fields):
            layer.register_buffer("gguf_" + name, field, persistent=False)
        if self.gpu_cache_enabled:
            self._bind_gpu_cache(layer)
        elif os.environ.get("SGLANG_XPU_GGUF_NATIVE_STAGE") == "1":
            from sglang.kernels.ops.quantization.gguf_expert_copy import (
                make_expert_copies,
            )

            self._native_staging = make_expert_copies(
                self.sources, layer, self.num_experts
            )

    def _bind_gpu_cache(self, layer):
        from sglang.kernels.ops.quantization.gguf_gpu_cache import (
            GPUExpertCache,
            encode_host_experts,
        )

        kind = self.sources[0].kind
        packed_gu = (
            kind in (16, 29) and os.environ.get("SGLANG_XPU_GGUF_LOWBIT_GU") == "1"
        )
        gu_soa = (kind == 21 and self.sources[2].kind in (8, 20)
                  and os.environ.get("SGLANG_XPU_GGUF_IQ3_GU_SOA") == "1")
        down_soa = (
            kind in (21, 23) and self.sources[2].kind == 20
            and os.environ.get("SGLANG_XPU_GGUF_IQ4_DOWN_SOA") == "1"
        )
        if (down_soa or gu_soa) and os.environ.get("SGLANG_XPU_GGUF_MIXED_PACKETS") != "1":
            raise ValueError("Expert field order requires mixed packet kernels")
        if not (
            self.fused_q8
            and self.hidden_size == 2560
            and self.intermediate == 640
            and kind in (16, 21, 22, 23, 29, 42)
            and (self.sources[2].kind == 42 or
                 (kind in (21, 23) and self.sources[2].kind in (8, 20)))
            and self.cache_slots >= min(40, self.num_experts)
            and (kind != 22 or self.iq2s_pvr)
        ):
            raise ValueError("GPU GGUF cache requires the bounded B70 Q8 expert path")
        mirrors = []
        buffers = tuple(
            getattr(layer, "gguf_" + name) for name in ("gate", "up", "down")
        )
        resident_ids = tuple(range(self.cache_slots))
        mirror_ids = tuple(range(self.cache_slots, self.num_experts))
        lru_order = None
        profile_path = os.environ.get("SGLANG_XPU_GGUF_EXPERT_PROFILE")
        if profile_path and kind in (21, 23) and self.cache_slots < self.num_experts:
            from sglang.srt.layers.quantization.gguf_expert_profile import (
                read_profile, initial_placement,
            )
            profile = read_profile(profile_path)
            if (profile['layers'], profile['experts']) != (48, 512):
                raise ValueError("Qwen4 IQ4 expert profile must describe 48x512 experts")
            layer_id = self.moe_runner_config.layer_id
            real = getattr(self, 'real_experts', self.num_experts)
            dummy = real if real != self.num_experts else None
            start = getattr(self, 'ep_rank', 0) * real
            resident_ids, mirror_ids, lru_order = initial_placement(
                profile, layer_id, start, real, self.cache_slots, dummy=dummy,
            )
            if layer_id == 0:
                logging.getLogger(__name__).info(
                    "GGUF initial expert profile sha256=%s: rank=%d slots=%d; "
                    "unchanged compressed bytes and capacity; first logical IDs=%s",
                    profile['sha256'], getattr(self, 'ep_rank', 0),
                    self.cache_slots, resident_ids[:12],
                )
        for field, (source, target) in enumerate(zip(self.sources, buffers)):
            rows = source.rows // self.num_experts
            if lru_order is None:
                resident = source.packed_row_range(
                    0, self.cache_slots * rows, pin_memory=True
                ).view(self.cache_slots, -1)
            else:
                from sglang.srt.layers.quantization.gguf_expert_profile import read_experts
                resident = read_experts(source, resident_ids, rows)
            encoded = encode_host_experts(resident, kind, field, packed_gu=packed_gu, down_kind=self.sources[2].kind, down_soa=down_soa, gu_soa=gu_soa)
            target.copy_(encoded)
            outside = self.num_experts - self.cache_slots
            cold = (source.packed_row_range(
                self.cache_slots * rows, source.rows, pin_memory=True
            ).view(outside, target.shape[1]) if lru_order is None
                    else read_experts(source, mirror_ids, rows))
            mirrors.append(encode_host_experts(cold, kind, field, packed_gu=packed_gu, down_kind=self.sources[2].kind, down_soa=down_soa, gu_soa=gu_soa))
        host_aos = (kind in (21, 23)
                    and os.environ.get("SGLANG_XPU_GGUF_HOST_EXPERT_AOS") == "1")
        if host_aos:
            from sglang.kernels.ops.quantization.gguf_gpu_cache import make_host_expert_aos

            mirrors = make_host_expert_aos(mirrors)
            if self.moe_runner_config.layer_id == 0:
                logging.getLogger(__name__).info("GGUF expert mirrors: one primary pinned expert-contiguous arena; original bytes, no secondary field store")
        self._gpu_cache = GPUExpertCache(
            kind,
            resident_ids,
            buffers,
            mirror_ids,
            mirrors,
            layout="lowbit_gu_packed"
            if packed_gu
            else "iq2s_q2_pvr"
            if kind == 22
            else "original",
            down_kind=self.sources[2].kind,
            down_soa=down_soa,
            gu_soa=gu_soa,
            lru_order=lru_order,
            host_layout="expert_aos" if host_aos else "fields",
            resident_layout="expert_aos" if (kind in (21, 23) and os.environ.get("SGLANG_XPU_GGUF_RESIDENT_EXPERT_AOS") == "1") else "fields",
        )
        if gu_soa and self.moe_runner_config.layer_id == 0:
            logging.getLogger(__name__).info(
                "GGUF IQ3 gate/up SoA: lossless four-row field order in slots and mirrors; unchanged bytes and capacity"
            )
        if down_soa and self.moe_runner_config.layer_id == 0:
            logging.getLogger(__name__).info(
                "GGUF IQ4 down SoA: lossless four-row field order in slots and mirrors; unchanged bytes and capacity"
            )

    def grow_cache(self, layer, slots):
        """Grow only at an idle eager boundary; preserve slot identity and LRU.

        Compressed routing is an eager graph boundary and resolves these buffer
        pointers on each call. Captured producer/consumer tensors do not change.
        Model initialization calls this after all speculative graphs are built,
        so temporary capture storage need not compete with the final cache.
        """
        if self.sources is None or not self.cache_slots <= slots <= self.num_experts:
            raise ValueError(
                "GGUF cache growth requires bound sources and valid capacity"
            )
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError("Cannot replace expert cache buffers during capture")
        if slots == self.cache_slots:
            return 0
        old = tuple(getattr(layer, "gguf_" + name) for name in ("gate", "up", "down"))
        if self._gpu_cache is not None:
            count = slots - self.cache_slots
            if self._gpu_cache.resident_layout == "expert_aos":
                from sglang.kernels.ops.quantization.gguf_gpu_cache import empty_expert_aos

                extra = empty_expert_aos(count, self._gpu_cache.sizes, self._gpu_cache.device)
            else:
                extra = tuple(
                    torch.empty((count, value.shape[1]), dtype=value.dtype, device=value.device)
                    for value in old
                )
            current = self._gpu_cache.expert_slots.cpu().tolist()
            selected = [e for e, slot in enumerate(current) if slot < 0][:count]
            profile_path = os.environ.get("SGLANG_XPU_GGUF_EXPERT_PROFILE")
            if profile_path and self.sources[0].kind in (21, 23):
                from sglang.srt.layers.quantization.gguf_expert_profile import (
                    read_profile, growth_placement,
                )

                real = getattr(self, "real_experts", self.num_experts)
                selected = growth_placement(
                    read_profile(profile_path), self.moe_runner_config.layer_id,
                    getattr(self, "ep_rank", 0) * real, real, current, count,
                    dummy=real if real != self.num_experts else None,
                )
            self._gpu_cache.grow(extra, selected)
            # Residency growth must keep captured wave shapes and output ownership.
            self._decode_cache_slots = getattr(self, "_decode_cache_slots", self.cache_slots)
            self.cache_slots = slots
            return sum(value.numel() for value in extra)
        # Allocate all three before replacing anything; allocation failure leaves
        # this layer's original buffers, capacity, LRU and packed markers intact.
        grown = tuple(
            torch.empty((slots, value.shape[1]), dtype=value.dtype, device=value.device)
            for value in old
        )
        stream = torch.xpu.current_stream(old[0].device)
        for before, after in zip(old, grown):
            after[: self.cache_slots].copy_(before, non_blocking=True)
            before.record_stream(stream)
        added = sum((slots - self.cache_slots) * value.shape[1] for value in old)
        for name, value in zip(("gate", "up", "down"), grown):
            setattr(layer, "gguf_" + name, value)
        self.cache_slots = slots
        return added

    def _ensure_expert(self, layer, expert):
        if expert in self.cache:
            slot = self.cache.pop(expert)
            self.cache[expert] = slot
            self.hits += 1
            return slot
        if len(self.cache) == self.cache_slots:
            _, slot = self.cache.popitem(last=False)
        else:
            slot = len(self.cache)
        if self._native_staging is not None:
            self._native_staging.copy(layer, expert, slot)
            self.transferred_bytes += self._native_staging.expert_bytes
        else:
            for name, source in zip(("gate", "up", "down"), self.sources):
                rows = source.rows // self.num_experts
                packed = source.packed_row_range(
                    expert * rows, (expert + 1) * rows, pin_memory=True
                )
                # The pinned allocator retains each source until its DMA completes.
                # Copies and expert kernels share the current stream, so a miss can
                # enqueue all three weights without a CPU synchronization per copy.
                getattr(layer, "gguf_" + name)[slot].copy_(
                    packed.reshape(-1), non_blocking=True
                )
                self.transferred_bytes += packed.numel()
        self._iq2s_packed_slots.discard(slot)
        self.cache[expert] = slot
        self.misses += 1
        return slot

    def _ensure_group(self, layer, experts, *, packed_gu=False):
        # Touch hits before inserting misses, so selected resident experts
        # cannot be evicted before this group's matrix kernels consume them.
        order = [expert for expert in experts if expert in self.cache] + [
            expert for expert in experts if expert not in self.cache
        ]
        resident = {expert: self._ensure_expert(layer, expert) for expert in order}
        changed = [
            slot
            for slot in resident.values()
            if (slot in self._iq2s_packed_slots) != packed_gu
        ]
        if changed:
            if not (
                self.sources[0].kind == 22
                and self.hidden_size == 2560
                and self.intermediate == 640
            ):
                raise ValueError("Packed IQ2S cache geometry mismatch")
            from sglang.kernels.ops.quantization.gguf_iq2s_pvr import repack_iq2s_cache

            repack_iq2s_cache(
                layer.gguf_gate,
                layer.gguf_up,
                changed,
                to_packed=packed_gu,
                down=layer.gguf_down,
            )
            if packed_gu:
                self._iq2s_packed_slots.update(changed)
            else:
                self._iq2s_packed_slots.difference_update(changed)
        return resident

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        weights, ids, _ = dispatch_output.topk_output
        x = dispatch_output.hidden_states
        short_f16_format = (
            self.sources[0].kind in (21, 23) and self.sources[2].kind == 8
        ) or (self.sources[0].kind == self.sources[2].kind == 42)
        if 0 < x.shape[0] <= 4 and short_f16_format and is_gguf_prefill():
            return StandardCombineInput(
                hidden_states=self._apply_routes(layer, x, weights, ids)
            )
        wave_rows = 4
        if self._gpu_cache is not None and 4 < x.shape[0] <= 64:
            from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

            try:
                mode = get_forward_batch().forward_mode
            except RuntimeError:
                mode = None
            mtp_waves = (
                mode is not None
                and mode.is_draft_extend_v2()
                and os.environ.get("SGLANG_XPU_GGUF_MTP_BATCH_WAVES") == "1"
                and self.fused_q8
                and self.cache_slots == self.num_experts
                and self.sources[0].kind == self.sources[2].kind == 42
            )
            if mode is not None and (
                mode.is_decode() or mode.is_target_verify() or mtp_waves
            ):
                if (not mtp_waves and self.sources[0].kind in (21, 23)
                        and self.sources[2].kind in (8, 20)
                        and getattr(self, "_resident_preparation", None) is None):
                    # The opt-in EP2 union keeps cold overflow in pinned host
                    # mirrors. Otherwise preserve the existing worst-case
                    # protected-set bound and post-capture slot growth.
                    maximum = getattr(self._gpu_cache, "max_rows", 4)
                    if (
                        getattr(self._gpu_cache, "union_overflow", False)
                        and x.shape[0] > 16
                        and maximum > 16
                    ):
                        # The overflow-capable native planner handles 17–32 rows.
                        wave_rows = maximum
                    else:
                        # Preserve the protected-set bound and output ownership
                        # for short batches (152 slots allow at most 15 rows).
                        protected_slots = min(
                            self.cache_slots,
                            getattr(self, "_decode_cache_slots", self.cache_slots),
                        )
                        wave_rows = min(16, maximum, protected_slots // ids.shape[1])
                    if wave_rows < 1:
                        raise ValueError("Expert cache cannot protect one routed row")
                # Each packet keeps the original DP4A/eight-chain arithmetic.
                # Larger mixed-format waves share grouping and encoded reads.
                # Fully resident Q2 MTP may opt into the same Q8 activation
                # arithmetic as BS1, avoiding its host-routed multirow prefill.
                # Keep this explicit: the former FP16 prefill is a different
                # numerical path and needs model-level acceptance validation.
                if x.shape[0] > wave_rows:
                    output = torch.empty_like(x)
                    topk = dispatch_output.topk_output
                    for first in range(0, x.shape[0], wave_rows):
                        stop = min(first + wave_rows, x.shape[0])
                        logits = topk.router_logits
                        wave = dispatch_output._replace(
                            hidden_states=x[first:stop],
                            topk_output=topk._replace(
                                topk_weights=weights[first:stop],
                                topk_ids=ids[first:stop],
                                router_logits=logits[first:stop]
                                if logits is not None
                                else None,
                            ),
                        )
                        part = self.apply(layer, wave)
                        # The native workspace is reused by the following wave.
                        output[first:stop].copy_(part.hidden_states)
                    return StandardCombineInput(hidden_states=output)
        if self._gpu_cache is not None and 1 <= x.shape[0] <= wave_rows:
            from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts

            preparation = getattr(self, "_resident_preparation", None)
            routes = (preparation.routes_dispatch(dispatch_output) if preparation is not None
                      else self._gpu_cache.prepare(ids))
            output = fused_q8_experts(
                x,
                layer.gguf_gate,
                layer.gguf_up,
                layer.gguf_down,
                self.sources[0].kind,
                routes.slots,
                weights,
                self._fused_q8_workspaces,
                grouping=routes.grouping
                if self.sources[0].kind in (21, 22, 23) or self._gpu_cache.packed_gu
                else None,
                slot_addresses=self._gpu_cache.slot_addresses,
                zero_slot=getattr(self, "_ep_zero_slot", None),
                paged_zero_expert=getattr(self, "_paged_zero_expert", None),
                down_kind=self.sources[2].kind,
            )
            if (x.shape[0] > 16
                    and getattr(self._gpu_cache, "union_overflow", False)):
                # The legacy wide path returned a dedicated Torch output after
                # copying each native wave. Preserve that ownership boundary.
                output = output.clone()
        else:
            output = self._apply_routes(layer, x, weights, ids)
        return StandardCombineInput(hidden_states=output)

    def _apply_gpu_prefill(self, x, weights, ids_cpu, experts, *, output_dtype=None):
        if (x.shape[0] > 0 and os.environ.get("SGLANG_XPU_GGUF_PREFILL_F16") == "1"
                and self.sources[0].kind in (16, 21, 22, 23, 29, 42)
                and self.sources[2].kind in (8, 20, 42)):
            from sglang.kernels.ops.quantization.gguf_prefill import apply_prefill

            if getattr(self, "_mtp_prefill_last_only", False) and x.shape[0] > 4:
                from sglang.kernels.ops.quantization.gguf_mtp_prefill import apply_tail
                return apply_tail(self, x, weights, ids_cpu, output_dtype=output_dtype)
            return apply_prefill(self, x, weights, ids_cpu, experts,
                                 output_dtype=output_dtype)
        # A shared bounded arena avoids retaining another expert cache per
        # layer. Gather from current GPU/mirror owners without mutating LRU;
        # restore original encoded bytes before the existing FP32 arithmetic.
        if not experts:
            return x.new_empty((x.shape[0], self.hidden_size))
        key = (x.device, torch.xpu.current_stream(x.device).sycl_queue)
        arena_sizes = (524800, 524800, 460800)
        if self.sources[0].kind in (21, 23):
            arena_sizes = self._gpu_cache.sizes
            key = (*key, *arena_sizes)
        limit = 32
        if key not in _gpu_prefill_arenas:
            _gpu_prefill_arenas[key] = tuple(
                torch.empty(limit * size, dtype=torch.uint8, device=x.device)
                for size in arena_sizes
            )
        arena = _gpu_prefill_arenas[key]
        activation = x.float().contiguous()
        flat_ids = ids_cpu.reshape(-1)
        parts = torch.empty(
            (flat_ids.size, self.hidden_size), dtype=torch.float32, device=x.device
        )
        for begin in range(0, len(experts), limit):
            group = experts[begin : begin + limit]
            encoded = tuple(
                value[: len(group) * size].view(len(group), size)
                for value, size in zip(arena, self._gpu_cache.sizes)
            )
            self._gpu_cache.stage_original(group, encoded)
            selected = np.flatnonzero(np.isin(flat_ids, group))
            destinations = torch.as_tensor(selected.astype(np.int32), device=x.device)
            tokens = torch.as_tensor(
                (selected // ids_cpu.shape[1]).astype(np.int32), device=x.device
            )
            resident = {expert: slot for slot, expert in enumerate(group)}
            slots = torch.tensor(
                [resident[int(e)] for e in flat_ids[selected]],
                dtype=torch.int32,
                device=x.device,
            )
            gate, up = expert_matvec_gguf(
                activation,
                encoded[0],
                self.sources[0].kind,
                slots,
                self.intermediate,
                up=encoded[1],
                tokens=tokens,
            )
            down = expert_matvec_gguf(
                (F.silu(gate) * up).contiguous(),
                encoded[2],
                self.sources[2].kind,
                slots,
                self.hidden_size,
            )
            scatter_expert_routes(down, destinations, parts)
        return (
            (
                parts.view(*ids_cpu.shape, self.hidden_size)
                * weights.float().unsqueeze(-1)
            )
            .sum(dim=1)
            .to(x.dtype if output_dtype is None else output_dtype)
        )

    @eager_on_graph(capture_stub=_capture_gguf_routes)
    def _apply_routes(self, layer, x, weights, ids):
        # Tensor-only graph boundaries preserve the dispatch/combine NamedTuple
        # types, which the graph's recursive weak-ref conversion cannot retain.
        if self.sources is None:
            raise RuntimeError("Compressed GGUF expert sources were not loaded")
        ids_cpu = ids.cpu().numpy()
        if np.any(ids_cpu < 0) or np.any(ids_cpu >= self.num_experts):
            raise ValueError("Invalid single-device GGUF expert routing ID")
        experts = np.unique(ids_cpu).tolist()
        if self._gpu_cache is not None:
            return self._apply_gpu_prefill(x, weights, ids_cpu, experts)
        if (
            self.fused_q8
            and 1 <= x.shape[0] <= 4
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and self.hidden_size == 2560
            and self.intermediate == 640
            and self.sources[0].kind in (16, 22, 29, 42)
            and self.sources[2].kind == 42
            and weights.dtype in (torch.bfloat16, torch.float32)
            and weights.is_contiguous()
            and ids.shape[1] <= 10
            and len(experts) <= self.cache_slots
        ):
            from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts

            # Only the measured four-row verify shape uses grouped packing.
            pvr = self.iq2s_pvr and self.sources[0].kind == 22 and x.shape[0] == 4
            resident = self._ensure_group(layer, experts, packed_gu=pvr)
            grouping = None
            if pvr:
                from sglang.kernels.ops.quantization.gguf_iq2s_pvr import (
                    group_iq2s_routes,
                )

                if self._native_staging is not None:
                    grouping = self._native_staging.group_routes(ids_cpu, resident)
                if grouping is None:
                    grouping = group_iq2s_routes(ids_cpu, resident, x.device)
                slots = grouping.route_slots
            else:
                slots = torch.tensor(
                    [resident[int(expert)] for expert in ids_cpu.reshape(-1)],
                    dtype=torch.int32,
                    device=x.device,
                )
            return fused_q8_experts(
                x,
                layer.gguf_gate,
                layer.gguf_up,
                layer.gguf_down,
                self.sources[0].kind,
                slots,
                weights,
                self._fused_q8_workspaces,
                grouping=grouping,
            )
        if x.shape[0] == 1 and len(experts) <= self.cache_slots:
            resident = self._ensure_group(layer, experts)
            slots = torch.tensor(
                [resident[expert] for expert in ids_cpu.reshape(-1)],
                dtype=torch.int32,
                device=x.device,
            )
            gate, up = expert_matvec_gguf(
                x.float().contiguous(),
                layer.gguf_gate,
                self.sources[0].kind,
                slots,
                self.intermediate,
                up=layer.gguf_up,
                q8_activations=self.decode_q8,
                q8_workspace=(
                    self._decode_q8_workspace(x.device, 1, self.hidden_size)
                    if self.decode_q8
                    else None
                ),
            )
            down = expert_matvec_gguf(
                (F.silu(gate) * up).contiguous(),
                layer.gguf_down,
                self.sources[2].kind,
                slots,
                self.hidden_size,
                q8_activations=self.decode_q8,
                q8_workspace=(
                    self._decode_q8_workspace(
                        x.device, slots.numel(), self.intermediate
                    )
                    if self.decode_q8
                    else None
                ),
            )
            parts = down.unsqueeze(0)
            output = (parts * weights.float().unsqueeze(-1)).sum(dim=1).to(x.dtype)
            return output
        hot = [expert for expert in experts if expert in self.cache]
        cold = [expert for expert in experts if expert not in self.cache]
        if not experts:
            return x.new_empty((x.shape[0], self.hidden_size))
        if len(experts) <= self.cache_slots:
            groups = [experts]
        else:
            # Consume all hits before cold groups can replace their slots.
            # Unlike sorted cyclic LRU, a slightly oversized working set then
            # transfers only its missing experts on the next invocation.
            groups = ([hot] if hot else []) + [
                cold[begin : begin + self.cache_slots]
                for begin in range(0, len(cold), self.cache_slots)
            ]
        activation = x.float().contiguous()
        use_q8 = self.decode_q8 and x.shape[0] <= 4
        parts = torch.empty(
            (ids.numel(), self.hidden_size), device=x.device, dtype=torch.float32
        )
        flat_ids = ids_cpu.reshape(-1)
        for group in groups:
            resident = self._ensure_group(layer, group)
            flat_cpu = np.flatnonzero(np.isin(flat_ids, group))
            destinations = torch.as_tensor(flat_cpu.astype(np.int32), device=x.device)
            tokens = torch.as_tensor(
                (flat_cpu // ids.shape[1]).astype(np.int32), device=x.device
            )
            slots = torch.tensor(
                [resident[int(expert)] for expert in flat_ids[flat_cpu]],
                dtype=torch.int32,
                device=x.device,
            )
            gate, up = expert_matvec_gguf(
                activation,
                layer.gguf_gate,
                self.sources[0].kind,
                slots,
                self.intermediate,
                up=layer.gguf_up,
                tokens=tokens,
                q8_activations=use_q8,
                q8_workspace=(
                    self._decode_q8_workspace(x.device, x.shape[0], self.hidden_size)
                    if use_q8
                    else None
                ),
            )
            down = expert_matvec_gguf(
                (F.silu(gate) * up).contiguous(),
                layer.gguf_down,
                self.sources[2].kind,
                slots,
                self.hidden_size,
                q8_activations=use_q8,
                q8_workspace=(
                    self._decode_q8_workspace(
                        x.device, slots.numel(), self.intermediate
                    )
                    if use_q8
                    else None
                ),
            )
            scatter_expert_routes(down, destinations, parts)
        output = (
            (parts.view(*ids.shape, self.hidden_size) * weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(x.dtype)
        )
        return output
