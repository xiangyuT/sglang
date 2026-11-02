# SPDX-License-Identifier: Apache-2.0
"""Native XPU QSA INT8 KV: encoded rows are the unit of every transfer."""

import torch

from sglang.kernels.ops.attention.xpu_qsa_int8 import RECORD_BYTES, quantize_and_store
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheQuantMethodBase,
    KVCacheAttentionAccess,
    KVCacheAttentionAccessKind,
    KVCacheAttentionPhase,
    KVCacheBackendMatcher,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, KvBufferDesc


class Int8QSAKVCacheMethod(KVCacheQuantMethodBase):
    name = "xpu_qsa_int8_block64"
    SCALE_BLOCK_SIZE = 64

    def attention_accesses(self):
        return tuple(
            KVCacheAttentionAccess(
                phase=phase,
                kind=KVCacheAttentionAccessKind.NATIVE_INT8,
                backend_matcher=KVCacheBackendMatcher(
                    exact=frozenset({"intel_xpu", "qwen_sparse_attn"})
                ),
                storage_dtype=torch.int8,
                attention_kv_dtype=torch.int8,
                scale_recipe=self.name,
            )
            for phase in KVCacheAttentionPhase
        )

    def create_buffers(self, size, head_num, head_dim, layer_num, device):
        if head_dim != 256 or head_num not in (1, 2):
            raise ValueError("INT8 QSA KV requires head_dim=256 and 1 or 2 KV heads")
        self.raw_k = [
            torch.zeros((size, head_num, RECORD_BYTES), dtype=torch.int8, device=device)
            for _ in range(layer_num)
        ]
        self.raw_v = [torch.zeros_like(t) for t in self.raw_k]
        self.owners = torch.empty(size, dtype=torch.int32, device=device)
        self.error = torch.zeros(1, dtype=torch.int32, device=device)
        return dict(
            k_buffer=[t[:, :, :256] for t in self.raw_k],
            v_buffer=[t[:, :, :256] for t in self.raw_v],
            k_scale_buffer=[t[:, :, 256:].view(torch.float16) for t in self.raw_k],
            v_scale_buffer=[t[:, :, 256:].view(torch.float16) for t in self.raw_v],
            store_dtype=torch.int8,
        )

    def quantize_and_store(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        **kwargs,
    ):
        # Dynamic block scales replace model-wide FP8 calibration scales.
        quantize_and_store(
            k_buffer, v_buffer, loc, cache_k, cache_v, self.owners, self.error
        )

    def dequantize_prev_kv(self, *args, **kwargs):
        raise NotImplementedError(
            "INT8 QSA reads encoded KV directly in the native attention kernel"
        )

    def compute_cell_size(self, head_num, head_dim, num_layers, kv_size):
        if head_dim != 256:
            raise ValueError("INT8 QSA KV requires head_dim=256")
        # One int32 owner per physical row supports deterministic duplicate
        # resolution in large prefill writes; it is shared across all layers.
        return 2 * head_num * RECORD_BYTES * num_layers + 4


class Int8QSATokenToKVPool(MHATokenToKVPool):
    def __init__(self, *args, quant_method=None, **kwargs):
        method = quant_method or Int8QSAKVCacheMethod()
        if not isinstance(method, Int8QSAKVCacheMethod):
            raise ValueError("INT8 QSA pool requires its block-64 quantization method")
        if kwargs.get("post_capture_active") or kwargs.get("kv_cache_layout") not in (
            None,
            "nhd",
        ):
            raise ValueError("INT8 QSA KV requires ordinary NHD allocations")
        super().__init__(*args, quant_method=method, **kwargs)
        if self.use_hnd or self.head_dim != self.v_head_dim:
            raise ValueError(
                "INT8 QSA KV supports equal K/V head dimensions in NHD only"
            )

    def _slot_move_pointer_buffers(self):
        return self.quant_method.raw_k + self.quant_method.raw_v

    def _build_kv_buffer_descs(self):
        return [
            KvBufferDesc(
                f"{prefix}{i}", tuple(t.shape), row_bytes=t.stride(0), tokens_per_row=1
            )
            for prefix, tensors in (
                ("k", self.quant_method.raw_k),
                ("v", self.quant_method.raw_v),
            )
            for i, t in enumerate(tensors)
        ]

    def _mha_buffer_info(self):
        from sglang.srt.mem_cache.device_pool_info import (
            MHABufferInfo,
            KVBuffers,
            KVBufferLayout,
        )

        return MHABufferInfo(
            buffers=KVBuffers(
                k=tuple(self.quant_method.raw_k),
                v=tuple(self.quant_method.raw_v),
                k_data_ptrs=self.k_data_ptrs,
                v_data_ptrs=self.v_data_ptrs,
                layout=KVBufferLayout.NHD,
            ),
            page_size=self.page_size,
            head_num=self.head_num,
            head_dim=RECORD_BYTES,
            v_head_dim=RECORD_BYTES,
            dtype=torch.int8,
            device=self.quant_method.raw_k[0].device
            if self.layer_num
            else torch.device(self.device),
            scales=None,
        )

    def _pd_registerable_tensors(self):
        return self._slot_move_pointer_buffers()

    def get_kv_scale_buffer(self, layer_id):
        i = layer_id - self.start_layer
        return self.k_scale_buffer[i], self.v_scale_buffer[i]

    def _move_kv_cache_impl(self, tgt_loc, src_loc):
        # Gather before scatter preserves overlapping moves and inline scales.
        for raw in self._slot_move_pointer_buffers():
            raw[tgt_loc] = raw[src_loc]

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.get_device_module(self.device).synchronize()
        result = [
            [
                (
                    k[indices[i : i + self.cpu_offloading_chunk_size]].cpu(),
                    v[indices[i : i + self.cpu_offloading_chunk_size]].cpu(),
                )
                for i in range(0, len(indices), self.cpu_offloading_chunk_size)
            ]
            for k, v in zip(self.quant_method.raw_k, self.quant_method.raw_v)
        ]
        return result

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        for layer, (k, v) in enumerate(
            zip(self.quant_method.raw_k, self.quant_method.raw_v)
        ):
            for j, (kc, vc) in enumerate(kv_cache_cpu[layer]):
                dst = indices[
                    j * self.cpu_offloading_chunk_size : (j + 1)
                    * self.cpu_offloading_chunk_size
                ]
                if (
                    kc.shape != (len(dst), self.head_num, RECORD_BYTES)
                    or vc.shape != kc.shape
                ):
                    raise ValueError("INT8 KV CPU copy must include inline scales")
                k[dst] = kc.to(k.device)
                v[dst] = vc.to(v.device)
        torch.get_device_module(self.device).synchronize()
