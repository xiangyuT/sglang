import logging

import torch
from sgl_kernel import gdn_attention as sgl_kernel_gdn_attention

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import mamba_cache_chunk_size

logger = logging.getLogger(__name__)


class XpuGDNAttnBackend(GDNAttnBackend):
    """XPU specialization of ``GDNAttnBackend``.

    Adds an optional fused path that dispatches the whole conv1d + gating +
    delta-rule pipeline to the vendored vLLM SYCL kernel exposed as
    ``torch.ops.sgl_kernel.gdn_attention``. This is opt-in via
    ``--linear-attn-backend intel_xpu`` (the default remains ``triton``, same
    as other platforms).
    """

    def _decode_initial_state(self, bs, device):
        if not envs.SGLANG_XPU_GDN_INITIAL_STATE_CACHE.get():
            return torch.ones(bs, dtype=torch.bool, device=device)

        cache = getattr(self, "_decode_initial_state_cache", None)
        if cache is None:
            cache = self._decode_initial_state_cache = {}
        key = (device, bs)
        state = cache.get(key)
        if state is None:
            if torch.xpu.is_current_stream_capturing():
                # Do not cache an unexecuted graph allocation for another graph
                # or stream to consume. Keep the original fill on a cold key.
                return torch.ones(bs, dtype=torch.bool, device=device)
            state = torch.ones(bs, dtype=torch.bool, device=device)
            # One-time warmup initialization; the shared constant must be ready
            # before any capture stream can read it. No per-token synchronize.
            torch.xpu.current_stream(device).synchronize()
            cache[key] = state
            logger.info(
                "XPU GDN decode initial-state cache: device=%s bs=%d; "
                "read-only true flags initialized before capture",
                device, bs,
            )
        state.record_stream(torch.xpu.current_stream(device))
        return state

    def supports_fused_gdn(self, layer, forward_batch: ForwardBatch) -> bool:
        """Conservative guard: only the plain decode / non-prefix-cached,
        non-speculative extend cases are handled by the fused kernel."""
        mode = forward_batch.forward_mode
        backends = self.linear_attn_backends
        selected = (
            backends.verify
            if mode.is_target_verify()
            else (backends.decode if mode.is_decode_or_idle() else backends.prefill)
        )
        if not selected.is_intel_xpu():
            return False
        if not hasattr(torch.ops.sgl_kernel, "gdn_attention"):
            # User explicitly asked for intel_xpu but the op isn't built.
            raise RuntimeError(
                "--linear-attn-backend intel_xpu requires the "
                "torch.ops.sgl_kernel.gdn_attention op, but it is not "
                "available. Rebuild sgl-kernel-xpu or use "
                "--linear-attn-backend triton."
            )
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return False
        fm = self.forward_metadata
        if getattr(fm, "has_mamba_track_mask", False) and not self._has_cpu_prefill_track_metadata(forward_batch):
            return False
        if getattr(fm, "query_start_loc", None) is None:
            return False
        # GDN (not KDA) shared weights must be plain tensors
        if not isinstance(layer.conv_weights, torch.Tensor):
            return False
        if layer.bias is not None and not isinstance(layer.bias, torch.Tensor):
            return False
        return True

    def forward_fused_gdn(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
        preconv=None,
    ):
        """Run the fused SYCL GDN op and return ``(core_attn_out, z)``.

        Caches stay in the SGLang pool layout and are updated in place. The conv
        pool is ``[cache, dim, width-1]``; we pass a transposed view so the op sees
        its logical ``[cache, width-1, dim]`` layout while the kernels index via
        explicit width/dim strides (no gather/transpose/scatter copies). The ssm
        pool already matches the op layout. ``mamba_cache_indices`` indexes the
        full pool directly for both conv and ssm.
        """
        fm = self.forward_metadata
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]  # [cache, dim, width-1]
        ssm_states = layer_cache.temporal  # [cache, nv, hv, hk]
        cache_indices = fm.mamba_cache_indices
        query_start_loc = fm.query_start_loc

        device = projected_states_qkvz.device
        dtype = projected_states_qkvz.dtype
        bs = forward_batch.batch_size
        num_actual_tokens = projected_states_qkvz.shape[0]

        if preconv is not None:
            mixed, z, state_pointer, index_pointer, layer_id = preconv
            assert state_pointer == conv_states.data_ptr()
            assert index_pointer == cache_indices.data_ptr() and layer_id == layer.layer_id
            assert forward_batch.forward_mode.is_decode() and bs == num_actual_tokens == 1
            from sglang.kernels.ops.attention.xpu_gdn_decode import recurrent_only
            return recurrent_only(
                mixed, projected_states_ba[:, 32:], projected_states_ba[:, :32],
                layer.A_log, layer.dt_bias, ssm_states, cache_indices, z,
            )

        if (
            envs.SGLANG_XPU_GDN_DECODE.get()
            and forward_batch.forward_mode.is_decode_or_idle()
            and bs == num_actual_tokens == 1
            and layer.activation in ("silu", "swish")
            and (layer.num_k_heads, layer.num_v_heads) == (16, 32)
            and (layer.head_k_dim, layer.head_v_dim) == (128, 128)
            and cache_indices.shape == (1,)
            and cache_indices.device == device
            and cache_indices.dtype in (torch.int32, torch.int64)
            and cache_indices.is_contiguous()
        ):
            from sglang.kernels.ops.attention.xpu_gdn_decode import (
                can_use_xpu_gdn_decode,
                xpu_gdn_decode,
            )

            if can_use_xpu_gdn_decode(
                projected_states_qkvz,
                projected_states_ba,
                conv_states,
                ssm_states,
                layer.conv_weights,
                layer.bias,
                layer.A_log,
                layer.dt_bias,
            ):
                if not getattr(self, "_b70_decode_logged", False):
                    logger.info(
                        "Using B70 single-token GDN decode: FP32 conv products, "
                        "BF16 a+dt_bias, FP32 beta."
                    )
                    self._b70_decode_logged = True
                return xpu_gdn_decode(
                    projected_states_qkvz,
                    projected_states_ba,
                    conv_states,
                    ssm_states,
                    layer.conv_weights,
                    layer.A_log,
                    layer.dt_bias,
                    cache_indices,
                )

        if getattr(fm, "has_mamba_track_mask", False):
            return self._forward_tracked_prefill(
                layer, forward_batch, projected_states_qkvz, projected_states_ba,
                conv_states, ssm_states, cache_indices,
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            num_decodes, num_prefills = bs, 0
            has_initial_state = self._decode_initial_state(bs, device)
        else:
            num_decodes, num_prefills = 0, bs
            has_initial_state = forward_batch.extend_prefix_lens > 0

        return self._run_pool_gdn(
            layer, projected_states_qkvz, projected_states_ba, conv_states,
            ssm_states, cache_indices, query_start_loc, has_initial_state,
            num_prefills=num_prefills, num_decodes=num_decodes,
        )

    def _forward_tracked_prefill(self, layer, fb, qkvz, ba, conv, ssm, indices):
        """Checkpoint the same native prefill at the requested radix boundary.

        Metadata is already on CPU. The common all-final case keeps the full
        batched kernel; an interior boundary splits only that request at an
        aligned checkpoint. Every segment writes its live private slot, and
        copying to the extra-buffer slot leaves the shared prefix immutable.
        """
        masks = fb.mamba_prefill_track_mask_cpu
        lengths = fb.extend_seq_lens_cpu
        prefixes = fb.extend_prefix_lens_cpu
        # Match ForwardBatch.mamba_track_aligned_lens(). The scheduler encodes
        # an unaligned sequence end (or boundary+1) to request an intermediate
        # state. Saving at that encoded offset corrupts the aligned radix state.
        chunk_size = mamba_cache_chunk_size()
        cuts = [((int(end) - int(prefix)) // chunk_size) * chunk_size
                if mask else int(length)
                for mask, end, prefix, length in zip(
                    masks, fb.mamba_track_seqlens_cpu, prefixes, lengths)]
        if any(mask and not 0 < cut <= int(length)
               for mask, cut, length in zip(masks, cuts, lengths)):
            raise ValueError("Invalid native GDN prefill checkpoint boundary")

        def save(row):
            src = indices[row:row + 1].long()
            dst = fb.mamba_track_indices[row:row + 1].long()
            conv.index_copy_(0, dst, conv.index_select(0, src))
            ssm.index_copy_(0, dst, ssm.index_select(0, src))

        if all(cut == int(length) for cut, length in zip(cuts, lengths)):
            result = self._run_pool_gdn(
                layer, qkvz, ba, conv, ssm, indices,
                self.forward_metadata.query_start_loc, fb.extend_prefix_lens > 0,
                num_prefills=fb.batch_size, num_decodes=0,
            )
            for row, mask in enumerate(masks):
                if mask:
                    save(row)
            return result

        output = torch.empty((qkvz.shape[0], layer.num_v_heads, layer.head_v_dim),
                             dtype=qkvz.dtype, device=qkvz.device)
        z = torch.empty_like(output)
        offset = 0
        for row, (length, cut, prefix, mask) in enumerate(zip(lengths, cuts, prefixes, masks)):
            length = int(length)
            segments = ((0, cut), (cut, length)) if cut < length else ((0, length),)
            for begin, end in segments:
                qsl = torch.tensor([0, end - begin], dtype=torch.int32, device=qkvz.device)
                has_initial = torch.tensor([int(prefix) > 0 or begin > 0], device=qkvz.device)
                part, part_z = self._run_pool_gdn(
                    layer, qkvz[offset + begin:offset + end], ba[offset + begin:offset + end],
                    conv, ssm, indices[row:row + 1], qsl, has_initial,
                    num_prefills=1, num_decodes=0,
                )
                output[offset + begin:offset + end].copy_(part)
                z[offset + begin:offset + end].copy_(part_z)
                if mask and end == cut:
                    save(row)
            offset += length
        return output, z

    @staticmethod
    def _run_pool_gdn(layer, projected_states_qkvz, projected_states_ba,
                      conv_states, ssm_states, cache_indices, query_start_loc,
                      has_initial_state, *, num_prefills, num_decodes):
        num_actual_tokens = projected_states_qkvz.shape[0]
        dtype, device = projected_states_qkvz.dtype, projected_states_qkvz.device
        # Full-pool, zero-copy: transposed view for conv + native ssm pool, indexed
        # directly by the full-pool cache indices.
        conv_view = conv_states.transpose(1, 2)  # [cache, width-1, dim] view
        state_idx = cache_indices.to(torch.int32).contiguous()

        core_attn_out = torch.empty(
            num_actual_tokens,
            layer.num_v_heads,
            layer.head_v_dim,
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)

        sgl_kernel_gdn_attention(
            core_attn_out=core_attn_out,
            z=z,
            projected_states_qkvz=projected_states_qkvz,
            projected_states_ba=projected_states_ba,
            num_k_heads=layer.num_k_heads,
            num_v_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            conv_state=conv_view,
            ssm_state=ssm_states,
            conv_weights=layer.conv_weights,
            conv_bias=layer.bias,
            activation=layer.activation,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_idx,
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=num_actual_tokens,
            # Heads/tensors are already per-rank sharded; kernel needs no
            # further in-kernel sharding, so this is always 1, not --tp-size.
            tp_size=1,
            reorder_input=True,
        )

        # conv/ssm states were updated in place via the pool views.
        return core_attn_out, z
