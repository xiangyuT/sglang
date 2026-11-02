# SPDX-License-Identifier: Apache-2.0
"""Bounded BF16/Q8 expert pipeline on the current XPU stream."""

import os

import torch

from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.kernels.ops.quantization.gguf_expert_geometry import expert_field_sizes
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES


def fused_q8_experts(
    x,
    gate,
    up,
    down,
    kind,
    slots,
    weights,
    workspaces,
    *,
    grouping=None,
    slot_addresses=None,
    zero_slot=None,
    paged_zero_expert=None,
    down_kind=42,
):
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("Fused Q8 activations and routing weights must be matrices")
    tokens, hidden = x.shape
    intermediate, topk = 640, weights.shape[1]
    max_mixed_rows = (32 if os.environ.get("SGLANG_XPU_GGUF_EXPERT_UNION_OVERFLOW") == "1"
                      else 16)
    geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
    sizes = expert_field_sizes(kind, down_kind)
    fields = (gate, up, down)
    encoded_layout_valid = all(t.is_contiguous() for t in fields)
    if (not encoded_layout_valid and slot_addresses is not None
            and kind in (21, 23) and gate.ndim == 2):
        from sglang.kernels.ops.quantization.gguf_gpu_cache import expert_aos_views_valid

        encoded_layout_valid = expert_aos_views_valid(fields, gate.shape[0], sizes, x.device)
    if not (
        kind in (16, 21, 22, 23, 29, 42)
        and geometry is not None
        and x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and hidden == 2560
        and (1 <= tokens <= 4 or (
            4 < tokens <= max_mixed_rows and kind in (21, 23) and down_kind in (8, 20)
            and grouping is not None and slot_addresses is not None
        ))
        and 1 <= topk <= 10
        and gate.dtype == up.dtype == down.dtype == torch.uint8
        and gate.ndim == up.ndim == down.ndim == 2
        and gate.shape == up.shape
        and gate.shape[0] == down.shape[0]
        and gate.shape[1] == intermediate * (hidden // geometry[0] * geometry[1])
        and down.shape[1] == sizes[2]
        and slots.dtype == torch.int32
        and slots.ndim == 1
        and slots.numel() == tokens * topk
        and weights.dtype in (torch.float32, torch.bfloat16)
        and weights.shape[0] == tokens
        and all(t.device == x.device for t in (gate, up, down, slots, weights))
        and encoded_layout_valid and slots.is_contiguous() and weights.is_contiguous()
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    ):
        raise ValueError("Unsupported B70 fused GGUF Q8 expert geometry")
    capacity = gate.shape[0]
    if slot_addresses is not None:
        if not (
            slot_addresses.device == x.device
            and slot_addresses.dtype == torch.uint64
            and slot_addresses.ndim == 2
            and slot_addresses.shape[1] == 3
            and capacity <= slot_addresses.shape[0] <= 512
            and slot_addresses.is_contiguous()
            and (kind != 22 or grouping is not None)
        ):
            raise ValueError("Invalid stable GGUF expert address table")
        # This is the table's bound. The live resident capacity and pointers
        # are maintained by GPUExpertCache, including chunks added after capture.
        capacity = slot_addresses.shape[0]
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_moe_q8"):
        raise ValueError("Rebuild the GGUF library for the fused Q8 expert pipeline")
    if (
        down_kind == 8 and grouping is not None and slot_addresses is not None
        and topk == 10
        and os.environ.get("SGLANG_XPU_GGUF_MIXED_PACKETS") == "1"
        and os.environ.get("SGLANG_XPU_GGUF_Q8_DOWN_PACKETS") == "1"
        and not hasattr(library, "xpu_gguf_q8_down_packets")
    ):
        raise ValueError("Rebuild the GGUF library for Q8 expert-down packets")
    if (
        kind == 22
        and tokens == 4
        and os.environ.get("SGLANG_XPU_GGUF_IQ2S_PLANE_SPLIT") == "1"
        and not hasattr(library, "xpu_gguf_iq2s_gu_ptr_split4")
    ):
        raise ValueError("Rebuild the GGUF library for IQ2S plane splitting")
    if (
        kind != 22
        and down_kind == 42
        and os.environ.get("SGLANG_XPU_GGUF_RAW_Q2_DPAS") == "1"
        and not hasattr(library, "xpu_raw_q2_down")
    ):
        raise ValueError("Rebuild the GGUF library for raw Q2 down DPAS")
    stream = torch.xpu.current_stream(x.device)
    arena_enabled = (
        os.environ.get("SGLANG_XPU_GGUF_EXPERT_WORKSPACE_ARENA") == "1"
        and paged_zero_expert is not None
        and kind in (21, 23) and down_kind in (8, 20) and topk == 10
    )
    if arena_enabled and not hasattr(library, "xpu_gguf_moe_q8_arena_supported"):
        raise ValueError("Rebuild the GGUF library for masked expert arena rows")
    key = (x.device, stream.sycl_queue, tokens, topk)
    if arena_enabled:
        key += ("arena",)
    if key not in workspaces:
        routes = tokens * topk

        def floats(*shape):
            return torch.empty(shape, device=x.device, dtype=torch.float32)

        down_buffer = floats(routes, hidden)
        if arena_enabled:
            # Gate/up/activation finish before quantized down consumes scratch.
            # Keep input FP32, output and every stream/shape separately owned.
            flat = down_buffer.view(-1)
            count = routes * intermediate
            gate_buffer, up_buffer, activation_buffer = (
                flat[i * count : (i + 1) * count].view(routes, intermediate)
                for i in range(3)
            )
        else:
            gate_buffer, up_buffer, activation_buffer = (
                floats(routes, intermediate) for _ in range(3)
            )
        workspaces[key] = dict(
            x=floats(tokens, hidden),
            gate=gate_buffer,
            up=up_buffer,
            activation=activation_buffer,
            down=down_buffer,
            scratch=torch.empty(
                max(tokens * hidden, routes * intermediate) // 32 * 36,
                device=x.device,
                dtype=torch.uint8,
            ),
            token_rows=torch.arange(
                tokens, device=x.device, dtype=torch.int32
            ).repeat_interleave(topk),
            output=torch.empty_like(x),
        )
    ws = workspaces[key]
    from sglang.kernels.ops.quantization.gguf_resident_prepare import ResidentRoutes

    input_prepared = False
    if paged_zero_expert is not None:
        if isinstance(grouping, ResidentRoutes) or grouping is None or zero_slot is not None:
            raise ValueError("Paged preparation requires the ordinary GPU cache grouping")
        from sglang.kernels.ops.quantization.gguf_resident_prepare import prepare_paged

        prepare_input = (
            os.environ.get("SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8") == "1"
            and os.environ.get("SGLANG_XPU_GGUF_FUSED_QUANT") == "1"
            and slot_addresses is not None
        )
        if prepare_input and not hasattr(library, "xpu_gguf_moe_q8_addressed_prepared"):
            raise ValueError("Rebuild the GGUF library for prepared Q8 input")
        grouping, input_prepared = prepare_paged(
            grouping, ws, stream, paged_zero_expert, x=x if prepare_input else None
        )
    elif isinstance(grouping, ResidentRoutes):
        prepare_input = (
            os.environ.get("SGLANG_XPU_GGUF_RESIDENT_INPUT_Q8") == "1"
            and os.environ.get("SGLANG_XPU_GGUF_FUSED_QUANT") == "1"
            and slot_addresses is not None
        )
        if prepare_input and not hasattr(library, "xpu_gguf_moe_q8_addressed_prepared"):
            raise ValueError("Rebuild the GGUF library for resident-produced Q8 input")
        input_prepared = grouping.prepare_resident(
            ws, stream, zero_slot, x=x if prepare_input else None
        )
    elif zero_slot is not None:
        # Resident EP sorts owned IDs before its final zero-encoded expert.
        # Clip that last active group and every inactive trailing range, without
        # changing any real expert's token order, address or reduction tree.
        if grouping is None or kind not in (16, 21, 22, 23, 29) or zero_slot != capacity - 1:
            raise ValueError("Zero-group clipping requires the final resident EP slot")
        if "ep_starts" not in ws:
            ws["ep_starts"] = torch.empty_like(grouping.starts)
            ws["ep_clip"] = torch.empty((), device=x.device, dtype=torch.int32)
        torch.amin(torch.where(grouping.group_slots == zero_slot,
                              grouping.starts[:-1], slots.numel()), out=ws["ep_clip"])
        torch.minimum(grouping.starts, ws["ep_clip"], out=ws["ep_starts"])
        from dataclasses import replace
        grouping = replace(grouping, starts=ws["ep_starts"])
        # Skipped rows remain finite through activation quantization and route
        # weighting; clearing also removes data left by the preceding replay.
        ws["gate"].zero_()
        ws["up"].zero_()
        ws["down"].zero_()
    arguments = (
        stream.sycl_queue,
        x.data_ptr(),
        gate.data_ptr(),
        up.data_ptr(),
        down.data_ptr(),
        slots.data_ptr(),
        ws["token_rows"].data_ptr(),
        weights.data_ptr(),
        ws["output"].data_ptr(),
        ws["x"].data_ptr(),
        ws["gate"].data_ptr(),
        ws["up"].data_ptr(),
        ws["activation"].data_ptr(),
        ws["down"].data_ptr(),
        ws["scratch"].data_ptr(),
        kind,
        hidden,
        intermediate,
        tokens,
        topk,
        capacity,
        gate.shape[1],
        gate.shape[1] // intermediate,
        down.shape[1],
        down.shape[1] // hidden,
        int(weights.dtype == torch.bfloat16),
    )
    grouped_arguments = (0, 0, 0, 0, 0, 4)
    if grouping is not None:
        if not (
            kind in (16, 21, 22, 23, 29)
            and 1 <= grouping.groups <= slots.numel()
            and grouping.group_slots.numel() == grouping.groups
            and grouping.starts.numel() == grouping.groups + 1
            and grouping.token_rows.numel() == slots.numel()
            and grouping.destinations.numel() == slots.numel()
            and all(
                t.dtype == torch.int32 and t.device == x.device and t.is_contiguous()
                for t in (
                    grouping.data,
                    grouping.group_slots,
                    grouping.starts,
                    grouping.token_rows,
                    grouping.destinations,
                )
            )
        ):
            raise ValueError("Invalid grouped packed expert routing metadata")
        grouped_arguments = (
            grouping.group_slots.data_ptr(),
            grouping.starts.data_ptr(),
            grouping.token_rows.data_ptr(),
            grouping.destinations.data_ptr(),
            grouping.groups,
            4,
        )
    if slot_addresses is not None:
        if not hasattr(library, "xpu_gguf_moe_q8_addressed"):
            raise ValueError("Rebuild the GGUF library for addressed expert MoE")
        mixed = kind in (21, 23) or down_kind != 42
        if mixed:
            function = getattr(library, "xpu_gguf_moe_q8_mixed_addressed", None)
            if function is None:
                raise ValueError("Rebuild the GGUF library for mixed expert MoE")
            status = function(
                *arguments, *grouped_arguments, slot_addresses.data_ptr(),
                down_kind, int(input_prepared),
            )
        else:
            function = (library.xpu_gguf_moe_q8_addressed_prepared if input_prepared
                        else library.xpu_gguf_moe_q8_addressed)
            status = function(
                *arguments, *grouped_arguments, slot_addresses.data_ptr()
            )
        slot_addresses.record_stream(stream)
    elif grouping is not None:
        if not hasattr(library, "xpu_gguf_moe_q8_pvr"):
            raise ValueError("Rebuild the GGUF library for grouped IQ2S MoE")
        status = library.xpu_gguf_moe_q8_pvr(*arguments, *grouped_arguments)
    else:
        status = library.xpu_gguf_moe_q8(*arguments)
    if grouping is not None:
        grouping.data.record_stream(stream)
    if status:
        raise RuntimeError((status, library.xpu_gguf_error().decode()))
    for tensor in (x, gate, up, down, slots, weights, *ws.values()):
        tensor.record_stream(stream)
    return ws["output"]
