# SPDX-License-Identifier: Apache-2.0
"""Opt-in MTP projections from the existing Strata runtime Q8_0 pack.

Only projection entries are imported. The raw GGUF remains authoritative for
norms, HC, router and experts; in particular, no runtime-pack Gemma +1 is added
to the model's existing norm Parameters.
"""
import hashlib
import os
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from sglang.srt.layers.quantization.gguf_dense_linear import (
    CompressedDenseProjectionMethod,
)


PROJECTIONS = {
    "fc_embedding": ("fc_embedding.weight",),
    "fc_hidden": ("fc_hidden.weight",),
    "model.layers.0.qkv_proj": tuple(
        f"self_attn.{name}_proj.weight" for name in ("q", "k", "v")
    ),
    "model.layers.0.o_proj": ("self_attn.o_proj.weight",),
    "model.layers.0.indexer.index_qk_proj": (
        "self_attn.indexer.index_qk_proj.weight",
    ),
    "model.layers.0.mlp.shared_expert.gate_up_proj": tuple(
        f"mlp.shared_expert.{name}_proj.weight" for name in ("gate", "up")
    ),
    "model.layers.0.mlp.shared_expert.down_proj": (
        "mlp.shared_expert.down_proj.weight",
    ),
}


def read_index(folder):
    folder = Path(folder)
    text = folder / "dense.txt"
    binary = folder / "dense.bin"
    limit = binary.stat().st_size
    result = {}
    intervals = []
    for line in text.read_text().splitlines():
        name, kind, rows, columns, offset, size = line.split()
        rows, columns, offset, size = map(int, (rows, columns, offset, size))
        if name in result or kind not in ("q8_0", "bf16", "f32"):
            raise ValueError("Duplicate or unsupported MTP runtime entry")
        expected = rows * columns * (2 if kind == "bf16" else 4)
        if kind == "q8_0":
            if columns % 32:
                raise ValueError("Q8_0 runtime columns must contain whole blocks")
            expected = rows * (columns // 32) * 34
        if rows < 1 or columns < 1 or offset < 0 or offset % 256 or size != expected or offset + size > limit:
            raise ValueError("Invalid MTP runtime shape, range or alignment")
        intervals.append((offset, offset + size))
        result[name] = dict(kind=kind, rows=rows, columns=columns, offset=offset, size=size)
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping MTP runtime entries")
    return result


def _forward(module, x):
    width, _ = module._gguf_mtp_original_shape
    if x.shape[-1] != width or x.dtype != torch.bfloat16:
        raise ValueError("MTP Q8 projection requires its original BF16 activation layout")
    shape = x.shape[:-1]
    if x.numel() == 0:
        output = x.new_empty((0, module._gguf_mtp_original_shape[1]))
    else:
        output = module.quant_method.apply(module, x.reshape(-1, width))
    output = output.reshape(*shape, output.shape[-1])
    return (output, None) if module._gguf_mtp_return_bias else output


def bind_projection(module, packed, *, width):
    """Retain the linear module/metadata, replacing only its inference method."""
    if (
        module.weight.device.type != "xpu"
        or torch.xpu.get_device_properties(module.weight.device).device_id != 0xE223
        or module.weight.dtype != torch.bfloat16
        or tuple(module.weight.shape) != (packed.shape[0], width)
        or packed.dtype != torch.uint8
        or packed.shape[1] != width // 32 * 34
        or getattr(module, "bias", None) is not None
    ):
        raise ValueError("MTP runtime projection does not match the loaded BF16 module")
    layout = "original_q8_0"
    device_weight = packed
    if os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_SOA_LIB"):
        from sglang.kernels.ops.quantization.mtp_q8_soa import MTPQ8SoAMethod, pack_rows

        method = MTPQ8SoAMethod(packed, width)
        device_weight = pack_rows(packed)
        if not torch.equal(pack_rows(device_weight, inverse=True), packed):
            raise ValueError("MTP SoA16 inverse byte check failed")
        layout = "q8_0_soa16"
    else:
        method = CompressedDenseProjectionMethod((8,), width, torch.bfloat16)
    original_bytes = module.weight.numel() * module.weight.element_size()
    module._gguf_mtp_original_shape = (width, packed.shape[0])
    module._gguf_mtp_return_bias = not isinstance(module, torch.nn.Linear)
    module.weight.requires_grad_(False)
    module.weight.data = device_weight.to(module.weight.device)
    module.quant_method = method
    method.process_weights_after_loading(module)
    module.forward = MethodType(_forward, module)
    return dict(original_BF16_bytes=original_bytes, packed_bytes=packed.numel(), layout=layout)


def install(model, folder):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Install MTP runtime projections before graph capture")
    folder = Path(folder)
    index = read_index(folder)
    modules = dict(model.named_modules())
    mapping = np.memmap(folder / "dense.bin", mode="r", dtype=np.uint8)
    pending = []
    # Validate and stage every entry before replacing any projection.
    for target, names in PROJECTIONS.items():
        module = modules[target]
        parts, widths = [], set()
        for name in names:
            entry = index[name]
            if entry["kind"] != "q8_0":
                raise ValueError(f"MTP projection is not Q8_0: {name}")
            widths.add(entry["columns"])
            raw = np.array(mapping[entry["offset"] : entry["offset"] + entry["size"]], copy=True)
            parts.append(torch.from_numpy(raw).view(entry["rows"], -1))
        if len(widths) != 1:
            raise ValueError("Merged MTP projection widths differ")
        width = widths.pop()
        packed = torch.cat(parts)
        if tuple(module.weight.shape) != (packed.shape[0], width):
            raise ValueError(f"MTP runtime geometry mismatch: {target}")
        pending.append((target, module, packed, width, names))
    rows = []
    for target, module, packed, width, names in pending:
        receipt = bind_projection(module, packed, width=width)
        rows.append(dict(module=target, sources=list(names), **receipt))
    return dict(
        dense_index_sha256=hashlib.sha256((folder / "dense.txt").read_bytes()).hexdigest(),
        dense_binary_sha256=hashlib.sha256(mapping).hexdigest(),
        projections=rows,
        contract="Existing Strata runtime Q8_0 projection bytes; BF16 activations. Larger rows use bounded Q8_0 reconstruction. Norms/HC/router/experts retained from raw GGUF. Numerical variant, not bitwise BF16 equivalence.",
    )
