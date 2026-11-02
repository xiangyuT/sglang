# SPDX-License-Identifier: Apache-2.0
"""Four-row B70 IQ4_XS row projections with BF16 activation operands.

A derivative cache retains nonlinear nibble codes and FP32 d*scale for each
32-value group. Decoding recovers the loaded BF16 weights exactly. Original
Parameters remain available to the existing prefill method. This saves active
weight-read traffic and adds a cache; it does not reduce total model VRAM.
"""

import ctypes
import os
from functools import lru_cache

import numpy as np
import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

_IQ4_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int16,
)


def pack_iq4_row_cache(raw, world, rank, *, gdn):
    """Pack an original 2560x6144 IQ4_XS matrix for one retained TP shard.

    Return CPU codes, FP32 gains and decoded BF16 reference values. The caller
    compares that reference against the independently loaded original weight.
    """
    source = np.asarray(raw)
    if (
        source.dtype != np.uint8
        or source.size != 2560 * 24 * 136
        or world not in (2, 4)
        or not 0 <= rank < world
        or not isinstance(gdn, bool)
    ):
        raise ValueError("IQ4 row cache requires original 2560x6144 bytes and TP2/4")
    source = source.reshape(2560, 24, 136)
    d = np.ascontiguousarray(source[:, :, :2]).view("<f2").reshape(2560, 24)
    d = d.astype(np.float32)
    high = np.ascontiguousarray(source[:, :, 2:4]).view("<u2").reshape(2560, 24)
    low, part = source[:, :, 4:8], np.arange(8)
    scales = ((low[:, :, part // 2] >> (4 * (part % 2))) & 15) | (
        ((high[:, :, None] >> (2 * part)) & 3) << 4
    )
    gains = d[:, :, None] * (scales.astype(np.int32) - 32).astype(np.float32)
    packed = source[:, :, 8:].reshape(2560, 24, 8, 16)
    codes = np.concatenate((packed & 15, packed >> 4), axis=-1).reshape(2560, 6144)
    columns = np.arange(6144)
    if gdn:
        columns = columns.reshape(3, 16, 128).transpose(1, 0, 2).reshape(-1)
    columns = columns.reshape(world, -1)[rank]
    codes = codes[:, columns]
    selected_gains = np.repeat(gains.reshape(2560, 192), 32, axis=1)[:, columns]
    groups = selected_gains.reshape(2560, -1, 32)
    if not np.all(groups == groups[:, :, :1]):
        raise ValueError("IQ4 TP shard crosses a scale group")
    gains = groups[:, :, 0]
    k = len(columns)
    tile = codes.reshape(160, 16, k // 32, 2, 16)
    pairs = (tile[..., 0::2] | (tile[..., 1::2] << 4)).transpose(0, 2, 3, 4, 1)
    scale_tiles = gains.reshape(160, 16, k // 32).transpose(0, 2, 1)
    reference = torch.from_numpy(gains.repeat(32, axis=1) * _IQ4_VALUES[codes])
    return (
        torch.from_numpy(pairs.copy().reshape(160, k // 32, 256)),
        torch.from_numpy(scale_tiles.copy()),
        reference.bfloat16(),
    )


@lru_cache(maxsize=1)
def _library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the IQ4 BF16 row library before graph capture")
    library = ctypes.CDLL(path)
    library.xpu_iq4_bf16_row.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
    library.xpu_iq4_bf16_row.restype = ctypes.c_int
    library.xpu_iq4_bf16_row_error.restype = ctypes.c_char_p
    return library


class _RowCacheMethod(UnquantizedLinearMethod):
    def __init__(self, parent, library):
        self.parent = parent
        self.library = library
        self._partials = {}

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def apply(self, layer, x, bias=None):
        if (
            x.device.type != "xpu"
            or x.device != layer.weight.device
            or x.ndim != 2
            or x.shape[0] != 4
            or x.shape[1] != layer.weight.shape[1]
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or bias is not None
        ):
            return self.parent.apply(layer, x, bias)
        stream = torch.xpu.current_stream(x.device)
        m, k = x.shape
        n = layer.weight.shape[0]
        key = (x.device, stream.sycl_queue)
        partial = self._partials.get(key)
        if partial is None:
            partial = torch.empty((8, m, n), dtype=torch.float32, device=x.device)
            self._partials[key] = partial
        output = torch.empty((m, n), dtype=x.dtype, device=x.device)
        status = self.library.xpu_iq4_bf16_row(
            stream.sycl_queue,
            layer.gguf_iq4_row_codes.data_ptr(),
            layer.gguf_iq4_row_gains.data_ptr(),
            x.data_ptr(),
            output.data_ptr(),
            None,
            partial.data_ptr(),
            m,
            n,
            k,
            8,
        )
        if status:
            raise RuntimeError(self.library.xpu_iq4_bf16_row_error().decode())
        for value in (
            x,
            output,
            partial,
            layer.gguf_iq4_row_codes,
            layer.gguf_iq4_row_gains,
        ):
            value.record_stream(stream)
        return output


def install_iq4_bf16_rows(model, shards, config, device):
    """Prepare eligible TP2 target caches after all original weights are loaded."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not (
        parallel.tp_size == parallel.moe_ep_size
        and parallel.tp_size == 2
        and parallel.moe_tp_size == 1
        and parallel.attn_dp_size == parallel.attn_cp_size == 1
        and (
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )
        == (16, 48, 128)
        and torch.xpu.get_device_properties(device).device_id == 0xE223
    ):
        raise ValueError("IQ4 BF16 row cache currently requires B70 native TP2/EP2")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_IQ4_ROW_BF16_LIB"))
    tensors = {
        tensor.name: (shard, tensor) for shard in shards for tensor in shard.tensors
    }
    records = []
    for index, layer in enumerate(model.model.layers):
        gdn = hasattr(layer, "linear_attn")
        suffix = "ssm_out" if gdn else "attn_output"
        name = f"blk.{index}.{suffix}.weight"
        shard, tensor = tensors[name]
        if tensor.type_id != 23:
            continue
        module = layer.linear_attn.out_proj if gdn else layer.o_proj
        if not (
            isinstance(module.quant_method, UnquantizedLinearMethod)
            and not isinstance(module.quant_method, _RowCacheMethod)
            and module.weight.dtype == torch.bfloat16
            and module.weight.shape == (2560, 6144 // parallel.tp_size)
            and module.weight.is_contiguous()
        ):
            raise ValueError(f"Unsupported original BF16 row shard: {name}")
        raw = np.frombuffer(
            shard.read_tensor_range(tensor, 0, tensor.expected_bytes()), dtype=np.uint8
        )
        codes, gains, reference = pack_iq4_row_cache(
            raw, parallel.tp_size, parallel.tp_rank, gdn=gdn
        )
        if not torch.equal(
            reference.view(torch.int16), module.weight.detach().cpu().view(torch.int16)
        ):
            raise ValueError(
                f"IQ4 derivative cache differs from loaded BF16 weights: {name}"
            )
        module.register_buffer("gguf_iq4_row_codes", codes.to(device), persistent=False)
        module.register_buffer("gguf_iq4_row_gains", gains.to(device), persistent=False)
        module.quant_method = _RowCacheMethod(module.quant_method, library)
        records.append(
            dict(
                name=name,
                reference_bytes=module.weight.numel() * 2,
                cache_bytes=codes.numel() + gains.numel() * 4,
                reconstructed_BF16_bits_exact=True,
                original_weight_retained=True,
            )
        )
    model.gguf_iq4_bf16_row_receipt = records
    return records
