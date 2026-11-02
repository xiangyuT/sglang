# SPDX-License-Identifier: Apache-2.0
"""B70 GGUF HC normalization and paired down/injection projection.

Retain the model's BF16 operands and original parameter views. Only the
bounded decode rows use the paired projection; larger prefill stays original.
"""

import ctypes
from functools import lru_cache
import os

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_sycl import _library


def eligible(x):
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4
        and x.shape[1] == 10240
        and x.is_contiguous()
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


def prefill_norm_eligible(x):
    return (
        os.environ.get("SGLANG_XPU_GGUF_HC_PREFILL_NORM") == "1"
        and x.device.type == "xpu" and x.dtype == torch.bfloat16
        and x.ndim == 2 and 5 <= x.shape[0] <= 4096
        and x.shape[1] == 10240 and x.is_contiguous()
        and torch.xpu.get_device_properties(x.device).device_id == 0xE223
    )


@lru_cache(None)
def _prefill_norm_library(path):
    if not path:
        raise RuntimeError("Set SGLANG_XPU_GGUF_HC_PREFILL_NORM_LIB")
    library = ctypes.CDLL(path)
    library.xpu_hc_prefill_norm.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int, ctypes.c_float, ctypes.c_int,
    ]
    library.xpu_hc_prefill_norm.restype = ctypes.c_int
    library.xpu_hc_prefill_norm_error.restype = ctypes.c_char_p
    return library


def normalize(x, weight, eps):
    prefill = prefill_norm_eligible(x)
    if not (
        (eligible(x) or prefill)
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and weight.shape == (10240,)
        and weight.is_contiguous()
        and eps > 0
    ):
        raise ValueError("Unsupported B70 HC normalization geometry")
    library = (_prefill_norm_library(os.environ.get("SGLANG_XPU_GGUF_HC_PREFILL_NORM_LIB"))
               if prefill else _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")))
    if not prefill and not hasattr(library, "xpu_gguf_hc_norm"):
        raise ValueError("Rebuild the GGUF library for native HC normalization")
    stream = torch.xpu.current_stream(x.device)
    output = torch.empty_like(x)
    means = torch.empty(x.shape[0] * 4, device=x.device, dtype=torch.float32)
    arguments = (
        stream.sycl_queue, x.data_ptr(), weight.data_ptr(), output.data_ptr(),
        means.data_ptr(), x.shape[0], eps,
    )
    if prefill:
        status = library.xpu_hc_prefill_norm(*arguments, 2560)
        error = library.xpu_hc_prefill_norm_error
    elif os.environ.get("SGLANG_XPU_GGUF_HC_ESIMD") == "1":
        if not hasattr(library, "xpu_hc_esimd"):
            raise RuntimeError("Rebuild the GGUF library for ESIMD HC normalization")
        status = library.xpu_hc_esimd(*arguments, 2560)
        error = library.xpu_hc_esimd_error
    else:
        status = library.xpu_gguf_hc_norm(*arguments)
        error = library.xpu_gguf_hc_error
    if status:
        raise RuntimeError((status, error().decode()))
    for value in (x, weight, output, means):
        value.record_stream(stream)
    return output


@torch.compile
def _mix_from_scores(normed, scores, up):
    low = F.silu(scores[:, :320] / 4)
    high = F.linear(low, up)
    output = (torch.sigmoid(high).view(-1, 4, 2560) * normed.view(-1, 4, 2560)).mean(
        dim=-2
    )
    return output, scores[:, 320:324]


@torch.compile
def _torch_mix(normed, down_gate, up):
    return _mix_from_scores(normed, F.linear(normed, down_gate), up)


@torch.compile
def _low_from_scores(scores):
    return F.silu(scores[:, :320] / 4)


@lru_cache(maxsize=1)
def _up_mix_library(path):
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the HC up/mix library before graph capture")
    library = ctypes.CDLL(path)
    library.xpu_hc_up_mix.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    library.xpu_hc_up_mix.restype = ctypes.c_int
    library.xpu_hc_up_mix_error.restype = ctypes.c_char_p
    if hasattr(library, "xpu_hc_down_low"):
        library.xpu_hc_down_low.argtypes = [ctypes.c_void_p] * 6
        library.xpu_hc_down_low.restype = ctypes.c_int
        library.xpu_hc_down_low_error.restype = ctypes.c_char_p
    return library


def _native_down_low(normed, down_gate, path):
    """Preserve the BF16 score boundary while fusing reduction and SiLU."""
    if not (
        eligible(normed)
        and normed.shape == (4, 10240)
        and down_gate.shape == (324, 10240)
        and down_gate.dtype == torch.bfloat16
        and down_gate.device == normed.device
        and down_gate.is_contiguous()
    ):
        raise ValueError("Unsupported native HC down/low geometry")
    library = _up_mix_library(path)
    if not hasattr(library, "xpu_hc_down_low"):
        raise ValueError("Rebuild the HC library for native down/low")
    scores = torch.empty((4, 324), dtype=torch.bfloat16, device=normed.device)
    low = torch.empty((4, 320), dtype=torch.bfloat16, device=normed.device)
    partial = torch.empty((10, 4, 324), dtype=torch.float32, device=normed.device)
    stream = torch.xpu.current_stream(normed.device)
    status = library.xpu_hc_down_low(
        stream.sycl_queue, normed.data_ptr(), down_gate.data_ptr(),
        scores.data_ptr(), partial.data_ptr(), low.data_ptr(),
    )
    if status:
        raise RuntimeError(library.xpu_hc_down_low_error().decode())
    for value in (normed, down_gate, scores, low, partial):
        value.record_stream(stream)
    return scores, low


def _native_mix_from_scores(normed, scores, up, path, low=None):
    """Keep the BF16 high projection and sequential four-branch mean."""
    if not (
        eligible(normed)
        and normed.shape == (4, 10240)
        and scores.shape == (4, 324)
        and up.shape == (10240, 320)
        and scores.dtype == up.dtype == torch.bfloat16
        and normed.device == scores.device == up.device
        and up.is_contiguous()
    ):
        raise ValueError("Unsupported native HC up/mix geometry")
    library = _up_mix_library(path)
    if low is None:
        low = _low_from_scores(scores)
    elif not (
        low.shape == (4, 320)
        and low.dtype == torch.bfloat16
        and low.device == normed.device
        and low.is_contiguous()
    ):
        raise ValueError("Unsupported native HC low activation layout")
    output = torch.empty((4, 2560), dtype=torch.bfloat16, device=normed.device)
    stream = torch.xpu.current_stream(normed.device)
    tile = int(os.environ.get("SGLANG_XPU_GGUF_HC_UP_MIX_TILE", "1"))
    if tile not in (1, 2, 4):
        raise ValueError("HC up/mix tile must be 1, 2 or 4")
    status = library.xpu_hc_up_mix(
        stream.sycl_queue, low.data_ptr(), up.data_ptr(), normed.data_ptr(),
        output.data_ptr(), tile, 1,
    )
    if status:
        raise RuntimeError(library.xpu_hc_up_mix_error().decode())
    for value in (normed, scores, up, low, output):
        value.record_stream(stream)
    return output, scores[:, 320:324]


def mix(normed, down_gate, up):
    from sglang.kernels.ops.gemm.gguf_hc_wide_sycl import (
        eligible as wide_eligible,
        mix as wide_mix,
    )

    if wide_eligible(normed):
        return wide_mix(normed, down_gate, up)
    if (
        os.environ.get("SGLANG_XPU_GGUF_HC_DOWN_DPAS") == "1"
        and eligible(normed)
        and normed.shape[0] == 4
        and down_gate.dtype == torch.bfloat16
        and down_gate.shape == (324, 10240)
        and down_gate.is_contiguous()
    ):
        up_mix_path = os.environ.get("SGLANG_XPU_GGUF_HC_UP_MIX_LIB")
        low = None
        if os.environ.get("SGLANG_XPU_GGUF_HC_DOWN_LOW") == "1":
            scores, low = _native_down_low(normed, down_gate, up_mix_path)
        else:
            library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
            scores = torch.empty((4, 324), dtype=torch.bfloat16, device=normed.device)
            partial = torch.empty((10, 4, 324), dtype=torch.float32, device=normed.device)
            stream = torch.xpu.current_stream(normed.device)
            rc = library.xpu_bf16_raw_2da_gemm(
                stream.sycl_queue, normed.data_ptr(), down_gate.data_ptr(),
                scores.data_ptr(), partial.data_ptr(), None, 4, 324, 10240, 10,
            )
            if rc:
                raise RuntimeError(library.xpu_bf16_raw_2da_error().decode())
            for tensor in (normed, down_gate, scores, partial):
                tensor.record_stream(stream)
        if up_mix_path:
            return _native_mix_from_scores(normed, scores, up, up_mix_path, low=low)
        return _mix_from_scores(normed, scores, up)
    return _torch_mix(normed, down_gate, up)


def combine_and_normalize(block, residual, scores, weight, eps):
    if not (
        eligible(residual)
        and block.shape == (residual.shape[0], 2560)
        and block.dtype == scores.dtype == weight.dtype == torch.bfloat16
        and block.is_contiguous()
        and scores.shape == (residual.shape[0], 4)
        and scores.stride(1) == 1
        and scores.stride(0) >= 4
        and weight.shape == (10240,)
        and weight.is_contiguous()
        and all(t.device == residual.device for t in (block, scores, weight))
        and eps > 0
    ):
        raise ValueError("Unsupported B70 HC combine/normalization layout")
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    combined, normed = torch.empty_like(residual), torch.empty_like(residual)
    means = torch.empty(residual.shape[0] * 4, dtype=torch.float32, device=residual.device)
    stream = torch.xpu.current_stream(residual.device)
    rc = library.xpu_gguf_hc_combine_norm(
        stream.sycl_queue, residual.data_ptr(), block.data_ptr(), scores.data_ptr(),
        weight.data_ptr(), combined.data_ptr(), normed.data_ptr(), means.data_ptr(),
        residual.shape[0], eps, scores.stride(0), 0,
    )
    if rc:
        raise RuntimeError(library.xpu_gguf_hc_combine_norm_error().decode())
    for tensor in (residual, block, scores, weight, combined, normed, means):
        tensor.record_stream(stream)
    return combined, normed


@torch.compile
def combine(block, residual, scores):
    gate = 2 * torch.sigmoid(scores / 4)
    return (
        residual.view(-1, 4, 2560) + block.unsqueeze(-2) * gate.unsqueeze(-1)
    ).flatten(-2)


def install(model):
    """Initialize the opt-in layout before any graph is captured."""
    from sglang.srt.layers.hyperconnection import GatedResidual

    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Install HC weight views before graph capture")
    from sglang.kernels.ops.gemm.gguf_hc_wide_sycl import warm_from_env

    warm_from_env()
    library = _library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"))
    if not hasattr(library, "xpu_gguf_hc_norm"):
        raise ValueError("Rebuild the GGUF library for native HC")
    up_mix_path = os.environ.get("SGLANG_XPU_GGUF_HC_UP_MIX_LIB")
    down_low = os.environ.get("SGLANG_XPU_GGUF_HC_DOWN_LOW") == "1"
    if down_low and not up_mix_path:
        raise ValueError("Native HC down/low requires SGLANG_XPU_GGUF_HC_UP_MIX_LIB")
    if up_mix_path:
        if os.environ.get("SGLANG_XPU_GGUF_HC_DOWN_DPAS") != "1":
            raise ValueError("Native HC up/mix requires SGLANG_XPU_GGUF_HC_DOWN_DPAS=1")
        native = _up_mix_library(up_mix_path)
        if down_low and not hasattr(native, "xpu_hc_down_low"):
            raise ValueError("Rebuild the HC library for native down/low")
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, GatedResidual):
            continue
        norm = module.hc_norm
        if getattr(norm, "_xpu_gguf_hc_enabled", False):
            records.append(
                dict(name=name, paired=hasattr(module, "_xpu_gguf_hc_down_gate"))
            )
            continue
        if not (
            module.hc_count == 4
            and module.hidden_size == 2560
            and module.config.hc_lowrank == 320
            and module.config.hc_per_branch_norm
            and norm.weight.device.type == "xpu"
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.shape == (10240,)
            and norm.group_size == 2560
            and torch.xpu.get_device_properties(norm.weight.device).device_id == 0xE223
        ):
            raise ValueError(f"Unsupported native GGUF HC module: {name}")
        paired = hasattr(module, "block_inject_weight")
        if paired:
            down, inject = (
                module.input_mix_weight_down.weight,
                module.block_inject_weight.weight,
            )
            if (
                down.shape != (320, 10240)
                or inject.shape != (4, 10240)
                or down.dtype != torch.bfloat16
                or inject.dtype != torch.bfloat16
            ):
                raise ValueError(f"Unsupported paired HC weights: {name}")
            packed = torch.cat((down.detach(), inject.detach()), dim=0).contiguous()
            # Original Parameters keep their shapes, values and loading hooks.
            # Share one storage so pairing does not retain a second weight copy.
            down.data = packed[:320]
            inject.data = packed[320:324]
            module.register_buffer("_xpu_gguf_hc_down_gate", packed, persistent=False)
        norm._xpu_gguf_hc_enabled = True
        records.append(dict(name=name, paired=paired))
    return records
