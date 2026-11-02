# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving Strata MTP Q8_0 SoA16 and bounded DPAS projection."""
import ctypes
import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_sycl import dequantize_gguf


def pack_rows(raw, *, inverse=False):
    if raw.device.type != "cpu" or raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[0] % 16 or not raw.is_contiguous():
        raise ValueError("SoA16 requires whole contiguous CPU Q8_0 row groups")
    rows, stride = raw.shape
    if stride % 34:
        raise ValueError("Incomplete Q8_0 rows")
    blocks = stride // 34
    output = torch.empty_like(raw)
    if not inverse:
        source = raw.numpy().reshape(rows // 16, 16, blocks, 34)
        target = output.numpy().reshape(rows // 16, blocks, 544)
        target[..., :32] = source[..., :2].transpose(0, 2, 1, 3).reshape(rows // 16, blocks, 32)
        target[..., 32:] = source[..., 2:].reshape(rows // 16, 16, blocks, 8, 4).transpose(0, 2, 3, 1, 4).reshape(rows // 16, blocks, 512)
    else:
        source = raw.numpy().reshape(rows // 16, blocks, 544)
        target = output.numpy().reshape(rows // 16, 16, blocks, 34)
        target[..., :2] = source[..., :32].reshape(rows // 16, blocks, 16, 2).transpose(0, 2, 1, 3)
        target[..., 2:] = source[..., 32:].reshape(rows // 16, blocks, 8, 16, 4).transpose(0, 3, 1, 2, 4).reshape(rows // 16, 16, blocks, 32)
    return output


@lru_cache(maxsize=1)
def library():
    path = os.environ.get("SGLANG_XPU_GGUF_MTP_Q8_SOA_LIB")
    if not path or torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Build and warm the MTP Q8 SoA library before capture")
    lib = ctypes.CDLL(path)
    lib.xpu_mtp_q8_quant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.xpu_mtp_q8_quant.restype = ctypes.c_int
    lib.xpu_mtp_q8_soa.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    lib.xpu_mtp_q8_soa.restype = ctypes.c_int
    lib.xpu_mtp_q8_soa_error.restype = ctypes.c_char_p
    return lib


class MTPQ8SoAMethod:
    def __init__(self, raw, width):
        if width not in (640, 2560, 6144) or raw.shape[1] != width // 32 * 34:
            raise ValueError("Unsupported MTP Q8 width")
        self.raw = raw.detach().cpu().clone()
        self.width = width
        self.rows = raw.shape[0]
        self.packed_soa = True
        self.lib = library()

    def process_weights_after_loading(self, module):
        module.gguf_dense_iq3_packed_parts = (False,)

    def reconstruct_weight(self, module):
        # Only the active larger-row projection expands; decode keeps SoA16.
        raw = self.raw.to(module.weight.device)
        return dequantize_gguf(raw.reshape(-1), 8, dtype=torch.float32).view(self.rows, self.width).bfloat16()

    def apply(self, module, x, bias=None):
        tokens = x.shape[0]
        if bias is not None:
            raise ValueError("MTP Q8 adapter supports bias-free projections only")
        if tokens not in (1, 4, 8, 16):
            from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

            try:
                mode = get_forward_batch().forward_mode
            except RuntimeError:
                mode = None
            if (mode is not None and (mode.is_decode() or mode.is_draft_extend_v2())
                    and 0 < tokens <= 256):
                output = torch.empty((tokens, self.rows), dtype=x.dtype, device=x.device)
                for first in range(0, tokens, 4):
                    count = min(4, tokens - first)
                    wave = x[first:first + count].contiguous()
                    if count != 4:
                        padded = x.new_zeros((4, self.width))
                        padded[:count].copy_(wave)
                        wave = padded
                    output[first:first + count].copy_(self.apply(module, wave)[:count])
                return output
            return F.linear(x, self.reconstruct_weight(module))
        if not x.is_contiguous() or x.dtype != torch.bfloat16:
            raise ValueError("MTP Q8 DPAS requires contiguous BF16 rows")
        scratch = torch.empty((tokens, self.width // 32 * 36), dtype=torch.uint8, device=x.device)
        output = torch.empty((tokens, self.rows), dtype=torch.bfloat16, device=x.device)
        stream = torch.xpu.current_stream(x.device)
        rc = self.lib.xpu_mtp_q8_quant(stream.sycl_queue, x.data_ptr(), scratch.data_ptr(), self.width, tokens)
        if not rc:
            rc = self.lib.xpu_mtp_q8_soa(stream.sycl_queue, module.weight.data_ptr(), scratch.data_ptr(), output.data_ptr(), self.width, self.rows, tokens)
        if rc:
            raise RuntimeError((rc, self.lib.xpu_mtp_q8_soa_error().decode()))
        for tensor in (x, module.weight, scratch, output):
            tensor.record_stream(stream)
        return output
