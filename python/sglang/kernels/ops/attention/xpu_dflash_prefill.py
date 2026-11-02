"""Native B70 causal verification with a smaller query tile."""

import ctypes
from functools import lru_cache

import torch

from sglang.kernels.ops.attention.dflash_verify_xpu import supports_dflash_verify_layout
from sglang.srt.environ import envs


@lru_cache(maxsize=1)
def load_native_prefill():
    path = envs.SGLANG_XPU_DFLASH_NATIVE_LIB.get()
    if not path:
        raise RuntimeError(
            "SGLANG_XPU_DFLASH_NATIVE_PREFILL requires SGLANG_XPU_DFLASH_NATIVE_LIB; "
            "build it with test/manual/kernels/build_xpu_dflash_native.py"
        )
    lib = ctypes.CDLL(path)
    lib.dflash_native_abi.argtypes = []
    lib.dflash_native_abi.restype = ctypes.c_int
    if lib.dflash_native_abi() != 1:
        raise RuntimeError("Unsupported native DFlash library ABI")
    lib.dflash_native_prefill.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3
    lib.dflash_native_prefill.restype = ctypes.c_int
    lib.dflash_prefill_error.argtypes = []
    lib.dflash_prefill_error.restype = ctypes.c_char_p
    return lib


def try_native_dflash_prefill(
    q,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale,
    out=None,
):
    """Return None for unsupported layouts; otherwise return the native output.

    The caller provides one causal, non-branching chain with current K/V already
    written and cu_seqlens_q=[0,M]. The original native query/KV strides,
    Q8/subgroup mainloop, KV64 tile, causal mask and BF16 math are preserved.
    Device lengths/page tables remain live inputs of the captured graph.
    """
    if not (
        supports_dflash_verify_layout(
            q, key_cache, value_cache, page_table, cache_seqlens, max_seqlen_q
        )
        and 0 < page_table.shape[1] <= 32
        and cu_seqlens_q.shape == (2,)
        and cu_seqlens_q.device == q.device
        and cu_seqlens_q.dtype == torch.int32
        and cu_seqlens_q.is_contiguous()
        and softmax_scale == 256**-0.5
        and all(t.data_ptr() % 16 == 0 for t in (q, key_cache, value_cache))
    ):
        return None
    if out is None:
        out = torch.empty_like(q)
    elif not (
        out.shape == q.shape
        and out.dtype == q.dtype
        and out.device == q.device
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        raise ValueError("Native DFlash prefill requires a contiguous matching output")
    lib = load_native_prefill()
    status = lib.dflash_native_prefill(
        q.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        page_table.data_ptr(),
        cache_seqlens.data_ptr(),
        cu_seqlens_q.data_ptr(),
        out.data_ptr(),
        q.shape[0],
        key_cache.shape[0],
        page_table.shape[1],
    )
    if status:
        raise RuntimeError(
            f"Native DFlash prefill failed ({status}): "
            + lib.dflash_prefill_error().decode()
        )
    return out
