"""Single-row block-FP8 W8A8 GEMV for XPU decode projections.

Consumes the existing per-128 activation quantization and 128x128 weight
scales. The FP8 checkpoint and activation quantization are unchanged.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_decode_xpu import fp8_e4m3_div_256
from sglang.srt.environ import envs


@triton.jit
def _gemv_kernel(
    X,
    W,
    XS,
    WS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BN: tl.constexpr,
    FP16_DEQUANT: tl.constexpr,
    SCALED_FP8: tl.constexpr = False,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, K)
    groups = cols // 128
    # Match oneDNN's Xe2 block-FP8 GEMM: scales and the scaled operands
    # round to FP16 before FP32 accumulation. Scaling an FP32 group dot
    # afterward is mathematically valid but changes the native result.
    xs = tl.load(XS + groups)
    value = tl.load(X + cols)
    if SCALED_FP8:
        x = fp8_e4m3_div_256(value)
    else:
        x = value.to(tl.float32)
    # The guarded shapes are multiples of the row tile. Avoid a masked FP8
    # load: Intel Triton cannot materialize its implicit integer zero in FP8.
    w = tl.load(W + rows[:, None] * K + cols[None, :])
    ws = tl.load(
        WS + (rows[:, None] // 128) * WS0 + groups[None, :] * WS1,
    )
    if FP16_DEQUANT:
        xs = xs.to(tl.float16).to(tl.float32)
        ws = ws.to(tl.float16).to(tl.float32)
        if SCALED_FP8:
            xs = xs * 256
            ws = ws * 256
            w = fp8_e4m3_div_256(w)
        else:
            w = w.to(tl.float32)
        x = (x * xs).to(tl.float16).to(tl.float32)
        w = (w * ws).to(tl.float16).to(tl.float32)
    else:
        # oneDNN chooses an FP32 SIMD path for the small shared gate/up.
        x = x * xs
        w = w.to(tl.float32) * ws
    out = tl.sum(w * x[None, :], 1)
    tl.store(Y + rows, out, rows < N)


def xpu_block_fp8_gemv(
    q_input, weight, activation_scale, weight_scale, *, rows=4, compensate_long_k=False
):
    """Return BF16 ``q_input[1,K] @ weight[N,K].T`` with block scales."""
    n, k = weight.shape
    assert rows in (1, 2, 4, 8) and n % rows == 0
    if (
        envs.SGLANG_XPU_AR_SMALL_DENSE_SYCL.get()
        and rows == 4
        and (n, k) in ((1024, 2048), (2048, 512))
        and not torch.compiler.is_compiling()
    ):
        from sglang.srt.layers.ar728_shared import try_run as try_shared728
        native = try_shared728(q_input, weight, activation_scale, weight_scale)
        if native is not None:
            return native
        from sglang.kernels.ops.gemm.xpu_ar_small_dense_sycl import (
            can_use_ar_small_dense,
            xpu_ar_small_dense,
        )
        from sglang.srt.runtime_context import get_spec

        if get_spec().speculative_algorithm is None and can_use_ar_small_dense(
            q_input, weight, activation_scale, weight_scale
        ):
            return xpu_ar_small_dense(q_input, weight, activation_scale, weight_scale)
    # The short-K projection benefits from fewer warps. Eight warps improve
    # the long-K projection; this changes FP32 reduction order and can change
    # greedy output near BF16 rounding boundaries (see the native-path flag).
    warps = 1 if rows == 4 and (n, k) == (2048, 512) else 4
    if (
        rows == 4
        and (n, k) == (2048, 4096)
        and envs.SGLANG_XPU_M1_BLOCK_FP8_LONG_K_WARP8.get()
    ):
        warps = 8
    # This preserves FP16 dequantization and the warp8 FP32 reduction tree.
    # Runtime owns the AR/K192 opt-in; other shapes and launch modes stay intact.
    scaled_fp8 = (n, k) in {(12288, 2048), (9216, 2048)} or (
        compensate_long_k and (n, k) == (2048, 4096) and rows == 4 and warps == 8
    )
    if rows == 4 and warps == 8 and scaled_fp8 and (n, k) == (2048, 4096):
        from sglang.srt.layers.ar671_output import try_run as try_output671
        native = try_output671(q_input, weight, activation_scale, weight_scale)
        if native is not None:
            return native
    out = torch.empty((1, n), dtype=torch.bfloat16, device=q_input.device)
    _gemv_kernel[(triton.cdiv(n, rows),)](
        q_input,
        weight,
        activation_scale,
        weight_scale,
        out,
        N=n,
        K=k,
        WS0=weight_scale.stride(0),
        WS1=weight_scale.stride(1),
        BN=rows,
        FP16_DEQUANT=(n, k) != (1024, 2048),
        SCALED_FP8=scaled_fp8,
        grf_mode="256" if scaled_fp8 and rows == 4 else "default",
        num_warps=warps,
    )
    return out
