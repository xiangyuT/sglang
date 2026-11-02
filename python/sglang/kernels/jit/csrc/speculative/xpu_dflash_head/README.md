# Experimental B70 DFlash target head

`SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL=1` selects FP8 top-64 proposals followed by
SYCL BF16 rescoring, conditional exclusion envelopes and GPU completion of
uncertain rows. It defaults to disabled and requires
`SGLANG_XPU_PAGED_DFLASH=1`, single-rank DFLASH with a four- or eight-token
window, B70, contiguous BF16 hidden `[M, 2048]`, and an unpadded BF16 head
`[248320, 2048]`. The window includes the anchor, leaving three or seven proposals.
Other layouts and forward modes retain the existing head. No host confidence
readback occurs in the verify loop.

The exclusion envelope assumes the tested FP8 native cast model, whose complete
numeric contract is not established. Complete BF16 rows also use a different
FP32 reduction from Torch. This is an opt-in numerical experiment, not an
arbitrary-input lossless guarantee or a dataset-quality claim. Original BF16
weights remain live and unchanged. The native implementation uses SYCL/ESIMD;
FP8 quantization and top-k reuse the existing implementations.

The DFlash worker restores original full scores before logits adjustments for
logprob, grammar, non-greedy/custom sampling or penalties and when the source
pointer/version changes. Supported greedy requests consume the conditional GPU
result. Restart and recapture after changing weights, flags or libraries. LoRA,
DP/TP gather, FP32 heads, softcap, bias, logit scale and torch.compile are excluded.
The state keeps separate eager and captured plans so a request fallback uses the
normalized hidden that produced its output, rather than auxiliary hidden states.

Build with the existing explicit builder in Omix 0.4.0, Torch 2.14 XPU and
oneAPI 2026.1. The caller supplies local, clean dependency checkouts pinned to
`sgl-kernel-xpu` commit `028ac5a4e92a4addabeedac1a94374515f68e4a5` and
`sycl-tla` commit `7b56b936b87f799f80dde60c2fd8d11c1128cebb`:

```bash
python test/manual/kernels/build_xpu_dflash_native.py \
  --kernel-source /path/to/sgl-kernel-xpu \
  --sycl-tla /path/to/sycl-tla \
  --compiler /opt/intel/oneapi/compiler/2026.1/bin/icpx \
  --output /path/to/build/native.so

export SGLANG_XPU_PAGED_DFLASH=1
export SGLANG_XPU_DFLASH_TARGET_HEAD_SYCL=1
export SGLANG_XPU_DFLASH_NATIVE_LIB=/path/to/build/native.so
```

Add `--window-size 8` for an M8 library and use
`--speculative-num-draft-tokens 8` when launching that server. The default build
remains M4. `dflash_native_window()` declares the selected width; old libraries
without this query are M4. Input/library mismatch fails before buffer writes.

The builder records source/dependency/compiler identities and the library hash;
it never downloads, installs or overwrites an existing result. The library must
export `dflash_head_abi=1`. A selected missing/incompatible library fails clearly.
INFO reports `Using experimental B70 SYCL target head M=4/top64 with GPU BF16
completion` (or M=8). The target and draft reuse one private FP8 candidate copy when the
draft head child is also enabled. Weight envelopes are prepared before capture.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/speculative \
  -p 'test_dflash_target_head_sycl.py'
```

Tests cover dynamic graph flags and inputs, BF16 numerical comparison, confidence
edge cases and dispatch/state guards. Unit evidence is distinct from model
quality and from normal end-to-end performance measurements.
