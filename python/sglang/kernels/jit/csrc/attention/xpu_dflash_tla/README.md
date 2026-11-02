# Opt-in SYCL-TLA DFlash draft attention

`SGLANG_XPU_DFLASH_ATTENTION_TLA=1` selects the native adapter for the existing
B70 single-request, four-token bidirectional draft path. It defaults to disabled
and requires `SGLANG_XPU_PAGED_DFLASH=1` and `SGLANG_XPU_DFLASH_ATTENTION=1`.
Target verification uses its existing backend. Other supported draft layouts
retain their existing attention implementation.

The contract is BF16 Q `[4,32,128]`, GQA4, contiguous current K/V `[4,8,128]`,
contiguous token-addressed prefix K/V `[slots,8,128]`, INT32 CSR pointer `[2]`
and contiguous INT64 slot IDs. Q row stride is 4096 or 6144 elements; Q head
and dimension strides are 128 and 1. Output is contiguous `[4,32,128]` with
separate storage. Prefix length is 0..2047; all referenced slots must be valid
backend metadata. Scale is `128**-0.5`, with no custom mask, sink or logit cap.
Every current token may attend all four current tokens. Query `t` additionally
sees prefix position `n` when `prefix_length+t <= n+2047`.

Build in the Torch XPU/oneAPI development environment from the pinned local
dependency checkouts; no dependency downloads or package changes are performed:

```bash
python test/manual/kernels/build_xpu_dflash_attention_tla.py \
  --kernel-source /path/to/sgl-kernel-xpu \
  --sycl-tla /path/to/sycl-tla \
  --compiler /opt/intel/oneapi/compiler/2026.1/bin/icpx \
  --output /path/to/attention-tla/attention.so

export SGLANG_XPU_PAGED_DFLASH=1
export SGLANG_XPU_DFLASH_ATTENTION=1
export SGLANG_XPU_DFLASH_ATTENTION_TLA=1
export SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB=/path/to/attention-tla/attention.so
```

The library is independent of the expert, GDN and small-projection libraries.
An enabled supported layout with a missing library or incompatible ABI fails
explicitly. Restart the server and recapture graphs after flag/library changes.
INFO reports `Using SYCL-TLA B70 M4 bidirectional draft attention` when selected.

One SYCL kernel gathers the live prefix K/V, appends current K/V, zeros the last
page padding and produces four-query metadata. The TLA decode reads the same
17 pages for four pseudo sequences. Its local-mask query position is adjusted
to the original draft token rather than the last KV position. BF16 operands,
FP32 accumulation and the existing TLA GQA packing are retained. Reduction
rounding can differ from the retained attention implementation. Kernel/component
results do not establish model-level throughput or speculative acceptance.

Scratch is allocated per invocation and recorded on the current device stream:
two BF16 `[2176,8,128]` buffers (8.5 MiB together), four INT32 lengths, five
INT32 Q offsets and a `[4,17]` INT32 page table. The graph allocator retains its
captured allocations; there is no source-weight cache or host readback. Inputs
and CSR values remain dynamic under graph replay.

The builder requires clean `sgl-kernel-xpu` commit
`028ac5a4e92a4addabeedac1a94374515f68e4a5` and `intel/sycl-tla` commit
`7b56b936b87f799f80dde60c2fd8d11c1128cebb`. It checks the exact input header
hashes, generates adapted headers alongside the output without editing the
dependency checkouts, and retains their Intel/Codeplay BSD-3-Clause notices.
The original attention kernel and decode runner are under
`src/sycl/kernels/flash_attention_v2`. The adaptation changes the query-position
calculation and gives the adapted kernel/namespace separate identities. Original
dependencies are still required for the unchanged mainloop and epilogue.
The builder records generated diffs, source/dependency/compiler/library hashes
and ABI 1 in the receipt and rejects existing output paths.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/speculative \
  -p 'test_dflash_attention_tla.py'
```

Set `SGLANG_XPU_DFLASH_ATTENTION_TLA_LIB` to the built library for GPU checks.
The checks exercise the production dispatcher flags and fallback guards,
noncontiguous Q, changing graph inputs/slots/lengths, exact window boundaries
and current-block future visibility.

## Saved paged KV child

`SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA=1` adds a second, default-disabled child
under the master, draft-attention and TLA flags above. Build its independent
library with the same builder and pins, adding `--paged-kv`, and set
`SGLANG_XPU_DFLASH_ATTENTION_PAGED_TLA_LIB` to that output. The packed library
remains required for supported packed fallbacks; restart and recapture after
changing either selection.

For an eight-token draft block, build the paged child with
`--paged-kv --window-size 8` and select that library. The default remains M4.
The M8 build has separate kernel/decoder identities and a query-position mask
for all eight tokens. ABI 1 signatures are preserved; a window query identifies
the compiled width. The Python adapter rejects an input/library width mismatch
before metadata or KV writes. Legacy libraries without the query remain M4.
The packed child still supports M4; missing saved metadata on M8 retains the
previous attention path.

The Triton draft backend passes saved-KV metadata only for DFlash draft `TARGET_VERIFY`,
`ENCODER_ONLY`, page size 128, the standard `MHATokenToKVPool`, and a forward
that saves current K/V before attention. Existing attention semantic/layout
checks still apply. Missing metadata, other pools/modes/page sizes and unsupported
layouts retain the packed or previous path. A selected missing/incompatible
paged library fails explicitly.

The full-prefix CSR must start on a logical 128-token page boundary. Each logical
page is internally contiguous, but physical pages may be reordered. Current
write locations continue the same logical sequence; the last allocated page
belongs exclusively to this request. These value/ownership properties come
from the backend allocator contract and are not validated by reading GPU
metadata onto the CPU. Do not call the low-level helper with arbitrary slot IDs.

The metadata kernel drops `floor(max(full_prefix-2047,0)/128)*128` tokens from
the left. It therefore preserves the full local window and at most 127 older
tokens, with local prefix length 0..2174 and at most 18 pages including current
K/V. TLA applies the original per-query 2047 left mask to this aligned suffix.
Full prefixes larger than the window remain dynamic graph inputs.

This path reads existing pool K/V and allocates M INT32 lengths, M+1
Q offsets and an `[M,18]` page table (324 bytes for M4, 644 for M8). It clears the invalid suffix
of the current request's final page to prevent stale NaNs from entering masked
PV products. Valid K/V is preserved; there is no complete prefix packing or
source-weight cache. INFO identifies `saved paged KV attention` when selected.
This changes an internal allocated page's invalid padding and must never be
used without the exclusive-page ownership guarantee.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/speculative \
  -p 'test_dflash_attention*paged_tla.py'
```

These checks require both explicitly built libraries. They cover flag/fallback
behavior, page permutations, NaN padding, exact unaffected-cache preservation,
dynamic full prefixes through 8188 tokens (M4) or 8184 tokens (M8), both Q
strides and exact masks after page-aligned clipping. The M8 checks additionally
require `SGLANG_TEST_DFLASH_PAGED_TLA_M8_LIB` and
`SGLANG_TEST_DFLASH_PAGED_TLA_LEGACY_LIB` for window and legacy-output guards.
Component results remain separate from model throughput.
