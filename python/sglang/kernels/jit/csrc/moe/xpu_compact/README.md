# Experimental native B70 DFlash kernels

The expert/metadata/head builder accepts `--window-size 4` (default) or
`--window-size 8`. Match this to `--speculative-num-draft-tokens`; each
library exports `dflash_native_window()`. Fixed-window children reject a
selected incompatible library before writing buffers. Older libraries without
the query are M4. Build into distinct output paths, then restart and recapture.
GDN and small-TLA have separate builders and support both widths in one library;
paged attention uses its own matching `--window-size` build.
An M8 process can still receive a four-row prompt or prompt tail: those dense
and planner calls retain the existing path, and route preparation retains the
M4 fused path. M8 verification still requires the matching M8 library.

`SGLANG_XPU_DFLASH_SYCL_PLANNER=1` selects the pure SYCL expert-residency
planner for E256/K192 and four or eight rows of top-8 int32 routes. It requires the
DFlash master and a library rebuilt with `planner.cpp`. Stable input order,
lowest-slot LRU ties, inverse mapping, invalid-route zero weights and plan
tails retain the existing integer policy. Relaxed atomic timestamp stores
make duplicate all-hit routes safe. Other shapes retain the existing planner.
INFO prints the selected R=32 or R=64 geometry with SG=32. The child
defaults to disabled; restart and recapture after changing flags or libraries.

An independent `SGLANG_XPU_DFLASH_GDN_NARROW=1` child selects BV4 for the
retained recurrent verify kernel on single-request, four-token B70 GDN with
FP32 intermediate checkpoints and the native BF16 gate boundary. It requires
the DFlash master. Tree verification and other shapes retain their existing
launch configuration. This changes the existing kernel's value tile; it does
not select a native GDN implementation or require another library build.
Restart and recapture after changing it. INFO prints
`Using B70 DFlash GDN verify BV=4 (T=4, FP32 checkpoints)`.

## Precise SYCL GDN verify and accepted-state commit

`SGLANG_XPU_DFLASH_GDN_SYCL=1` selects a separate SYCL/ESIMD M4/M8/BV2
implementation. It defaults to disabled and requires the DFlash master,
B70, TP1/PP1, one running request, radix cache disabled, BF16 Q/K/V and
gates, and FP32 state shaped `[slots,32,128,128]`. Linear four- and eight-token
verify windows are supported. KDA, tree drafts, PP stable rows, tracking and
ReplaySSM are not supported by this path. Existing ReplaySSM guards remain.

Build its library independently in the same Omix 0.4.0 / Torch 2.14 XPU /
oneAPI 2026.1 environment:

```bash
python test/manual/kernels/build_xpu_dflash_gdn.py \
  --compiler /opt/intel/oneapi/compiler/2026.1/bin/icpx \
  --output /path/to/gdn-build/gdn.so

export SGLANG_XPU_PAGED_DFLASH=1
export SGLANG_XPU_DFLASH_GDN_SYCL=1
export SGLANG_XPU_DFLASH_GDN_LIB=/path/to/gdn-build/gdn.so
```

The builder uses `-fp-model=precise -ffp-contract=off`, preserving the expert
library's independent compiler options. It records the command, compiler,
Torch version, source and library hashes in `gdn.so.json`, and never
overwrites existing outputs. This child does not use `NATIVE_LIB` for GDN.
A selected missing or incompatible GDN library fails explicitly.
M8 requires the rebuilt library's `dflash_gdn_supports_steps(8)` capability;
legacy M4 libraries remain usable for M4. Pool scratch and backend metadata
must agree on the selected window. Other experimental children have their own
shape constraints; enabling M8 GDN does not extend those children.

The Mamba pool owns prepared Q/K/gates, saved FP32 deltas, per-layer source
indices and BF16 outputs. Verify does not change the live SSM checkpoint;
commit replays the accepted prefix with the original FP32 gate FMA order.
Source indices are snapshotted before their caller buffers can be reused.
Source layout identity is checked before submission. Original conv rollback,
PLE continuation, non-verify paths and full checkpoint allocation remain.
The transient records are not slot state or host-transfer buffers; their
memory is charged once to the owning pool. A selected unsupported verify
layout fails rather than reading stale snapshots.

Restart and recapture after changing flags, source/scratch buffers or either
library. INFO reports the selected width, for example
`Using SYCL B70 DFlash GDN M=8/BV=2; pool-owned delta replay`.
The separate `GDN_NARROW` child remains the retained-kernel control;
it does not enable this library. This opt-in implementation does not claim
other shapes or model-level throughput from component results alone.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/attention \
  -p 'test_dflash_gdn_sycl.py'
```

The checks cover full thirty-layer states, BF16/FP32 raw-bit equivalence,
changing graph inputs/source slots/accepted prefixes, in-place commit,
input/index reuse, independent pools, memory accounting, original conv
rollback and snapshot fallback, and fail-closed mapping/tracking/layout
guards. Run them with `SGLANG_XPU_DFLASH_GDN_LIB` set to the built library.

## Asynchronous SYCL graph-input copies

`SGLANG_XPU_DFLASH_SMALL_COPY_SYCL=1` independently selects a single SYCL
submission for five contiguous INT64 graph inputs with M/M/M/1/1 elements,
where M is 4 or 8.
It defaults to disabled and requires the DFlash master on B70. It uses
`SGLANG_XPU_DFLASH_GDN_LIB`, built by the same command above; the builder
now includes `xpu_dflash_small_copy.cpp` alongside the unchanged precise GDN
unit. The existing GDN ABI1 is preserved, with an additional small-copy ABI1.
Enabling copies does not require enabling the GDN child.

```bash
export SGLANG_XPU_DFLASH_SMALL_COPY_SYCL=1
export SGLANG_XPU_DFLASH_GDN_LIB=/path/to/gdn-build/gdn.so
```

The graph buffer registry selects this path before its original foreach
copies. No value conversions occur. Sources and destinations are recorded on
the current stream for allocator lifetime; cross-copy/partial-overlap hazards
are rejected, while an identical source/destination view is safe. Other
shapes, dtypes, strides, devices, prefill rows and disabled flags retain the
existing path. The GPU copies stay ordered on the current queue without a
host wait. This permits preparation to overlap previously queued draft work;
the time spent waiting in a host foreach call is not itself a speedup budget.

The first selected call logs `Using SYCL B70 M4 graph-input copies: 5 INT64
buffers (4/4/4/1/1)` or the corresponding M8 geometry. M8 uses the additional
`dflash_small_copy_m8` symbol; the M4 symbol and ABI1 remain supported.
A selected library missing the required symbols raises an
explicit rebuild error. Restart and recapture after changing flags/libraries.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/memory \
  -p 'test_xpu_small_copy_sycl.py'
```

The checks cover INT64 extremes, changing graph inputs, source release under
allocator pressure, registry dispatch, master/child and unsupported-layout
fallbacks, alias hazards and missing libraries/symbols.

## Native expert and attention library

The expert/attention library preserves the native BF16/FP8 DPAS mainloop and replaces
its scan across all resident expert slots with compact active-expert indices.
It reads weights and scales in their existing resident-slot layout. Stable
routing, native Silu, native combine, and external routed scaling are preserved.
The supported layout is top-8, 2–8 BF16 rows, hidden size 2048, intermediate
size 512, FP8 E4M3 weights and FP32 block-128 scales on B70. Other layouts keep
the original paged executor. The single-row optimization is independent. Non-power-of-two route counts
are padded only for sorting; native GEMMs keep the actual route count.

The same library also instantiates the original causal target-attention kernel
with Q16/KV64/SG2, retaining the original Q8/subgroup mainloop. This path is
limited to single-request TARGET_VERIFY, BF16 Q16/KV2 heads, head size 256,
page128, at most 4096 tokens, and 2–8 queries. Other modes and layouts use the
existing attention backend.

## Build explicitly

Use the Omix 0.4.0 development environment with oneAPI 2026.1 `icpx` and
Torch 2.14 XPU. The native attention headers require C++20. Create
local dependency checkouts at these exact revisions (the build never downloads
or installs anything):

- `https://github.com/sgl-project/sgl-kernel-xpu.git`,
  `028ac5a4e92a4addabeedac1a94374515f68e4a5` (v0.4.0).
- `https://github.com/intel/sycl-tla.git`,
  `7b56b936b87f799f80dde60c2fd8d11c1128cebb` (the dependency pinned by v0.4.0).

From the SGLang repository root, replace the three paths with your local
checkouts and a new output path. A sparse kernel checkout must include
`src/sycl`, `src/jit` and `include`:

```bash
python test/manual/kernels/build_xpu_dflash_native.py \
  --kernel-source /path/to/sgl-kernel-xpu \
  --sycl-tla /path/to/sycl-tla \
  --compiler /opt/intel/oneapi/compiler/2026.1/bin/icpx \
  --output /path/to/build/native.so
```

The command creates `native.so` and `native.so.json`, recording the full
compiler command, dependency revisions, source hashes, exit status and library
hash. Existing outputs are not overwritten. The generic SPIR-V target enables
the split-barrier extension used by the native mainloop. Dependency checkouts
must have no tracked changes.

## Enable and verify

Set these before starting the DFlash server:

```bash
export SGLANG_XPU_PAGED_DFLASH=1
export SGLANG_XPU_DFLASH_COMPACT_MOE=1
export SGLANG_XPU_DFLASH_NATIVE_PREFILL=1
export SGLANG_XPU_DFLASH_NATIVE_LIB=/path/to/build/native.so
```

The library path is the output created by the preceding build command. The
startup log reports `Using compact native W8A16 scheduling` when the supported
paged layout is selected, and `Using B70 native DFlash prefill Q16/KV64/SG2`
when the attention path is used. Missing libraries or an incompatible ABI fail at
startup when the feature is selected. The library is not loaded when disabled.
The module records SYCL submissions into the existing full XPU graph. It does
not support torch.compile. This is a development integration, not a new wheel.

```bash
python -m pytest -q \
  test/registered/kernels/ops/moe/test_xpu_compact_moe.py \
  test/registered/kernels/ops/attention/test_xpu_dflash_prefill.py
```

This test requires the library environment variable and checks eager and graph
outputs exactly against native experts, including changing routes, high slot
IDs, duplicate experts, zero weights and zero input. Attention tests compare
eager/graph outputs exactly with native prefill while changing Q/K/V, page
tables and cache lengths through page boundaries. Remove the library flag
or set `SGLANG_XPU_DFLASH_COMPACT_MOE=0` for the native grouped-GEMM comparison.
Set `SGLANG_XPU_DFLASH_NATIVE_PREFILL=0` for the original attention tile. Both
features default to disabled, and require the experimental DFlash master flag.

The separate `SGLANG_XPU_DFLASH_VERIFY_DECODE=1` child flag, disabled by default
and requiring the experimental DFlash master flag, reframes a supported single causal
verification block as independent native decode queries. Each query receives
its own causal KV length. It can round differently from native prefill; use
`SGLANG_XPU_DFLASH_VERIFY_DECODE=0` to retain the original target attention.
It remains opt-in because it can change model outputs and speculative
acceptance. Select either native prefill or verify decode; enabling both raises
a configuration error. Component equivalence to independent decode queries
does not establish model quality or end-to-end superiority.

## Host fetch and device-prefix context writes

Two additional experimental children default to disabled:

```bash
export SGLANG_XPU_DFLASH_PAGED_MOE=1
export SGLANG_XPU_DFLASH_PREFIX_KV=1
```

Both require `SGLANG_XPU_PAGED_DFLASH=1`. The paged child additionally
requires compact MoE and a freshly built library exporting `dflash_paged_abi`.
Its supported expert layout is the compact layout above, with contiguous,
64-byte-aligned pinned W13/W2/S13/S2 host planes. The planner's unique missing
pairs must all occur among the active routes. The two native GEMMs read missing
host weights and fill the resident slots while retaining DPAS/decode/Silu/combine
arithmetic. The first M tile owns cache writes; repeated routes are supported.
The K32 mainloop fills each missing raw cache row with a contiguous 32-byte
load/store per lane. It retains the separate host read, original decode, DPAS,
scale and accumulation order. Cache ownership remains with the first M tile,
including repeated routes that span multiple M tiles.

Host stores and pointer tables must outlive every graph replay. Unsupported
layouts keep the original gather/expert executor. INFO reports `compact native
DFlash host fetch/cache fill` when selected.

The independent prefix child selects the existing device-prefix byte-copy kernel
for equal-width BF16 K/V caches stored as BF16. It bypasses the XPU fallback's
dynamic `nonzero` and `index_select` per draft layer. Other dtypes and asymmetric
rows retain the original path. INFO reports `DFlash BF16 device-prefix KV writes`.
This does not by itself capture the complete context append into a graph.

Restart and recapture after changing either flag. Set either to zero for its
respective control. Run the dynamic pager and prefix tests with the same library:

```bash
python -m pytest -q \
  test/registered/kernels/ops/moe/test_paged_experts_xpu.py::TestXpuPagedExperts::test_compact_native_paged_verify_graph \
  test/registered/kernels/ops/kvcache/test_xpu_dflash_prefix_kv.py
```

The pager test covers M=2..8, hits/misses/padded routes, duplicate routes spanning
multiple M tiles, and all-padding replay. It compares outputs exactly to native
fully resident experts and checks every resident cache byte against host stores.
Prefix tests update commit lengths and locations after capture, test strided
source views, preserve uncommitted cache rows, and use invalid locations in
uncommitted entries to check that those entries are never accessed.

## Pure SYCL route preparation

`SGLANG_XPU_DFLASH_SYCL_ROUTES=1` selects a pure SYCL replacement for M=4/M=8
sorting, compact counts/offsets, host classification and scale copies. It
requires the DFlash master and native paged MoE, and defaults to disabled.
Rebuild the library with the command above; it must export
`dflash_routes_abi`. The path logs its selected M4/M8 route metadata and
scale copies with WG=128 when first used. Matrix arithmetic remains unchanged.

This child enables fused preparation independently of the optional Triton
`SGLANG_XPU_DFLASH_FUSED_ROUTES` flag. With the SYCL child off, that flag selects
the Triton fused control; both off retain the original separate preparation.
Other M values retain their existing path. Restart and recapture after changing
flags or the library. Test both fused backends with:

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/moe \
  -p 'test_xpu_dflash_*routes.py'
```

The tests compare metadata, outputs and every cache byte under changing graph
inputs/plans, including duplicate experts and master/shape/paging fallbacks.

## Selected sequential SYCL dense projections

`SGLANG_XPU_DFLASH_DENSE_SYCL=1` additionally selects native M=4/M=8 projections
for weights shaped 12288x2048 and 9216x2048. It defaults to disabled and requires
both `SGLANG_XPU_PAGED_DFLASH=1` and `SGLANG_XPU_DFLASH_DENSE_DPAS=1`.
Rebuild this library before enabling it. INFO reports `Using sequential SYCL
M=4 dense: 12288x2048 and 9216x2048 (SG=4, GRF=128, lane-major FP16 A,
vector FP8 B)` on first use. Rebuild the native library: the sequential dense
interface requires ABI 2; M8 logs its corresponding width and requires the
matching M8 library. Other layouts retain the existing dispatch.

Both shapes use the same subgroup matrix instruction type as the retained
M4 path, extended to M8: M=4/8, N=16, K=16, packed FP16 A/B and FP32 accumulation. The
FP16 scale/operand rounding and sequential K order are preserved, with BF16
output. A pure SYCL ESIMD kernel prepares the live activation operands in a
16/32 KiB lane-major FP16 buffer, followed by vector FP8 weight decoding and the
original sequential K=16 DPAS mainloop. Both shapes use SG=4 and GRF=128. There
are no split partial sums or second reduction kernel. Other
projection shapes retain their existing paths, including the shared
1024x2048 projection whose dequantization contract is FP32.

Packing retains every FP8 weight byte and keeps an extra layout copy on each
selected weight tensor. On this 40-layer model the selected copies total about
0.879 GiB; source FP8 parameters remain available. Scales stay dynamic inputs.
Warm these immutable
weights before graph capture; the implementation rejects first-time packing
during capture. A weight-version change rebuilds its layout in eager mode,
but existing graphs must be discarded. Restart and recapture after changing
source weights, flags or the library. The target's original FP8 weights remain
available for prefill. This does not support training or live weight updates
under an existing graph.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/gemm \
  -p 'test_xpu_dflash_dense_sycl.py'
```

The tests include selected native/CPU references, changing graph inputs and
scales, byte-preserving weight packing, eager invalidation before recapture,
warmup requirements, and master/parent/child/shape guards.

## Small SYCL dense projection

`SGLANG_XPU_DFLASH_SMALL_DENSE_SYCL=1` selects the M=4, N2048/K512
projection through pure SYCL ESIMD with VL128. It defaults to disabled and
requires the DFlash master and `SGLANG_XPU_DFLASH_DENSE_DPAS=1`; the separate
large-dense SYCL child is independent. Rebuild this library, restart and
recapture graphs before enabling it. INFO reports `Using SYCL M=4 small dense:
2048x512 (ESIMD VL=128, live FP8)` when selected.

FP8 weights and FP32 block-128 scales remain live inputs. Scales are rounded to
FP16, dequantized operands are rounded to FP16, and vector products/reductions
use FP32 before BF16 output. A 4 KiB activation prepare buffer is replayed on
device; there is no derived weight cache. Other shapes retain their current
dispatch. FP32 reduction order differs from DPAS and can change greedy output
and speculative acceptance, so component speedup alone is not an E2E claim.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/gemm \
  -p 'test_xpu_dflash_small_dense_sycl.py'
```

Tests compare native and selected FP64 references, changing input/weight/scale
values under graph replay, finite FP8 patterns, zeros, strided/contiguous
scales, dispatch fallbacks and an explicitly missing native library.

## Cached small sycl-tla projection

`SGLANG_XPU_DFLASH_SMALL_TLA=1` selects a separate B70 M=4 or M=8 small projection
for contiguous BF16 `[M,512]` activations, FP8 E4M3 `[2048,512]` weights,
FP32 128x128 block scales and no bias. It defaults to disabled and requires
the DFlash master and `SGLANG_XPU_DFLASH_DENSE_DPAS=1`. The large-dense SYCL
child is independent. Prequantized inputs and other layouts retain their
existing paths.

Build its independent library from the same pinned dependency checkouts:

```bash
python test/manual/kernels/build_xpu_dflash_small_tla.py \
  --kernel-source /path/to/sgl-kernel-xpu \
  --sycl-tla /path/to/sycl-tla \
  --compiler /opt/intel/oneapi/compiler/2026.1/bin/icpx \
  --output /path/to/small-tla/small_tla.so

export SGLANG_XPU_PAGED_DFLASH=1
export SGLANG_XPU_DFLASH_DENSE_DPAS=1
export SGLANG_XPU_DFLASH_SMALL_TLA=1
export SGLANG_XPU_DFLASH_SMALL_TLA_LIB=/path/to/small-tla/small_tla.so
```

`SMALL_TLA_LIB` is this builder's output, independent of the expert and GDN
libraries. The builder never downloads or installs dependencies, rejects
existing outputs, and records source/dependency/compiler/library identities
in `small_tla.so.json`. Quantization remains a separate translation unit. This
library uses `-O3 -fp-model=precise -ffp-contract=off`, matching the tested
quantization options. Missing selected symbols or ABI1 fail before
submission. INFO reports the selected M4 or M8 projection. M8 requires the
additional `dflash_small_tla_gemm_m8` symbol; old M4 libraries remain usable
for M4 and fail explicitly before packing weights if selected for M8.

The three SYCL kernels quantize live activations, prepare FP16 activation
operands and run sycl-tla DPAS with FP32 accumulation and BF16 output. Scales
are rounded to FP16 before dequantized operands are rounded to FP16. The two
original Triton kernels are replaced by three SYCL kernels; this is not a
claim that total GPU kernel count or end-to-end latency decreases.

Each selected weight tensor owns an additional 2 MiB FP16 operand cache;
forty such projections add 80 MiB. M4 and M8 share the same cache for a given
weight/scale identity. Original FP8 weights remain available.
Weights and scales are static during graph replay. Warm on the capture stream
before capture; cross-stream eager use waits for the cache's ready event.
Before capture, eager weight/scale version changes rebuild the cache. After
capture, later Python calls reject changed weight, scale or library identities.
Raw graph replay does not execute these Python checks. Restart and recapture
after changing weights, scales, flags or libraries; live updates under an
existing graph are unsupported. Cache/library/scale ownership remains attached
to the model's source weight through the graph lifecycle.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/gemm \
  -p 'test_xpu_dflash_small_tla.py'
```

Tests cover quantized bytes and FP32 scale bits, finite FP16 cache bits,
dynamic graph inputs, full-output reference and FP64 bounds, cache invalidation
and ownership, allocator pressure, cross-stream readiness and dispatch guards.
Component results do not establish model-level throughput or acceptance.

## Single-request SYCL draft metadata

With the DFlash master enabled, `SGLANG_XPU_DFLASH_SYCL_METADATA=1` combines
the M=4/M=8 draft verify QO/KV/window/mask buffer updates into one SYCL launch.
It defaults to disabled. Rebuild this library, restart and recapture before
enabling it. INFO reports `Using SYCL single-request DFlash M=4 verify metadata
(window=2047)` when selected.

The supported contract is B70, one draft request, four or eight tokens, page size128,
window2047, no custom mask or DCP, and full-space KV storage. Separate SWA
pools keep the original backend path. INT32/INT64 source ids, lengths and rows
are accepted; QO/mask buffers can be INT32 or INT64. KV/window streams remain
INT64, with INT32 CSR pointers and window offsets.

Every call asks the translator for its current read source. Token entries,
already translated page entries and live v2p tables retain the original
addressing and padding rules. The implementation does not cache a host-fed
plan or enable the incompatible metadata glue graph. Unwritten stream tails
remain untouched. Unsupported contracts retain the original path; a requested
native library with missing symbols fails explicitly.

The tests are in
`test/registered/kernels/ops/attention/test_dflash_metadata_sycl.py`, covering
whole-buffer equality with the existing backend, changing lengths and live
v2p entries under graph replay, and master/child/layout fallbacks.

## Source provenance

The independent shared-projection experiment uses the existing Triton
installation. With the DFlash master enabled,
`SGLANG_XPU_DFLASH_SHARED_GEMV=1` selects FP32 vector decoding for the shared
gate/up. It defaults to disabled. The guards require B70, M=2..8, BF16 output,
contiguous quantized input and weights, the existing 128x128 weight blocks and
matching activation scales; unsupported layouts retain the original path.
FP32 operand dequantization is preserved, but reduction order can differ.
This flag needs no additional native library. Restart and recapture graphs
after changing it; INFO logs identify `shared FP32 vector decode` when selected.

The projection checks are in
`test/registered/kernels/ops/gemm/test_xpu_dflash_gemm.py`, including independent
FP64 selected-row references, exact decoding of every FP8 bit pattern, dynamic
graph input/weight/scale changes and fallback guards. Component parity does not
establish model quality or speculative acceptance.

`compact_kernel.hpp` adapts the BSD-3-Clause Intel kernel at
`src/sycl/kernels/moe/xe20/w8a16/moe_kernel.hpp` in the pinned kernel repository.
`compact.cpp` instantiates its original 16x64x32 W8A16 geometry. The original compact mainloop,
block copies and SYCL-TLA headers are read from the pinned external checkouts.
The additional `jit/csrc/attention/xpu_dflash_prefill.cpp` instantiates
`xe_fmha_fwd_prefill_runner.hpp` from that same pinned dependency; its full
Intel 2026 BSD notice is retained. See `LICENSE-BSD-3-Clause.txt` for the expert
wrapper; original copyright/SPDX notices are retained. The build receipt also
records the Torch version and C++ ABI used to link its libraries.

`dense_sequential.cpp` implements the selected dense shapes directly in SYCL.
It retains the Apache-2.0 SGLang notice for the FP8 layout/decoder experiment;
the subgroup matrix intrinsic follows the
[SPIR-V extension](https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_subgroup_matrix_multiply_accumulate.html).
The build explicitly enables this extension in addition to the native experts'
split-barrier extension. The receipt records this source and its library hash.

`paged_kernel.hpp` adapts the compact scheduler, and `paged_mainloop.hpp` adapts
the pinned `moe_mainloop.hpp` with raw-word cache fills after the native block
load. Their Intel BSD notices are preserved. `paged.cpp` instantiates the same
16x64x32 geometry; all three added source hashes are in the build receipt.


## DFlash SYCL activation quantization

`SGLANG_XPU_DFLASH_QUANT_SYCL=1` independently selects the bundled SYCL
block128 quantizer for contiguous BF16 target inputs `(4, 2048)`, `(4, 4096)`,
`(8, 2048)`, `(8, 4096)` and draft inputs `(3, 2048)`, `(7, 2048)` on B70.
It defaults to disabled and requires the DFlash master.
Set `SGLANG_XPU_DFLASH_SMALL_TLA_LIB` to the library built by
`test/manual/kernels/build_xpu_dflash_small_tla.py`; its existing ABI 1 and
precise quantization compiler flags are unchanged. A selected missing library
or symbol fails clearly. This child does not enable the small projection or
allocate its derived weight cache. Prequantized inputs and unsupported layouts
retain their existing dispatch. Restart and recapture after flag/library changes.

The raw FP8 outputs and FP32 scale bits are checked against the retained
quantizer, including zero groups, scale extremes, nonfinite values and dynamic
graph inputs. INFO reports `Using SYCL B70 DFlash block128 quant` only when this
child runs. On the measured M4 target/M3 draft, it replaces the remaining 121/1
Triton quant calls when combined with the separate small TLA projection.
Component time was approximately unchanged; fewer Triton calls are not an
end-to-end throughput improvement claim.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/quantization \
  -p 'test_dflash_quant_sycl.py'
```

## Long SYCL projection

`SGLANG_XPU_DFLASH_LONG_DENSE_SYCL=1` selects the M=4, N2048/K4096
projection. It defaults to disabled and requires the DFlash master,
`SGLANG_XPU_DFLASH_DENSE_DPAS=1` and `SGLANG_XPU_DFLASH_DENSE_SYCL=1`.
Rebuild with `dense_long.cpp` before enabling it; the existing sequential-dense
ABI 2 is unchanged, and the added long-dense symbols use their own ABI 1.
A selected missing library/symbol fails clearly. Other shapes retain their
existing dispatch. Restart and recapture after changing weights, flags or libraries.

The three SYCL kernels prepare 32 KiB of lane-major FP16 activations, execute
eight K=512 partials using M4/N16/K16 FP16 DPAS with SG=4 and GRF=128, and
reduce 256 KiB of FP32 partials to BF16. Scale and operand FP16 rounding is
preserved; FP32 reduction order changes and can affect greedy outputs and
speculative acceptance. Component speedup is not an end-to-end or quality claim.
INFO reports `Using SYCL M=4 long dense: 2048x4096 (split-K=8, SG=4, GRF=128)`.

The existing byte-preserving FP8 packing helper retains one extra layout copy
on each selected source tensor. On this 40-layer model the new copies total
0.3125 GiB, in addition to the existing 0.879 GiB for the large N projections.
Source FP8 parameters remain available. Warm weights before graph capture;
weight-version changes require eager repacking and discarded/rebuilt graphs.
There is no derived FP16 weight cache. Activation/scales remain dynamic.

```bash
python -m unittest discover -v \
  -s test/registered/kernels/ops/gemm \
  -p 'test_xpu_dflash_long_dense_sycl.py'
```

The tests check full-output CPU FP64 bounds for the documented FP16 operands
and FP32 accumulation, changing graph inputs/scales, byte-preserving packing,
eager weight invalidation and recapture, missing libraries, and opt-in guards.
