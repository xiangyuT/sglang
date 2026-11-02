# Intel XPU paged expert and GGUF development snapshot

This branch records the SGLang fork's Intel XPU development state on 2026-11-02.
The main serving path is Qwen3.8-Flash-Next GGUF on Intel Arc Pro B70, with
compressed expert paging, multi-GPU TP/EP, NEXTN/MTP and agentic KV reuse.
Earlier Qwen3.6 FP8 and DFlash2 experiments are retained as development history.

This is a source snapshot for review and continued development. Experimental
switches remain opt-in; inclusion here does not establish model correctness or
performance for every combination.

## Source lineage

- Upstream ancestor: `fdd89d182a82fc78cc2c1b6dd0251d5bb7893172`.
- Previous fork head: `c429ab8f368a5accc9e26c9cbac1587f3afdcf98`.
  Its three Intel XPU commits add native expert paging/decode graphs,
  single-token block-FP8 GEMV and fused paged expert compute.
- This snapshot preserves that development lineage. It is not rebased onto the
  current fork `main`.
- Package versions are unchanged. Model files, compiled libraries, local logs,
  request/response dumps and private development records are not included.

## Implemented areas

| Area | Implementation |
| --- | --- |
| Expert paging | Expert-level GPU slots, logical expert identity and device residency/planning. FP8 uses a pinned host backing store; GGUF uses resident GPU slots and cold expert mirrors with exchange. |
| GGUF loading | Native sharded metadata/byte readers, compressed expert rows, mixed quantization geometry, IQ2/IQ4 paths, original encoded bytes and PLE handling. |
| Multi-GPU | XPU TP/EP integration, local expert partitions including dummy expert handling, per-rank pagers, peer collectives and prefill precision boundaries. |
| Decode kernels | SYCL expert gate/up/down, packed low-bit and Q8 activation paths, dense/head operations, hyperconnections, routing/TopK and GDN. |
| Attention and KV | QSA kernels and metadata, INT8 KV pools, sparse attention integration, long-context and batched verification paths. |
| Graph execution | XPU breakable graph backend, target/draft graph state, explicit eager boundaries for paging and stable captured buffers. |
| Speculative decoding | GGUF NEXTN/MTP loading and execution, Q8 draft pack support, token maps and draft/verify integration. |
| Agentic serving | Radix and session prefix reuse, Mamba state ownership/checkpoints, HiCache host-pool handling and MTP/target KV layout separation. |
| Earlier experiments | Qwen3.6 FP8 kernels, CPU expert paths and DFlash2 SYCL/TLA prototypes. These are not the selected current GGUF NEXTN serving path. |

The selected two-card IQ4 development profile uses BF16 activations, INT8 KV,
TP2/EP2, NEXTN with 3 speculative steps / 4 draft tokens / top-k 1, and
breakable decode graphs for batch sizes 1–8. It enables radix cache, session
radix cache, HiCache and overlap. Its per-rank expert cache is initialized with
152 slots and grows to 160 after capture; those values describe this profile
and are not universal capacity recommendations.

The configured context is 262144 with a 270336-token shared pool and at most
8 running requests. This does not assert capacity for eight simultaneous
256K histories.

## Source map

- `python/sglang/srt/model_loader/qwen4_gguf*.py`,
  `gguf_native_reader.py`, `gguf_rows.py`: model loading and encoded rows.
- `python/sglang/srt/layers/quantization/qwen4_gguf_xpu.py`,
  `gguf_ep.py`, `gguf_tp_prefill.py`: compressed experts and parallel execution.
- `python/sglang/kernels/ops/quantization/gguf_gpu_cache.py`:
  GPU residency, host mirrors, growth and compaction.
- `python/sglang/kernels/jit/csrc/quantization/xpu_gguf/`:
  low-bit SYCL operations and expert cache kernels.
- `python/sglang/kernels/jit/csrc/attention/xpu_qsa/` and
  `kernels/ops/attention/xpu_qsa*.py`: QSA and INT8 attention.
- `python/sglang/srt/hardware_backend/xpu/graph_runner/` and
  `srt/speculative/`: graph and speculative decoding integration.
- `python/sglang/srt/mem_cache/`: KV storage and prefix/session ownership.
- `test/manual/kernels/build_xpu_*.py`: component library build entry points.
- `test/registered/`: component and integration contract tests.

## Native build and runtime dependencies

The development environment uses Linux and the Omix 0.4.0 image with Intel
oneAPI, Torch XPU and sgl-kernel-xpu. Native libraries must match the selected
Torch ABI and GPU architecture; a source snapshot does not contain those binaries.

The component build scripts accept an explicit output path and write build
receipts. For example, from the repository root in the development environment:

```bash
source /opt/intel/oneapi/setvars.sh
python test/manual/kernels/build_xpu_gguf.py --output build/xpu/gguf.so
python test/manual/kernels/build_xpu_qsa.py --output build/xpu/qsa.so
```

These are component build entry points, not the entire tuned serving setup.
The implementation loads optional libraries through the corresponding
`SGLANG_XPU_*_LIB` environment variables. Merely launching the ordinary server
without those libraries and feature flags does not reproduce the tuned profile.

The existing native GDN prefill fence shim is also archived under
[`test/manual/kernels/xpu_gdn_prefill_fence/`](test/manual/kernels/xpu_gdn_prefill_fence/).
It preserves the BF16-activation / FP32-state dispatcher and adds global memory
fences to its pinned upstream header. Its process-local loader preserves the
installed package.

Pinned external sources:

- [sgl-project/sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu/tree/028ac5a4e92a4addabeedac1a94374515f68e4a5)
  at `028ac5a4e92a4addabeedac1a94374515f68e4a5`, package version 0.4.0.
- [intel/sycl-tla](https://github.com/intel/sycl-tla/tree/7b56b936b87f799f80dde60c2fd8d11c1128cebb)
  at `7b56b936b87f799f80dde60c2fd8d11c1128cebb`.

To build that compatibility shim, supply the local checkouts explicitly:

```bash
python test/manual/kernels/xpu_gdn_prefill_fence/build.py \
  --sgl-kernel-source /path/to/sgl-kernel-xpu \
  --sycl-tla-source /path/to/sycl-tla \
  --output build/xpu/gdn-prefill-fence.so
```

The helper verifies the original and patched header hashes and writes its
patched copy beside the output, without changing either external checkout.
Set `SGLANG_XPU_GDN_PREFILL_FENCE_LIB` to that library's absolute path when using
the adjacent `serve.py` compatibility launcher with the normal server arguments.
Other tuned libraries and model assets still need their own explicit paths.

## Validation for this publication

- Python syntax checks cover the exported additions and modifications.
- The selected CPU-only loader, expert partition, KV/cache ownership, precision
  guard and speculative geometry tests pass: **271 passed, 9 subtests passed**.
- The new portable GDN build helper's preparation-only path reproduces the
  existing patched header SHA-256 exactly. A fresh native build from that
  helper was not run for publication.
- Two stale test expectations were updated in this export: growth before expert
  sources are bound must fail, and the prefill guard's current error text is checked.
  The active serving source was not changed.
- No complete GPU CI matrix or new end-to-end acceptance run was performed for
  this publication. Current AgentPerf C1 collection is separate and has no
  finalized comparison in this branch.

## Follow-up work

- Unify FP8 and GGUF host-store/mirror management behind a shared residency
  abstraction while retaining their encoding and layout differences.
- Consolidate the component build scripts and environment-driven library selection
  into a maintained reproducible serving profile.
- Separate adopted paths from disabled experiments before preparing small upstream
  contributions; reconcile this snapshot with newer upstream changes independently.
- Validate larger concurrency/context combinations and experimental kernels with
  their own correctness and performance gates.
- DFlash2 and CPU expert experiments remain outside the selected NEXTN path.

## Attribution

The Strata-derived GGUF and QSA code retains its Strata MIT license notices.
GGML-derived decoding tables retain the GGML MIT notices. Other vendored code
retains the adjacent license files; the repository's Apache-2.0 license remains
unchanged. See the corresponding native source directories for the original
copyright and license text.

