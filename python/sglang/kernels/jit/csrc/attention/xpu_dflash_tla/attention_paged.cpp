// SPDX-License-Identifier: BSD-3-Clause
// Adapt the pinned sgl-kernel-xpu SYCL-TLA decode; see README.md for provenance.
#define SYCL_INTEL_TARGET 20
#ifndef DFLASH_TLA_WINDOW
#define DFLASH_TLA_WINDOW 4
#endif
static_assert(DFLASH_TLA_WINDOW == 4 || DFLASH_TLA_WINDOW == 8);
#include <c10/core/DeviceGuard.h>
#include <sycl/ext/intel/esimd.hpp>

#include "dflash_tla_decode.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

#if DFLASH_TLA_WINDOW == 8
namespace dflash_paged_tla_decode = dflash_paged_tla_decode_m8;
#endif

namespace dflash_paged_tla {
thread_local std::string error;
template <int Window>
class Pack;

void metadata(
    sycl::queue& queue,
    uint16_t* kb,
    uint16_t* vb,
    const int* ptr,
    const int64_t* ids,
    const int64_t* loc,
    int* lengths,
    int* cuq,
    int* table) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Pack<DFLASH_TLA_WINDOW>>(sycl::range<1>(128 * 1024 / 64), [=](sycl::id<1> gid) SYCL_ESIMD_KERNEL {
      namespace es = sycl::ext::intel::esimd;
      int start = ptr[0], n = ptr[1] - start, g = gid[0];
      // Retain the entire local window plus a page-aligned left margin.
      // n becomes 0..2174 regardless of the full prefix length.
      int discarded = n > 2047 ? ((n - 2047) / 128) * 128 : 0;
      start += discarded;
      n -= discarded;
      if (g < DFLASH_TLA_WINDOW) lengths[g] = n + DFLASH_TLA_WINDOW;
      if (g < DFLASH_TLA_WINDOW + 1) cuq[g] = g;
      if (g < DFLASH_TLA_WINDOW * 18) {
        int pos = (g % 18) * 128;
        int64_t slot = pos < n ? ids[start + pos] : (pos < n + DFLASH_TLA_WINDOW ? loc[pos - n] : 0);
        table[g] = int(slot / 128);
      }
      // Current request owns its final allocated 128-token page. Zero only
      // the invalid tail there, preventing masked 0*NaN from stale padding.
      int token = g / 16, d = (g % 16) * 64;
      int64_t last = loc[DFLASH_TLA_WINDOW - 1];
      int count = 127 - int(last % 128);
      if (token < count) {
        es::simd<uint16_t, 64> zero(0);
        es::block_store<uint16_t, 64>(kb + (last + 1 + token) * 1024 + d, zero);
        es::block_store<uint16_t, 64>(vb + (last + 1 + token) * 1024 + d, zero);
      }
    });
  });
}

int decode(
    void* q, void* pk, void* pv, int* cuq, int* lengths, int* table, void* out, int qstride, int device, int slots) {
  using namespace cute;
  dflash_paged_tla_decode::Arguments p{};
  p.q_ptr = q;
  p.k_ptr = pk;
  p.v_ptr = pv;
  p.o_ptr = out;
  p.q_row_stride = qstride;
  p.q_head_stride = 128;
  p.k_row_stride = p.v_row_stride = 1024;
  p.k_head_stride = p.v_head_stride = 128;
  p.v_dim_stride = 1;
  p.k_stride_page = p.v_stride_page = 128 * 1024;
  p.k_stride_seq = p.v_stride_seq = 1024;
  p.k_stride_heads = p.v_stride_heads = 128;
  p.o_row_stride = 4096;
  p.o_head_stride = 128;
  p.b = p.b_k = DFLASH_TLA_WINDOW;
  p.h = 32;
  p.h_k = 8;
  p.q_group_size = 4;
  p.d = p.dv = p.d_rounded = p.dv_rounded = 128;
  p.total_q = DFLASH_TLA_WINDOW;
  p.seqlen_q = 1;
  p.total_k = slots;
  p.seqlen_k = 2175 + DFLASH_TLA_WINDOW;
  p.softmax_scale = 0.08838834764831845f;
  p.cu_seqlens_q = cuq;
  p.cu_seqlens_k = lengths;
  p.page_table = table;
  p.max_num_pages_per_seq = 18;
  p.page_table_batch_stride = 18;
  p.page_size = 128;
  p.num_pages = slots / 128;
  p.pagedkv_tma = true;
  p.is_bf16 = true;
  p.is_local = true;
  p.window_size_left = p.window_size_right = 2047;
  p.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kXPU, device));
  using QK = Shape<Int<4>, Int<128>, _64>;
  using PV = Shape<Int<4>, _32, Int<128>>;
  using Out = Shape<Int<4>, Int<128>>;
  constexpr int subgroups = dflash_paged_tla_decode::kv_subgroups_for_slm(8, 4, 128);
  using Layout = cute::Layout<Shape<_1, Int<subgroups>, _1>>;
  return dflash_paged_tla_decode::DecodeConfig<
      false,
      true,
      false,
      false,
      false,
      QK,
      PV,
      Out,
      Layout,
      void,
      1,
      false,
      cutlass::bfloat16_t,
      cutlass::bfloat16_t,
      cutlass::bfloat16_t,
      cutlass::bfloat16_t>::run_paged(p);
}
}  // namespace dflash_paged_tla

extern "C" int dflash_attention_paged_tla_abi() {
  return 1;
}
extern "C" int dflash_attention_paged_tla_window() {
  return DFLASH_TLA_WINDOW;
}
extern "C" const char* dflash_attention_paged_tla_error() {
  return dflash_paged_tla::error.c_str();
}
extern "C" int dflash_attention_paged_tla_run(
    void* queue,
    void* q,
    void* kb,
    void* vb,
    const int* ptr,
    const int64_t* ids,
    const int64_t* loc,
    void* out,
    int* lengths,
    int* cuq,
    int* table,
    int qstride,
    int device,
    int slots) {
  try {
    if ((qstride != 4096 && qstride != 6144) || slots < 128 || slots % 128)
      throw std::runtime_error("Unsupported direct KV layout");
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device));
    dflash_paged_tla::metadata(
        *static_cast<sycl::queue*>(queue),
        static_cast<uint16_t*>(kb),
        static_cast<uint16_t*>(vb),
        ptr,
        ids,
        loc,
        lengths,
        cuq,
        table);
    return dflash_paged_tla::decode(q, kb, vb, cuq, lengths, table, out, qstride, device, slots);
  } catch (const std::exception& e) {
    dflash_paged_tla::error = e.what();
    return 1;
  }
}
