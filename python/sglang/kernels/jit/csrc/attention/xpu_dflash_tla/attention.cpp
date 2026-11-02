// SPDX-License-Identifier: BSD-3-Clause
// Adapt the pinned sgl-kernel-xpu SYCL-TLA decode; see README.md for provenance.
#define SYCL_INTEL_TARGET 20
#include <c10/core/DeviceGuard.h>
#include <sycl/ext/intel/esimd.hpp>

#include "dflash_tla_decode.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dflash_tla {
thread_local std::string error;
class Pack;

void pack(
    sycl::queue& queue,
    const uint16_t* k,
    const uint16_t* v,
    const uint16_t* kb,
    const uint16_t* vb,
    const int* ptr,
    const int64_t* ids,
    uint16_t* pk,
    uint16_t* pv,
    int* lengths,
    int* cuq,
    int* table) {
  queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<Pack>(sycl::range<1>(2176 * 1024 / 64), [=](sycl::id<1> gid) SYCL_ESIMD_KERNEL {
      namespace es = sycl::ext::intel::esimd;
      int i = gid[0] * 64, token = i / 1024, d = i % 1024;
      int start = ptr[0], length = ptr[1] - start;
      es::simd<uint16_t, 64> key(0), value(0);
      if (token < length) {
        int64_t slot = ids[start + token];
        key = es::block_load<uint16_t, 64>(kb + slot * 1024 + d);
        value = es::block_load<uint16_t, 64>(vb + slot * 1024 + d);
      } else if (token < length + 4) {
        key = es::block_load<uint16_t, 64>(k + (token - length) * 1024 + d);
        value = es::block_load<uint16_t, 64>(v + (token - length) * 1024 + d);
      }
      // Zero the last page: masked K/V lanes must not retain NaN payloads.
      es::block_store<uint16_t, 64>(pk + i, key);
      es::block_store<uint16_t, 64>(pv + i, value);
      if (gid[0] == 0) {
        es::block_store<int, 4>(lengths, es::simd<int, 4>(length + 4));
      }
      if (gid[0] < 5) cuq[gid[0]] = gid[0];
      if (gid[0] < 68) table[gid[0]] = gid[0] % 17;
    });
  });
}

int decode(void* q, void* pk, void* pv, int* cuq, int* lengths, int* table, void* out, int qstride, int device) {
  using namespace cute;
  dflash_tla_decode::Arguments p{};
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
  p.b = p.b_k = 4;
  p.h = 32;
  p.h_k = 8;
  p.q_group_size = 4;
  p.d = p.dv = p.d_rounded = p.dv_rounded = 128;
  p.total_q = 4;
  p.seqlen_q = 1;
  p.total_k = 2176;
  p.seqlen_k = 2051;
  p.softmax_scale = 0.08838834764831845f;
  p.cu_seqlens_q = cuq;
  p.cu_seqlens_k = lengths;
  p.page_table = table;
  p.max_num_pages_per_seq = 17;
  p.page_table_batch_stride = 17;
  p.page_size = 128;
  p.num_pages = 17;
  p.pagedkv_tma = true;
  p.is_bf16 = true;
  p.is_local = true;
  p.window_size_left = p.window_size_right = 2047;
  p.tensor_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kXPU, device));
  using QK = Shape<Int<4>, Int<128>, _64>;
  using PV = Shape<Int<4>, _32, Int<128>>;
  using Out = Shape<Int<4>, Int<128>>;
  constexpr int subgroups = dflash_tla_decode::kv_subgroups_for_slm(8, 4, 128);
  using Layout = cute::Layout<Shape<_1, Int<subgroups>, _1>>;
  return dflash_tla_decode::DecodeConfig<
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
}  // namespace dflash_tla

extern "C" int dflash_attention_tla_abi() {
  return 1;
}

extern "C" const char* dflash_attention_tla_error() {
  return dflash_tla::error.c_str();
}

extern "C" int dflash_attention_tla_run(
    void* queue,
    void* q,
    const void* k,
    const void* v,
    const void* kb,
    const void* vb,
    const int* ptr,
    const int64_t* ids,
    void* out,
    void* pk,
    void* pv,
    int* lengths,
    int* cuq,
    int* table,
    int qstride,
    int device) {
  try {
    if (qstride != 4096 && qstride != 6144) throw std::runtime_error("Unsupported M4 query stride");
    c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device));
    dflash_tla::pack(
        *static_cast<sycl::queue*>(queue),
        static_cast<const uint16_t*>(k),
        static_cast<const uint16_t*>(v),
        static_cast<const uint16_t*>(kb),
        static_cast<const uint16_t*>(vb),
        ptr,
        ids,
        static_cast<uint16_t*>(pk),
        static_cast<uint16_t*>(pv),
        lengths,
        cuq,
        table);
    return dflash_tla::decode(q, pk, pv, cuq, lengths, table, out, qstride, device);
  } catch (const std::exception& e) {
    dflash_tla::error = e.what();
    return 1;
  }
}
