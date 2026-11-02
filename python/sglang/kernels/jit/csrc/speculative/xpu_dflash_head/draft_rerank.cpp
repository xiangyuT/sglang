/* Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Draft BF16 shortlist rescore and global-ID top16, SYCL.
 * Math follows dflash_head_xpu.py; build separately with precise FP32 math.
 * Calls sharing counters must be ordered on one queue. Each ID row is unique.
 */
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off) reassociate(off)
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int, int, float) noexcept;
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int, int, float, unsigned) noexcept;
namespace dflash_draft_rerank_impl {
inline float reduce32(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFAdd(3, 0, x);
#else
  return x;
#endif
}
inline float reduce4(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFAdd(3, 3, x, 4u);
#else
  return x;
#endif
}
thread_local std::string error;
class Rerank;
inline float bf16(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
inline uint16_t rounded(float x) {
  uint32_t b = sycl::bit_cast<uint32_t>(x);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
inline uint64_t key(uint16_t value, int64_t id) {
  uint16_t ordered = (value & 0x8000) ? uint16_t(~value) : uint16_t(value | 0x8000);
  return (uint64_t(ordered) << 32) | uint64_t(0xffffffffu - uint32_t(id));
}
}  // namespace dflash_draft_rerank_impl
extern "C" int dflash_draft_rerank_abi() {
  return 1;
}
extern "C" const char* dflash_draft_rerank_error() {
  return dflash_draft_rerank_impl::error.c_str();
}
extern "C" int dflash_draft_rerank_run(
    void* queue_ptr,
    const uint16_t* hidden,
    const uint16_t* weight,
    const int64_t* ids,
    uint16_t* scores,
    uint16_t* values,
    int64_t* output,
    uint32_t* counters,
    int rows) {
  try {
    if (!queue_ptr || !hidden || !weight || !ids || !scores || !values || !output || !counters ||
        (rows != 3 && rows != 7))
      throw std::runtime_error(
          "Expected M3/M7 contiguous BF16 H2048/weight248320x2048,unique valid top64 IDs,padded zero counters");
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    namespace one = sycl::ext::oneapi::experimental;
    namespace intel = sycl::ext::intel::experimental;
    one::properties props{one::sub_group_size<32>, intel::grf_size<128>};
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<int, 1> last(sycl::range<1>(1), h);
      sycl::local_accessor<uint64_t, 1> keys(sycl::range<1>(64), h);
      sycl::local_accessor<float, 1> partials(sycl::range<1>(16), h);
      h.parallel_for<dflash_draft_rerank_impl::Rerank>(
          sycl::nd_range<1>(sycl::range<1>(rows * 16 * 128), sycl::range<1>(128)), props, [=](sycl::nd_item<1> item) {
            unsigned group = item.get_group_linear_id(), local = item.get_local_linear_id();
            unsigned row = group / 16, first = (group % 16) * 4, lane = local % 32, sg = local / 32;
            sycl::vec<uint16_t, 8> h0, h1;
            h0.load(0, hidden + row * 2048 + local * 8);
            h1.load(0, hidden + row * 2048 + local * 8 + 1024);
#pragma unroll
            for (int member = 0; member < 4; ++member) {
              int64_t vocab_id = ids[row * 64 + first + member];
              sycl::vec<uint16_t, 8> w0, w1;
              w0.load(0, weight + vocab_id * 2048 + local * 8);
              w1.load(0, weight + vocab_id * 2048 + local * 8 + 1024);
              float product[16];
#pragma unroll
              for (int j = 0; j < 8; ++j) {
                product[j] = dflash_draft_rerank_impl::bf16(h0[j]) * dflash_draft_rerank_impl::bf16(w0[j]);
                product[j + 8] = dflash_draft_rerank_impl::bf16(h1[j]) * dflash_draft_rerank_impl::bf16(w1[j]);
              }
#pragma unroll
              for (int count = 8; count > 0; count /= 2)
#pragma unroll
                for (int j = 0; j < count; ++j)
                  product[j] = product[j * 2] + product[j * 2 + 1];
              float sum = dflash_draft_rerank_impl::reduce32(product[0]);
              if (lane == 0) partials[member * 4 + sg] = sum;
            }
            sycl::group_barrier(item.get_group());
            float cross = local < 16 ? partials[local] : 0.f;
            float value = dflash_draft_rerank_impl::reduce4(cross);
            if (local < 16 && local % 4 == 0)
              scores[row * 64 + first + local / 4] = dflash_draft_rerank_impl::rounded(value);
            sycl::group_barrier(item.get_group(), sycl::memory_scope::device);
            if (local == 0) {
              sycl::atomic_ref<
                  uint32_t,
                  sycl::memory_order::acq_rel,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>
                  ticket(counters[row * 16]);
              last[0] = ticket.fetch_add(1) == 15;
              if (last[0]) ticket.store(0);
            }
            sycl::group_barrier(item.get_group(), sycl::memory_scope::device);
            if (last[0]) {
              if (local < 64)
                keys[local] = dflash_draft_rerank_impl::key(scores[row * 64 + local], ids[row * 64 + local]);
              sycl::group_barrier(item.get_group());
              if (local < 64) {
                uint64_t mine = keys[local];
                int rank = 0;
#pragma unroll 4
                for (int j = 0; j < 64; ++j)
                  rank += keys[j] > mine;
                if (rank < 16) {
                  values[row * 16 + rank] = scores[row * 64 + local];
                  output[row * 16 + rank] = ids[row * 64 + local];
                }
              }
            }
          });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_draft_rerank_impl::error = e.what();
    return 1;
  }
}
