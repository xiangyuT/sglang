// SPDX-License-Identifier: Apache-2.0
// Optional B70 M4/M8 graph-input copies, submitted on the current SYCL queue.
// Decode geometry: M/M/M/1/1 INT64 elements. No dtype conversion.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
namespace sglang_dflash_small_copy_impl {
thread_local std::string error;
struct Args {
  int64_t* dst[5];
  const int64_t* src[5];
};
template <int Window>
class Copy;
}  // namespace sglang_dflash_small_copy_impl
extern "C" int dflash_small_copy_abi() {
  return 1;
}
extern "C" const char* dflash_small_copy_error() {
  return sglang_dflash_small_copy_impl::error.c_str();
}
template <int Window>
int run_small_copy(
    void* queue,
    void* d0,
    void* d1,
    void* d2,
    void* d3,
    void* d4,
    const void* s0,
    const void* s1,
    const void* s2,
    const void* s3,
    const void* s4) {
  try {
    if (!queue || !d0 || !d1 || !d2 || !d3 || !d4 || !s0 || !s1 || !s2 || !s3 || !s4)
      throw std::runtime_error("Invalid small-copy pointer");
    sglang_dflash_small_copy_impl::Args a{
        {static_cast<int64_t*>(d0),
         static_cast<int64_t*>(d1),
         static_cast<int64_t*>(d2),
         static_cast<int64_t*>(d3),
         static_cast<int64_t*>(d4)},
        {static_cast<const int64_t*>(s0),
         static_cast<const int64_t*>(s1),
         static_cast<const int64_t*>(s2),
         static_cast<const int64_t*>(s3),
         static_cast<const int64_t*>(s4)}};
    auto& q = *static_cast<sycl::queue*>(queue);
    q.submit([&](sycl::handler& h) {
      h.parallel_for<sglang_dflash_small_copy_impl::Copy<Window>>(sycl::range<1>(32), [=](sycl::id<1> id) {
        int i = id[0];
        if (i >= 3 * Window + 2) return;
        int job = i < 3 * Window ? i / Window : i - 3 * Window + 3;
        int offset = i < 3 * Window ? i % Window : 0;
        a.dst[job][offset] = a.src[job][offset];
      });
    });
    return 0;
  } catch (const std::exception& e) {
    sglang_dflash_small_copy_impl::error = e.what();
    return 1;
  }
}

extern "C" int dflash_small_copy(
    void* queue, void* d0, void* d1, void* d2, void* d3, void* d4,
    const void* s0, const void* s1, const void* s2, const void* s3, const void* s4) {
  return run_small_copy<4>(queue, d0, d1, d2, d3, d4, s0, s1, s2, s3, s4);
}
extern "C" int dflash_small_copy_m8(
    void* queue, void* d0, void* d1, void* d2, void* d3, void* d4,
    const void* s0, const void* s1, const void* s2, const void* s3, const void* s4) {
  return run_small_copy<8>(queue, d0, d1, d2, d3, d4, s0, s1, s2, s3, s4);
}
