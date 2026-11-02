// SPDX-License-Identifier: Apache-2.0
// Eager routed-prefill reduction. Preserve the Omix Torch vt0=4 sum order.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>
namespace sglang_route_reduce {
using BF = sycl::ext::oneapi::bfloat16;
thread_local std::string error;
template <int K, class O> class Kernel;
template <int K, class O>
void launch(sycl::queue& q, const float* sorted, const int32_t* inverse,
            const float* weights, O* out, int T, int H, int N) {
  const size_t units = size_t(T) * H / 4;
  q.parallel_for<Kernel<K, O>>(
      sycl::nd_range<1>((units + 127) / 128 * 128, 128),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t u = it.get_global_linear_id();
        if (u >= units) return;
        const int t = u / (H / 4), h = (u % (H / 4)) * 4;
        float acc[4][4] = {};
        for (int k = 0; k < K; ++k) {
          int row = inverse[t * K + k];
          float w = weights[t * K + k];
          sycl::vec<float, 4> values(0.0f);
          if (row >= 0 && row < N) values = *reinterpret_cast<const sycl::vec<float, 4>*>(sorted + size_t(row) * H + h);
          for (int j = 0; j < 4; ++j) {
            float p = values[j] * w;
            acc[k % 4][j] = acc[k % 4][j] + p;
          }
        }
        for (int j = 0; j < 4; ++j) {
          float v = acc[0][j];
          v = ((v + acc[1][j]) + acc[2][j]) + acc[3][j];
          out[u * 4 + j] = O(v);
        }
      });
}
template <class O>
void dispatch(sycl::queue& q, const float* sorted, const int32_t* inverse,
              const float* weights, O* out, int T, int K, int H, int N) {
#define CASE(C) case C: launch<C>(q, sorted, inverse, weights, out, T, H, N); break
  switch (K) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); CASE(9); CASE(10); }
#undef CASE
}
}
extern "C" const char* xpu_gguf_route_reduce_error() { return sglang_route_reduce::error.c_str(); }
extern "C" int xpu_gguf_route_reduce(void* stream, const float* sorted,
    const int32_t* inverse, const float* weights, void* out,
    int T, int K, int H, int dtype, int N) {
  if (!stream || !sorted || !inverse || !weights || !out || T < 5 || T > 4096 ||
      K < 1 || K > 10 || N < 1 || N > T * K || H != 2560 || (dtype != 0 && dtype != 1)) return 2;
  try {
    auto& q = *static_cast<sycl::queue*>(stream);
    if (!q.has_property<sycl::property::queue::in_order>()) return 2;
    if (dtype == 0) sglang_route_reduce::dispatch(q, sorted, inverse, weights,
        static_cast<sglang_route_reduce::BF*>(out), T, K, H, N);
    else sglang_route_reduce::dispatch(q, sorted, inverse, weights,
        static_cast<float*>(out), T, K, H, N);
    return 0;
  } catch (const std::exception& e) { sglang_route_reduce::error = e.what(); return 1; }
}
