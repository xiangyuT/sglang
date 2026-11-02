// SPDX-License-Identifier: Apache-2.0
// B70 ordinary AR: BF16 [1,256] -> FP32/int32 top8, SG32/GRF128.
// Compile for BMG with -cl-fp32-correctly-rounded-divide-sqrt.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>

#pragma clang fp contract(off) reassociate(off)
namespace ar_topk_b70_abi1 {
thread_local std::string error;
class Kernel;

void run(sycl::queue& queue, const uint16_t* x, float* weights, int* ids, bool renorm) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<32>, intel::grf_size<128>};
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Kernel>(sycl::nd_range<1>(32, 32), props, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      int lane = it.get_local_linear_id();
      sycl::vec<uint16_t, 8> raw;
      raw.load(0, x + lane * 8);
      float a[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        a[e] = sycl::bit_cast<float>(uint32_t(raw[e]) << 16);
      float maximum = a[0];
#pragma unroll
      for (int e = 1; e < 8; ++e)
        maximum = maximum > a[e] ? maximum : a[e];
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        float other = sycl::permute_group_by_xor(sg, maximum, delta);
        maximum = maximum > other ? maximum : other;
      }
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        a[e] = sycl::native::exp(a[e] - maximum);
        sum += a[e];
      }
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2)
        sum += sycl::permute_group_by_xor(sg, sum, delta);
      float inv = 1.f / sum;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        a[e] *= inv;
      float selected[8];
      int indices[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float value;
        int expert;
        if (sycl::isfinite(sum) && sum > 0.f) {
          uint32_t bits = 0;
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (a[e] >= 0.f) bits = sycl::max(bits, sycl::bit_cast<uint32_t>(a[e]));
          bits = sycl::reduce_over_group(sg, bits, sycl::maximum<uint32_t>());
          value = sycl::bit_cast<float>(bits);
          expert = 256;
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (a[e] == value) expert = sycl::min(expert, lane * 8 + e);
          expert = sycl::reduce_over_group(sg, expert, sycl::minimum<int>());
        } else {
          // Preserve the native float argmax behavior for nonfinite softmax.
          value = a[0];
          expert = lane * 8;
#pragma unroll
          for (int e = 1; e < 8; ++e)
            if (a[e] > value) {
              value = a[e];
              expert = lane * 8 + e;
            }
#pragma unroll
          for (int delta = 16; delta > 0; delta /= 2) {
            float other = sycl::permute_group_by_xor(sg, value, delta);
            int index = sycl::permute_group_by_xor(sg, expert, delta);
            if (other > value || (other == value && index < expert)) {
              value = other;
              expert = index;
            }
          }
        }
        selected[k] = value;
        indices[k] = expert;
        if (lane == expert / 8) a[expert % 8] = -10000.f;
      }
      if (lane == 0) {
        float selected_sum = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
          selected_sum += selected[k];
        float norm = renorm ? 1.f / selected_sum : 1.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          weights[k] = selected[k] * norm;
          ids[k] = indices[k];
        }
      }
    });
  });
}
}  // namespace ar_topk_b70_abi1

extern "C" int ar_topk_abi() {
  return 1;
}
extern "C" const char* ar_topk_error() {
  return ar_topk_b70_abi1::error.c_str();
}
extern "C" int ar_topk_run(void* queue, const uint16_t* x, float* weights, int* ids, int renorm) {
  try {
    if (!queue || !x || !weights || !ids || (renorm != 0 && renorm != 1)) return 2;
    ar_topk_b70_abi1::run(*static_cast<sycl::queue*>(queue), x, weights, ids, bool(renorm));
    return 0;
  } catch (const std::exception& e) {
    ar_topk_b70_abi1::error = e.what();
    return 1;
  }
}
