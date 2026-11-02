// SPDX-License-Identifier: Apache-2.0
// Experimental BF16 -> block128 FP8 quantization for B70, with FP32 scales.
// Keeps the retained kernel's two FP32 divisions and maxnum/clamp semantics.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFMax(int, int, float) noexcept;
namespace dflash_small_tla_quant {
thread_local std::string error;
struct Args {
  const uint16_t* x;
  uint8_t* q;
  float* scale;
  int groups;
  float eps, limit;
};
inline float reduce_max(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFMax(3, 0, x);
#else
  return x;
#endif
}
inline uint8_t encode(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x), mag = bits & 0x7fffffffu;
  if (mag > 0x7f800000u) return 127;
  uint32_t sign = (bits >> 24) & 128u;
  uint32_t value;
  if (mag < 0x3c800000u)
    value = uint32_t(sycl::rint(sycl::fabs(x) * 512.f));
  else {
    uint32_t exp = mag >> 23, mant = mag & 0x7fffffu;
    value = ((exp > 120 ? exp - 120 : 0) << 3) + uint32_t(sycl::rint(float(mant) * 0x1p-20f));
  }
  return uint8_t(sign | sycl::min(value, 126u));
}
template <int SGs>
class Quant;
template <int SGs>
void launch(sycl::queue& queue, Args a) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  int global = ((a.groups + SGs - 1) / SGs) * SGs * 32;
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Quant<SGs>>(sycl::nd_range<1>(global, SGs * 32), props, [=](sycl::nd_item<1> it) {
      unsigned g = it.get_global_linear_id() / 32, lane = it.get_global_linear_id() % 32;
      if (g >= unsigned(a.groups)) return;
      sycl::vec<uint16_t, 4> raw;
      raw.load(0, a.x + g * 128 + lane * 4);
      float y[4], mx = 0.f;
      for (int i = 0; i < 4; ++i) {
        y[i] = sycl::bit_cast<float>(uint32_t(raw[i]) << 16);
        mx = sycl::fmax(mx, sycl::fabs(y[i]));
      }
      float maximum = sycl::fmax(reduce_max(mx), a.eps);
      float scale = maximum / a.limit, inv = 1.f / scale;
      sycl::vec<uint8_t, 4> out;
      for (int i = 0; i < 4; ++i)
        out[i] = encode(sycl::fmin(sycl::fmax(y[i] * inv, -a.limit), a.limit));
      out.store(0, a.q + g * 128 + lane * 4);
      if (lane == 0) a.scale[g] = scale;
    });
  });
}
}  // namespace dflash_small_tla_quant
extern "C" int dflash_small_tla_quantize(
    void* queue, const uint16_t* x, uint8_t* q, float* scales, int groups, float eps, float limit, int mode) {
  try {
    if (!queue || !x || !q || !scales || groups <= 0 || eps <= 0 || limit != 448.f) return 2;
    auto& stream = *static_cast<sycl::queue*>(queue);
    dflash_small_tla_quant::Args a{x, q, scales, groups, eps, limit};
    if (mode == 0)
      dflash_small_tla_quant::launch<1>(stream, a);
    else if (mode == 1)
      dflash_small_tla_quant::launch<2>(stream, a);
    else if (mode == 2)
      dflash_small_tla_quant::launch<4>(stream, a);
    else
      return 2;
    return 0;
  } catch (const std::exception& e) {
    dflash_small_tla_quant::error = e.what();
    return 1;
  }
}
extern "C" const char* dflash_small_tla_quant_error() {
  return dflash_small_tla_quant::error.c_str();
}
