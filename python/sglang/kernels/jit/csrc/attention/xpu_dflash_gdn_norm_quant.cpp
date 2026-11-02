// SPDX-License-Identifier: Apache-2.0
// Opt-in B70 GDN gated RMSNorm128 + block128 FP8 quantization.
// Match installed Triton blocked[1,8]/threads[2,16] reduction and BF16 boundary.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
SYCL_EXTERNAL __attribute__((const, convergent)) float __spirv_GroupNonUniformFAdd(int, int, float, unsigned) noexcept;
SYCL_EXTERNAL __attribute__((const, convergent)) float __spirv_GroupNonUniformFMax(int, int, float, unsigned) noexcept;
namespace dflash_gdn_norm_quant_impl {
thread_local std::string error;
inline float bf16(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
inline uint16_t rounded(float x) {
  auto b = sycl::bit_cast<uint32_t>(x);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
inline float sum16(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFAdd(3, 3, x, 16u);
#else
  return x;
#endif
}
inline float max16(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFMax(3, 3, x, 16u);
#else
  return x;
#endif
}
inline uint8_t encode(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x), mag = bits & 0x7fffffffu;
  if (mag > 0x7f800000u) return 127;
  uint32_t sign = (bits >> 24) & 128u, value;
  if (mag < 0x3c800000u)
    value = uint32_t(sycl::rint(sycl::fabs(x) * 512.f));
  else {
    uint32_t exp = mag >> 23, mant = mag & 0x7fffffu;
    value = ((exp > 120 ? exp - 120 : 0) << 3) + uint32_t(sycl::rint(float(mant) * 0x1p-20f));
  }
  return uint8_t(sign | sycl::min(value, 126u));
}
template <int SGs, int GRF, bool Debug>
class Kernel;
template <int SGs, int GRF, bool Debug>
void run(
    sycl::queue& stream,
    const uint16_t* x,
    const uint16_t* z,
    const uint16_t* w,
    uint8_t* q,
    float* s,
    uint16_t* y,
    float* rstd,
    int rows,
    float eps,
    float limit) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<32>, intel::grf_size<GRF>};
  stream.submit([&](sycl::handler& h) {
    h.parallel_for<Kernel<SGs, GRF, Debug>>(
        sycl::nd_range<1>((rows / 2) * 32, SGs * 32), props, [=](sycl::nd_item<1> it) {
          unsigned row = it.get_global_linear_id() / 16, lane = it.get_global_linear_id() % 16;
          sycl::vec<uint16_t, 8> rawx, rawz, raww;
          rawx.load(0, x + row * 128 + lane * 8);
          rawz.load(0, z + row * 128 + lane * 8);
          raww.load(0, w + lane * 8);
          float a[8], squares[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            a[i] = bf16(rawx[i]);
            squares[i] = a[i] * a[i];
          }
          // Local adjacent tree followed by clustered16. Do not reassociate.
          float p0 = squares[0] + squares[1], p1 = squares[2] + squares[3], p2 = squares[4] + squares[5],
                p3 = squares[6] + squares[7];
          float reduced = sum16((p0 + p1) + (p2 + p3));
          float invstd = sycl::rsqrt(reduced * 0x1p-7f + eps);
          float values[8], maximum = 0.f;
          sycl::vec<uint16_t, 8> norm;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float gate = bf16(rawz[i]);
            float e = sycl::exp2(gate * sycl::bit_cast<float>(0xbfb8aa3bu));
            float sigmoid = 1.f / (1.f + e);
            float value = ((a[i] * invstd) * bf16(raww[i])) * (gate * sigmoid);
            norm[i] = rounded(value);
            values[i] = bf16(norm[i]);
            maximum = sycl::fmax(maximum, sycl::fabs(values[i]));
          }
          float scale = sycl::fmax(max16(maximum), 1e-10f) / limit, inv = 1.f / scale;
          sycl::vec<uint8_t, 8> quant;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            quant[i] = encode(sycl::fmin(sycl::fmax(values[i] * inv, -limit), limit));
          quant.store(0, q + row * 128 + lane * 8);
          if (lane == 0) s[row] = scale;
          if constexpr (Debug) {
            norm.store(0, y + row * 128 + lane * 8);
            if (lane == 0) rstd[row] = invstd;
          }
        });
  });
}
}  // namespace dflash_gdn_norm_quant_impl
extern "C" int dflash_gdn_norm_quant_abi() {
  return 1;
}
extern "C" const char* dflash_gdn_norm_quant_error() {
  return dflash_gdn_norm_quant_impl::error.c_str();
}
extern "C" int dflash_gdn_norm_quant_run(
    void* queue,
    const uint16_t* x,
    const uint16_t* z,
    const uint16_t* w,
    uint8_t* q,
    float* s,
    uint16_t* y,
    float* rstd,
    int rows,
    float eps,
    int debug,
    float limit) {
  try {
    if (!queue || !x || !z || !w || !q || !s || (rows != 128 && rows != 256) || eps <= 0 || limit != 448.f ||
        (debug != 0 && debug != 1) || (debug && (!y || !rstd)))
      return 2;
    auto& stream = *static_cast<sycl::queue*>(queue);
    if (debug)
      dflash_gdn_norm_quant_impl::run<2, 256, true>(stream, x, z, w, q, s, y, rstd, rows, eps, limit);
    else
      dflash_gdn_norm_quant_impl::run<2, 256, false>(stream, x, z, w, q, s, y, rstd, rows, eps, limit);
    return 0;
  } catch (const std::exception& e) {
    dflash_gdn_norm_quant_impl::error = e.what();
    return 1;
  }
}
