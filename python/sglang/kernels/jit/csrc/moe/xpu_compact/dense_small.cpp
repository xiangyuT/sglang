/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off) reassociate(off)

namespace native_small_dense {
namespace es = sycl::ext::intel::esimd;
thread_local std::string error;
template <int VL>
ESIMD_INLINE es::simd<sycl::half, VL> input_operand(es::simd<uint8_t, VL> raw, float scale) {
  auto mag = es::convert<uint16_t>(raw) & 127;
  auto sign = (es::convert<uint16_t>(raw) & 128) << 8;
  es::simd<uint16_t, VL> bits = (mag << 7) | sign;
  bits.merge(sign | 0x7e00, mag == 127);
  es::simd<sycl::half, VL> small = bits.template bit_cast_view<sycl::half>();
  return es::convert<sycl::half>(es::convert<float>(small) * scale);
}
inline uint16_t bf16(float x) {
  unsigned bits = sycl::bit_cast<unsigned>(x);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
class Prepare;
template <int VL>
class Gemv;
void prepare(sycl::queue& queue, const uint8_t* x, const float* xs, sycl::half* dst) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Prepare>(sycl::range<1>(16), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      unsigned block = id[0];
      float scale = float(sycl::half(xs[block])) * 256.f;
      auto raw = es::block_load<uint8_t, 128>(x + block * 128);
      es::block_store<sycl::half, 128>(dst + block * 128, input_operand<128>(raw, scale));
    });
  });
}
template <int VL>
void launch(sycl::queue& queue, const sycl::half* a, const uint8_t* w, const float* ws, uint16_t* out, int s0, int s1) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Gemv<VL>>(sycl::range<1>(2048), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      unsigned row = id[0];
      es::simd<float, VL> accum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t)
        accum[t] = 0.f;
#pragma unroll
      for (int begin = 0; begin < 512; begin += VL) {
        auto raw = es::block_load<uint8_t, VL>(w + row * 512 + begin);
        es::simd<float, VL> weight;
        if constexpr (VL <= 128) {
          float scale = float(sycl::half(ws[(row / 128) * s0 + (begin / 128) * s1])) * 256.f;
          weight = es::convert<float>(input_operand<VL>(raw, scale));
        } else {
#pragma unroll
          for (int group = 0; group < VL / 128; ++group) {
            float scale = float(sycl::half(ws[(row / 128) * s0 + (begin / 128 + group) * s1])) * 256.f;
            es::simd<uint8_t, 128> fragment = raw.template select<128, 1>(group * 128);
            weight.template select<128, 1>(group * 128) = es::convert<float>(input_operand<128>(fragment, scale));
          }
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          auto input = es::convert<float>(es::block_load<sycl::half, VL>(a + t * 512 + begin));
          accum[t] += weight * input;
        }
      }
#pragma unroll
      for (int t = 0; t < 4; ++t)
        out[t * 2048 + row] = bf16(es::reduce<float>(accum[t], std::plus<>()));
    });
  });
}
}  // namespace native_small_dense
extern "C" int dflash_small_dense_abi() {
  return 1;
}
extern "C" const char* dflash_small_dense_error() {
  return native_small_dense::error.c_str();
}
extern "C" int dflash_small_dense_gemm(
    void* q,
    const uint8_t* x,
    const float* xs,
    const uint8_t* w,
    const float* ws,
    uint16_t* prepared,
    uint16_t* y,
    int s0,
    int s1) {
  try {
    auto& queue = *static_cast<sycl::queue*>(q);
    auto* a = reinterpret_cast<sycl::half*>(prepared);
    native_small_dense::prepare(queue, x, xs, a);
    native_small_dense::launch<128>(queue, a, w, ws, y, s0, s1);
    return 0;
  } catch (const std::exception& e) {
    native_small_dense::error = e.what();
    return 1;
  }
}
