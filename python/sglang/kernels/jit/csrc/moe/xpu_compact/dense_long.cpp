/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// M4/N2048/K4096 live-FP8 projection. FP16 scale/operand rounding follows
// dense_sequential.cpp. Eight independent FP32 K partials change reduction order.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
using F4 = float __attribute__((ext_vector_type(4)));
using S4 = short __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL __attribute__((const, convergent)) F4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int, S4, I8, F4, int);
namespace native_long_dense {
thread_local std::string error;
class Prepare;
class Gemm;
class Reduce;
void prepare(sycl::queue& queue, const uint8_t* q, const float* xs, sycl::half* a) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Prepare>(sycl::range<1>(128), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      namespace es = sycl::ext::intel::esimd;
      unsigned begin = id[0] * 32;
      es::simd<sycl::half, 128> dst;
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        float scale = float(sycl::half(xs[row * 32 + begin / 128])) * 256.f;
        auto raw = es::block_load<uint8_t, 32>(q + row * 4096 + begin);
        auto u = es::convert<uint16_t>(raw);
        auto mag = u & 127;
        auto sign = (u & 128) << 8;
        es::simd<uint16_t, 32> bits = (mag << 7) | sign;
        bits.merge(sign | 0x7e00, mag == 127);
        es::simd<sycl::half, 32> small = bits.template bit_cast_view<sycl::half>();
        dst.template select<32, 4>(row) = es::convert<sycl::half>(es::convert<float>(small) * scale);
      }
      es::block_store<sycl::half, 128>(a + begin * 4, dst);
    });
  });
}
void gemm(
    sycl::queue& queue, const sycl::half* a, const uint32_t* packed, const float* ws, float* partial, int s0, int s1) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<16>, intel::grf_size<128>};
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Gemm>(
        sycl::nd_range<1>(sycl::range<1>(16384), sycl::range<1>(64)), props, [=](sycl::nd_item<1> item) {
          unsigned global = item.get_global_linear_id(), lane = global & 15, task = global >> 4;
          unsigned part = task / 128, column = (task % 128) * 16 + lane;
          F4 acc{};
#pragma unroll 1
          for (int group = part * 4; group < (part + 1) * 4; ++group) {
            float sb = float(sycl::half(ws[(column / 128) * s0 + group * s1])) * 256.f;
#pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
              unsigned begin = group * 128 + chunk * 16;
              sycl::vec<short, 4> av;
              av.load(0, reinterpret_cast<const short*>(a) + (begin + lane) * 4);
              S4 ax = sycl::bit_cast<S4>(av);
              unsigned index = (column >> 4) * 16384 + (begin >> 5) * 128 + ((begin & 31) >> 2) * 16 + (column & 15);
              sycl::vec<uint8_t, 16> raw;
#pragma unroll
              for (int word = 0; word < 4; ++word) {
                uint32_t value = packed[index + word * 16];
#pragma unroll
                for (int byte = 0; byte < 4; ++byte)
                  raw[word * 4 + byte] = uint8_t(value >> (byte * 8));
              }
              auto u = raw.template convert<uint16_t>();
              auto mag = u & uint16_t(127);
              auto sign = (u & uint16_t(128)) << uint16_t(8);
              auto bits = (mag << uint16_t(7)) | sign;
              bits = sycl::select(bits, sign | uint16_t(0x7e00), mag == uint16_t(127));
              auto small = bits.template as<sycl::vec<sycl::half, 16>>().template convert<float>();
              auto rounded = (small * sb).template convert<sycl::half, sycl::rounding_mode::rte>();
              auto bw = rounded.template as<sycl::vec<int, 8>>();
              I8 bx{};
#pragma unroll
              for (int word = 0; word < 8; ++word)
                bx[word] = bw[word];
#ifdef __SYCL_DEVICE_ONLY__
              acc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, ax, bx, acc, 3072);
#endif
            }
          }
#pragma unroll
          for (int row = 0; row < 4; ++row)
            partial[(part * 4 + row) * 2048 + column] = acc[row];
        });
  });
}
void reduce(sycl::queue& queue, const float* partial, uint16_t* out) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Reduce>(sycl::range<1>(8192), [=](sycl::id<1> id) {
      unsigned i = id[0];
      float value = 0.f;
#pragma unroll
      for (int part = 0; part < 8; ++part)
        value += partial[part * 8192 + i];
      out[i] = sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(value));
    });
  });
}
}  // namespace native_long_dense
extern "C" int dflash_long_dense_abi() {
  return 1;
}
extern "C" const char* dflash_long_dense_error() {
  return native_long_dense::error.c_str();
}
extern "C" int dflash_long_dense_gemm(
    void* q,
    const uint8_t* x,
    const uint32_t* packed,
    const float* xs,
    const float* ws,
    void* prepared,
    uint16_t* out,
    float* partial,
    int s0,
    int s1) {
  try {
    auto& queue = *static_cast<sycl::queue*>(q);
    auto* a = static_cast<sycl::half*>(prepared);
    native_long_dense::prepare(queue, x, xs, a);
    native_long_dense::gemm(queue, a, packed, ws, partial, s0, s1);
    native_long_dense::reduce(queue, partial, out);
    return 0;
  } catch (const std::exception& e) {
    native_long_dense::error = e.what();
    return 1;
  }
}
