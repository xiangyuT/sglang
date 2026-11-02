/* Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Sequential-K SYCL projections retain the FP8 byte-preserving layout and
// FP16 operand recipe. A is prepared once in lane-major order; B uses vector
// decoding. No split reduction or alternate DPAS accumulation order is used.
#include "dflash_window.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

using F4 = float __attribute__((ext_vector_type(DFLASH_NATIVE_WINDOW)));
using S4 = short __attribute__((ext_vector_type(DFLASH_NATIVE_WINDOW)));
using I8 = int __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL __attribute__((const, convergent)) F4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int, S4, I8, F4, int);

namespace native_seq08 {
thread_local std::string error;
struct Arguments {
  const uint8_t* x;
  const uint16_t* prepared;
  const uint32_t* packed;
  const float *xs, *ws;
  uint16_t* out;
  int n, k, s0, s1;
};
struct Frame {
  S4 a;
  I8 b;
};

inline uint16_t operand(uint8_t raw, float scale) {
  uint16_t magnitude = raw & 127;
  uint16_t sign = uint16_t(raw & 128) << 8;
  uint16_t bits = (magnitude << 7) | sign;
  if (magnitude == 127) bits = sign | 0x7e00;
  auto value = sycl::bit_cast<sycl::half>(bits);
  return sycl::bit_cast<uint16_t>(sycl::half(float(value) * scale));
}

using U4 = unsigned __attribute__((ext_vector_type(4)));
template <int K>
inline U4 read_weights(Arguments a, unsigned column, unsigned begin) {
  unsigned index = (column >> 4) * unsigned(K * 4) + (begin >> 5) * 128 + ((begin & 31) >> 2) * 16 + (column & 15);
  U4 words{};
#pragma unroll
  for (int word = 0; word < 4; ++word)
    words[word] = a.packed[index + word * 16];
  return words;
}
template <int K, bool Prepared, bool Vector>
inline Frame convert(Arguments a, unsigned begin, unsigned input_lane, F4 sa, float sb, U4 words) {
  Frame frame{};
  sycl::vec<short, DFLASH_NATIVE_WINDOW> av;
  av.load(0, reinterpret_cast<const short*>(a.prepared) + (begin + input_lane) * DFLASH_NATIVE_WINDOW);
  frame.a = sycl::bit_cast<S4>(av);
  if constexpr (Vector) {
    sycl::vec<uint8_t, 16> raw;
#pragma unroll
    for (int word = 0; word < 4; ++word)
#pragma unroll
      for (int byte = 0; byte < 4; ++byte)
        raw[word * 4 + byte] = uint8_t(words[word] >> (byte * 8));
    auto u = raw.template convert<uint16_t>();
    auto mag = u & uint16_t(127);
    auto sign = (u & uint16_t(128)) << uint16_t(8);
    auto bits = (mag << uint16_t(7)) | sign;
    bits = sycl::select(bits, sign | uint16_t(0x7e00), mag == uint16_t(127));
    auto values = bits.template as<sycl::vec<sycl::half, 16>>().template convert<float>();
    auto half = (values * sb).template convert<sycl::half, sycl::rounding_mode::rte>();
    auto packed = half.template as<sycl::vec<int, 8>>();
#pragma unroll
    for (int word = 0; word < 8; ++word)
      frame.b[word] = packed[word];
  } else {
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      unsigned raw = words[word];
      unsigned h0 = operand(uint8_t(raw), sb), h1 = operand(uint8_t(raw >> 8), sb);
      unsigned h2 = operand(uint8_t(raw >> 16), sb), h3 = operand(uint8_t(raw >> 24), sb);
      frame.b[2 * word] = sycl::bit_cast<int>(h0 | (h1 << 16));
      frame.b[2 * word + 1] = sycl::bit_cast<int>(h2 | (h3 << 16));
    }
  }
  return frame;
}

inline F4 mma(Frame f, F4 acc) {
#ifdef __SYCL_DEVICE_ONLY__
  // 0x400|0x800: packed FP16 A/B. The same intrinsic/type tuple as C04.
  return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, f.a, f.b, acc, 3072);
#else
  return acc;
#endif
}

template <int K, int SG, int Stage, int GRF, int Cols, bool Prepared, bool Vector>
class Dense;
template <int VL>
class Prepare;
class Pack;
class Decode;
template <int K, int SG, int Stage, int GRF, int Cols, bool Prepared, bool Vector>
void launch(sycl::queue& queue, Arguments a) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<16>, intel::grf_size<GRF>};
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Dense<K, SG, Stage, GRF, Cols, Prepared, Vector>>(
        sycl::nd_range<1>(sycl::range<1>(size_t(a.n) * 16 / Cols), sycl::range<1>(SG * 16)),
        props,
        [=](sycl::nd_item<1> item) {
          unsigned global = item.get_global_linear_id(), input_lane = global & 15;
          unsigned column = (global >> 4) * Cols + input_lane % Cols;
          F4 acc{};
          U4 ring[Stage > 0 ? Stage : 1];
          if constexpr (Stage > 0) {
#pragma unroll
            for (int slot = 0; slot < Stage; ++slot)
              ring[slot] = read_weights<K>(a, column, slot * 16);
          }
#pragma unroll 1
          for (int group = 0; group < K / 128; ++group) {
            F4 sa{};
#pragma unroll
            for (int row = 0; row < DFLASH_NATIVE_WINDOW; ++row)
              if constexpr (!Prepared) sa[row] = float(sycl::half(a.xs[row * (K / 128) + group])) * 256.f;
            float sb = float(sycl::half(a.ws[(column / 128) * a.s0 + group * a.s1])) * 256.f;
#pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
              unsigned begin = group * 128 + chunk * 16;
              U4 words{};
              if constexpr (Stage == 0)
                words = read_weights<K>(a, column, begin);
              else {
                words = ring[chunk % Stage];
                unsigned future = begin + Stage * 16;
                if (future < K) ring[chunk % Stage] = read_weights<K>(a, column, future);
              }
              acc = mma(convert<K, Prepared, Vector>(a, begin, input_lane, sa, sb, words), acc);
            }
          }
          if (input_lane < Cols) {
#pragma unroll
            for (int row = 0; row < DFLASH_NATIVE_WINDOW; ++row) {
              sycl::ext::oneapi::bfloat16 value(acc[row]);
              a.out[row * a.n + column] = sycl::bit_cast<uint16_t>(value);
            }
          }
        });
  });
}
template <int K>
void dispatch(sycl::queue& queue, Arguments a, int mode) {
  if (mode != 0 && mode != 1) throw std::runtime_error("Unknown sequential production geometry");
  launch<K, 4, 0, 128, 16, true, true>(queue, a);
}
}  // namespace native_seq08

extern "C" int dflash_seq_dense_abi() {
  return 2;
}
extern "C" const char* dflash_seq_dense_error() {
  return native_seq08::error.c_str();
}
extern "C" int dflash_seq_dense_gemm(
    void* q,
    const void* x,
    const void* packed,
    const float* xs,
    const float* ws,
    void* prepared,
    void* y,
    int n,
    int k,
    int s0,
    int s1,
    int mode) {
  try {
    if (n <= 0 || n % 128) throw std::runtime_error("N must be positive and divisible by128");
    if (k != 128 && k != 2048) throw std::runtime_error("Unsupported matrix K");
    if (mode != 0 && mode != 1) throw std::runtime_error("Unknown sequential production geometry");
    auto& queue = *static_cast<sycl::queue*>(q);
    native_seq08::Arguments a{
        static_cast<const uint8_t*>(x),
        static_cast<const uint16_t*>(prepared),
        static_cast<const uint32_t*>(packed),
        xs,
        ws,
        static_cast<uint16_t*>(y),
        n,
        k,
        s0,
        s1};
    auto prepare = [&]<int VL>() {
      queue.submit([&](sycl::handler& h) {
        h.parallel_for<native_seq08::Prepare<VL>>(
            sycl::range<1>(size_t(k / VL)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
              namespace es = sycl::ext::intel::esimd;
              unsigned begin = id[0] * VL;
              es::simd<sycl::half, VL * DFLASH_NATIVE_WINDOW> dst;
#pragma unroll
              for (int row = 0; row < DFLASH_NATIVE_WINDOW; ++row) {
                float scale = float(sycl::half(xs[row * (k / 128) + begin / 128])) * 256.f;
                auto raw = es::block_load<uint8_t, VL>(static_cast<const uint8_t*>(x) + row * k + begin);
                auto u = es::convert<uint16_t>(raw);
                auto mag = u & 127;
                auto sign = (u & 128) << 8;
                es::simd<uint16_t, VL> bits = (mag << 7) | sign;
                bits.merge(sign | 0x7e00, mag == 127);
                es::simd<sycl::half, VL> decoded = bits.template bit_cast_view<sycl::half>();
                dst.template select<VL, DFLASH_NATIVE_WINDOW>(row) = es::convert<sycl::half>(es::convert<float>(decoded) * scale);
              }
              es::block_store<sycl::half, VL * DFLASH_NATIVE_WINDOW>(reinterpret_cast<sycl::half*>(prepared) + begin * DFLASH_NATIVE_WINDOW, dst);
            });
      });
    };
    prepare.template operator()<32>();
    switch (k) {
      case 128:
        native_seq08::dispatch<128>(queue, a, mode);
        break;
      case 2048:
        native_seq08::dispatch<2048>(queue, a, mode);
        break;
      default:
        throw std::runtime_error("Unsupported matrix K");
    }
    return 0;
  } catch (const std::exception& e) {
    native_seq08::error = e.what();
    return 1;
  }
}
extern "C" int dflash_seq_dense_pack(void* q, const void* weight, void* packed, int n, int k) {
  try {
    if (n <= 0 || k <= 0 || n % 128 || k % 128) throw std::runtime_error("Invalid pack shape");
    auto& queue = *static_cast<sycl::queue*>(q);
    auto* src = static_cast<const uint32_t*>(weight);
    auto* dst = static_cast<uint32_t*>(packed);
    queue.submit([&](sycl::handler& h) {
      h.parallel_for<native_seq08::Pack>(sycl::range<1>(size_t(n) * k / 4), [=](sycl::id<1> item) {
        int i = item[0];
        int row = (i / (k / 32 * 128)) * 16 + i % 16;
        int col = ((i / 128) % (k / 32)) * 8 + (i / 16) % 8;
        dst[i] = src[row * (k / 4) + col];
      });
    });
    return 0;
  } catch (const std::exception& e) {
    native_seq08::error = e.what();
    return 1;
  }
}
extern "C" int dflash_seq_dense_decode(void* q, const uint8_t* raw, const float* scales, uint16_t* y) {
  try {
    auto& queue = *static_cast<sycl::queue*>(q);
    queue.submit([&](sycl::handler& h) {
      h.parallel_for<native_seq08::Decode>(sycl::range<1>(65536 * 256), [=](sycl::id<1> item) {
        unsigned i = item[0];
        float scale = float(sycl::half(scales[i >> 8])) * 256.f;
        y[i] = native_seq08::operand(raw[i & 255], scale);
      });
    });
    return 0;
  } catch (const std::exception& e) {
    native_seq08::error = e.what();
    return 1;
  }
}
