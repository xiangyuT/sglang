#define DELTA_RECORD 1
// SPDX-License-Identifier: Apache-2.0
// Qwen BF16 GDN verify and accepted-prefix SSM commit for single-request B70.
// Recurrence follows SGLang/FLA's fused sigmoid-gated delta rule. Explicit
// FP32 rounding, reduction order, BF16 gates and OpenCL math are intentional.
// Compile this unit with -fp-model=precise -ffp-contract=off, independently
// of the expert/DPAS library. Verify saves delta; commit replays the same FMA.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int, int, float) noexcept;
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#ifndef GGUF_GDN_TP
#define GGUF_GDN_TP 1
#endif
#if GGUF_GDN_TP == 2
#define sglang_nextn_gdn48 sglang_nextn_gdn24
#define nextn_gdn48_abi nextn_gdn24_abi
#define nextn_gdn48_supports_steps nextn_gdn24_supports_steps
#define nextn_gdn48_error nextn_gdn24_error
#define nextn_gdn48_verify nextn_gdn24_verify
#define nextn_gdn48_verify_batch nextn_gdn24_verify_batch
#define nextn_gdn48_commit nextn_gdn24_commit
#elif GGUF_GDN_TP == 4
#define sglang_nextn_gdn48 sglang_nextn_gdn12
#define nextn_gdn48_abi nextn_gdn12_abi
#define nextn_gdn48_supports_steps nextn_gdn12_supports_steps
#define nextn_gdn48_error nextn_gdn12_error
#define nextn_gdn48_verify nextn_gdn12_verify
#define nextn_gdn48_verify_batch nextn_gdn12_verify_batch
#define nextn_gdn48_commit nextn_gdn12_commit
#elif GGUF_GDN_TP != 1
#error Unsupported GGUF GDN tensor parallel width
#endif
namespace sglang_nextn_gdn48 {
namespace es = sycl::ext::intel::esimd;
thread_local std::string error;
struct Args {
  const uint16_t *q, *k, *v, *a, *b;
  const float* alog;
  const uint16_t* bias;
  const float* state;
  const int32_t* index;
  float* cache;
  const int32_t* cache_index;
  uint16_t* out;
  float* prepared;
  int steps;
  int64_t state_pitch;
  int64_t cache_pitch;
  int a_stride, b_stride;
  int32_t* saved_index;
  float rounding_zero;
  int batch = 1;
};
template <int N>
ESIMD_INLINE es::simd<float, N> bf16(const uint16_t* ptr) {
  auto v = es::block_load<uint16_t, N>(ptr);
  es::simd<uint32_t, N> bits = es::convert<uint32_t>(v) << 16;
  return bits.template bit_cast_view<float>();
}
ESIMD_INLINE float bf16_scalar(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
ESIMD_INLINE uint16_t rounded_bf16(float x) {
  unsigned bits = sycl::bit_cast<unsigned>(x);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
ESIMD_INLINE float sum128(es::simd<float, 128> x) {
  es::simd<float, 32> a = x.template select<32, 4>(0).read() + x.template select<32, 4>(1).read();
  es::simd<float, 32> b = x.template select<32, 4>(2).read() + x.template select<32, 4>(3).read();
  es::simd<float, 32> c = a + b;
  es::simd<float, 16> r16 = c.template select<16, 1>(0).read() + c.template select<16, 1>(16).read();
  es::simd<float, 8> r8 = r16.template select<8, 1>(0).read() + r16.template select<8, 1>(8).read();
  es::simd<float, 4> r4 = r8.template select<4, 1>(0).read() + r8.template select<4, 1>(4).read();
  es::simd<float, 2> r2 = r4.template select<2, 1>(0).read() + r4.template select<2, 1>(2).read();
  return r2[0] + r2[1];
}
ESIMD_INLINE float dot128(es::simd<float, 128> x, es::simd<float, 128> y) {
  namespace ex = sycl::ext::intel::experimental::esimd;
  es::simd<float, 32> x0 = x.template select<32, 4>(0).read(), x1 = x.template select<32, 4>(1).read(),
                      x2 = x.template select<32, 4>(2).read(), x3 = x.template select<32, 4>(3).read();
  es::simd<float, 32> y0 = y.template select<32, 4>(0).read(), y1 = y.template select<32, 4>(1).read(),
                      y2 = y.template select<32, 4>(2).read(), y3 = y.template select<32, 4>(3).read();
  es::simd<float, 32> c = ex::fma<float, 32>(x0, y0, x1 * y1) + ex::fma<float, 32>(x2, y2, x3 * y3);
  es::simd<float, 16> r16 = c.template select<16, 1>(0).read() + c.template select<16, 1>(16).read();
  es::simd<float, 8> r8 = r16.template select<8, 1>(0).read() + r16.template select<8, 1>(8).read();
  es::simd<float, 4> r4 = r8.template select<4, 1>(0).read() + r8.template select<4, 1>(4).read();
  es::simd<float, 2> r2 = r4.template select<2, 1>(0).read() + r4.template select<2, 1>(2).read();
  return r2[0] + r2[1];
}
ESIMD_INLINE es::simd<float, 128> rounded_product(es::simd<float, 128> x, es::simd<float, 128> y, float zero) {
  es::simd<uint32_t, 128> xb = x.template bit_cast_view<uint32_t>(), yb = y.template bit_cast_view<uint32_t>();
  es::simd<uint32_t, 128> bits = ((xb ^ yb) & 0x80000000u) | sycl::bit_cast<uint32_t>(zero);
  es::simd<float, 128> signed_zero = bits.template bit_cast_view<float>();
  return sycl::ext::intel::experimental::esimd::fma<float, 128>(x, y, signed_zero);
}
inline float scalar_bf16(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
inline uint16_t scalar_rounded_bf16(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline float regular_exp(float x) {
  return sycl::native::exp2(x * 1.4426950408889634f);
}
inline float regular_reduce(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFAdd(3, 0, x);
#else
  return x;
#endif
}
template <bool Ieee>
class Prepare;
template <bool Ieee>
class PrepareBatch;
template <bool Ieee, bool Batched = false>
void prepare(sycl::queue& q, Args base) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<std::conditional_t<Batched, PrepareBatch<Ieee>, Prepare<Ieee>>>(
        sycl::nd_range<1>(sycl::range<1>((Batched ? base.batch : 1) * base.steps * (16 / GGUF_GDN_TP) * 32), sycl::range<1>(32)), props, [=](sycl::nd_item<1> item) {
          Args a = base;
          unsigned group = item.get_group_linear_id();
          if constexpr (Batched) {
            const unsigned request = group / (base.steps * (16 / GGUF_GDN_TP));
            a.q += int64_t(request) * a.steps * (10240 / GGUF_GDN_TP);
            a.k += int64_t(request) * a.steps * (10240 / GGUF_GDN_TP);
            a.v += int64_t(request) * a.steps * (10240 / GGUF_GDN_TP);
            a.a += int64_t(request) * a.steps * a.a_stride;
            a.b += int64_t(request) * a.steps * a.b_stride;
            a.index += request;
            a.saved_index += request;
            a.prepared += int64_t(request) * (82688 / GGUF_GDN_TP);
            group %= base.steps * (16 / GGUF_GDN_TP);
          }
          unsigned t = group / (16 / GGUF_GDN_TP), kh = group % (16 / GGUF_GDN_TP),
                   lane = item.get_local_linear_id(), kk = lane * 4;
          if (t == 0 && kh == 0 && lane == 0 && a.saved_index) *a.saved_index = *a.index;
          float query[4], key[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            query[i] = scalar_bf16(a.q[t * (10240 / GGUF_GDN_TP) + kh * 128 + kk + i]);
            key[i] = scalar_bf16(a.k[t * (10240 / GGUF_GDN_TP) + kh * 128 + kk + i]);
          }
          float qs =
              regular_reduce((query[0] * query[0] + query[1] * query[1]) + (query[2] * query[2] + query[3] * query[3]));
          float ks = regular_reduce((key[0] * key[0] + key[1] * key[1]) + (key[2] * key[2] + key[3] * key[3]));
          float qr = 1.f / sycl::sqrt(qs + 1e-6f), kr = 1.f / sycl::sqrt(ks + 1e-6f);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            a.prepared[t * (4096 / GGUF_GDN_TP) + kh * 256 + kk + i] = (query[i] * qr) * 0.08838834764831845f;
            a.prepared[t * (4096 / GGUF_GDN_TP) + kh * 256 + 128 + kk + i] = key[i] * kr;
          }
#pragma unroll
          for (int member = 0; member < 3; ++member) {
            unsigned hv = kh * 3 + member;
#pragma unroll
            for (int i = 0; i < 4; ++i)
              a.prepared[(33536 / GGUF_GDN_TP) + t * (6144 / GGUF_GDN_TP) + hv * 128 + kk + i] = scalar_bf16(a.v[t * (10240 / GGUF_GDN_TP) + hv * 128 + kk + i]);
            if (lane == 0) {
              float x =
                  scalar_bf16(scalar_rounded_bf16(scalar_bf16(a.a[t * a.a_stride + hv]) + scalar_bf16(a.bias[hv])));
              float soft = x <= 20.f ? sycl::log(1.f + regular_exp(x)) : x;
              float g = -regular_exp(a.alog[hv]) * soft;
              a.prepared[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2] = regular_exp(g);
              a.prepared[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2 + 1] =
                  1.f / (1.f + regular_exp(-scalar_bf16(a.b[t * a.b_stride + hv])));
            }
          }
        });
  });
}

template <int BV, bool Ieee, bool Checkpoint>
class Verify;
template <int BV, bool Ieee, bool Checkpoint>
class VerifyBatch;
template <int BV, bool Ieee, bool Checkpoint, bool Batched = false>
void launch(sycl::queue& q, Args base, bool do_prepare = true) {
  if (do_prepare) prepare<Ieee, Batched>(q, base);
  q.submit([&](sycl::handler& h) {
    h.parallel_for<std::conditional_t<Batched, VerifyBatch<BV, Ieee, Checkpoint>, Verify<BV, Ieee, Checkpoint>>>(
        sycl::range<1>((Batched ? base.batch : 1) * (48 / GGUF_GDN_TP) * (128 / BV)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
          Args a = base;
          unsigned local_id = id[0];
          if constexpr (Batched) {
            const unsigned request = local_id / ((48 / GGUF_GDN_TP) * (128 / BV));
            a.index += request;
            a.prepared += int64_t(request) * (82688 / GGUF_GDN_TP);
            a.out += int64_t(request) * a.steps * (6144 / GGUF_GDN_TP);
            local_id %= (48 / GGUF_GDN_TP) * (128 / BV);
          }
          unsigned hv = local_id / (128 / BV), vstart = (local_id % (128 / BV)) * BV;
          int idx = *a.index, cidx = -1;
          if constexpr (Checkpoint) cidx = *a.cache_index;
          es::simd<float, BV * 128> state = 0.f;
          if (idx >= 0)
            state = es::block_load<float, BV * 128>(a.state + int64_t(idx) * a.state_pitch + hv * 16384 + vstart * 128);
          state += 0.f;
#pragma unroll 1
          for (int t = 0; t < a.steps; ++t) {
            auto query = es::block_load<float, 128>(a.prepared + t * (4096 / GGUF_GDN_TP) + (hv / 3) * 256);
            auto key = es::block_load<float, 128>(a.prepared + t * (4096 / GGUF_GDN_TP) + (hv / 3) * 256 + 128);
            float beta = a.prepared[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2 + 1];
            float gate = a.prepared[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2];
#pragma unroll
            for (int v = 0; v < BV; ++v) {
              es::simd<float, 128> row = state.template select<128, 1>(v * 128).read();
              es::simd<float, 128> oldrow = row;
              row = rounded_product(oldrow, es::simd<float, 128>(gate), a.rounding_zero);
              float dot = dot128(row, key);
              float delta = (a.prepared[(33536 / GGUF_GDN_TP) + t * (6144 / GGUF_GDN_TP) + hv * 128 + vstart + v] - dot) * beta;
              if constexpr (DELTA_RECORD) a.prepared[(33536 / GGUF_GDN_TP) + t * (6144 / GGUF_GDN_TP) + hv * 128 + vstart + v] = delta;
              row = sycl::ext::intel::experimental::esimd::fma<float, 128>(
                  oldrow,
                  es::simd<float, 128>(gate),
                  rounded_product(key, es::simd<float, 128>(delta), a.rounding_zero));
              state.template select<128, 1>(v * 128) = row;
              a.out[t * (6144 / GGUF_GDN_TP) + hv * 128 + vstart + v] = rounded_bf16(dot128(row, query));
            }
            if constexpr (Checkpoint) {
              if (cidx >= 0)
                es::block_store<float, BV * 128>(
                    a.cache + int64_t(cidx) * a.cache_pitch + t * (786432 / GGUF_GDN_TP) + hv * 16384 + vstart * 128, state);
            }
          }
        });
  });
}

// Fold reads only the persistent prepared records. Inputs may be reused after verify.
template <int BV>
class Fold;
template <int BV>
void fold(
    sycl::queue& q,
    const float* source,
    const int32_t* source_index,
    int64_t source_pitch,
    const float* records,
    float* target,
    const int32_t* target_index,
    int64_t target_pitch,
    const int32_t* last_correct,
    int steps,
    int layers = 1,
    int64_t source_layer_pitch = 0,
    int64_t record_layer_pitch = 0,
    int64_t target_layer_pitch = 0,
    int source_index_stride = 0,
    float rounding_zero = 0.f) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Fold<BV>>(sycl::range<1>(layers * (48 / GGUF_GDN_TP) * (128 / BV)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      int count = *last_correct + 1, dst = *target_index;
      if (count <= 0 || count > steps || dst < 0) return;
      unsigned layer = id[0] / ((48 / GGUF_GDN_TP) * (128 / BV)), local = id[0] % ((48 / GGUF_GDN_TP) * (128 / BV));
      unsigned hv = local / (128 / BV), vstart = (local % (128 / BV)) * BV;
      const float* layer_records = records + int64_t(layer) * record_layer_pitch;
      int src = source_index[int64_t(layer) * source_index_stride];
      es::simd<float, BV * 128> state = 0.f;
      if (src >= 0)
        state = es::block_load<float, BV * 128>(
            source + int64_t(layer) * source_layer_pitch + int64_t(src) * source_pitch + hv * 16384 + vstart * 128);
      state += 0.f;
#pragma unroll 1
      for (int t = 0; t < count; ++t) {
        auto key = es::block_load<float, 128>(layer_records + t * (4096 / GGUF_GDN_TP) + (hv / 3) * 256 + 128);
        float beta = layer_records[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2 + 1];
        float gate = layer_records[(32768 / GGUF_GDN_TP) + t * (96 / GGUF_GDN_TP) + hv * 2];
#pragma unroll
        for (int v = 0; v < BV; ++v) {
          es::simd<float, 128> row = state.template select<128, 1>(v * 128).read();
          es::simd<float, 128> oldrow = row;
          float delta;
          if constexpr (DELTA_RECORD) {
            delta = layer_records[(33536 / GGUF_GDN_TP) + t * (6144 / GGUF_GDN_TP) + hv * 128 + vstart + v];
          } else {
            row = rounded_product(oldrow, es::simd<float, 128>(gate), rounding_zero);
            float dot = dot128(row, key);
            delta = (layer_records[(33536 / GGUF_GDN_TP) + t * (6144 / GGUF_GDN_TP) + hv * 128 + vstart + v] - dot) * beta;
          }
          row = sycl::ext::intel::experimental::esimd::fma<float, 128>(
              oldrow, es::simd<float, 128>(gate), rounded_product(key, es::simd<float, 128>(delta), rounding_zero));
          state.template select<128, 1>(v * 128) = row;
        }
      }
      es::block_store<float, BV * 128>(
          target + int64_t(layer) * target_layer_pitch + int64_t(dst) * target_pitch + hv * 16384 + vstart * 128,
          state);
    });
  });
}
}  // namespace sglang_nextn_gdn48
extern "C" int nextn_gdn48_abi() {
  return 1;
}
extern "C" int nextn_gdn48_supports_steps(int steps) {
  return steps >= 1 && steps <= 4;
}
extern "C" const char* nextn_gdn48_error() {
  return sglang_nextn_gdn48::error.c_str();
}
extern "C" int nextn_gdn48_verify(
    void* queue,
    const void* query,
    const void* key,
    const void* value,
    const void* a,
    const void* b,
    const float* alog,
    const void* bias,
    const float* state,
    const int32_t* index,
    float* cache,
    const int32_t* cidx,
    void* out,
    float* records,
    int32_t* saved_index,
    int steps,
    int mode,
    int checkpoints,
    int64_t state_pitch,
    int64_t cache_pitch,
    int a_stride,
    int b_stride) {
  try {
    if (!queue || !query || !key || !value || !a || !b || !alog || !bias || !state || !index || !out || !records ||
        !nextn_gdn48_supports_steps(steps) || mode < 0 || mode > 2 || !saved_index || (a_stride != (48 / GGUF_GDN_TP) && a_stride != (96 / GGUF_GDN_TP)) ||
        (b_stride != (48 / GGUF_GDN_TP) && b_stride != (96 / GGUF_GDN_TP)) || state_pitch < (786432 / GGUF_GDN_TP) || checkpoints < 0 || checkpoints > 1 ||
        (checkpoints && (!cache || !cidx || cache_pitch < int64_t(steps) * (786432 / GGUF_GDN_TP))))
      throw std::runtime_error("Invalid raw-window SYCL GDN contract");
    sglang_nextn_gdn48::Args args{
        static_cast<const uint16_t*>(query),
        static_cast<const uint16_t*>(key),
        static_cast<const uint16_t*>(value),
        static_cast<const uint16_t*>(a),
        static_cast<const uint16_t*>(b),
        alog,
        static_cast<const uint16_t*>(bias),
        state,
        index,
        cache,
        cidx,
        static_cast<uint16_t*>(out),
        records,
        steps,
        state_pitch,
        cache_pitch,
        a_stride,
        b_stride,
        saved_index,
        0.f};
    auto& q = *static_cast<sycl::queue*>(queue);
    if (checkpoints) {
      if (mode == 0)
        sglang_nextn_gdn48::launch<1, true, true>(q, args);
      else if (mode == 1)
        sglang_nextn_gdn48::launch<2, true, true>(q, args);
      else
        sglang_nextn_gdn48::launch<4, true, true>(q, args);
    } else {
      if (mode == 0)
        sglang_nextn_gdn48::launch<1, true, false>(q, args);
      else if (mode == 1)
        sglang_nextn_gdn48::launch<2, true, false>(q, args);
      else
        sglang_nextn_gdn48::launch<4, true, false>(q, args);
    }
    return 0;
  } catch (const std::exception& e) {
    sglang_nextn_gdn48::error = e.what();
    return 1;
  }
}
// Preserve the single-request ABI and its specialization for BS1.
#if GGUF_GDN_TP == 2
extern "C" int nextn_gdn48_verify_batch(
    void* queue,
    const void* query,
    const void* key,
    const void* value,
    const void* a,
    const void* b,
    const float* alog,
    const void* bias,
    const float* state,
    const int32_t* index,
    float* cache,
    const int32_t* cidx,
    void* out,
    float* records,
    int32_t* saved_index,
    int steps,
    int mode,
    int checkpoints,
    int64_t state_pitch,
    int64_t cache_pitch,
    int a_stride,
    int b_stride,
    int batch) {
  try {
    if (batch < 2 || batch > 8 || steps != 4 || mode != 2 || checkpoints != 0)
      throw std::runtime_error("Batched GGUF GDN requires TP2, BS2..8, four steps, BV4, delta records");
    if (!queue || !query || !key || !value || !a || !b || !alog || !bias || !state || !index || !out || !records ||
        !nextn_gdn48_supports_steps(steps) || mode < 0 || mode > 2 || !saved_index || (a_stride != (48 / GGUF_GDN_TP) && a_stride != (96 / GGUF_GDN_TP)) ||
        (b_stride != (48 / GGUF_GDN_TP) && b_stride != (96 / GGUF_GDN_TP)) || state_pitch < (786432 / GGUF_GDN_TP) || checkpoints < 0 || checkpoints > 1 ||
        (checkpoints && (!cache || !cidx || cache_pitch < int64_t(steps) * (786432 / GGUF_GDN_TP))))
      throw std::runtime_error("Invalid raw-window SYCL GDN contract");
    sglang_nextn_gdn48::Args args{
        static_cast<const uint16_t*>(query),
        static_cast<const uint16_t*>(key),
        static_cast<const uint16_t*>(value),
        static_cast<const uint16_t*>(a),
        static_cast<const uint16_t*>(b),
        alog,
        static_cast<const uint16_t*>(bias),
        state,
        index,
        cache,
        cidx,
        static_cast<uint16_t*>(out),
        records,
        steps,
        state_pitch,
        cache_pitch,
        a_stride,
        b_stride,
        saved_index,
        0.f};
    auto& q = *static_cast<sycl::queue*>(queue);
    args.batch = batch;
    sglang_nextn_gdn48::launch<4, true, false, true>(q, args);
    return 0;
  } catch (const std::exception& e) {
    sglang_nextn_gdn48::error = e.what();
    return 1;
  }
}
#endif
extern "C" int nextn_gdn48_commit(
    void* queue,
    const float* source,
    const int32_t* source_index,
    const float* records,
    float* target,
    const int32_t* target_index,
    const int32_t* last_correct,
    int steps,
    int mode,
    int layers,
    int source_index_stride,
    int64_t source_pitch,
    int64_t target_pitch,
    int64_t source_layer_pitch,
    int64_t record_layer_pitch,
    int64_t target_layer_pitch) {
  try {
    if (!queue || !source || !source_index || !records || !target || !target_index || !last_correct ||
        !nextn_gdn48_supports_steps(steps) ||
        mode < 0 || mode > 2 || source_index_stride < 0 || source_index_stride > 8 || layers < 1 || layers > 48 ||
        source_pitch < (786432 / GGUF_GDN_TP) || target_pitch < (786432 / GGUF_GDN_TP) ||
        (layers > 1 && (source_layer_pitch <= 0 || target_layer_pitch <= 0 || record_layer_pitch < (82688 / GGUF_GDN_TP))))
      throw std::runtime_error("Invalid fused accepted-prefix SYCL GDN contract");
    auto& q = *static_cast<sycl::queue*>(queue);
    if (mode == 0)
      sglang_nextn_gdn48::fold<1>(
          q,
          source,
          source_index,
          source_pitch,
          records,
          target,
          target_index,
          target_pitch,
          last_correct,
          steps,
          layers,
          source_layer_pitch,
          record_layer_pitch,
          target_layer_pitch,
          source_index_stride);
    else if (mode == 1)
      sglang_nextn_gdn48::fold<2>(
          q,
          source,
          source_index,
          source_pitch,
          records,
          target,
          target_index,
          target_pitch,
          last_correct,
          steps,
          layers,
          source_layer_pitch,
          record_layer_pitch,
          target_layer_pitch,
          source_index_stride);
    else
      sglang_nextn_gdn48::fold<4>(
          q,
          source,
          source_index,
          source_pitch,
          records,
          target,
          target_index,
          target_pitch,
          last_correct,
          steps,
          layers,
          source_layer_pitch,
          record_layer_pitch,
          target_layer_pitch,
          source_index_stride);
    return 0;
  } catch (const std::exception& e) {
    sglang_nextn_gdn48::error = e.what();
    return 1;
  }
}
