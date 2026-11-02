// SPDX-License-Identifier: Apache-2.0
// B70 IQ4_XS derivative row cache, BF16 activations and exact decoded BF16 weights.
// Nonlinear codebook from the existing ggml-common.h; see GGML-LICENSE.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
namespace sglang_iq4_bf16_row {
namespace es = sycl::ext::intel::esimd;
thread_local std::string error;
static constexpr int16_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
template <int N>
SYCL_ESIMD_FUNCTION inline es::simd<uint16_t, N> rounded(es::simd<float, N> v) {
  es::simd<sycl::ext::oneapi::bfloat16, N> values = v;
  return values.template bit_cast_view<uint16_t>().read();
}
SYCL_ESIMD_FUNCTION inline es::simd<uint16_t, 256> decode(const uint8_t* codes, const float* gain) {
#pragma clang fp contract(off)
  auto bytes = es::block_load<uint8_t, 128>(codes);
  es::simd<uint16_t, 256> idx;
  idx.template select<128, 2>(0) = es::convert<uint16_t>(bytes & 15u);
  idx.template select<128, 2>(1) = es::convert<uint16_t>(bytes >> 4);
  es::simd<int16_t, 256> values;
  es::simd<uint32_t, 256> off = es::convert<uint32_t>(idx) * 2u;
  values = es::gather<int16_t, 256>(table, off);
  auto gs = es::block_load<float, 16>(gain);
  es::simd<float, 32> twice;
  twice.template select<16, 2>(0) = gs;
  twice.template select<16, 2>(1) = gs;
  es::simd<float, 256> scales;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    scales.template select<32, 1>(j * 32) = twice;
  es::simd<float, 256> w = es::convert<float>(values) * scales;
  return rounded(w);
}

template <int E>
class RowSplit;
class Reduce;
template <int E>
void launch(
    sycl::queue& q,
    const uint8_t* codes,
    const float* gain,
    const uint16_t* x,
    float* partial,
    int m,
    int n,
    int k,
    int splits) {
  const int groups = k / 32, tiles = n / 16;
  q.parallel_for<RowSplit<E>>(sycl::nd_range<1>(tiles * splits, 4), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
    const int task = it.get_global_linear_id(), part = task / tiles, tile = task % tiles, row = tile * 16;
    es::simd<float, E * 16> sum = 0.f;
    for (int kb = part * (groups / splits); kb < (part + 1) * (groups / splits); ++kb) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        auto b = sglang_iq4_bf16_row::decode(
            codes + (size_t(tile) * groups + kb) * 256 + half * 128, gain + (size_t(tile) * groups + kb) * 16);
        auto aw = es::load_2d<uint32_t, 8, E, 1, false, false>(
            reinterpret_cast<const uint32_t*>(x), k * 2 - 1, m - 1, k * 2 - 1, kb * 16 + half * 8, 0);
        auto a = aw.template bit_cast_view<uint16_t>().read();
        using P = es::xmx::dpas_argument_type;
        sum = es::xmx::dpas<8, E, float, float, uint16_t, uint16_t, P::bf16, P::bf16>(sum, b, a);
      }
    }
#pragma unroll
    for (int e = 0; e < E; ++e)
      if (e < m) {
        es::simd<float, 16> v = sum.template select<16, 1>(e * 16);
        es::block_store<float, 16>(partial + (size_t(part) * m + e) * n + row, v);
      }
  });
}
void reduce(sycl::queue& q, const float* partial, uint16_t* y, float* raw, int m, int n, int splits) {
  q.parallel_for<Reduce>(sycl::range<1>(m * n / 16), [=](sycl::id<1> it) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
    const int offset = it[0] * 16;
    es::simd<float, 16> v = 0.f;
    for (int p = 0; p < splits; ++p)
      v += es::block_load<float, 16>(partial + size_t(p) * m * n + offset);
    es::block_store<uint16_t, 16>(y + offset, sglang_iq4_bf16_row::rounded(v));
    if (raw) es::block_store<float, 16>(raw + offset, v);
  });
}
}  // namespace sglang_iq4_bf16_row
extern "C" int xpu_iq4_bf16_row(
    void* queue,
    const uint8_t* codes,
    const float* gain,
    const uint16_t* x,
    uint16_t* y,
    float* raw,
    float* partial,
    int m,
    int n,
    int k,
    int splits) {
  if (!queue || !codes || !gain || !x || !y || !partial || m != 4 || n < 64 || n % 64 || k < 32 || k % 32 ||
      splits < 1 || splits > 128 || (k / 32) % splits)
    return 2;
  try {
    auto& q = *static_cast<sycl::queue*>(queue);
    if (!q.has_property<sycl::property::queue::in_order>()) return 2;
#define RUN(E) sglang_iq4_bf16_row::launch<E>(q, codes, gain, x, partial, m, n, k, splits)
    RUN(4);
#undef RUN
    sglang_iq4_bf16_row::reduce(q, partial, y, raw, m, n, splits);
    return 0;
  } catch (const std::exception& e) {
    sglang_iq4_bf16_row::error = e.what();
    return 1;
  }
}

extern "C" const char* xpu_iq4_bf16_row_error() {
  return sglang_iq4_bf16_row::error.c_str();
}
