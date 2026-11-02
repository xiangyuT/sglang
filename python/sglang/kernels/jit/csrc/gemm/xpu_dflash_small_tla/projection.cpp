// SPDX-License-Identifier: Apache-2.0
// Experimental B70 static FP16 operand cache and M4 projection.
// Uses the unmodified pinned Intel sgl-kernel-xpu xe_gemm (BSD-3-Clause)
// and sycl-tla tensor/copy/MMA headers; their upstream source is retained.
#define SYCL_INTEL_TARGET 20
#define CUTLASS_ENABLE_SYCL
#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "kernels/moe/xe20/common/xe_gemm.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dflash_small_tla {
using namespace cute;
thread_local std::string error;
class Pack;
template <int Rows>
class Prepare;
template <int Rows>
class Dense;

inline uint16_t operand(uint8_t raw, float scale) {
  uint16_t mag = raw & 127, sign = uint16_t(raw & 128) << 8;
  uint16_t bits = (mag << 7) | sign;
  if (mag == 127) bits = sign | 0x7e00;
  return sycl::bit_cast<uint16_t>(sycl::half(float(sycl::bit_cast<sycl::half>(bits)) * scale));
}

template <int Rows>
void prepare(sycl::queue& queue, const uint8_t* q, const float* xs, uint16_t* dst) {
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Prepare<Rows>>(sycl::range<1>(512 * Rows), [=](sycl::id<1> item) {
      unsigned i = item[0];
      dst[i] = operand(q[i], float(sycl::half(xs[i / 128])) * 256.f);
    });
  });
}

template <int Rows>
void launch(sycl::queue& queue, const uint16_t* input, const uint16_t* weight, uint16_t* output) {
  using Tile = Shape<_8, _64, _32>;
  using SLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
  using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, half_t>>, Layout<Tile>, SLayout>::TiledMMA;
  auto a = make_tensor(
      make_gmem_ptr(reinterpret_cast<const half_t*>(input)), make_layout(make_shape(Rows, 512), make_stride(512, _1{})));
  auto b = make_tensor(
      make_gmem_ptr(reinterpret_cast<const half_t*>(weight)),
      make_layout(make_shape(2048, 512), make_stride(512, _1{})));
  auto d = make_tensor(
      make_gmem_ptr(reinterpret_cast<bfloat16_t*>(output)), make_layout(make_shape(Rows, 2048), make_stride(2048, _1{})));
  constexpr int threads = size(MMA{});
  sycl::range<3> local(1, 1, threads), groups(1, 2048 / 64, 1);
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<16>, intel::grf_size<256>};
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<Dense<Rows>>(sycl::nd_range<3>(groups * local, local), props, [=](sycl::nd_item<3> item) {
      auto output_tensor = d;
      moe_xe20::xe_gemm<void, void, void>(
          a,
          b,
          static_cast<const float*>(nullptr),
          static_cast<const float*>(nullptr),
          output_tensor,
          make_coord(0, int(item.get_group_linear_id()), _, 0),
          MMA{});
    });
  });
}
}  // namespace dflash_small_tla

extern "C" int dflash_small_tla_abi() {
  return 1;
}
extern "C" const char* dflash_small_tla_error() {
  return dflash_small_tla::error.c_str();
}

extern "C" int
dflash_small_tla_pack(void* q, const uint8_t* w, const float* ws, uint16_t* out, int n, int k, int s0, int s1) {
  try {
    if (!q || !w || !ws || !out || n != 2048 || k != 512 || s0 <= 0 || s1 <= 0)
      throw std::runtime_error("Unsupported small TLA cache shape, stride or pointer");
    auto& queue = *static_cast<sycl::queue*>(q);
    queue.submit([&](sycl::handler& h) {
      h.parallel_for<dflash_small_tla::Pack>(sycl::range<1>(size_t(n) * k), [=](sycl::id<1> item) {
        unsigned i = item[0], row = i / k, col = i % k;
        out[i] = dflash_small_tla::operand(w[i], float(sycl::half(ws[(row / 128) * s0 + (col / 128) * s1])) * 256.f);
      });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_small_tla::error = e.what();
    return 1;
  }
}

template <int Rows>
int run_small_tla(
    void* q, const uint8_t* x, const uint16_t* w, const float* xs, uint16_t* a, uint16_t* y, int n, int k) {
  try {
    if (!q || !x || !w || !xs || !a || !y || n != 2048 || k != 512)
      throw std::runtime_error("Unsupported small TLA projection shape or pointer");
    auto& queue = *static_cast<sycl::queue*>(q);
    dflash_small_tla::prepare<Rows>(queue, x, xs, a);
    dflash_small_tla::launch<Rows>(queue, a, w, y);
    return 0;
  } catch (const std::exception& e) {
    dflash_small_tla::error = e.what();
    return 1;
  }
}
extern "C" int dflash_small_tla_gemm(
    void* q, const uint8_t* x, const uint16_t* w, const float* xs, uint16_t* a, uint16_t* y, int n, int k) {
  return run_small_tla<4>(q, x, w, xs, a, y, n, k);
}
extern "C" int dflash_small_tla_gemm_m8(
    void* q, const uint8_t* x, const uint16_t* w, const float* xs, uint16_t* a, uint16_t* y, int n, int k) {
  return run_small_tla<8>(q, x, w, xs, a, y, n, k);
}
