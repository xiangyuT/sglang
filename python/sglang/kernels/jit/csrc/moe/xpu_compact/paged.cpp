// Experimental adaptation of sgl-project/sgl-kernel-xpu v0.4.0 W8A16 launcher.
// Original kernel copyright (C) 2025 Intel Corporation. BSD-3-Clause.
#define SYCL_INTEL_TARGET 20
#define CUTLASS_ENABLE_SYCL
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "paged_kernel.hpp"
#include <exception>
#include <string>
using namespace cute;
class DFlashCompactMoePaged;
namespace {
thread_local std::string last_error;
}
extern "C" int dflash_paged_abi() {
  return 1;
}
extern "C" const char* dflash_paged_error() {
  return last_error.c_str();
}
extern "C" int dflash_paged_gemm(
    void* queue_ptr,
    const void* activations,
    const void* weights,
    const void* scales,
    void* output,
    const int* counts,
    const int* expert_ids,
    const int* offsets,
    const void* host_weights,
    const int* host_ids,
    int experts,
    int n,
    int k,
    int tile_major) {
  try {
    using Element = cutlass::bfloat16_t;
    auto dummy = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };
    using A = decltype(dummy(Element{}, Stride<int, _1>{}));
    using B = decltype(dummy(cutlass::float_e4m3_t{}, Stride<int, _1>{}));
    using D = decltype(dummy(Element{}, Stride<int, _1>{}));
    using Tile = Shape<_16, _64, _32>;
    using SG = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
    using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, Element>>, Layout<Tile>, SG>::TiledMMA;
    using Kernel = moe_w8a16::PagedCompactW8A16Kernel<Tile, SG, A, B, D, MMA, false, true>;
    typename Kernel::Params p{
        static_cast<const uint8_t*>(activations),
        static_cast<const uint8_t*>(weights),
        scales,
        nullptr,
        static_cast<Element*>(output),
        counts,
        expert_ids,
        offsets,
        static_cast<const uint8_t*>(host_weights),
        host_ids,
        n,
        k,
        experts,
        nullptr,
        MMA{},
        k,
        static_cast<int>(moe_w8a16::ScaleMode::BlockFP32),
        true,
        bool(tile_major)};
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    constexpr int threads = size(MMA{});
    sycl::range<3> local(1, 1, threads), groups(1, experts * ((n + 63) / 64), 1);
    namespace one = sycl::ext::oneapi::experimental;
    namespace intel = sycl::ext::intel::experimental;
    one::properties props{one::sub_group_size<16>, intel::grf_size<256>};
    q.submit([&](sycl::handler& h) {
      h.parallel_for<DFlashCompactMoePaged>(
          sycl::nd_range<3>(groups * local, local), props, [=](sycl::nd_item<3> item) { Kernel{}(p, item, nullptr); });
    });
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return 1;
  }
}
