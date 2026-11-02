/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "../../moe/xpu_compact/dflash_window.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off) reassociate(off)

namespace dflash_head_rescore_impl {
namespace es = sycl::ext::intel::esimd;
thread_local std::string error;
class Rescore;
template <int N>
ESIMD_INLINE es::simd<float, N> bf16_load(const uint16_t* source) {
  auto raw = es::block_load<uint16_t, N>(source);
  es::simd<uint32_t, N> bits = es::convert<uint32_t>(raw) << 16;
  return bits.template bit_cast_view<float>();
}
}  // namespace dflash_head_rescore_impl
extern "C" const char* dflash_head_rescore_error() {
  return dflash_head_rescore_impl::error.c_str();
}
extern "C" int dflash_head_rescore(
    void* stream,
    const uint16_t* hidden,
    const uint16_t* weight,
    const int64_t* ids,
    float* scores,
    float* absolute_sum) {
  try {
    auto& queue = *static_cast<sycl::queue*>(stream);
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<dflash_head_rescore_impl::Rescore>(
          sycl::range<1>(DFLASH_NATIVE_WINDOW * 64), [=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            unsigned index = item[0], token = index / 64;
            int64_t row = ids[index];
            es::simd<float, 128> accum = 0.f, absolute = 0.f;
#pragma unroll 1
            for (int begin = 0; begin < 2048; begin += 128) {
              auto h = dflash_head_rescore_impl::bf16_load<128>(hidden + token * 2048 + begin);
              auto w = dflash_head_rescore_impl::bf16_load<128>(weight + row * 2048 + begin);
              auto product = h * w;
              accum += product;
              absolute += es::abs(product);
            }
            scores[index] = es::reduce<float>(accum, std::plus<>());
            absolute_sum[index] = es::reduce<float>(absolute, std::plus<>());
          });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_head_rescore_impl::error = e.what();
    return 1;
  }
}
