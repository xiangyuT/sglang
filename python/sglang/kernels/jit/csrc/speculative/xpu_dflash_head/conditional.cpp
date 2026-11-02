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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "head_errors.hpp"
#include <cstdint>
#include <functional>
#pragma clang fp contract(off) reassociate(off)
namespace dflash_head_conditional_impl {
namespace es = sycl::ext::intel::esimd;
class Head;
template <int N>
ESIMD_INLINE es::simd<float, N> load(const uint16_t* p) {
  auto raw = es::block_load<uint16_t, N>(p);
  es::simd<uint32_t, N> bits = es::convert<uint32_t>(raw) << 16;
  return bits.template bit_cast_view<float>();
}
ESIMD_INLINE float rounded(float x) {
  uint32_t b = sycl::bit_cast<uint32_t>(x);
  if ((b & 0x7fffffff) > 0x7f800000u) return sycl::bit_cast<float>(0x7fc00000u);
  b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
  return sycl::bit_cast<float>(b);
}
}  // namespace dflash_head_conditional_impl
extern "C" int dflash_head_complete(
    void* stream,
    const uint16_t* hidden,
    const uint16_t* weight,
    const int64_t* predictions,
    const int* flags,
    float* output,
    int* counts) {
  try {
    auto& queue = *static_cast<sycl::queue*>(stream);
    queue.submit([&](sycl::handler& h) {
      h.parallel_for<dflash_head_conditional_impl::Head>(
          sycl::range<1>(248320),
#if DFLASH_NATIVE_WINDOW == 8
          sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
#endif
          [=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            using namespace dflash_head_conditional_impl;
            int row = item[0];
#if DFLASH_NATIVE_WINDOW == 4
            bool f0 = flags[0], f1 = flags[1], f2 = flags[2], f3 = flags[3];
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            if (!f0 || !f1 || !f2 || !f3) {
              es::simd<float, 128> a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 1
              for (int k = 0; k < 2048; k += 128) {
                auto w = load<128>(weight + row * 2048 + k);
                if (!f0) a0 += w * load<128>(hidden + k);
                if (!f1) a1 += w * load<128>(hidden + 2048 + k);
                if (!f2) a2 += w * load<128>(hidden + 4096 + k);
                if (!f3) a3 += w * load<128>(hidden + 6144 + k);
              }
              if (!f0) s0 = rounded(es::reduce<float>(a0, std::plus<>()));
              if (!f1) s1 = rounded(es::reduce<float>(a1, std::plus<>()));
              if (!f2) s2 = rounded(es::reduce<float>(a2, std::plus<>()));
              if (!f3) s3 = rounded(es::reduce<float>(a3, std::plus<>()));
            }
            float neg = sycl::bit_cast<float>(0xff800000u);
            output[row] = f0 ? (row == predictions[0] ? 0.f : neg) : s0;
            output[248320 + row] = f1 ? (row == predictions[1] ? 0.f : neg) : s1;
            output[496640 + row] = f2 ? (row == predictions[2] ? 0.f : neg) : s2;
            output[744960 + row] = f3 ? (row == predictions[3] ? 0.f : neg) : s3;
            if (row == 0) {
              counts[0] += 1;
              counts[1] += int(!f0 || !f1 || !f2 || !f3);
              counts[2] += int(!f0) + int(!f1) + int(!f2) + int(!f3);
            }
#else
            bool fast[8]; bool any=false; int missing=0;
#pragma unroll
            for(int token=0;token<8;++token){fast[token]=flags[token];any|=!fast[token];missing+=int(!fast[token]);}
            float scores[8]{};
            if(any){
              es::simd<float,128> acc[8];
#pragma unroll
              for(int token=0;token<8;++token)acc[token]=0.f;
#pragma unroll 1
              for(int k=0;k<2048;k+=128){
                auto w=load<128>(weight+row*2048+k);
#pragma unroll
                for(int token=0;token<8;++token)if(!fast[token])acc[token]+=w*load<128>(hidden+token*2048+k);
              }
#pragma unroll
              for(int token=0;token<8;++token)if(!fast[token])scores[token]=rounded(es::reduce<float>(acc[token],std::plus<>()));
            }
            float neg=sycl::bit_cast<float>(0xff800000u);
#pragma unroll
            for(int token=0;token<8;++token)output[token*248320+row]=fast[token]?(row==predictions[token]?0.f:neg):scores[token];
            if(row==0){counts[0]+=1;counts[1]+=int(any);counts[2]+=missing;}

#endif
          });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_target_head_impl::error = e.what();
    return 1;
  }
}
