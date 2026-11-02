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
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off) reassociate(off)

namespace dflash_target_head_impl {
constexpr int N = 248320, K = 2048, Blocks = N / 256;
thread_local std::string error;
class Prepare;
class Bound;
class Select;
class Fill;
inline float bf16(uint16_t v) {
  return sycl::bit_cast<float>(uint32_t(v) << 16);
}
inline float cast(float v, int dtype) {
  if (dtype == 1) return float(sycl::half(v));
  if (dtype == 2) return float(sycl::ext::oneapi::bfloat16(v));
  return v;
}
inline uint16_t round_bf16(float v) {
  return sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(v));
}
inline float fp8(uint8_t raw) {
  uint16_t mag = raw & 127, sign = uint16_t(raw & 128) << 8;
  uint16_t bits = (mag << 7) | sign;
  if (mag == 127) bits = sign | 0x7e00;
  return float(sycl::bit_cast<sycl::half>(bits)) * 256.f;
}
inline uint64_t key(uint16_t bits, int64_t id) {
  if ((bits & 0x7fff) == 0) bits = 0;
  uint32_t ordered = bits & 0x8000 ? (~uint32_t(bits)) & 0xffff : bits ^ 0x8000;
  return (uint64_t(ordered) << 32) | (0xffffffffu - uint32_t(id));
}
}  // namespace dflash_target_head_impl
extern "C" const char* dflash_head_error() {
  return dflash_target_head_impl::error.c_str();
}
extern "C" int dflash_head_finish(
    void* stream,
    const uint16_t* h,
    const uint8_t* q,
    const float* xs,
    uint16_t* coarse,
    const int64_t* ids,
    const float* coeff,
    const float* scores,
    const float* absolute,
    float* state,
    float* partial,
    int64_t* predictions,
    int* flags) {
  try {
    using namespace dflash_target_head_impl;
    auto& queue = *static_cast<sycl::queue*>(stream);
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<Prepare>(sycl::nd_range<1>(DFLASH_NATIVE_WINDOW * 256, 256), [=](sycl::nd_item<1> item) {
        int token = item.get_group_linear_id(), lane = item.get_local_linear_id();
        float hn = 0.f, norms[9]{}, errors[9]{};
        int valid = 1;
        for (int d = lane; d < K; d += 256) {
          float original = bf16(h[token * K + d]);
          float raw = fp8(q[token * K + d]), scale = xs[token * 16 + d / 128];
          valid &= sycl::isfinite(original) && sycl::isfinite(raw) && sycl::isfinite(scale);
          hn += original * original;
#pragma unroll
          for (int s = 0; s < 3; ++s) {
            float value = raw * cast(scale, s);
#pragma unroll
            for (int o = 0; o < 3; ++o) {
              int j = s * 3 + o;
              float operand = cast(value, o), delta = original - operand;
              valid &= sycl::isfinite(operand);
              norms[j] += operand * operand;
              errors[j] += delta * delta;
            }
          }
        }
        hn = sycl::reduce_over_group(item.get_group(), hn, sycl::plus<float>());
        float an = 0.f, en = 0.f;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          float n = sycl::reduce_over_group(item.get_group(), norms[j], sycl::plus<float>());
          float e = sycl::reduce_over_group(item.get_group(), errors[j], sycl::plus<float>());
          valid &= sycl::isfinite(n) && sycl::isfinite(e);
          an = sycl::fmax(an, n);
          en = sycl::fmax(en, e);
        }
        valid &= sycl::isfinite(hn);
        valid = sycl::reduce_over_group(item.get_group(), valid, sycl::minimum<int>());
        if (lane == 0) {
          state[token * 4] = sycl::sqrt(hn) * 1.001f + 1e-10f;
          state[token * 4 + 1] = sycl::sqrt(an) * 1.001f + 1e-10f;
          state[token * 4 + 2] = sycl::sqrt(en) * 1.001f + 1e-10f;
          state[token * 4 + 3] = float(valid);
        }
        if (lane < 64) coarse[token * N + ids[token * 64 + lane]] = 0xff80;
      });
    });
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<Bound>(sycl::nd_range<1>(DFLASH_NATIVE_WINDOW * N, 256), [=](sycl::nd_item<1> item) {
        int group = item.get_group_linear_id(), token = group / Blocks;
        int row = (group % Blocks) * 256 + item.get_local_linear_id();
        float value = bf16(coarse[token * N + row]), upper = -std::numeric_limits<float>::infinity();
        constexpr float gamma = (2048.f / 16777216.f) / (1.f - 2048.f / 16777216.f);
        constexpr float u = 1.f / 256.f;
        if (value != -std::numeric_limits<float>::infinity()) {
          float norm = state[token * 4], an = state[token * 4 + 1], error = state[token * 4 + 2];
          float wn = coeff[row], we = coeff[N + row], wa = coeff[2 * N + row];
          float codec = error * wn + an * we;
          float accum = gamma * (an * wa + norm * wn);
          float a_round = (u / (1.f - u)) * sycl::fabs(value) + 1e-10f;
          float limit = (1.f + u) * (codec + accum + a_round) + u * sycl::fabs(value) + 1e-10f;
          upper = value + limit;
          if (!sycl::isfinite(value) || !sycl::isfinite(upper)) upper = std::numeric_limits<float>::infinity();
        }
        float largest = sycl::reduce_over_group(item.get_group(), upper, sycl::maximum<float>());
        if (item.get_local_linear_id() == 0) partial[group] = largest;
      });
    });
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<Select>(sycl::nd_range<1>(DFLASH_NATIVE_WINDOW * 64, 64), [=](sycl::nd_item<1> item) {
        int token = item.get_group_linear_id(), lane = item.get_local_linear_id();
        int index = token * 64 + lane;
        float score = scores[index], sum = absolute[index];
        constexpr float gn = (2048.f / 16777216.f) / (1.f - 2048.f / 16777216.f);
        constexpr float go = (23.f / 16777216.f) / (1.f - 23.f / 16777216.f);
        float error = (gn + go) * sum * 1.001f + 1e-30f;
        uint64_t mine = key(round_bf16(score), ids[index]);
        uint64_t best = sycl::reduce_over_group(item.get_group(), mine, sycl::maximum<uint64_t>());
        uint64_t lower = key(round_bf16(score - error), ids[index]);
        uint64_t upper = mine == best ? 0 : key(round_bf16(score + error), ids[index]);
        uint64_t best_lower =
            sycl::reduce_over_group(item.get_group(), mine == best ? lower : uint64_t(0), sycl::maximum<uint64_t>());
        uint64_t rival_upper = sycl::reduce_over_group(item.get_group(), upper, sycl::maximum<uint64_t>());
        float best_low_score = sycl::reduce_over_group(
            item.get_group(),
            mine == best ? bf16(round_bf16(score - error)) : -std::numeric_limits<float>::infinity(),
            sycl::maximum<float>());
        float excluded = -std::numeric_limits<float>::infinity();
        for (int b = lane; b < Blocks; b += 64)
          excluded = sycl::fmax(excluded, partial[token * Blocks + b]);
        excluded = sycl::reduce_over_group(item.get_group(), excluded, sycl::maximum<float>());
        int valid = sycl::isfinite(score) && sycl::isfinite(sum) && sum >= 0.f;
        valid = sycl::reduce_over_group(item.get_group(), valid, sycl::minimum<int>());
        if (lane == 0) {
          predictions[token] = int64_t(0xffffffffu - uint32_t(best));
          flags[token] = valid && state[token * 4 + 3] == 1.f && best_lower > rival_upper && best_low_score > excluded;
        }
      });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_target_head_impl::error = e.what();
    return 1;
  }
}
extern "C" int dflash_head_fill(void* stream, const int64_t* predictions, float* logits) {
  try {
    auto& queue = *static_cast<sycl::queue*>(stream);
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<dflash_target_head_impl::Fill>(
          sycl::range<1>(DFLASH_NATIVE_WINDOW * dflash_target_head_impl::N), [=](sycl::id<1> item) {
            int index = item[0], token = index / dflash_target_head_impl::N,
                column = index % dflash_target_head_impl::N;
            logits[index] = column == predictions[token] ? 0.f : -std::numeric_limits<float>::infinity();
          });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_target_head_impl::error = e.what();
    return 1;
  }
}

extern "C" int dflash_head_abi() {
  return 1;
}
