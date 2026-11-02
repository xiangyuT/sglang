// SPDX-License-Identifier: Apache-2.0
// QSA indexer RoPE: retain each BF16 product rounding from apply_rotary_emb.
// The BF16 conversion/NeoX pairing follows xpu_tp_qk_norm_rope.cpp, without
// that kernel's normalization or its fixed TP2/head geometry.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>

#pragma clang fp contract(off) reassociate(off)

namespace sglang_qsa {
using BF = sycl::ext::oneapi::bfloat16;
inline float as_float(uint16_t value) {
    return sycl::bit_cast<float>(uint32_t(value) << 16);
}
inline uint16_t round_bf(float value) {
    return sycl::bit_cast<uint16_t>(BF(value));
}
class IndexRoPE;
}

extern "C" int xpu_qsa_rope(void* queue, const uint16_t* input,
    const void* cache, const int64_t* positions, const int64_t* axes,
    uint16_t* output, int rows, int heads, int cache_rows, int cache_bytes,
    int position_axes, int64_t input_row_stride, int64_t input_head_stride,
    int64_t cache_row_stride, int64_t position_axis_stride,
    int64_t position_row_stride) {
    if (!queue || !input || !cache || !positions || !output || rows < 1 ||
        (heads != 1 && heads != 4) || cache_rows < 1 ||
        (cache_bytes != 2 && cache_bytes != 4) ||
        (position_axes != 1 && position_axes != 3) ||
        (position_axes == 3 && !axes)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        q.parallel_for<sglang_qsa::IndexRoPE>(
            sycl::nd_range<1>(size_t(rows) * heads * 128, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int group = item.get_group_linear_id();
                const int row = group / heads, head = group % heads;
                const int column = item.get_local_linear_id();
                const auto* src = input + row * input_row_stride + head * input_head_stride;
                auto* dst = output + int64_t(group) * 128;
                if (column >= 64) {
                    dst[column] = src[column];
                    return;
                }
                const int lane = column % 32;
                const int axis = position_axes == 1 ? 0 : int(axes[lane]);
                // The opt-in fixed-cache route requires all scheduler-provided
                // coordinates to be in the preallocated RoPE table. Never read
                // out of bounds if that producer contract is violated.
                if (axis < 0 || axis >= position_axes) {
                    dst[column] = sglang_qsa::round_bf(sycl::nan(0u));
                    return;
                }
                const int64_t position = positions[axis * position_axis_stride + row * position_row_stride];
                if (position < 0 || position >= cache_rows) {
                    dst[column] = sglang_qsa::round_bf(sycl::nan(0u));
                    return;
                }
                const int64_t offset = position * cache_row_stride;
                float cosine, sine;
                if (cache_bytes == 2) {
                    auto* table = static_cast<const uint16_t*>(cache);
                    cosine = sglang_qsa::as_float(table[offset + lane]);
                    sine = sglang_qsa::as_float(table[offset + 32 + lane]);
                } else {
                    auto* table = static_cast<const float*>(cache);
                    cosine = sglang_qsa::as_float(sglang_qsa::round_bf(table[offset + lane]));
                    sine = sglang_qsa::as_float(sglang_qsa::round_bf(table[offset + 32 + lane]));
                }
                const bool upper = column >= 32;
                const float self = sglang_qsa::as_float(src[column]);
                const float partner = sglang_qsa::as_float(src[column + (upper ? -32 : 32)]);
                const float left = sglang_qsa::as_float(sglang_qsa::round_bf(self * cosine));
                const float right = sglang_qsa::as_float(sglang_qsa::round_bf(partner * sine));
                dst[column] = sglang_qsa::round_bf(upper ? left + right : left - right);
            });
        return 0;
    } catch (const std::exception&) { return 1; }
}
