// SPDX-License-Identifier: MIT
// IQ4_NL formulas and codebook: ggml / attributed Strata decoder port.
// Lossless field order for four output rows and four adjacent K blocks.
#include "dequant.hpp"

namespace sglang_iq4_expert_soa {
constexpr int64_t encoded_bytes = 921600, values = 2560 * 640;

inline int64_t offset(int row, int block, int byte) {
    const int64_t tile = int64_t(row / 4) * 1440 + (block / 4) * 288;
    const int lane = (row % 4) * 8 + (block % 4) * 2;
    if (byte < 2) return tile + lane + byte;
    const int q = byte - 2;
    return tile + 32 + ((q / 4) % 2) * 128 + (lane + q / 8) * 4 + q % 4;
}

class Permute;
template<typename D> class Decode;
template<typename D>
void decode(sycl::queue& q, const uint8_t* source, D* destination, int64_t count) {
    q.parallel_for<Decode<D>>(sycl::range<1>(count), [=](sycl::id<1> id) {
        const int64_t i = id[0], expert = i / values;
        const int row = (i / 640) % 2560, k = i % 640, block = k / 32;
        const auto* p = source + expert * encoded_bytes;
        const float d = float(*reinterpret_cast<const sycl::half*>(p + offset(row, block, 0)));
        const uint8_t bits = p[offset(row, block, 2 + k % 16)];
        const int code = k % 32 < 16 ? bits & 15 : bits >> 4;
        destination[i] = D(d * kvalues_iq4nl[code]);
    });
}
}

extern "C" int xpu_gguf_iq4_down_permute(
    void* queue, const uint8_t* source, uint8_t* destination, int experts, int inverse) {
    using namespace sglang_iq4_expert_soa;
    if (!queue || !source || !destination || source == destination || experts < 1
        || experts > 512 || (inverse != 0 && inverse != 1)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        q.parallel_for<Permute>(sycl::range<1>(int64_t(experts) * encoded_bytes),
            [=](sycl::id<1> id) {
                const int64_t raw = id[0], expert = raw / encoded_bytes;
                const int row = (raw % encoded_bytes) / 360;
                const int block = (raw % 360) / 18, byte = raw % 18;
                const int64_t packed = expert * encoded_bytes + offset(row, block, byte);
                if (inverse) destination[raw] = source[packed];
                else destination[packed] = source[raw];
            });
        return 0;
    } catch (const std::exception&) { return 1; }
}

extern "C" int xpu_gguf_dequant(void*, const void*, void*, int, int64_t, int);
// For cache-owned prefill buffers only. Ordinary GGUF dequantization retains
// its original row format; other cache quantizations delegate unchanged.
extern "C" int xpu_gguf_iq4_cache_dequant(
    void* queue, const void* source, void* destination, int kind, int64_t count, int dtype) {
    using namespace sglang_iq4_expert_soa;
    if (kind != 20) return xpu_gguf_dequant(queue, source, destination, kind, count, dtype);
    if (!queue || !source || !destination || count < values || count % values
        || (dtype != 0 && dtype != 1)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        if (dtype == 0) decode(q, static_cast<const uint8_t*>(source), static_cast<sycl::half*>(destination), count);
        else decode(q, static_cast<const uint8_t*>(source), static_cast<float*>(destination), count);
        return 0;
    } catch (const std::exception&) { return 1; }
}
