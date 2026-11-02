// SPDX-License-Identifier: MIT
// IQ3_S formulas/codebook: ggml / attributed Strata decoder port.
// Lossless four-output-row field order. Cache owners use this order in both tiers.
#include "iq3_expert_soa.hpp"

namespace sglang_iq3_expert_soa {
constexpr int64_t encoded_bytes = 704000, values = 640 * 2560;


class Permute;
template<typename D> class Decode;
template<typename D>
void decode(sycl::queue& q, const uint8_t* source, D* destination, int64_t count) {
    q.parallel_for<Decode<D>>(sycl::range<1>(count), [=](sycl::id<1> id) {
        const int64_t i = id[0], expert = i / values;
        const int row = i / 2560 % 640, k = i % 2560, block = k / 256;
        destination[i] = D(value(source + expert * encoded_bytes, row, block, k));
    });
}
}

extern "C" int xpu_gguf_iq3_gu_permute(
    void* queue, const uint8_t* source, uint8_t* destination, int experts, int inverse) {
    using namespace sglang_iq3_expert_soa;
    if (!queue || !source || !destination || source == destination || experts < 1
        || experts > 512 || (inverse != 0 && inverse != 1)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.is_in_order()) return 2;
        q.parallel_for<Permute>(sycl::range<1>(int64_t(experts) * encoded_bytes),
            [=](sycl::id<1> id) {
                const int64_t raw = id[0], expert = raw / encoded_bytes;
                const int row = raw % encoded_bytes / 1100;
                const int block = raw % 1100 / 110, byte = raw % 110;
                const int64_t packed = expert * encoded_bytes + offset(row, block, byte);
                if (inverse) destination[raw] = source[packed];
                else destination[packed] = source[raw];
            });
        return 0;
    } catch (const std::exception&) { return 1; }
}

extern "C" int xpu_gguf_dequant(void*, const void*, void*, int, int64_t, int);
extern "C" int xpu_gguf_iq3_cache_dequant(
    void* queue, const void* source, void* destination, int kind, int64_t count, int dtype) {
    using namespace sglang_iq3_expert_soa;
    if (kind != 21) return xpu_gguf_dequant(queue, source, destination, kind, count, dtype);
    if (!queue || !source || !destination || count < values || count % values
        || (dtype != 0 && dtype != 1)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.is_in_order()) return 2;
        if (dtype == 0) decode(q, static_cast<const uint8_t*>(source), static_cast<sycl::half*>(destination), count);
        else decode(q, static_cast<const uint8_t*>(source), static_cast<float*>(destination), count);
        return 0;
    } catch (const std::exception&) { return 1; }
}
