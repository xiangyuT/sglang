// SPDX-License-Identifier: MIT
// Reuses resident_prepare.cpp / moe_quant_fused.cpp finite Q8_1 arithmetic.
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Prepare an existing GPU-cache grouping without changing placement or LRU.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>

namespace paged1362 {
thread_local std::string error;
using bf16 = sycl::ext::oneapi::bfloat16;
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81Block) == 36);
inline float finite_half(float x) {
    return sycl::fabs(x) > 65504.f ? sycl::copysign(65504.f, x) : x;
}
inline float warp_sum(sycl::sub_group sg, float x) {
    for (int o = 16; o; o >>= 1) x += sycl::permute_group_by_xor(sg, x, o);
    return x;
}
inline float warp_max(sycl::sub_group sg, float x) {
    for (int o = 16; o; o >>= 1) x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, o));
    return x;
}
template<bool Quant, bool Arena> class PagedWorkspace;
template<bool Quant, bool Arena>
void prepare(sycl::queue& q, const int32_t* meta, int32_t* starts,
             float* gate, float* up, float* down, int T, int topk, int zero,
             const bf16* input, float* xfp, Q81Block* xq) {
    const int routes = T * topk, gu = routes * 640;
    const int total = Arena ? routes * 2560 : 2 * gu + routes * 2560;
    q.parallel_for<PagedWorkspace<Quant, Arena>>(
        sycl::nd_range<1>((128 + (Quant ? T * 10 : 0)) * 64, 64),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int index = it.get_global_linear_id(), lane = it.get_local_linear_id();
            if constexpr (Quant) {
                if (it.get_group_linear_id() >= 128) {
                    auto sg = it.get_sub_group();
                    const int first = (it.get_group_linear_id() - 128) * 256;
                    for (int step = 0; step < 4; ++step) {
                        const int i = first + step * 64 + lane;
                        const float xi = float(input[i]); xfp[i] = xi;
                        const float amax = warp_max(sg, sycl::fabs(xi));
                        const float sum = warp_sum(sg, xi), d = finite_half(amax / 127.f);
                        float code = amax == 0.f ? 0.f : sycl::round(xi / d);
                        code = code > 127.f ? 127.f : code < -127.f ? -127.f : code;
                        xq[i / 32].qs[i % 32] = static_cast<int8_t>(code);
                        if (i % 32 == 0) xq[i / 32].ds = sycl::vec<float, 2>(
                            finite_half(d), finite_half(sum)).convert<sycl::half, sycl::rounding_mode::automatic>();
                    }
                    return;
                }
            }
            const sycl::vec<float, 4> zeros(0.f);
            for (int i = index; i < total / 4; i += 128 * 64) {
                if constexpr (Arena) {
                    // One clear owns each byte; aliased GU/down clears would race.
                    reinterpret_cast<sycl::vec<float, 4>*>(down)[i] = zeros;
                } else {
                    if (i < gu / 4) reinterpret_cast<sycl::vec<float, 4>*>(gate)[i] = zeros;
                    else if (i < gu / 2) reinterpret_cast<sycl::vec<float, 4>*>(up)[i - gu / 4] = zeros;
                    else reinterpret_cast<sycl::vec<float, 4>*>(down)[i - gu / 2] = zeros;
                }
            }
            if (index <= routes) {
                int clip = routes;
                const int groups = meta[6 * routes + 1];
                if (!meta[6 * routes + 2] && groups >= 1 && groups <= routes) {
                    // Native grouping sorts logical IDs. The dummy is the final
                    // logical expert even when its physical GPU slot changes.
                    for (int g = 0; g < groups; ++g)
                        if (meta[5 * routes + 1 + g] == zero) clip = meta[2 * routes + g];
                }
                // Keep pager metadata intact for subsequent planning/readback.
                starts[index] = sycl::min(meta[2 * routes + index], clip);
            }
        });
}
}

using namespace paged1362;
extern "C" int paged_prepare(void* queue, const int32_t* meta, int32_t* starts,
    float* gate, float* up, float* down, const void* input, float* xfp, void* xq,
    int T, int topk, int zero) {
    try {
        if (!queue || !meta || !starts || !gate || !up || !down || T < 1 || T > 32 ||
            topk < 1 || topk > 10 || zero < 1 || zero > 511 ||
            (input && (!xfp || !xq)) || (!input && (xfp || xq))) return 2;
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.is_in_order()) return 2;
        const bool arena = gate == down;
        if (arena && up != down + T * topk * 640) return 2;
        if (input) {
            if (arena) prepare<true, true>(q, meta, starts, gate, up, down, T, topk, zero,
                static_cast<const bf16*>(input), xfp, static_cast<Q81Block*>(xq));
            else prepare<true, false>(q, meta, starts, gate, up, down, T, topk, zero,
                static_cast<const bf16*>(input), xfp, static_cast<Q81Block*>(xq));
        } else {
            if (arena) prepare<false, true>(q, meta, starts, gate, up, down, T, topk, zero,
                nullptr, nullptr, nullptr);
            else prepare<false, false>(q, meta, starts, gate, up, down, T, topk, zero,
                nullptr, nullptr, nullptr);
        }
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}
extern "C" const char* paged_error() { return error.c_str(); }
extern "C" int paged_prepare_max_rows() { return 32; }

extern "C" int paged_prepare_arena_supported() { return 1; }
