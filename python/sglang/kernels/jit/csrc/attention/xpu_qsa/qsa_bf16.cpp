// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Adapted from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd,
// sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp: chunk/merge attention.
// See the accompanying STRATA-LICENSE. The adapter reads SGLang's physical
// BF16 token slots, supports a query batch, and writes BF16 output. It retains
// the 64-cell chunks and 12-query-head KV reuse of the original algorithm.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cfloat>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <cstring>
#ifdef GGUF_QSA_INT8
#include "strata_fp16_bits.hpp"
#endif

// Local TP head partitions retain the full-head reduction order. TP4 replicates
// each global KV head on two ranks, each of which computes six query heads.
#ifndef GGUF_QSA_TP
#define GGUF_QSA_TP 1
#endif
#ifdef GGUF_QSA_INT8
#if GGUF_QSA_TP == 1
#define QSA_NAMESPACE sglang_qsa_int8
#define QSA_ENTRY xpu_qsa_int8
#elif GGUF_QSA_TP == 2
#define QSA_NAMESPACE sglang_qsa_int8_tp2
#define QSA_ENTRY xpu_qsa_int8_tp2
#elif GGUF_QSA_TP == 4
#define QSA_NAMESPACE sglang_qsa_int8_tp4
#define QSA_ENTRY xpu_qsa_int8_tp4
#else
#error "GGUF_QSA_TP must be 1, 2 or 4"
#endif
#else
#if GGUF_QSA_TP == 1
#define QSA_NAMESPACE sglang_qsa
#define QSA_ENTRY xpu_qsa_bf16
#elif GGUF_QSA_TP == 2
#define QSA_NAMESPACE sglang_qsa_tp2
#define QSA_ENTRY xpu_qsa_bf16_tp2
#elif GGUF_QSA_TP == 4
#define QSA_NAMESPACE sglang_qsa_tp4
#define QSA_ENTRY xpu_qsa_bf16_tp4
#else
#error "GGUF_QSA_TP must be 1, 2 or 4"
#endif
#endif
namespace QSA_NAMESPACE {
constexpr int QH = 24 / GGUF_QSA_TP, KVH = GGUF_QSA_TP == 1 ? 2 : 1;
constexpr int HD = 256, G = QH / KVH, CHUNK = 64, THREADS = 256;
#ifdef GGUF_QSA_INT8
using CacheElement = int8_t;
constexpr int CACHE_HD = HD + (HD / 64) * 2;
inline float cache_value(const int8_t* base, int d) {
    const auto* scales = reinterpret_cast<const uint16_t*>(base + HD);
    return float(base[d]) * strata::kernels::f32_from_f16(scales[d / 64]);
}
#else
using CacheElement = uint16_t;
constexpr int CACHE_HD = HD;
#endif
class ChunkBF16;
class MergeBF16;
inline float bf16(uint16_t value) {
    return sycl::bit_cast<float>(uint32_t(value) << 16);
}
inline float warp_sum(sycl::sub_group sg, float x) {
    for (int offset = 16; offset; offset >>= 1)
        x += sycl::permute_group_by_xor(sg, x, offset);
    return x;
}
inline float warp_max(sycl::sub_group sg, float x) {
    for (int offset = 16; offset; offset >>= 1)
        x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, offset));
    return x;
}
}

using QSA_NAMESPACE::CacheElement;

extern "C" int QSA_ENTRY(
    void* queue, const uint16_t* q, const CacheElement* k, const CacheElement* v,
    const int32_t* ids, float* part_acc, float* part_m, float* part_l,
    uint16_t* output, int queries, int width, int pool_rows,
    int64_t k_stride, int64_t v_stride, float scale) {
    using namespace QSA_NAMESPACE;
    if (!queue || !q || !k || !v || !ids || !part_acc || !part_m || !part_l ||
        !output || queries <= 0 || width <= 0 || pool_rows <= 0 ||
        k_stride < KVH * CACHE_HD || v_stride < KVH * CACHE_HD) return 2;
    try {
        auto& stream = *static_cast<sycl::queue*>(queue);
        const int chunks = (width + CHUNK - 1) / CHUNK;
        const char* setting = std::getenv("SGLANG_XPU_QSA_SKIP_EMPTY_CHUNKS");
        const bool skip_empty = queries > 4 && setting && std::strcmp(setting, "1") == 0;
        stream.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sq(G * HD, h), sp(G * CHUNK, h);
            sycl::local_accessor<int32_t, 1> rows(CHUNK, h);
            h.parallel_for<ChunkBF16>(
                sycl::nd_range<1>(size_t(queries) * KVH * chunks * THREADS, THREADS),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const int group = it.get_group_linear_id();
                    const int chunk = group % chunks;
                    const int kvh = (group / chunks) % KVH;
                    const int query = group / (KVH * chunks);
                    const int t = it.get_local_linear_id(), lane = t & 31, warp = t >> 5;
                    const int count = sycl::min(CHUNK, width - chunk * CHUNK);
                    const size_t partial = size_t(group) * G;
                    for (int i = t; i < G * HD; i += THREADS)
                        sq[i] = bf16(q[(size_t(query) * QH + kvh * G) * HD + i]);
                    if (t < CHUNK) {
                        int row = t < count ? ids[size_t(query) * width + chunk * CHUNK + t] : -1;
                        rows[t] = row >= 0 && row < pool_rows ? row : -1;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    const auto sg = it.get_sub_group();
                    // Strata emits these sentinels for chunks beyond a query's
                    // selected width. SGLang supplies masked physical slots,
                    // so every subgroup votes over the same complete 64 rows.
                    // The vote is identical across the workgroup; merge never
                    // reads part_acc for a -FLT_MAX chunk.
                    if (skip_empty && !sycl::any_of_group(
                            sg, rows[lane] >= 0 || rows[lane + 32] >= 0)) {
                        if (t < G) {
                            part_m[partial + t] = -FLT_MAX;
                            part_l[partial + t] = 0.0f;
                        }
                        return;
                    }
                    for (int c = warp; c < CHUNK; c += THREADS / 32) {
                        if (rows[c] < 0) {
                            if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                            continue;
                        }
                        const size_t base = int64_t(rows[c]) * k_stride + kvh * CACHE_HD;
                        const size_t offset = base + lane * 8;
                        float key[8];
                        #pragma unroll
                        for (int d = 0; d < 8; ++d) {
#ifdef GGUF_QSA_INT8
                            key[d] = cache_value(k + base, lane * 8 + d);
#else
                            key[d] = bf16(k[offset + d]);
#endif
                        }
                        #pragma unroll
                        for (int head = 0; head < G; ++head) {
                            const int qi = head * HD + lane * 8;
                            float dot = key[0] * sq[qi] + key[1] * sq[qi + 1] +
                                key[2] * sq[qi + 2] + key[3] * sq[qi + 3] +
                                key[4] * sq[qi + 4] + key[5] * sq[qi + 5] +
                                key[6] * sq[qi + 6] + key[7] * sq[qi + 7];
                            dot = warp_sum(sg, dot);
                            if (lane == 0) sp[head * CHUNK + c] = dot * scale;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int head = warp; head < G; head += THREADS / 32) {
                        const float a = sp[head * CHUNK + lane];
                        const float b = sp[head * CHUNK + lane + 32];
                        const float m = warp_max(sg, sycl::fmax(a, b));
                        const float ea = rows[lane] >= 0 ? sycl::exp(a - m) : 0.0f;
                        const float eb = rows[lane + 32] >= 0 ? sycl::exp(b - m) : 0.0f;
                        sp[head * CHUNK + lane] = ea;
                        sp[head * CHUNK + lane + 32] = eb;
                        const float l = warp_sum(sg, ea + eb);
                        if (lane == 0) { part_m[partial + head] = m; part_l[partial + head] = l; }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    float acc[G] = {};
                    for (int c = 0; c < count; ++c) {
                        if (rows[c] < 0) continue;
                        #ifdef GGUF_QSA_INT8
                        const float value = cache_value(v + int64_t(rows[c]) * v_stride + kvh * CACHE_HD, t);
#else
                        const float value = bf16(v[int64_t(rows[c]) * v_stride + kvh * HD + t]);
#endif
                        #pragma unroll
                        for (int head = 0; head < G; ++head)
                            acc[head] = sycl::fma(sp[head * CHUNK + c], value, acc[head]);
                    }
                    #pragma unroll
                    for (int head = 0; head < G; ++head)
                        part_acc[(partial + head) * HD + t] = acc[head];
                });
        });
        stream.parallel_for<MergeBF16>(
            sycl::nd_range<1>(size_t(queries) * QH * HD, HD),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int head = it.get_group_linear_id() % QH;
                const int query = it.get_group_linear_id() / QH;
                const int d = it.get_local_linear_id();
                const int kvh = head / G, local_head = head % G;
                const size_t base = (size_t(query) * KVH + kvh) * chunks * G + local_head;
                float M = -FLT_MAX;
                for (int c = 0; c < chunks; ++c) M = sycl::fmax(M, part_m[base + c * G]);
                float L = 0.0f, acc = 0.0f;
                for (int c = 0; c < chunks; ++c) {
                    const size_t slot = base + c * G;
                    const float m = part_m[slot];
                    if (m == -FLT_MAX) continue;
                    const float w = sycl::exp(m - M);
                    L = sycl::fma(part_l[slot], w, L);
                    acc = sycl::fma(part_acc[slot * HD + d], w, acc);
                }
                const sycl::ext::oneapi::bfloat16 y(L > 0.0f ? acc / L : 0.0f);
                output[(size_t(query) * QH + head) * HD + d] = sycl::bit_cast<uint16_t>(y);
            });
        return 0;
    } catch (const std::exception&) { return 1; }
}
