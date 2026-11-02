// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Ported from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd,
// sycl/src/kernels/cuda/iq_kernels.dp.cpp (dequant functions).
// Original quantization formulas/tables: ggml, see GGML-LICENSE.
// Strata permission notice: STRATA-LICENSE.
#pragma once
#include <sycl/sycl.hpp>
#include <type_traits>
#include <cstring>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
namespace sglang_gguf {
template <typename dst_t> inline dst_t cvt(float v);
template <> inline float cvt<float>(float v) { return v; }
template <>
inline sycl::half cvt<sycl::half>(float v) {
    return sycl::vec<float, 1>(v)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

// SYCL port: the dequant functions below wrote one 2-byte value at a time (dq_* = llama.cpp's per-thread loops); on Xe
// the FP16 expert dequant ran at ~80 GB/s of writes, 40% of the prompt path. Each thread's run of values goes out as one
// vector store instead (the offsets are multiples of the run length, so the stores are aligned).
template <typename dst_t, int N>
inline void store_run(dst_t* y, const float* v) {
    if constexpr (std::is_same_v<dst_t, sycl::half> && (N == 4 || N == 8)) {
        sycl::vec<sycl::half, N> h;
#pragma unroll
        for (int j = 0; j < N; ++j) h[j] = sycl::half(v[j]);
        *reinterpret_cast<sycl::vec<sycl::half, N>*>(y) = h;
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) y[j] = cvt<dst_t>(v[j]);
    }
}
template <typename dst_t>
inline void dq_iq2_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_xxs_metadata(const void *vx, int64_t ibs, dst_t *yy, int tid, sycl::sub_group sg) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t lane = sg.get_local_linear_id();
    uint32_t aux32 = 0;
    if((lane & 3u) == 0u) aux32 = q2[2] | (q2[3] << 16);
    aux32 = sycl::select_from_group(sg, aux32, lane & ~3u);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq2_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_xxs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq3_s(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    float v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        v[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
    store_run<dst_t, 8>(y, v);
}
template <typename dst_t>
inline void dq_iq1_m(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template <typename dst_t>
inline void dq_iq4_nl(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    // SYCL port: thread (ib, il) writes the block's values [8 il, 8 il + 8) as one run: il < 2 the low nibbles of
    // qs[8 il ..], il >= 2 the high nibbles of qs[8 (il - 2) ..] (the same values as llama.cpp's 4 + 4 split)
    const int64_t il = tid % 4, ib = tid / 4;   // consecutive lanes write consecutive 16-byte runs (coalesced stores)
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q4 = x[ib].qs + 8 * (il & 1);
    const float d = (float) x[ib].d;
    float v[8];
    if (il < 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] & 0xf];
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 8>(y, v);
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template <typename dst_t>
inline void dq_q3_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
#pragma unroll
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(
            dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template <typename dst_t>
inline void dq_iq4_xs(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    float lo[4], hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = d * kvalues_iq4nl[q4[j] & 0xf];
        hi[j] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    store_run<dst_t, 4>(y, lo);
    store_run<dst_t, 4>(y + 16, hi);
}
template <typename dst_t>
inline void dq_q2_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        v[j] = d * (float) (code - 1);
    }
    store_run<dst_t, 8>(yy + b * 64 + part * 8, v);
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid / 4; it was block
// tid % 8, which scattered a sub-group's 2-byte stores 64 bytes apart: B70 Q5_1 / Q8_0 at 88-93 GB/s of FP16 against 250+).
inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d,
                                      uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template <typename dst_t>
inline void dq_q4_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = x[ibs].dm[0];
    const float dmin = x[ibs].dm[1];
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
#pragma unroll
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_k(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = x[ibs].dm[0];
        const float dmin = x[ibs].dm[1];
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template <typename dst_t>
inline void dq_q5_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        v0[j] = ((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d;
        v1[j] = ((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d;
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q4_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_0* x = (const block_q4_0*) vx + ibs * (QK_K / QK4_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_0 for value pairs iqs, iqs + 16
        v0[j] = ((float) (x[ib].qs[iqs] & 0xf) - 8.0f) * d;
        v1[j] = ((float) (x[ib].qs[iqs] >> 4) - 8.0f) * d;
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q4_1(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q4_1* x = (const block_q4_1*) vx + ibs * (QK_K / QK4_1);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const sycl::float2 dm =
        (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_1 for value pairs iqs, iqs + 16
        v0[j] = (float)(x[ib].qs[iqs] & 0xf) * dm.x() + dm.y();
        v1[j] = (float)(x[ib].qs[iqs] >> 4) * dm.x() + dm.y();
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template <typename dst_t>
inline void dq_q5_1(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const sycl::float2 dm =
        (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    float v0[4], v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        v0[j] = (float)((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x() + dm.y();
        v1[j] = (float)((x[ib].qs[iqs] >> 4) | xh_1) * dm.x() + dm.y();
    }
    store_run<dst_t, 4>(y + 4 * il, v0);
    store_run<dst_t, 4>(y + 16 + 4 * il, v1);
}
template<typename dst_t>
void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // llama.cpp's dequantize_q6_K (dequantize.cuh, written for 64 threads) folded onto the kernel's 32: each of the
    // 64 thread roles runs at tid and tid + 32.
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d =
            sycl::vec<sycl::half, 1>(x[ibs].d)
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]  = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template <typename dst_t>
inline void dq_q8_0(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid / 4, il = tid % 4;   // lanes 4b..4b+3 write block b: neighbouring lanes, neighbouring values
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    dst_t* y = yy + 32 * ib + 8 * il;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (float)x[ib].qs[8 * il + j] * d;
    store_run<dst_t, 8>(y, v);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template <typename dst_t>
inline void dq_bf16(const void *vx, int64_t ibs, dst_t *yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] =
        cvt<dst_t>(sycl::bit_cast<float>((uint32_t)x[j] << 16));
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template <typename dst_t>
inline void
dq_dispatch(int ty, const void *vx, int64_t ibs, dst_t *y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid);
            break;
        case 17: dq_iq2_xs(vx, ibs, y, tid);
            break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid);
            break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 14: dq_q6_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 2: dq_q4_0(vx, ibs, y, tid); break;
        case 3: dq_q4_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        default: break;
    }
}


}
