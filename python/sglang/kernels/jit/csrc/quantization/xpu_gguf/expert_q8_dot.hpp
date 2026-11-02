// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Ported from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd:
// iq_kernels.dp.cpp dot products, lane8 expert scheduling and q2_0_decode.hpp.
// Original compressed bytes, Q8_1 activations, FP32 row outputs.
// See STRATA-LICENSE and GGML-LICENSE. Optional GGUF decode arithmetic.
#include "dequant.hpp"
#include <string>
#include <limits>
namespace sglang_expert_q8 {
thread_local std::string error;
inline int get_int_b2(const void* x,int i) {const auto* p=(const uint16_t*)x;return int(uint32_t(p[2*i])|(uint32_t(p[2*i+1])<<16));}
inline int get_int_b4(const void* x,int i) {return ((const int*)x)[i];}
inline uint32_t unpack_ksigns(uint8_t v){return uint32_t(v^((sycl::popcount(v)&1)<<7))*0x01010101u;}
inline int swar_ne4(unsigned x){x=(x|(x>>4))&0x0f0f0f0fu;x=(x|(x>>2))&0x03030303u;x=(x|(x>>1))&0x01010101u;return int(x*255u);}
inline int swar_sub4(unsigned a,unsigned s){return int(((a&0x7f7f7f7fu)+(s&0x01010101u))^(a&0x80808080u));}
inline int dp4a(int a,int b,int c) {
#pragma unroll
 for(int i=0;i<4;++i)c+=int(int8_t(uint32_t(a)>>(8*i)))*int(int8_t(uint32_t(b)>>(8*i)));
 return c;
}
#define ggml_cuda_dp4a(a,b,c) dp4a(a,b,c)
inline float vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                                           const block_q8_1 *__restrict__ bq8_1,
                                           const int &kbx, const int &iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid0 = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid1 = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

inline float vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = swar_sub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}

inline float vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                                         const block_q8_1 *__restrict__ bq8_1,
                                         const int &kbx, const int &iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = sycl::vec<sycl::half, 1>(scale.f16)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs].ds[0];
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

// Reuse the established mixed dense dot formulas, retaining lane8 order.
inline float dot_iq3(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_iq3_s*>(row)+block;
    const sycl::int2 packed(get_int_b2(w->qs,part),get_int_b2(w->qs,part+1));
    const auto* qs=reinterpret_cast<const uint8_t*>(&packed);
    const int qh=w->qh[part/2],signs32=get_int_b2(w->signs,part/2);
    const auto* signs=reinterpret_cast<const uint8_t*>(&signs32);
    int sum=0;
#pragma unroll
    for(int l=0;l<8;l+=2) {
        const int grid0=iq3s_grid[qs[l]|((qh<<(8-l))&256)];
        const int grid1=iq3s_grid[qs[l+1]|((qh<<(7-l))&256)];
        const int s0=swar_ne4(((signs[l/2]&3)<<7)|((signs[l/2]&12)<<21));
        const int s1=swar_ne4(((signs[l/2]&48)<<3)|((signs[l/2]&192)<<17));
        sum=dp4a(swar_sub4(grid0^s0,s0),get_int_b4(a[part/2].qs,l),sum);
        sum=dp4a(swar_sub4(grid1^s1,s1),get_int_b4(a[part/2].qs,l+1),sum);
    }
    sum*=1+2*((w->scales[part/4]>>((part<<1)&4))&15);
    const float d=float(w->d)*float(a[part/2].ds[0]);
    return d*sum;
}
inline float dot_iq4nl(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_iq4_nl*>(row)+block;
    int sum=0;
#pragma unroll
    for(int l=0;l<2;++l) {
        const uint32_t codes=get_int_b2(w->qs,part+l);
        uint32_t lo=0,hi=0;
#pragma unroll
        for(int j=0;j<4;++j) {
            lo|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*j))&15]))<<(8*j);
            hi|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*j+4))&15]))<<(8*j);
        }
        sum=dp4a(int(lo),get_int_b4(a->qs,part+l),sum);
        sum=dp4a(int(hi),get_int_b4(a->qs,part+l+4),sum);
    }
    const float d=float(w->d)*float(a->ds[0]);
    return d*sum;
}
inline float dot_q8(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_q8_0*>(row)+block;
    int sum=0;
#pragma unroll
    for(int i=0;i<2;++i)sum=dp4a(get_int_b2(w->qs,part+i),get_int_b4(a->qs,part+i),sum);
    return float(w->d)*float(a->ds[0])*float(sum);
}
// IQ4_XS uses one 32-value sub-block per call, part=4*subblock.
inline float dot_iq4xs(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_iq4_xs*>(row)+block;
    int sum=0;
#pragma unroll
    for(int j=0;j<4;++j) {
        const uint32_t codes=get_int_b4(w->qs,part+j);
        uint32_t lo=0,hi=0;
#pragma unroll
        for(int k=0;k<4;++k) {
            lo|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*k))&15]))<<(8*k);
            hi|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*k+4))&15]))<<(8*k);
        }
        sum=dp4a(int(lo),get_int_b4(a[part/4].qs,j),sum);
        sum=dp4a(int(hi),get_int_b4(a[part/4].qs,j+4),sum);
    }
    const int scale=((w->scales_l[part/8]>>(part&4))&15)|(((w->scales_h>>(part/2))&3)<<4);
    sum*=scale-32;
    const float d=float(w->d)*float(a[part/4].ds[0]);
    return d*sum;
}
}
