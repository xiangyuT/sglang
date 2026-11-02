// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Reuse the attributed lowbit_gu.cpp packet schedule and IQ3/Q4 dot formulas.
// Original encoded rows, eight FP32 chains and stable SGLang slot addresses.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
#include <cstdlib>
#include <string>

namespace sglang_mixed_packets1323 {
namespace es = sycl::ext::intel::esimd;
thread_local std::string error;

// IQ3_S sign fields retain eight readable bytes inside the encoded row.
// One aligned vector load replaces two halfword gathers without unaligned u32 access.
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t,32> word(const uint8_t* w,es::simd<uint32_t,32> offset) {
    auto words=es::gather<uint32_t,64,2>(reinterpret_cast<const uint32_t*>(w),offset&~3u);
    es::simd<uint32_t,32> a=words.template select<32,1>(0),b=words.template select<32,1>(32),shift=(offset&2u)<<3;
    return (a>>shift)|((b<<16)<<(16u-shift));
}

// Two adjacent 32-bit words at a 2-byte-aligned source. Reuse the middle
// aligned dword and load the carry only for halfword-offset lanes. The mask
// prevents a read past the last IQ4_NL row when both requested words are aligned.
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t,64> word_pair(const uint8_t* w, es::simd<uint32_t,32> offset) {
    es::simd<uint32_t,32> aligned=offset&~3u,shift=(offset&2u)<<3;
    auto words=es::gather<uint32_t,64,2>(reinterpret_cast<const uint32_t*>(w),aligned);
    es::simd<uint32_t,32> a=words.template select<32,1>(0),b=words.template select<32,1>(32);
    auto carry=es::gather<uint32_t,32>(reinterpret_cast<const uint32_t*>(w),aligned+8u,shift!=0u);
    es::simd<uint32_t,64> result;
    result.template select<32,1>(0)=(a>>shift)|((b<<16)<<(16u-shift));
    result.template select<32,1>(32)=(b>>shift)|((carry<<16)<<(16u-shift));
    return result;
}

SYCL_ESIMD_FUNCTION inline es::simd<int32_t, 32> sign_word(
    es::simd<uint32_t, 32> value, es::simd<uint32_t, 32> signs) {
    es::simd<uint32_t, 32> bit = ((signs & 15u) * 0x00204081u) & 0x01010101u;
    return (value ^ (bit * 255u)) + bit;
}

SYCL_ESIMD_FUNCTION inline es::simd<uint32_t, 32> iq4_lookup(
    es::simd<uint32_t, 32> q) {
    constexpr uint32_t table[] = {0x81818181u, 0x98989898u, 0xadadadadu, 0xbfbfbfbfu,
        0xcfcfcfcfu, 0xddddddddu, 0xeaeaeaeau, 0xf6f6f6f6u, 0x01010101u,
        0x0d0d0d0du, 0x19191919u, 0x26262626u, 0x35353535u, 0x45454545u,
        0x59595959u, 0x71717171u};
    constexpr auto select = (es::bfn_t::x & ~es::bfn_t::z)
                          | (es::bfn_t::y & es::bfn_t::z);
    es::simd<uint32_t, 32> masks[4], a[8], b[4], c[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        es::simd<uint32_t, 32> bit = (q >> i) & 0x01010101u;
        masks[i] = (bit << 8) - bit;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        a[i] = es::bfn<select>(es::simd<uint32_t, 32>(table[2*i]),
                              es::simd<uint32_t, 32>(table[2*i+1]), masks[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = es::bfn<select>(a[2*i], a[2*i+1], masks[1]);
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = es::bfn<select>(b[2*i], b[2*i+1], masks[2]);
    return es::bfn<select>(c[0], c[1], masks[3]);
}

template<int E, int Rows>
SYCL_ESIMD_FUNCTION inline void store(
    es::simd<float, 32> (&sum)[E], float* dst, int row, int first,
    const int32_t* destinations, int routes, const int32_t* tokens = nullptr, int input_rows = 0) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
        es::simd<float, 4> p[8];
#pragma unroll
        for (int lane = 0; lane < 8; ++lane)
            p[lane] = sum[e].template select<4, 8>(lane);
        // Match the original lane8 XOR tree, including FP32 rounding order.
        auto v = ((p[0] + p[4]) + (p[2] + p[6]))
               + ((p[1] + p[5]) + (p[3] + p[7]));
        if constexpr(Rows == 640) {
            if (tokens && (tokens[first+e] < 0 || tokens[first+e] >= input_rows))
                v = std::numeric_limits<float>::quiet_NaN();
        }
        const int destination = destinations[first + e];
        if (destination >= 0 && destination < routes)
            es::block_store<float, 4>(dst + size_t(destination) * Rows + row, v);
    }
}

template<int E, bool GridSLM, bool GuSoA = false>
SYCL_ESIMD_FUNCTION inline void iq3_packet(
    const uint8_t* w, int row, int first, const int32_t* tokens,
    const int32_t* destinations, const block_q8_1* x, float* dst, int routes, int input_rows) {
#pragma clang fp contract(off)
    es::simd<uint32_t, 32> rows, lane;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        rows.template select<8, 1>(r*8) = uint32_t(row + r);
        lane.template select<8, 1>(r*8) = es::simd<uint32_t, 8>(0, 1);
    }
    es::simd<float, 32> acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
#pragma unroll 1
    for (int kb = 0; kb < 10; ++kb) {
        es::simd<uint32_t,32> qs0,qs1,qh,signs;
        es::simd<float,32> dw;
        es::simd<int32_t,32> odd;
        if constexpr(GuSoA) {
        const auto* base=w+size_t(row/4)*4400;
        qs0=es::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(base+1840+kb*256));
        qs1=es::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(base+1968+kb*256));
        qh=es::block_load<uint8_t,32>(base+240+kb*32);
        signs=es::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(base+560+kb*128));
        auto scales=es::block_load<uint32_t,4>(reinterpret_cast<const uint32_t*>(base+80+kb*16));
        es::simd<uint32_t,32> scale;
        auto db=es::gather<uint16_t,4>(reinterpret_cast<const uint16_t*>(base),es::simd<uint32_t,4>(uint32_t(kb*8),2));
        es::simd<float,4> dh=db.template bit_cast_view<sycl::half>().read();
#pragma unroll
        for(int r=0;r<4;++r) {
            const uint32_t packed_scale=scales[r]; const float d=dh[r];
            scale.template select<8,1>(r*8)=es::simd<uint32_t,8>(packed_scale)>>(es::simd<uint32_t,8>(0,1)/2u*8u);
            dw.template select<8,1>(r*8)=d;
        }
        odd=1u+2u*((scale>>(4u*(lane&1u)))&15u);
        } else {
        auto off = rows * 1100u + uint32_t(kb * 110);
        auto pair = word_pair(w,off+2u+lane*8u);
        qs0=pair.template select<32,1>(0);qs1=pair.template select<32,1>(32);
        qh = es::gather<uint8_t, 32>(w, off + 66u + lane);
        signs = word(w, off + 74u + lane*4u);
        es::simd<uint32_t, 32> scale = es::gather<uint8_t, 32>(w, off + 106u + lane/2u);
        odd = 1u + 2u*((scale >> (4u*(lane & 1u))) & 15u);
        auto db = es::gather<uint16_t, 32>(reinterpret_cast<const uint16_t*>(w), off);
        dw = db.template bit_cast_view<sycl::half>().read();
        }
        es::simd<int32_t, 32> codes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            es::simd<uint32_t, 32> index = (((j < 4 ? qs0 : qs1) >> (8*(j%4))) & 255u)
                                       | ((qh << (8-j)) & 256u);
            es::simd<uint32_t,32> grid;
            if constexpr(GridSLM) grid=es::slm_gather<uint32_t,32>(index*4u);
            else grid=es::gather<uint32_t,32>(iq3s_grid,index*4u);
            codes[j] = sign_word(grid, signs >> (4*j));
        }
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int token = tokens[first+e];
            const auto* raw = reinterpret_cast<const uint8_t*>(x + (token >= 0 && token < input_rows ? token : 0)*80 + kb*8);
            es::simd<uint32_t, 8> offsets(0, 36);
            auto words = es::gather<int32_t, 64, 8>(reinterpret_cast<const int32_t*>(raw), offsets+4u);
            auto dsb = es::gather<uint16_t, 8>(reinterpret_cast<const uint16_t*>(raw), offsets);
            es::simd<float, 8> ds8 = dsb.template bit_cast_view<sycl::half>().read();
            es::simd<float, 32> ds;
            es::simd<int32_t, 32> dot = 0;
#pragma unroll
            for (int r = 0; r < 4; ++r) ds.template select<8, 1>(r*8) = ds8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                es::simd<int32_t, 32> a;
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    a.template select<8, 1>(r*8) = words.template select<8, 1>(j*8);
                dot = es::dp4a<int32_t>(dot, codes[j], a);
            }
            dot *= odd;
            es::simd<float, 32> f = dot, factor = dw*ds, term = factor*f;
            // Match existing packet kernels: retain a row predicate between
            // multiplication and addition. IGC otherwise contracts ESIMD ops
            // into MAD despite the front-end no-contraction option.
            term.merge(0.f, rows >= 640u);
            acc[e] = acc[e] + term;
        }
    }
    store<E, 640>(acc, dst, row, first, destinations, routes, tokens, input_rows);
}

template<int E, bool SoA = false>
SYCL_ESIMD_FUNCTION inline void iq4_down_packet(
    const uint8_t* w, int row, int first, const int32_t* destinations,
    const block_q8_1* x, float* dst, int routes) {
#pragma clang fp contract(off)
    es::simd<uint32_t, 32> rows, lane;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        rows.template select<8, 1>(r*8) = uint32_t(row + r);
        lane.template select<8, 1>(r*8) = es::simd<uint32_t, 8>(0, 1);
    }
    es::simd<float, 32> acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
#pragma unroll 1
    for (int step = 0; step < 5; ++step) {
        es::simd<uint32_t,32> q0, q1;
        es::simd<float,32> dw;
        if constexpr (SoA) {
            const auto* base = w + size_t(row/4)*1440 + step*288;
            q0 = es::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(base+32));
            q1 = es::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(base+160));
            auto header = es::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(base));
            es::simd<uint16_t,32> db;
            db.template select<16,2>(0) = header;
            db.template select<16,2>(1) = header;
            dw = db.template bit_cast_view<sycl::half>().read();
        } else {
            auto block = lane/2u + uint32_t(step*4);
            auto off = rows*360u + block*18u;
            auto pair = word_pair(w,off+2u+(lane&1u)*8u);
            q0 = pair.template select<32,1>(0);
            q1 = pair.template select<32,1>(32);
            auto db = es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(w),off);
            dw = db.template bit_cast_view<sycl::half>().read();
        }
        es::simd<int32_t, 32> codes[4];
        codes[0] = iq4_lookup(q0 & 0x0f0f0f0fu);
        codes[1] = iq4_lookup((q0 >> 4) & 0x0f0f0f0fu);
        codes[2] = iq4_lookup(q1 & 0x0f0f0f0fu);
        codes[3] = iq4_lookup((q1 >> 4) & 0x0f0f0f0fu);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            es::simd<uint32_t, 8> p(0, 1);
            es::simd<uint32_t, 8> aoff = (p/2u+uint32_t(step*4))*36u;
            const int destination = destinations[first+e];
            const auto* raw = reinterpret_cast<const uint8_t*>(x + (destination >= 0 && destination < routes ? destination : 0)*20);
            auto dsb = es::gather<uint16_t, 8>(reinterpret_cast<const uint16_t*>(raw), aoff);
            es::simd<float, 8> ds8 = dsb.template bit_cast_view<sycl::half>().read();
            es::simd<float, 32> ds;
            es::simd<int32_t, 32> dot = 0;
#pragma unroll
            for (int r = 0; r < 4; ++r) ds.template select<8, 1>(r*8) = ds8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word_index = (j/2) + (j%2)*4;
                auto av = es::gather<int32_t, 8>(reinterpret_cast<const int32_t*>(raw),
                    aoff + 4u + ((p&1u)*2u+uint32_t(word_index))*4u);
                es::simd<int32_t, 32> a;
#pragma unroll
                for (int r = 0; r < 4; ++r) a.template select<8, 1>(r*8) = av;
                dot = es::dp4a<int32_t>(dot, codes[j], a);
            }
            es::simd<float, 32> f = dot, factor = dw*ds, term = factor*f;
            term.merge(0.f, rows >= 2560u);
            acc[e] = acc[e] + term;
        }
    }
    store<E, 2560>(acc, dst, row, first, destinations, routes);
}

template<bool Down, bool GridSLM, bool DownSoA, bool GuSoA, int Reuse> class PacketKernel;
template<bool Down, bool GridSLM, bool DownSoA, bool GuSoA, int Reuse>
void launch_impl(sycl::queue& q, const uint64_t* addresses, const block_q8_1* x,
    const int32_t* slots, const int32_t* starts, const int32_t* tokens,
    const int32_t* destinations, float* g, float* u,
    int groups, int routes, int capacity, int input_rows) {
    constexpr int Output = Down ? 2560 : 1280;
    q.parallel_for<PacketKernel<Down,GridSLM,DownSoA,GuSoA,Reuse>>(
        sycl::nd_range<2>(sycl::range<2>(groups, Output/4), sycl::range<2>(1, 4)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int group = it.get_group(0), slot = slots[group];
            const int first = starts[group], end = starts[group+1];
            if (first < 0 || end <= first || end > routes) return;
            const int v = it.get_group(1)*16 + it.get_local_id(1)*4;
            const bool high = !Down && v >= 640;
            const int row = high ? v-640 : v;
            float* dst = high ? u : g;
            const uint64_t address = slot >= 0 && slot < capacity
                ? addresses[int64_t(slot)*3 + (Down ? 2 : int(high))] : 0;
            if (!address) {
                for (int e = first; e < end; ++e) {
                    const int destination = destinations[e];
                    if (destination >= 0 && destination < routes) {
                        es::simd<float, 4> value = std::numeric_limits<float>::quiet_NaN();
                        es::block_store<float, 4>(dst + size_t(destination) * (Down ? 2560 : 640) + row, value);
                    }
                }
                return;
            }
            if constexpr(!Down && GridSLM) {
                es::slm_init<2048>();
                const int local=it.get_local_id(1);
#pragma unroll
                for(int packet=0;packet<4;++packet) {
                    es::simd<uint32_t,32> offsets(uint32_t(local*512+packet*128),4);
                    auto table=es::gather<uint32_t,32>(iq3s_grid,offsets);
                    es::slm_block_store<uint32_t,32>(local*512+packet*128,table);
                }
                es::barrier();
            }
            const auto* w = reinterpret_cast<const uint8_t*>(address);
            int e = first;
#define CALL(E) if constexpr(Down) iq4_down_packet<E,DownSoA>(w,row,e,destinations,x,dst,routes); else iq3_packet<E,GridSLM,GuSoA>(w,row,e,tokens,destinations,x,dst,routes,input_rows)
            if constexpr(Reuse == 8) for (; end-e >= 8; e += 8) { CALL(8); }
            for (; end-e >= 4; e += 4) { CALL(4); }
            if constexpr(Reuse != 0) if (end-e == 3) { CALL(3); e += 3; }
            if (end-e >= 2) { CALL(2); e += 2; }
            for (; e < end; ++e) { CALL(1); }
#undef CALL
        });
}


template<bool Down, bool GridSLM, bool DownSoA = false, bool GuSoA = false>
void launch(sycl::queue& q, const uint64_t* addresses, const block_q8_1* x,
    const int32_t* slots, const int32_t* starts, const int32_t* tokens,
    const int32_t* destinations, float* g, float* u,
    int groups, int routes, int capacity, int input_rows) {
    const char* value = std::getenv("SGLANG_XPU_GGUF_PACKET_REUSE");
    const int reuse = value ? std::atoi(value) : 0;
#define LAUNCH(R) launch_impl<Down,GridSLM,DownSoA,GuSoA,R>(q,addresses,x,slots,starts,tokens,destinations,g,u,groups,routes,capacity,input_rows)
    if(reuse==8) { LAUNCH(8); }
    else if(reuse==3) { LAUNCH(3); }
    else { LAUNCH(0); }
#undef LAUNCH
}

extern "C" int xpu_gguf_mixed_packets_soa(void* queue, const uint64_t* addresses,
    const void* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int kind, int groups, int routes, int capacity, int input_rows) {
    if (!queue || !addresses || !x || !slots || !starts || !destinations || !g
        || kind != 20 || groups < 1 || groups > routes || routes < 1 || routes > 320
        || input_rows < 1 || input_rows > 32 || routes != input_rows*10
        || capacity < 1 || capacity > 512) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        sglang_mixed_packets1323::launch<true,false,true>(q,addresses,
            static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,
            g,u,groups,routes,capacity,input_rows);
        return 0;
    } catch (const std::exception& e) {
        sglang_mixed_packets1323::error = e.what(); return 1;
    }
}
}

extern "C" const char* xpu_gguf_mixed_packets_error() { return sglang_mixed_packets1323::error.c_str(); }
extern "C" int xpu_gguf_mixed_packets(void* queue, const uint64_t* addresses,
    const void* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int kind, int groups, int routes, int capacity, int input_rows) {
    if (!queue || !addresses || !x || !slots || !starts || !destinations || !g
        || groups < 1 || groups > routes || routes < 1 || routes > 320
        || input_rows < 1 || input_rows > 32 || routes != input_rows*10
        || capacity < 1 || capacity > 512 || (kind != 21 && kind != 20)
        || (kind == 21 && (!tokens || !u))) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        const char* setting=std::getenv("SGLANG_XPU_GGUF_IQ3_GRID_SLM");
        const bool grid_slm=setting && std::string(setting)=="1";
        if (kind==21 && grid_slm) sglang_mixed_packets1323::launch<false,true>(q,addresses,
            static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,
            g,u,groups,routes,capacity,input_rows);
        else if (kind == 21) sglang_mixed_packets1323::launch<false,false>(q, addresses,
            static_cast<const block_q8_1*>(x), slots, starts, tokens, destinations,
            g, u, groups, routes, capacity, input_rows);
        else sglang_mixed_packets1323::launch<true,false>(q, addresses,
            static_cast<const block_q8_1*>(x), slots, starts, tokens, destinations,
            g, u, groups, routes, capacity, input_rows);
        return 0;
    } catch (const std::exception& e) {
        sglang_mixed_packets1323::error = e.what(); return 1;
    }
}

extern "C" int xpu_gguf_mixed_packets_gu_soa(void* queue, const uint64_t* addresses,
    const void* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int kind, int groups, int routes, int capacity, int input_rows) {
    if (!queue || !addresses || !x || !slots || !starts || !tokens || !destinations || !g || !u
        || kind != 21 || groups < 1 || groups > routes || routes < 1 || routes > 320
        || input_rows < 1 || input_rows > 32 || routes != input_rows*10
        || capacity < 1 || capacity > 512) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.is_in_order()) return 2;
        const char* setting=std::getenv("SGLANG_XPU_GGUF_IQ3_GRID_SLM");
        if (setting && std::string(setting)=="1") sglang_mixed_packets1323::launch<false,true,false,true>(q,addresses,
            static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,g,u,groups,routes,capacity,input_rows);
        else sglang_mixed_packets1323::launch<false,false,false,true>(q,addresses,
            static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,g,u,groups,routes,capacity,input_rows);
        return 0;
    } catch (const std::exception& e) { sglang_mixed_packets1323::error=e.what(); return 1; }
}

// Q8_0 expert-down: same AoS bytes and eight FP32 lane chains.
namespace sglang_mixed_packets1323 {
template<int E>
SYCL_ESIMD_FUNCTION inline void q8_down_packet(const uint8_t* w,int row,int first,
 const int32_t* destinations,const block_q8_1* x,float* out,int routes) {
#pragma clang fp contract(off)
 es::simd<uint32_t,32> rows,lane;
#pragma unroll
 for(int r=0;r<4;++r) {
  rows.template select<8,1>(r*8)=uint32_t(row+r);
  lane.template select<8,1>(r*8)=es::simd<uint32_t,8>(0,1);
 }
 es::simd<float,32> acc[E];
#pragma unroll
 for(int e=0;e<E;++e)acc[e]=0.f;
#pragma unroll 1
 for(int step=0;step<10;++step) {
  auto block=lane/4u+uint32_t(step*2);
  auto off=rows*680u+block*34u;
  auto pair=word_pair(w,off+2u+(lane%4u)*8u);
  es::simd<int32_t,32> q0=pair.template select<32,1>(0),q1=pair.template select<32,1>(32);
  auto db=es::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(w),off);
  es::simd<float,32> dw=db.template bit_cast_view<sycl::half>().read();
#pragma unroll
  for(int e=0;e<E;++e) {
   const int dest=destinations[first+e];
   const auto* raw=reinterpret_cast<const uint8_t*>(x+(dest>=0&&dest<routes?dest:0)*20);
   es::simd<uint32_t,8> p(0,1);
   auto aoff=(p/4u+uint32_t(step*2))*36u;
   auto words=es::gather<int32_t,16,2>(reinterpret_cast<const int32_t*>(raw),aoff+4u+(p%4u)*8u);
   auto db=es::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(raw),aoff);
   es::simd<float,8> ds8=db.template bit_cast_view<sycl::half>().read();
   es::simd<float,32> ds;es::simd<int32_t,32> a0,a1;
#pragma unroll
   for(int r=0;r<4;++r) {
    ds.template select<8,1>(r*8)=ds8;
    a0.template select<8,1>(r*8)=words.template select<8,1>(0);
    a1.template select<8,1>(r*8)=words.template select<8,1>(8);
   }
   auto dot=es::dp4a<int32_t>(es::simd<int32_t,32>(0),q0,a0);
   dot=es::dp4a<int32_t>(dot,q1,a1);
   es::simd<float,32> f=dot,factor=dw*ds,term=factor*f;
   term.merge(0.f,rows>=2560u);acc[e]=acc[e]+term;
  }
 }
 store<E,2560>(acc,out,row,first,destinations,routes);
}
template<int Reuse> class Q8DownPacket;
template<int Reuse>
void launch_q8_down_packet_impl(sycl::queue& q,const uint64_t* addresses,const block_q8_1* x,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,
 float* out,int groups,int routes,int capacity) {
 q.parallel_for<Q8DownPacket<Reuse>>(sycl::nd_range<2>(sycl::range<2>(groups,640),sycl::range<2>(1,4)),
 [=](sycl::nd_item<2> it)[[intel::sycl_explicit_simd]] {
  const int group=it.get_group(0),slot=slots[group],first=starts[group],end=starts[group+1];
  if(first<0||end<=first||end>routes)return;
  const int row=it.get_group(1)*16+it.get_local_id(1)*4;
  const uint64_t base=slot>=0&&slot<capacity?addresses[int64_t(slot)*3+2]:0;
  if(!base) {
   for(int e=first;e<end;++e) {
    int d=destinations[e];
    if(d>=0&&d<routes)es::block_store<float,4>(out+size_t(d)*2560+row,es::simd<float,4>(std::numeric_limits<float>::quiet_NaN()));
   }
   return;
  }
  const auto* w=reinterpret_cast<const uint8_t*>(base);int e=first;
  if constexpr(Reuse == 8) for(;end-e>=8;e+=8)q8_down_packet<8>(w,row,e,destinations,x,out,routes);
  for(;end-e>=4;e+=4)q8_down_packet<4>(w,row,e,destinations,x,out,routes);
  if constexpr(Reuse != 0) if(end-e==3){q8_down_packet<3>(w,row,e,destinations,x,out,routes);e+=3;}
  if(end-e>=2){q8_down_packet<2>(w,row,e,destinations,x,out,routes);e+=2;}
  for(;e<end;++e)q8_down_packet<1>(w,row,e,destinations,x,out,routes);
 });
}

void launch_q8_down_packet(sycl::queue& q,const uint64_t* addresses,const block_q8_1* x,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,
 float* out,int groups,int routes,int capacity) {
 const char* value=std::getenv("SGLANG_XPU_GGUF_PACKET_REUSE");
 const int reuse=value?std::atoi(value):0;
#define LAUNCH(R) launch_q8_down_packet_impl<R>(q,addresses,x,slots,starts,destinations,out,groups,routes,capacity)
 if(reuse==8){LAUNCH(8);}
 else if(reuse==3){LAUNCH(3);}
 else {LAUNCH(0);}
#undef LAUNCH
}

}
extern "C" int xpu_gguf_q8_down_packets(void* queue,const uint64_t* addresses,const void* x,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,float* out,
 int groups,int routes,int capacity) {
 if(!queue||!addresses||!x||!slots||!starts||!destinations||!out||groups<1||
 groups>routes||routes<1||routes>320||capacity<1||capacity>512)return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  if(!q.has_property<sycl::property::queue::in_order>())return 2;
  sglang_mixed_packets1323::launch_q8_down_packet(q,addresses,(const block_q8_1*)x,slots,starts,destinations,out,groups,routes,capacity);
  return 0;
 }catch(const std::exception& e){sglang_mixed_packets1323::error=e.what();return 1;}
}
