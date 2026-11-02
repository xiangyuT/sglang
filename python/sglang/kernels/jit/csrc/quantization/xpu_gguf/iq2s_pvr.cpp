// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Adapted from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd,
// sycl/src/kernels/cuda/iq_kernels.dp.cpp (grid_s4_basis9_packet).
// Original grid: ggml authors; see STRATA-LICENSE and GGML-LICENSE.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>

namespace sglang_gguf { extern thread_local std::string error; }
namespace sglang_iq2s_pvr {
namespace pvr = sycl::ext::intel::esimd;
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81Block) == 36 && offsetof(Q81Block, qs) == 4);
#include "iq2s_s4_grid.inc"

template<int E>
SYCL_ESIMD_FUNCTION inline void grid_s4_basis9_packet(
 const uint8_t* weights,int row,int first,const int32_t* tokens,const int32_t* destinations,
 const Q81Block* xq,float* dst,int row_limit,int input_rows,int routes){
#pragma clang fp contract(off)
 pvr::simd<uint32_t,16> rows(row,1);auto valid=rows<uint32_t(row_limit);
 pvr::simd<float,16> sum[8][E];
#pragma unroll
 for(int p=0;p<8;++p){
#pragma unroll
  for(int e=0;e<E;++e)sum[p][e]=0.f;
 }
 const auto* tile=weights+size_t(row/16)*13120;
 for(int kb=0;kb<10;++kb){
  const auto* block=tile+kb*1312;
  auto bits=pvr::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block+1280));
  pvr::simd<float,16> d=bits.template bit_cast_view<sycl::half>().read();
#pragma unroll
  for(int p=0;p<8;++p){
   const auto* plane=block+p*160;
   auto qs=pvr::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(plane));
   auto signs=pvr::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(plane+64));
   auto aux=pvr::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(plane+128));
   pvr::simd<uint32_t,16> qh=aux&255u,scale=aux>>8;
   pvr::simd<uint32_t,64> indices,signs_all;
#pragma unroll
   for(int j=0;j<4;++j){
    indices.template select<16,1>(j*16)=((qs>>(8*j))&255u)|((qh<<(8-2*j))&0x300u);
    signs_all.template select<16,1>(j*16)=signs>>(8*j);
   }
   auto packed=pvr::slm_gather<uint32_t,64>(indices*4u);
   pvr::simd<uint32_t,64> residual=0x11111111u+(((packed>>1)|(packed>>2))&0x11111111u);
   pvr::simd<uint32_t,64> bits=signs_all&255u;
   bits=(bits|(bits<<12))&0x000f000fu;
   bits=(bits|(bits<<6))&0x03030303u;
   bits=(bits|(bits<<3))&0x11111111u;
   pvr::simd<uint32_t,64> mask=bits*15u;
   pvr::simd<uint32_t,64> coarse=(packed^mask)+bits;
   residual=(residual^mask)+bits;
   pvr::simd<uint16_t,128> Bc=coarse.template bit_cast_view<uint16_t>().read();
   pvr::simd<uint16_t,128> Br=residual.template bit_cast_view<uint16_t>().read();
   pvr::simd<int32_t,16> a=scale&15u,b=scale>>4;
   pvr::simd<int32_t,E*16> aw=0;
   pvr::simd<float,E> activation_scale;
#pragma unroll
   for(int e=0;e<E;++e){
    const int token=tokens[first+e];
    const auto* raw=reinterpret_cast<const uint8_t*>(xq+size_t(token>=0&&token<input_rows?token:0)*80+kb*8+p);
    pvr::simd<uint32_t,8> offset(4,4);
    auto u=pvr::gather<int32_t,8>(reinterpret_cast<const int32_t*>(raw),offset);
    aw.template select<4,1>(e*16)=u.template select<4,1>(0);
    aw.template select<4,1>(e*16+12)=u.template select<4,1>(4);
    pvr::simd<uint32_t,1> zero_offset=0;
    auto dsbits=pvr::gather<uint16_t,1>(reinterpret_cast<const uint16_t*>(raw),zero_offset);
    pvr::simd<float,1> ds=dsbits.template bit_cast_view<sycl::half>().read();
    activation_scale[e]=ds[0];
   }
   pvr::simd<int8_t,E*64> A=aw.template bit_cast_view<int8_t>();
   pvr::simd<int32_t,E*32> zero=0;
   using P=pvr::xmx::dpas_argument_type;
   auto Cc=pvr::xmx::dpas<8,E*2,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Bc,A);
   auto Cr=pvr::xmx::dpas<8,E*2,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Br,A);
   auto pair_dots=Cc*9-Cr;
#pragma unroll
   for(int e=0;e<E;++e){
    pvr::simd<int32_t,16> s0=pair_dots.template select<16,1>(e*32);
    pvr::simd<int32_t,16> s1=pair_dots.template select<16,1>(e*32+16);
    pvr::simd<int32_t,16> si=(s0*a+s1*b+(s0+s1)/2)/4;
    pvr::simd<float,16> dot=si;
    const float scalar_ds=activation_scale[e];pvr::simd<float,16> factor=d*scalar_ds;
    pvr::simd<float,16> term=factor*dot;term.merge(0.f,!valid);
    sum[p][e]=sum[p][e]+term;
   }
  }
 }
#pragma unroll
 for(int e=0;e<E;++e){
  auto value=((sum[0][e]+sum[4][e])+(sum[2][e]+sum[6][e]))+
             ((sum[1][e]+sum[5][e])+(sum[3][e]+sum[7][e]));
  const int dest=destinations[first+e],token=tokens[first+e];
  if(token<0||token>=input_rows)value=std::numeric_limits<float>::quiet_NaN();
  if(dest>=0&&dest<routes)pvr::block_store<float,16>(dst+size_t(dest)*640+row,value);
 }
}



template<int WG> class IQ2SGrouped;
template<int WG>
void launch(sycl::queue& q, const uint8_t* gate, const uint8_t* up,
    const Q81Block* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int groups, int routes, int input_rows, int capacity) {
    q.parallel_for<IQ2SGrouped<WG>>(
        sycl::nd_range<2>(sycl::range<2>(groups, 80), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int group = int(it.get_group(0));
            const int slot = slots[group], first = starts[group], end = starts[group+1];
            if (first < 0 || end <= first || end > routes) return;
            const int lid = int(it.get_local_id(1));
            const int v = int(it.get_group(1)) * WG * 16 + lid * 16;
            const bool high = v >= 640;
            const int row = high ? v - 640 : v;
            float* dst = high ? u : g;
            if (slot < 0 || slot >= capacity) {
                for (int e=first; e<end; ++e) {
                    const int target=destinations[e];
                    if (target >= 0 && target < routes) {
                        pvr::simd<float,16> bad=std::numeric_limits<float>::quiet_NaN();
                        pvr::block_store<float,16>(dst+size_t(target)*640+row,bad);
                    }
                }
                return;
            }
            pvr::slm_init<4096>();
            if constexpr (WG == 16) {
                pvr::simd<uint32_t,64> offsets(lid*256,4);
                auto grid=pvr::gather<uint32_t,64>(b60_s4_grid,offsets);
                pvr::slm_block_store<uint32_t,64>(lid*256,grid);
            } else {
                pvr::simd<uint32_t,128> offsets(lid*1024,4);
                auto lo=pvr::gather<uint32_t,128>(b60_s4_grid,offsets);
                auto hi=pvr::gather<uint32_t,128>(b60_s4_grid,offsets+512u);
                pvr::slm_block_store<uint32_t,128>(lid*1024,lo);
                pvr::slm_block_store<uint32_t,128>(lid*1024+512,hi);
            }
            pvr::barrier();
            const auto* w=(high?up:gate)+size_t(slot)*524800;
            int e=first;
            for (;end-e>=4;e+=4)
                grid_s4_basis9_packet<4>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
            if (end-e>=2) {
                grid_s4_basis9_packet<2>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
                e+=2;
            }
            for (;e<end;++e)
                grid_s4_basis9_packet<1>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
        });
}

class RepackIQ2S;
// In a packed tile each 16-row block consists of eight 160-byte planes and
// 32 bytes of FP16 scales. This is a permutation of the same 16*10*82 bytes.
inline int original_byte(int packed) {
    const int kb=packed/1312, cell=packed%1312;
    if (cell >= 1280) {
        const int r=(cell-1280)/2, byte=(cell-1280)%2;
        return r*820+kb*82+byte;
    }
    const int plane=cell/160, field=cell%160;
    if (field < 64) return (field/4)*820+kb*82+2+plane*4+field%4;
    if (field < 128) return ((field-64)/4)*820+kb*82+34+plane*4+(field-64)%4;
    const int r=(field-128)/2;
    return r*820+kb*82+((field-128)%2?74:66)+plane;
}
}

extern "C" int xpu_gguf_iq2s_pack(void* queue, void* gate, void* up,
    const int32_t* slots, int count, int capacity, int inverse) {
    if (!queue || !gate || !up || !slots || count < 1 || capacity < 1 ||
        (inverse != 0 && inverse != 1)) return 2;
    try {
        using namespace sglang_iq2s_pvr;
        auto& q=*static_cast<sycl::queue*>(queue);
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint8_t,1> tile(13120,h);
            h.parallel_for<RepackIQ2S>(
                sycl::nd_range<1>(size_t(count)*2*40*256,256),
                [=](sycl::nd_item<1> it) {
                    const int group=int(it.get_group_linear_id());
                    const int slot=slots[group/80];
                    if (slot < 0 || slot >= capacity) return;
                    const int weight=(group/40)%2, index=group%40;
                    auto* bytes=static_cast<uint8_t*>(weight?up:gate)+
                        size_t(slot)*524800+size_t(index)*13120;
                    const int lid=int(it.get_local_linear_id());
                    for (int j=lid;j<13120;j+=256) tile[j]=bytes[j];
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int j=lid;j<13120;j+=256) {
                        const int source=original_byte(j);
                        if (inverse) bytes[source]=tile[j];
                        else bytes[j]=tile[source];
                    }
                });
        });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf::error=e.what();return 1;
    }
}

extern "C" int xpu_gguf_iq2s_gu(void* queue, const void* gate, const void* up,
    const void* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int groups, int routes, int input_rows, int capacity, int workgroup) {
    if (!queue || !gate || !up || !x || !slots || !starts || !tokens ||
        !destinations || !g || !u || groups < 1 || routes < groups ||
        routes > 40 || input_rows < 1 || input_rows > 4 || capacity < 1 ||
        (workgroup != 4 && workgroup != 16)) return 2;
    try {
        using namespace sglang_iq2s_pvr;
        auto& q=*static_cast<sycl::queue*>(queue);
#define SGL_IQ2S_LAUNCH(WG) launch<WG>(q,static_cast<const uint8_t*>(gate),static_cast<const uint8_t*>(up),static_cast<const Q81Block*>(x),slots,starts,tokens,destinations,g,u,groups,routes,input_rows,capacity)
        if (workgroup == 16) SGL_IQ2S_LAUNCH(16);
        else SGL_IQ2S_LAUNCH(4);
#undef SGL_IQ2S_LAUNCH
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf::error=e.what();return 1;
    }
}

namespace sglang_iq2s_pvr {
template<int WG> class IQ2SGroupedPointers;
template<int WG>
void launch_pointers(sycl::queue& q, const uint64_t* addresses,
    const Q81Block* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* g, float* u,
    int groups, int routes, int input_rows, int capacity) {
    q.parallel_for<IQ2SGroupedPointers<WG>>(
        sycl::nd_range<2>(sycl::range<2>(groups, 80), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int group = int(it.get_group(0));
            const int slot = slots[group], first = starts[group], end = starts[group+1];
            if (first < 0 || end <= first || end > routes) return;
            const int lid = int(it.get_local_id(1));
            const int v = int(it.get_group(1)) * WG * 16 + lid * 16;
            const bool high = v >= 640;
            const int row = high ? v - 640 : v;
            float* dst = high ? u : g;
            const uint64_t address=slot>=0&&slot<capacity?addresses[int64_t(slot)*3+(high?1:0)]:0;
            if (!address) {
                for (int e=first; e<end; ++e) {
                    const int target=destinations[e];
                    if (target >= 0 && target < routes) {
                        pvr::simd<float,16> bad=std::numeric_limits<float>::quiet_NaN();
                        pvr::block_store<float,16>(dst+size_t(target)*640+row,bad);
                    }
                }
                return;
            }
            pvr::slm_init<4096>();
            if constexpr (WG == 16) {
                pvr::simd<uint32_t,64> offsets(lid*256,4);
                auto grid=pvr::gather<uint32_t,64>(b60_s4_grid,offsets);
                pvr::slm_block_store<uint32_t,64>(lid*256,grid);
            } else {
                pvr::simd<uint32_t,128> offsets(lid*1024,4);
                auto lo=pvr::gather<uint32_t,128>(b60_s4_grid,offsets);
                auto hi=pvr::gather<uint32_t,128>(b60_s4_grid,offsets+512u);
                pvr::slm_block_store<uint32_t,128>(lid*1024,lo);
                pvr::slm_block_store<uint32_t,128>(lid*1024+512,hi);
            }
            pvr::barrier();
            const auto* w=reinterpret_cast<const uint8_t*>(address);
            int e=first;
            for (;end-e>=4;e+=4)
                grid_s4_basis9_packet<4>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
            if (end-e>=2) {
                grid_s4_basis9_packet<2>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
                e+=2;
            }
            for (;e<end;++e)
                grid_s4_basis9_packet<1>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);
        });
}

}
extern "C" int xpu_gguf_iq2s_gu_ptr(void* queue,const uint64_t* addresses,
 const void* x,const int32_t* slots,const int32_t* starts,const int32_t* tokens,
 const int32_t* destinations,float* g,float* u,int groups,int routes,int input_rows,int capacity,int workgroup) {
 if(!queue||!addresses||!x||!slots||!starts||!tokens||!destinations||!g||!u||groups<1||routes<groups||routes>40||input_rows<1||input_rows>4||capacity<1||(workgroup!=4&&workgroup!=16))return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  if(workgroup==16)sglang_iq2s_pvr::launch_pointers<16>(q,addresses,static_cast<const sglang_iq2s_pvr::Q81Block*>(x),slots,starts,tokens,destinations,g,u,groups,routes,input_rows,capacity);
  else sglang_iq2s_pvr::launch_pointers<4>(q,addresses,static_cast<const sglang_iq2s_pvr::Q81Block*>(x),slots,starts,tokens,destinations,g,u,groups,routes,input_rows,capacity);
  return 0;
 }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}
