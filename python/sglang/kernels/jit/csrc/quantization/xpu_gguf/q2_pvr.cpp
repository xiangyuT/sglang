// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Original GGUF grid: ggml authors; see GGML-LICENSE.
// Ported from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd, sycl/src/kernels/cuda/iq_kernels.dp.cpp.
// See STRATA-LICENSE.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <limits>
#include <string>
namespace sglang_gguf { extern thread_local std::string error; }
namespace sglang_q2_pvr {
namespace pvr=sycl::ext::intel::esimd;
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81Block)==36);
SYCL_ESIMD_FUNCTION inline pvr::simd<uint32_t,16> p4_q2_eight(pvr::simd<uint32_t,16> q){
 q&=65535u;q=(q|(q<<8))&0x00ff00ffu;
 q=(q|(q<<4))&0x0f0f0f0fu;q=(q|(q<<2))&0x33333333u;
 return (q+0x77777777u)^0x88888888u;
}
template<int E>
SYCL_ESIMD_FUNCTION inline void p4_down_packet(
 const uint8_t* weights,int row,int first,const int32_t* destinations,
 const Q81Block* xq,float* dst,int routes){
#pragma clang fp contract(off)
 pvr::simd<uint32_t,16> rows(row,1);auto valid=rows<2560u;
 pvr::simd<float,16> sum[8][E];
#pragma unroll
 for(int p=0;p<8;++p){
#pragma unroll
  for(int e=0;e<E;++e)sum[p][e]=0.f;
 }
 const auto* tile=weights+size_t(row/16)*2880;
 for(int step=0;step<3;++step){
#pragma unroll
  for(int p=0;p<8;++p){
   const int k=step*8+p;if(k>=20)continue;
   const auto* block=tile+(k/2)*288;
   auto q0=pvr::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(block+(k%2)*128));
   auto q1=pvr::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(block+(k%2)*128+64));
   auto bits=pvr::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block+256));
   pvr::simd<float,16> d=bits.template bit_cast_view<sycl::half>().read();
   pvr::simd<uint32_t,64> packed;
#pragma unroll
   for(int j=0;j<4;++j){
    pvr::simd<uint32_t,16> v=(j<2?q0:q1)>>(16*(j%2));
    packed.template select<16,1>(j*16)=p4_q2_eight(v);
   }
   // Keep the natural16-row dword order from the qualified IQ4 INT4 path.
   pvr::simd<uint16_t,128> B=packed.template bit_cast_view<uint16_t>().read();
   pvr::simd<uint32_t,E*8> aw;
   float scales[E];
#pragma unroll
   for(int e=0;e<E;++e){
    const int dest=destinations[first+e];
    const auto* raw=reinterpret_cast<const uint8_t*>(xq+size_t(dest>=0&&dest<routes?dest:0)*20+k);
    pvr::simd<uint32_t,8> offset(4,4);
    aw.template select<8,1>(e*8)=pvr::gather<uint32_t,8>(reinterpret_cast<const uint32_t*>(raw),offset);
    pvr::simd<uint32_t,1> zero_offset=0;
    auto dsbits=pvr::gather<uint16_t,1>(reinterpret_cast<const uint16_t*>(raw),zero_offset);
    pvr::simd<float,1> ds=dsbits.template bit_cast_view<sycl::half>().read();scales[e]=ds[0];
   }
   pvr::simd<int8_t,E*32> A=aw.template bit_cast_view<int8_t>();
   pvr::simd<int32_t,E*16> zero=0;
   using P=pvr::xmx::dpas_argument_type;
   // Q2 nibbles have natural K order. IQ4 GGML's K0/K16 interleave is not used.
   auto product=pvr::xmx::dpas<8,E,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,B,A);
#pragma unroll
   for(int e=0;e<E;++e){
    pvr::simd<int32_t,16> si=product.template select<16,1>(e*16);
    pvr::simd<float,16> dot=si;
    const float scalar_ds=scales[e];pvr::simd<float,16> factor=d*scalar_ds;
    pvr::simd<float,16> term=factor*dot;term.merge(0.f,!valid);
    sum[p][e]=sum[p][e]+term;
   }
  }
 }
#pragma unroll
 for(int e=0;e<E;++e){
  auto value=((sum[0][e]+sum[4][e])+(sum[2][e]+sum[6][e]))+
             ((sum[1][e]+sum[5][e])+(sum[3][e]+sum[7][e]));
  const int dest=destinations[first+e];
  if(dest>=0&&dest<routes)pvr::block_store<float,16>(dst+size_t(dest)*2560+row,value);
 }
}

template<int WG> class Q2Grouped;
template<int WG>
void launch(sycl::queue& q, const uint8_t* weight, const Q81Block* x,
    const int32_t* slots, const int32_t* starts, const int32_t* destinations,
    float* out, int groups, int routes, int capacity) {
    q.parallel_for<Q2Grouped<WG>>(
        sycl::nd_range<2>(sycl::range<2>(groups,160),sycl::range<2>(1,WG)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int group=int(it.get_group(0)),slot=slots[group];
            const int first=starts[group],end=starts[group+1];
            if(first<0 || end<=first || end>routes)return;
            const int row=int(it.get_group(1))*WG*16+int(it.get_local_id(1))*16;
            if(slot<0 || slot>=capacity) {
                for(int e=first;e<end;++e) {
                    const int dest=destinations[e];
                    if(dest>=0 && dest<routes) {
                        pvr::simd<float,16> bad=std::numeric_limits<float>::quiet_NaN();
                        pvr::block_store<float,16>(out+size_t(dest)*2560+row,bad);
                    }
                }
                return;
            }
            const auto* w=weight+size_t(slot)*460800;
            int e=first;
            for(;end-e>=4;e+=4)p4_down_packet<4>(w,row,e,destinations,x,out,routes);
            if(end-e>=2){p4_down_packet<2>(w,row,e,destinations,x,out,routes);e+=2;}
            for(;e<end;++e)p4_down_packet<1>(w,row,e,destinations,x,out,routes);
        });
}
class RepackQ2;
inline int original_byte(int packed) {
    const int kb=packed/288,cell=packed%288;
    if(cell>=256)return ((cell-256)/2)*180+kb*18+(cell-256)%2;
    const int part=cell/128,field=cell%128;
    if(field<64)return (field/4)*180+kb*18+2+part*8+field%4;
    return ((field-64)/4)*180+kb*18+6+part*8+(field-64)%4;
}
}
extern "C" int xpu_gguf_q2_pack(void* queue,void* weight,const int32_t* slots,
    int count,int capacity,int inverse) {
    if(!queue || !weight || !slots || count<1 || capacity<1 || (inverse!=0&&inverse!=1))return 2;
    try {
        using namespace sglang_q2_pvr;
        auto& q=*static_cast<sycl::queue*>(queue);
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint8_t,1> tile(2880,h);
            h.parallel_for<RepackQ2>(sycl::nd_range<1>(size_t(count)*160*128,128),
                [=](sycl::nd_item<1> it) {
                    const int group=int(it.get_group_linear_id()),slot=slots[group/160];
                    if(slot<0 || slot>=capacity)return;
                    auto* bytes=static_cast<uint8_t*>(weight)+size_t(slot)*460800+size_t(group%160)*2880;
                    const int lid=int(it.get_local_linear_id());
                    for(int j=lid;j<2880;j+=128)tile[j]=bytes[j];
                    it.barrier(sycl::access::fence_space::local_space);
                    for(int j=lid;j<2880;j+=128) {
                        const int source=original_byte(j);
                        if(inverse)bytes[source]=tile[j];else bytes[j]=tile[source];
                    }
                });
        });
        return 0;
    }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}
extern "C" int xpu_gguf_q2_down(void* queue,const void* weight,const void* x,
    const int32_t* slots,const int32_t* starts,const int32_t* destinations,
    float* out,int groups,int routes,int capacity,int workgroup) {
    if(!queue || !weight || !x || !slots || !starts || !destinations || !out ||
        groups<1 || routes<groups || routes>40 || capacity<1 || (workgroup!=4&&workgroup!=16))return 2;
    try {
        using namespace sglang_q2_pvr;
        auto& q=*static_cast<sycl::queue*>(queue);
#define SGL_Q2_LAUNCH(WG) launch<WG>(q,static_cast<const uint8_t*>(weight),static_cast<const Q81Block*>(x),slots,starts,destinations,out,groups,routes,capacity)
        if(workgroup==16)SGL_Q2_LAUNCH(16);else SGL_Q2_LAUNCH(4);
#undef SGL_Q2_LAUNCH
        return 0;
    }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}

extern "C" int xpu_gguf_iq2s_pack(void*,void*,void*,const int32_t*,int,int,int);
extern "C" int xpu_gguf_iq2s_pack_expert(void* queue,void* gate,void* up,void* down,
    const int32_t* slots,int count,int capacity,int inverse) {
    if(!down)return 2;
    int status=xpu_gguf_iq2s_pack(queue,gate,up,slots,count,capacity,inverse);
    if(status)return status;
    return xpu_gguf_q2_pack(queue,down,slots,count,capacity,inverse);
}

namespace sglang_q2_pvr {
template<int WG> class Q2GroupedPointers;
template<int WG>
void launch_pointers(sycl::queue& q, const uint64_t* addresses, const Q81Block* x,
    const int32_t* slots, const int32_t* starts, const int32_t* destinations,
    float* out, int groups, int routes, int capacity) {
    q.parallel_for<Q2GroupedPointers<WG>>(
        sycl::nd_range<2>(sycl::range<2>(groups,160),sycl::range<2>(1,WG)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int group=int(it.get_group(0)),slot=slots[group];
            const int first=starts[group],end=starts[group+1];
            if(first<0 || end<=first || end>routes)return;
            const int row=int(it.get_group(1))*WG*16+int(it.get_local_id(1))*16;
            const uint64_t address=slot>=0&&slot<capacity?addresses[int64_t(slot)*3+2]:0;
            if(!address) {
                for(int e=first;e<end;++e) {
                    const int dest=destinations[e];
                    if(dest>=0 && dest<routes) {
                        pvr::simd<float,16> bad=std::numeric_limits<float>::quiet_NaN();
                        pvr::block_store<float,16>(out+size_t(dest)*2560+row,bad);
                    }
                }
                return;
            }
            const auto* w=reinterpret_cast<const uint8_t*>(address);
            int e=first;
            for(;end-e>=4;e+=4)p4_down_packet<4>(w,row,e,destinations,x,out,routes);
            if(end-e>=2){p4_down_packet<2>(w,row,e,destinations,x,out,routes);e+=2;}
            for(;e<end;++e)p4_down_packet<1>(w,row,e,destinations,x,out,routes);
        });
}
}
extern "C" int xpu_gguf_q2_down_ptr(void* queue,const uint64_t* addresses,const void* x,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,float* out,int groups,int routes,int capacity,int workgroup) {
 if(!queue||!addresses||!x||!slots||!starts||!destinations||!out||groups<1||routes<groups||routes>40||capacity<1||(workgroup!=4&&workgroup!=16))return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  if(workgroup==16)sglang_q2_pvr::launch_pointers<16>(q,addresses,static_cast<const sglang_q2_pvr::Q81Block*>(x),slots,starts,destinations,out,groups,routes,capacity);
  else sglang_q2_pvr::launch_pointers<4>(q,addresses,static_cast<const sglang_q2_pvr::Q81Block*>(x),slots,starts,destinations,out,groups,routes,capacity);
  return 0;
 }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}
