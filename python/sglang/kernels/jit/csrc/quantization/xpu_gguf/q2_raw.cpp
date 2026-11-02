// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Q2 S4 expansion and eight FP32 chains from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd.
// Read original Q2_0 row bytes via 16-bit gathers rather than a packed copy.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <limits>
#include <string>
namespace sglang_raw_q2 {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t,16> expand(es::simd<uint32_t,16> q) {
 q&=65535u;q=(q|(q<<8))&0x00ff00ffu;q=(q|(q<<4))&0x0f0f0f0fu;
 q=(q|(q<<2))&0x33333333u;return(q+0x77777777u)^0x88888888u;
}
template<int E,bool Grouped>
SYCL_ESIMD_FUNCTION inline void packet(const uint8_t* w,const block_q8_1* acts,
 const int32_t* destinations,float* out,int row,int first,int routes) {
#pragma clang fp contract(off)
 es::simd<uint32_t,16> rows(row,1),base=rows*180u;
 es::simd<float,16> sums[8][E];
#pragma unroll
 for(int p=0;p<8;++p){
#pragma unroll
  for(int e=0;e<E;++e)sums[p][e]=0.f;
 }
 for(int step=0;step<3;++step){
#pragma unroll
  for(int p=0;p<8;++p){
   const int seg=step*8+p;if(seg>=20)continue;
   const int block=seg/2,part=seg%2;
   es::simd<uint16_t,64> raw;
#pragma unroll
   for(int j=0;j<4;++j)raw.template select<16,1>(j*16)=es::gather<uint16_t,16>(
       reinterpret_cast<const uint16_t*>(w),base+uint32_t(block*18+2+part*8+j*2));
   es::simd<uint32_t,64> words;
#pragma unroll
   for(int j=0;j<4;++j){
    es::simd<uint32_t,16> codes=raw.template select<16,1>(j*16);
    words.template select<16,1>(j*16)=expand(codes);
   }
   es::simd<uint16_t,128> b=words.template bit_cast_view<uint16_t>().read();
   auto scale_bits=es::gather<uint16_t,16>(reinterpret_cast<const uint16_t*>(w),base+uint32_t(block*18));
   es::simd<float,16> d=scale_bits.template bit_cast_view<sycl::half>().read();
   es::simd<uint32_t,E*8> aw;float scales[E];
#pragma unroll
   for(int e=0;e<E;++e){
    const int dest=Grouped?destinations[first+e]:first+e;
    const auto* a=reinterpret_cast<const uint8_t*>(acts+size_t(dest>=0&&dest<routes?dest:0)*20+seg);
    es::simd<uint32_t,8> offsets(4,4);
    aw.template select<8,1>(e*8)=es::gather<uint32_t,8>(reinterpret_cast<const uint32_t*>(a),offsets);
    es::simd<uint32_t,1> zero_offset=0;
    auto ds_bits=es::gather<uint16_t,1>(reinterpret_cast<const uint16_t*>(a),zero_offset);
    es::simd<float,1> ds=ds_bits.template bit_cast_view<sycl::half>().read();scales[e]=ds[0];
   }
   es::simd<int8_t,E*32> a=aw.template bit_cast_view<int8_t>();es::simd<int32_t,E*16> zero=0;
   using P=es::xmx::dpas_argument_type;
   auto product=es::xmx::dpas<8,E,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,b,a);
#pragma unroll
   for(int e=0;e<E;++e){
    es::simd<int32_t,16> integer=product.template select<16,1>(e*16);
    es::simd<float,16> dot=integer,factor=d*scales[e];
    es::simd<float,16> term=factor*dot;
    // Retain the qualified Q2 PVR validity predicate between MUL and ADD.
    // The ESIMD backend otherwise contracts these operations into FMA,
    // despite -ffp-contract=off, changing the original FP32 chain bits.
    term.merge(0.f,rows>=2560u);sums[p][e]=sums[p][e]+term;
   }
  }
 }
#pragma unroll
 for(int e=0;e<E;++e){
  auto value=((sums[0][e]+sums[4][e])+(sums[2][e]+sums[6][e]))+
             ((sums[1][e]+sums[5][e])+(sums[3][e]+sums[7][e]));
  const int dest=Grouped?destinations[first+e]:first+e;
  if(dest>=0&&dest<routes)es::block_store<float,16>(out+size_t(dest)*2560+row,value);
 }
}
template<bool Grouped>class Down;
template<bool Grouped>
void launch(sycl::queue& q,const uint8_t* weights,const uint64_t* addresses,const block_q8_1* acts,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,float* out,int groups,int routes,int capacity) {
 q.parallel_for<Down<Grouped>>(sycl::nd_range<2>(sycl::range<2>(groups,160),sycl::range<2>(1,4)),
  [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
   const int group=int(item.get_group(0)),row=int(item.get_group(1))*64+int(item.get_local_id(1))*16;
   const int first=Grouped?starts[group]:group,end=Grouped?starts[group+1]:group+1;
   if(first<0||end<=first||end>routes)return;
   const int slot=slots[group];const uint8_t* w=nullptr;
   if(slot>=0&&slot<capacity)w=addresses?reinterpret_cast<const uint8_t*>(addresses[slot*3+2]):weights+size_t(slot)*460800;
   if(!w){
    for(int e=first;e<end;++e){const int dest=Grouped?destinations[e]:e;
     if(dest>=0&&dest<routes){es::simd<float,16> bad=std::numeric_limits<float>::quiet_NaN();es::block_store<float,16>(out+size_t(dest)*2560+row,bad);}}
    return;
   }
   int e=first;for(;end-e>=4;e+=4)packet<4,Grouped>(w,acts,destinations,out,row,e,routes);
   if(end-e>=2){packet<2,Grouped>(w,acts,destinations,out,row,e,routes);e+=2;}
   for(;e<end;++e)packet<1,Grouped>(w,acts,destinations,out,row,e,routes);
  });
}
}
extern "C" const char* xpu_raw_q2_down_error(){return sglang_raw_q2::error.c_str();}
extern "C" int xpu_raw_q2_down(void* queue,const void* weight,const uint64_t* addresses,const void* acts,
 const int32_t* slots,const int32_t* starts,const int32_t* destinations,float* out,int groups,int routes,int capacity) {
 if(!queue||(!weight&&!addresses)||!acts||!slots||!out||groups<1||routes<1||routes>40||groups>routes||capacity<1||capacity>512||bool(starts)!=bool(destinations))return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
  if(starts)sglang_raw_q2::launch<true>(q,static_cast<const uint8_t*>(weight),addresses,static_cast<const block_q8_1*>(acts),slots,starts,destinations,out,groups,routes,capacity);
  else{if(groups!=routes)return 2;sglang_raw_q2::launch<false>(q,static_cast<const uint8_t*>(weight),addresses,static_cast<const block_q8_1*>(acts),slots,nullptr,nullptr,out,groups,routes,capacity);}
  return 0;
 }catch(const std::exception& e){sglang_raw_q2::error=e.what();return 1;}
}
