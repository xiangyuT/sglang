// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Adapted from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd,
// iq_kernels.dp.cpp p16_packet/i1_packet. Original GGUF byte permutations,
// Q8_1 activations and eight-lane FP32 reduction tree. See STRATA-LICENSE
// and GGML-LICENSE. Destinations and stable expert addresses are SGLang's.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
#include <string>
namespace sglang_gguf { extern thread_local std::string error; }
namespace sglang_lowbit_gu {
namespace pvr = sycl::ext::intel::esimd;
SYCL_ESIMD_FUNCTION inline pvr::simd<int32_t,32>
p16_sign(pvr::simd<uint32_t,32> positive,pvr::simd<uint32_t,32> sign){
 pvr::simd<uint32_t,32> bit=((sign&15u)*0x00204081u)&0x01010101u;
 return (positive^(bit*255u))+bit;
}
template<int E>
SYCL_ESIMD_FUNCTION inline void p16_packet(
 const uint8_t* weights,int row,int first,const int32_t* tokens,const int32_t* destinations,
 const block_q8_1* xq,float* dst,int row_limit,int input_rows,int routes){
#pragma clang fp contract(off)
 pvr::simd<uint32_t,32> rows;
#pragma unroll
 for(int r=0;r<4;++r)rows.template select<8,1>(r*8)=uint32_t(row+r);
 auto valid=rows<uint32_t(row_limit);
 pvr::simd<float,32> sum[E];
#pragma unroll
 for(int e=0;e<E;++e)sum[e]=0.f;
 const auto* tile=weights+size_t(row/8)*5280;
 const int half=(row%8)/4;
#pragma unroll 1
 for(int kb=0;kb<10;++kb){
  const auto* block=tile+kb*528;
  auto qs=pvr::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(block+half*128));
  auto aux=pvr::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(block+256+half*128));
  pvr::simd<uint32_t,4> doff(0,2);
  auto dbits=pvr::gather<uint16_t,4>(reinterpret_cast<const uint16_t*>(block+512+half*8),doff);
  pvr::simd<float,4> d4=dbits.template bit_cast_view<sycl::half>().read();
  pvr::simd<float,32> d;
#pragma unroll
  for(int r=0;r<4;++r)d.template select<8,1>(r*8)=float(d4[r]);
  pvr::simd<int32_t,32> w[8];
#pragma unroll
  for(int j=0;j<4;++j){
   pvr::simd<uint32_t,32> index=(qs>>(8*j))&255u;
   auto grid=pvr::gather<uint64_t,32>(iq2xxs_grid,index*8u);
   pvr::simd<uint32_t,32> lo=grid,hi=grid>>32;
   pvr::simd<uint32_t,32> s7=(aux>>(7*j))&127u;
   pvr::simd<uint32_t,32> signs=s7|((pvr::cbit(s7)&1u)<<7);
   w[2*j]=p16_sign(lo,signs);w[2*j+1]=p16_sign(hi,signs>>4);
  }
  pvr::simd<int32_t,32> odd=(aux>>27)|1u;
#pragma unroll
  for(int e=0;e<E;++e){
   const auto* raw=reinterpret_cast<const uint8_t*>(xq+size_t(tokens[first+e]>=0&&tokens[first+e]<input_rows?tokens[first+e]:0)*80+kb*8);
   pvr::simd<uint32_t,8> offsets(0,36);
   auto u=pvr::gather<int32_t,64,8>(reinterpret_cast<const int32_t*>(raw),offsets+4u);
   auto dsbits=pvr::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(raw),offsets);
   pvr::simd<float,8> ds8=dsbits.template bit_cast_view<sycl::half>().read();
   pvr::simd<float,32> ds;
#pragma unroll
   for(int r=0;r<4;++r)ds.template select<8,1>(r*8)=ds8;
   pvr::simd<int32_t,32> s0=0,s1=0;
#pragma unroll
   for(int j=0;j<8;++j){
    pvr::simd<int32_t,32> word;
#pragma unroll
    for(int r=0;r<4;++r)word.template select<8,1>(r*8)=u.template select<8,1>(j*8);
    if(j<4)s0=pvr::dp4a<int32_t>(s0,w[j],word);else s1=pvr::dp4a<int32_t>(s1,w[j],word);
   }
   pvr::simd<int32_t,32> si=(s0+s1)*odd/8;
   pvr::simd<float,32> dot=si,factor=d*ds;
   pvr::simd<float,32> term=factor*dot;term.merge(0.f,!valid);
   sum[e]=sum[e]+term;
  }
 }
#pragma unroll
 for(int e=0;e<E;++e){
  pvr::simd<float,4> s[8];
#pragma unroll
  for(int p=0;p<8;++p)s[p]=sum[e].template select<4,8>(p);
  auto value=((s[0]+s[4])+(s[2]+s[6]))+((s[1]+s[5])+(s[3]+s[7]));
  if(tokens[first+e]<0||tokens[first+e]>=input_rows)value=std::numeric_limits<float>::quiet_NaN();
  const int destination=destinations[first+e];
  if(destination>=0&&destination<routes)pvr::block_store<float,4>(dst+size_t(destination)*640+row,value);
 }
}
template<int E>
SYCL_ESIMD_FUNCTION inline void i1_packet(
 const uint8_t* weights,int row,int first,const int32_t* tokens,const int32_t* destinations,
 const block_q8_1* xq,float* dst,int row_limit,int input_rows,int routes){
#pragma clang fp contract(off)
 pvr::simd<uint32_t,32> rows;
#pragma unroll
 for(int r=0;r<4;++r)rows.template select<8,1>(r*8)=uint32_t(row+r);
 auto valid=rows<uint32_t(row_limit);
 pvr::simd<float,32> sum[E];
#pragma unroll
 for(int e=0;e<E;++e)sum[e]=0.f;
 const auto* tile=weights+size_t(row/8)*4480;
 const int half=(row%8)/4;
#pragma unroll 1
 for(int kb=0;kb<10;++kb){
  const auto* block=tile+kb*448;
  auto qs=pvr::block_load<uint32_t,32>(reinterpret_cast<const uint32_t*>(block+half*128));
  auto qh=pvr::block_load<uint16_t,32>(reinterpret_cast<const uint16_t*>(block+256+half*64));
  auto sc=pvr::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block+384+half*32));
  pvr::simd<uint16_t,4> dbits=(sc.template select<4,4>(0)>>12)|((sc.template select<4,4>(1)>>8)&0x00f0u)|((sc.template select<4,4>(2)>>4)&0x0f00u)|(sc.template select<4,4>(3)&0xf000u);
  pvr::simd<float,4> d4=dbits.template bit_cast_view<sycl::half>().read();
  pvr::simd<float,32> d;
  pvr::simd<uint32_t,32> tmp;
#pragma unroll
  for(int r=0;r<4;++r){
   d.template select<8,1>(r*8)=float(d4[r]);
#pragma unroll
   for(int p=0;p<8;++p)tmp[r*8+p]=uint32_t(sc[r*4+p/2])>>(6*(p%2));
  }
  pvr::simd<int32_t,32> sc0=2*(tmp&7u)+1,sc1=2*((tmp>>3)&7u)+1;
  pvr::simd<int32_t,32> w[8];
#pragma unroll
  for(int j=0;j<4;++j){
   pvr::simd<uint32_t,32> high=qh>>(4*j);
   pvr::simd<uint32_t,32> index=((qs>>(8*j))&255u)|((high&7u)<<8);
   auto grid=pvr::gather<uint32_t,32>(iq1s_grid_gpu,index*4u);
   pvr::simd<uint32_t,32> offset=(7u+((high&8u)>>2))*0x01010101u;
   pvr::simd<uint32_t,32> lo=(grid&0x0f0f0f0fu)<<3,hi=((grid>>4)&0x0f0f0f0fu)<<3;
   w[2*j]=((lo|0x80808080u)-offset)^0x80808080u;
   w[2*j+1]=((hi|0x80808080u)-offset)^0x80808080u;
  }
#pragma unroll
  for(int e=0;e<E;++e){
   const auto* raw=reinterpret_cast<const uint8_t*>(xq+size_t(tokens[first+e]>=0&&tokens[first+e]<input_rows?tokens[first+e]:0)*80+kb*8);
   pvr::simd<uint32_t,8> offsets(0,36);
   auto u=pvr::gather<int32_t,64,8>(reinterpret_cast<const int32_t*>(raw),offsets+4u);
   auto dsbits=pvr::gather<uint16_t,8>(reinterpret_cast<const uint16_t*>(raw),offsets);
   pvr::simd<float,8> ds8=dsbits.template bit_cast_view<sycl::half>().read();
   pvr::simd<float,32> ds;
#pragma unroll
   for(int r=0;r<4;++r)ds.template select<8,1>(r*8)=ds8;
   pvr::simd<int32_t,32> s0=0,s1=0;
#pragma unroll
   for(int j=0;j<8;++j){
    pvr::simd<int32_t,32> word;
#pragma unroll
    for(int r=0;r<4;++r)word.template select<8,1>(r*8)=u.template select<8,1>(j*8);
    if(j<4)s0=pvr::dp4a<int32_t>(s0,w[j],word);else s1=pvr::dp4a<int32_t>(s1,w[j],word);
   }
   pvr::simd<int16_t,32> a16=s0,b16=s1;
   pvr::simd<int32_t,32> a32=a16,b32=b16;
   pvr::simd<int32_t,32> scaled=a32*sc0+b32*sc1;
   pvr::simd<float,32> integer_value=scaled;
   pvr::simd<float,32> dot=integer_value*0.125f,factor=d*ds;
   pvr::simd<float,32> term=factor*dot;term.merge(0.f,!valid);
   sum[e]=sum[e]+term;
  }
 }
#pragma unroll
 for(int e=0;e<E;++e){
  pvr::simd<float,4> s[8];
#pragma unroll
  for(int p=0;p<8;++p)s[p]=sum[e].template select<4,8>(p);
  auto value=((s[0]+s[4])+(s[2]+s[6]))+((s[1]+s[5])+(s[3]+s[7]));
  if(tokens[first+e]<0||tokens[first+e]>=input_rows)value=std::numeric_limits<float>::quiet_NaN();
  const int destination=destinations[first+e];
  if(destination>=0&&destination<routes)pvr::block_store<float,4>(dst+size_t(destination)*640+row,value);
 }
}

template<int Kind> class PackedGU;
template<int Kind>
void submit(sycl::queue& q,const uint8_t* gate,const uint8_t* up,
 const uint64_t* addresses,const block_q8_1* x,const int32_t* slots,
 const int32_t* starts,const int32_t* tokens,const int32_t* destinations,
 float* g,float* u,int groups,int routes,int input_rows,int capacity) {
 constexpr int bytes=Kind==16?422400:358400;
 q.parallel_for<PackedGU<Kind>>(
  sycl::nd_range<2>(sycl::range<2>(groups,320),sycl::range<2>(1,4)),
  [=](sycl::nd_item<2> item) [[intel::sycl_explicit_simd]] {
   const int group=int(item.get_group(0)),slot=slots[group];
   const int first=starts[group],end=starts[group+1];
   if(first<0||end<=first||end>routes)return;
   const int v=int(item.get_group(1))*16+int(item.get_local_id(1))*4;
   const bool high=v>=640;const int row=high?v-640:v;
   float* dst=high?u:g;
   uint64_t address=0;
   if(slot>=0&&slot<capacity)address=addresses?addresses[slot*3+int(high)]:
      uint64_t((high?up:gate)+int64_t(slot)*bytes);
   if(!address) {
    for(int e=first;e<end;++e) {
     const int destination=destinations[e];
     if(destination>=0&&destination<routes) {
      pvr::simd<float,4> value=std::numeric_limits<float>::quiet_NaN();
      pvr::block_store<float,4>(dst+size_t(destination)*640+row,value);
     }
    }
    return;
   }
   const auto* w=reinterpret_cast<const uint8_t*>(address);
   int e=first;
#define LOWBIT_CALL(E) if constexpr(Kind==16)p16_packet<E>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes);else i1_packet<E>(w,row,e,tokens,destinations,x,dst,640,input_rows,routes)
   for(;end-e>=4;e+=4){LOWBIT_CALL(4);}
   if(end-e>=2){LOWBIT_CALL(2);e+=2;}
   for(;e<end;++e){LOWBIT_CALL(1);}
#undef LOWBIT_CALL
  });
}
}
extern "C" int xpu_gguf_lowbit_gu(void* queue,const void* gate,const void* up,
 const uint64_t* addresses,const void* x,const int32_t* slots,const int32_t* starts,
 const int32_t* tokens,const int32_t* destinations,float* g,float* u,
 int kind,int groups,int routes,int input_rows,int capacity,int workgroup) {
 if(!queue||(!addresses&&(!gate||!up))||!x||!slots||!starts||!tokens||!destinations||
    !g||!u||(kind!=16&&kind!=29)||groups<1||groups>routes||routes>40||
    input_rows<1||input_rows>4||capacity<1||capacity>512||workgroup!=4)return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  if(!q.has_property<sycl::property::queue::in_order>())return 2;
  using namespace sglang_lowbit_gu;
  if(kind==16)submit<16>(q,static_cast<const uint8_t*>(gate),static_cast<const uint8_t*>(up),addresses,static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,g,u,groups,routes,input_rows,capacity);
  else submit<29>(q,static_cast<const uint8_t*>(gate),static_cast<const uint8_t*>(up),addresses,static_cast<const block_q8_1*>(x),slots,starts,tokens,destinations,g,u,groups,routes,input_rows,capacity);
  return 0;
 }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}
