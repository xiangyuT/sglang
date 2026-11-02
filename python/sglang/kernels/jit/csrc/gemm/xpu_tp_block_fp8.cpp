// SPDX-License-Identifier: Apache-2.0
// B70 TP2/TP4 block-FP8 projections, row tiles4/8/16; GRF256.
// FP16 dequantization, FP32 tree accumulation, BF16 output.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include "xpu_tp_fp8_codec.hpp"
#include <cstdint>
#include <string>
#include <type_traits>
#pragma clang fp contract(off) reassociate(off)
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int,int,float,unsigned int) noexcept;
namespace tp_dense01 {
inline float fp8half(uint8_t v) {
 uint16_t mag=v&127,sign=uint16_t(v&128)<<8;
 uint16_t bits=mag==127?sign|0x7e00:sign|(mag<<7);
 return float(sycl::bit_cast<sycl::half>(bits));
}
template<bool Direct> inline float fp8div(uint8_t v) {
 if constexpr(!Direct) return fp8half(v);
 uint32_t mag=v&127;
 uint32_t bits=(mag<<20)+0x38000000u;
 if(mag<8)bits=sycl::bit_cast<uint32_t>(float(mag)*0x1p-17f);
 if(mag==127)bits=0x7fc00000u;
 return sycl::bit_cast<float>(bits|(uint32_t(v&128)<<24));
}
inline uint16_t bf16(float x) {
 uint32_t u=sycl::bit_cast<uint32_t>(x);
 if((u&0x7fffffffU)>0x7f800000U)return uint16_t((u>>16)|64);
 return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);
}
inline float sum16(float (&p)[16]) {
#pragma unroll
 for(int s=1;s<16;s*=2) {
#pragma unroll
  for(int i=0;i<16;i+=2*s)p[i]=p[i]+p[i+s];
 }
 return p[0];
}

template<int Rows,bool Packed,bool Direct,class Scratch> inline void body(sycl::nd_item<1> it,Scratch partial,int workgroup,
 const uint8_t* __restrict x,const uint8_t* __restrict w,const float* __restrict xs,const float* __restrict ws,uint16_t* __restrict y,int ws0,int ws1) {
 constexpr int K=2048,R=Rows,WARP=4;
 auto sg=it.get_sub_group();int tid=it.get_local_linear_id(),col=tid*16,group=col/128;
   int row0=workgroup*R;
   sycl::vec<uint32_t,4> xv;xv.load(0,reinterpret_cast<const uint32_t*>(x+col));
   float scale=float(sycl::half(xs[group]))*256.f;
   using Operand=std::conditional_t<Packed,sycl::half,float>;
   Operand xf[16];
#pragma unroll
   for(int word=0;word<4;++word){auto values=word540::div256(xv[word]);
#pragma unroll
    for(int j=0;j<4;++j){float v=values[j]*scale;
     v=float(sycl::half(v));xf[word*4+j]=v;}}
   // Every row tile stays inside one 128-row quantization block.
   float sw=float(sycl::half(ws[(row0/128)*ws0+group*ws1]))*256.f;
#pragma unroll
   for(int r=0;r<R;++r){
    int row=row0+r;sycl::vec<uint32_t,4> v;
    v.load(0,reinterpret_cast<const uint32_t*>(w+row*K+col));
    float p[16];
#pragma unroll
    for(int word=0;word<4;++word){auto values=word540::div256(v[word]);
#pragma unroll
     for(int j=0;j<4;++j){int index=word*4+j;float z=values[j]*sw;
      z=float(sycl::half(z));p[index]=z*float(xf[index]);}}
    float value=sycl::reduce_over_group(sg,sum16(p),sycl::plus<float>());
    if constexpr(WARP==1){if(tid==0)y[row]=bf16(value);}
    else {if(tid%32==0)partial[r*WARP+tid/32]=value;}
   }
   if constexpr(WARP>1){
    it.barrier(sycl::access::fence_space::local_space);
    float part=tid<R*WARP?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
    float value=__spirv_GroupNonUniformFAdd(3,3,part,4);
#else
    float value=part;
#endif
    if(tid<R*WARP && tid%WARP==0)y[row0+tid/WARP]=bf16(value);
   }
}
}

namespace tp_dense01 {
thread_local std::string error;
template<int Rows,int GRF> class Projection;
template<int Rows,int GRF> void run(sycl::queue& q,const uint8_t* x,const uint8_t* w,const float* xs,const float* ws,uint16_t* y,int n,int ws0,int ws1) {
 namespace one=sycl::ext::oneapi::experimental;
 namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<GRF>};
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<float,1> partial(Rows*4,h);
  h.parallel_for<Projection<Rows,GRF>>(sycl::nd_range<1>((n/Rows)*128,128),props,[=](sycl::nd_item<1> it){
   body<Rows,true,false>(it,partial,it.get_group_linear_id(),x,w,xs,ws,y,ws0,ws1);
  });
 });
}
}
extern "C" const char* tp_dense01_error(){return tp_dense01::error.c_str();}
extern "C" int tp_dense01_run(void* queue,const uint8_t* x,const uint8_t* w,const float* xs,const float* ws,uint16_t* y,int n,int ws0,int ws1,int mode) {
 if(!queue||!x||!w||!xs||!ws||!y||(n!=6144&&n!=3072&&n!=4608&&n!=2560)||ws0<1||ws1<1||mode<0||mode>2)return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  switch(mode){
   case 0:tp_dense01::run<4,256>(q,x,w,xs,ws,y,n,ws0,ws1);break;
   case 1:tp_dense01::run<8,256>(q,x,w,xs,ws,y,n,ws0,ws1);break;
   case 2:tp_dense01::run<16,256>(q,x,w,xs,ws,y,n,ws0,ws1);break;
  }
  return 0;
 }catch(const std::exception& e){tp_dense01::error=e.what();return 1;}
}

namespace tp_dense01 { class DecodeCheck; }
extern "C" int tp_dense01_decode(void* queue,const uint8_t* raw,float* old,float* direct) {
 if(!queue||!raw||!old||!direct)return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  q.parallel_for<tp_dense01::DecodeCheck>(sycl::range<1>(256),[=](sycl::id<1> i){old[i]=tp_dense01::fp8div<false>(raw[i]);direct[i]=tp_dense01::fp8div<true>(raw[i]);});
  return 0;
 }catch(const std::exception& e){tp_dense01::error=e.what();return 1;}
}
