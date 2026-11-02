// SPDX-License-Identifier: Apache-2.0
// Same Torch MeanOps FP32 reduction ordering as xpu_gguf/hc.cpp.
// ESIMD row packet: 4 explicit scalar chains, width-half tree then adjacent warp tree.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <string>
#ifndef SGLANG_HC_MAX_TOKENS
#define SGLANG_HC_MAX_TOKENS 4
#endif
namespace sglang_hc_bf256g128 {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
template<int V>SYCL_ESIMD_FUNCTION inline float adjacent(es::simd<float,V>v){
 if constexpr(V==1)return v[0];
 else if constexpr(V==2){float a=v[0],b=v[1];return a+b;}
 else{es::simd<float,V/2>a=v.template select<V/2,2>(0),b=v.template select<V/2,2>(1);return adjacent<V/2>(a+b);}
}
template<int V>SYCL_ESIMD_FUNCTION inline float mean_tree(es::simd<float,V>v){
 if constexpr(V==32)return adjacent<32>(v);
 else{es::simd<float,V/2>a=v.template select<V/2,1>(0),b=v.template select<V/2,1>(V/2);return mean_tree<V/2>(a+b);}
}
template<int V>SYCL_ESIMD_FUNCTION inline es::simd<float,V> from_bf16(es::simd<uint16_t,V>b){
 es::simd<sycl::ext::oneapi::bfloat16,V>values=b.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();es::simd<float,V>out=values;return out;
}
template<int V>SYCL_ESIMD_FUNCTION inline es::simd<uint16_t,V> to_bf16(es::simd<float,V>v){
 es::simd<sycl::ext::oneapi::bfloat16,V>values=v;return values.template bit_cast_view<uint16_t>().read();
}
template<int Width>class HCPacket;
template<int Width>
void launch(sycl::queue&q,const uint16_t*x,const uint16_t*w,uint16_t*y,float*means,int T,float eps,int K){
 q.parallel_for<HCPacket<Width>>(sycl::nd_range<1>(size_t(T)*4,1),[=](sycl::nd_item<1>it)[[intel::sycl_explicit_simd]]{
#pragma clang fp contract(off)
  const int row=it.get_global_id(0);const auto*input=x+size_t(row)*K;
  es::simd<float,Width>v[4];
#pragma unroll
  for(int k=0;k<4;++k)v[k]=0.f;
  if constexpr(Width==32){
   for(int start=0;start<K;start+=128){
    auto b=es::block_load<uint16_t,128>(input+start);auto a=from_bf16(b);
#pragma unroll
    for(int k=0;k<4;++k){es::simd<float,32> z=a.template select<32,4>(k);v[k]=v[k]+z*z;}
   }
  }else{
  for(int first=0;first<K;first+=Width*4){
#pragma unroll
   for(int sub=0;sub<Width/64;++sub){
    const int start=first+sub*256;if(start>=K)continue;
    es::simd<uint32_t,256>cols(start,1);auto valid=cols<uint32_t(K);
    es::simd<uint16_t,256>b;
    if(start+256<=K)b=es::block_load<uint16_t,256>(input+start);
    else b=es::gather<uint16_t,256>(input,cols*2u,valid);
    auto a=from_bf16(b);
#pragma unroll
    for(int k=0;k<4;++k){
     es::simd<float,64>z=a.template select<64,4>(k);es::simd<float,64>sq=z*z;
     es::simd<uint32_t,64>index(start+k,4);sq.merge(0.f,index>=uint32_t(K));
     es::simd<float,64>old=v[k].template select<64,1>(sub*64);v[k].template select<64,1>(sub*64)=old+sq;
    }
   }
  }
  }
  es::simd<float,Width>s=v[0]+v[1];s=s+v[2];s=s+v[3];
  const float mean=mean_tree<Width>(s)*(1.f/2560.f);means[row]=mean;
  es::simd<float,1>den=mean;es::simd_mask<1>bad=row>=T*4;den.merge(0.f,bad);den=den+eps;auto ri=es::rsqrt<float>(den);const float inv=ri[0];
  for(int start=0;start<K;start+=256){
   es::simd<uint32_t,256>cols(start,1);auto valid=cols<uint32_t(K);
   es::simd<uint16_t,256>b,wb;
   if(start+256<=K){b=es::block_load<uint16_t,256>(input+start);wb=es::block_load<uint16_t,256>(w+(row%4)*K+start);}
   else{b=es::gather<uint16_t,256>(input,cols*2u,valid);wb=es::gather<uint16_t,256>(w+(row%4)*K,cols*2u,valid);}
   auto a=from_bf16(b),gain=from_bf16(wb)+1.f;es::simd<float,256>norm=a*inv,result=norm*gain;auto out=to_bf16(result);
   if(start+256<=K)es::block_store<uint16_t,256>(y+size_t(row)*K+start,out);
   else es::scatter<uint16_t,256>(y+size_t(row)*K,cols*2u,out,valid);
  }
 });
}
}
extern "C" const char*xpu_hc_esimd_error(){return sglang_hc_bf256g128::error.c_str();}
extern "C" int xpu_hc_esimd(void*queue,const void*x,const void*w,void*y,float*means,int T,float eps,int K){
 if(!queue||!x||!w||!y||!means||T<1||T>SGLANG_HC_MAX_TOKENS||!(eps>0)||K!=2560)return 2;
 try{
  auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define RUN(W) sglang_hc_bf256g128::launch<W>(q,static_cast<const uint16_t*>(x),static_cast<const uint16_t*>(w),static_cast<uint16_t*>(y),means,T,eps,K)
  if(T==1){RUN(256);}else if(T<=3){RUN(128);}else if(T<8){RUN(64);}else{RUN(32);}
#undef RUN
  return 0;
 }catch(const std::exception&e){sglang_hc_bf256g128::error=e.what();return 1;}
}
