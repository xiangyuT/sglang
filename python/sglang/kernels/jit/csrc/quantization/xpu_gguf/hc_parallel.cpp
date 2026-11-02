// SPDX-License-Identifier: Apache-2.0
// Same Torch MeanOps FP32 reduction ordering as xpu_gguf/hc.cpp.
// ESIMD row packet: 4 explicit scalar chains, width-half tree then adjacent warp tree.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <string>
namespace sglang_hc_parallel {
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
template<int WG>class HCParallel;
template<int WG>void launch(sycl::queue&q,const uint16_t*x,const uint16_t*block,const uint16_t*scores,const uint16_t*w,uint16_t*combined,uint16_t*y,float*means,int T,float eps,int K,int score_stride,int mode){
 q.parallel_for<HCParallel<WG>>(sycl::nd_range<1>(size_t(T)*4*WG,WG),[=](sycl::nd_item<1>it)[[intel::sycl_explicit_simd]]{
#pragma clang fp contract(off)
  es::slm_init<WG*64+16>();
  const int row=it.get_group_linear_id(),sid=it.get_local_linear_id(),token=row/4,stream=row%4;
  es::simd<uint16_t,1>sb;sb[0]=scores[token*score_stride+stream];auto sf=from_bf16(sb);es::simd<float,1>z=-sf*.25f;
  auto gatev=es::inv<float>(es::exp<float>(z)+1.f);const float gate=gatev[0]*2.f;
  es::simd<float,16>v[4];
#pragma unroll
  for(int k=0;k<4;++k)v[k]=0.f;
  for(int first=0;first<K;first+=WG*64){
   const int start=first+sid*64;if(start>=K)continue;
   auto b=es::block_load<uint16_t,64>(x+size_t(row)*K+start),bb=es::block_load<uint16_t,64>(block+size_t(token)*K+start);
   auto a=from_bf16(b),bv=from_bf16(bb);es::simd<float,64>cv;
   if(mode==0)cv=sycl::ext::intel::experimental::esimd::fma(bv,es::simd<float,64>(gate),a);else{cv=bv*gate;cv=cv+a;}
   auto rounded=to_bf16(cv);es::block_store<uint16_t,64>(combined+size_t(row)*K+start,rounded);a=from_bf16(rounded);
#pragma unroll
   for(int k=0;k<4;++k){es::simd<float,16>part=a.template select<16,4>(k),sq=part*part;es::simd<uint32_t,16>idx(start+k,4);sq.merge(0.f,idx>=uint32_t(K));v[k]=v[k]+sq;}
  }
  es::simd<float,16>s=v[0]+v[1];s=s+v[2];s=s+v[3];es::slm_block_store<float,16>(sid*64,s);es::barrier();
  if(sid==0){
   auto all=es::slm_block_load<float,WG*16>(0);const float mean=mean_tree<WG*16>(all)*(1.f/2560.f);means[row]=mean;
   es::simd<float,1>den=mean;es::simd_mask<1>bad=row>=T*4;den.merge(0.f,bad);den=den+eps;auto ri=es::rsqrt<float>(den);es::simd<uint32_t,1>off=WG*64;es::slm_scatter<float,1>(off,ri);
  }
  es::barrier();es::simd<uint32_t,1>off=WG*64;auto ri=es::slm_gather<float,1>(off);const float inv=ri[0];
  for(int first=0;first<K;first+=WG*64){
   const int start=first+sid*64;if(start>=K)continue;
   auto b=es::block_load<uint16_t,64>(combined+size_t(row)*K+start),wb=es::block_load<uint16_t,64>(w+size_t(stream)*K+start);
   auto a=from_bf16(b),gain=from_bf16(wb)+1.f;es::simd<float,64>norm=a*inv,result=norm*gain;auto out=to_bf16(result);es::block_store<uint16_t,64>(y+size_t(row)*K+start,out);
  }
 });
}
}
extern "C" const char*xpu_gguf_hc_combine_norm_error(){return sglang_hc_parallel::error.c_str();}
extern "C" int xpu_gguf_hc_combine_norm(void*queue,const void*x,const void*block,const void*scores,const void*w,void*combined,void*y,float*means,int T,float eps,int score_stride,int mode){
 if(!queue||!x||!block||!scores||!w||!combined||!y||!means||T<1||T>4||!(eps>0)||score_stride<4||mode<0||mode>1)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define RUN(W) sglang_hc_parallel::launch<W>(q,static_cast<const uint16_t*>(x),static_cast<const uint16_t*>(block),static_cast<const uint16_t*>(scores),static_cast<const uint16_t*>(w),static_cast<uint16_t*>(combined),static_cast<uint16_t*>(y),means,T,eps,2560,score_stride,mode)
  if(T==1){RUN(16);}else if(T<=3){RUN(8);}else{RUN(4);}
#undef RUN
 return 0;}catch(const std::exception&e){sglang_hc_parallel::error=e.what();return 1;}
}
