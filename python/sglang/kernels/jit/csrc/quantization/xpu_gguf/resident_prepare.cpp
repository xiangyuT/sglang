// SPDX-License-Identifier: MIT
// Derived from the resident hot case of xpu_gguf/gpu_cache.cpp.
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Fixed identity expert->slot mapping only; no mirrors, eviction or LRU mutation.
// The optional input producer retains moe_quant_fused.cpp finiteQ8_1 arithmetic.
// Quantization uses disjoint workgroups; all metadata and scratch stay on the consumer queue.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>
namespace {
thread_local std::string error;
using bf16=sycl::ext::oneapi::bfloat16;
struct Q81Block{sycl::half2 ds;int8_t qs[32];};static_assert(sizeof(Q81Block)==36);
inline float finite_half(float x){return sycl::fabs(x)>65504.f?sycl::copysign(65504.f,x):x;}
inline float warp_sum(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x+=sycl::permute_group_by_xor(sg,x,o);return x;}
inline float warp_max(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x=sycl::fmax(x,sycl::permute_group_by_xor(sg,x,o));return x;}

template<bool Wide, bool Mapped = false, bool Quant = false> class Resident;
template<bool Wide, bool Mapped = false, bool Quant = false>
void prepare(sycl::queue&q,const void*ids,int32_t*meta,float*gate,float*up,float*down,
             int T,int topk,int zero,const int32_t*mapping=nullptr,int experts=0,
             const bf16*input=nullptr,float*xfp=nullptr,Q81Block*xq=nullptr) {
 const int routes=T*topk,gu=routes*640,total=2*gu+routes*2560;
 q.submit([&](sycl::handler&h) {
  sycl::local_accessor<int,1> original(64,h),sorted(64,h),invalid(1,h);
  h.parallel_for<Resident<Wide,Mapped,Quant>>(sycl::nd_range<1>((128+(Quant?T*10:0))*64,64),
   [=](sycl::nd_item<1>it) [[sycl::reqd_sub_group_size(32)]] {
    const int index=it.get_global_linear_id(),lane=it.get_local_linear_id();
    if constexpr(Quant) {
     if(it.get_group_linear_id()>=128) {
      auto sg=it.get_sub_group();const int first=(it.get_group_linear_id()-128)*256;
      for(int step=0;step<4;++step) {
       const int i=first+step*64+lane;const float xi=float(input[i]);xfp[i]=xi;
       const float amax=warp_max(sg,sycl::fabs(xi)),sum=warp_sum(sg,xi),d=finite_half(amax/127.f);
       float code=amax==0.f?0.f:sycl::round(xi/d);code=code>127.f?127.f:code< -127.f?-127.f:code;
       xq[i/32].qs[i%32]=static_cast<int8_t>(code);
       if(i%32==0)xq[i/32].ds=sycl::vec<float,2>(finite_half(d),finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
      }
      return;
     }
    }
    const sycl::vec<float,4> zeros(0.f);
    for(int i=index;i<total/4;i+=128*64) {
     if(i<gu/4)reinterpret_cast<sycl::vec<float,4>*>(gate)[i]=zeros;
     else if(i<gu/2)reinterpret_cast<sycl::vec<float,4>*>(up)[i-gu/4]=zeros;
     else reinterpret_cast<sycl::vec<float,4>*>(down)[i-gu/2]=zeros;
    }
    if(it.get_group_linear_id()!=0)return;
    int id=zero+1+lane;
    if(lane<routes) {
     int64_t v=Wide?static_cast<const int64_t*>(ids)[lane]:static_cast<const int32_t*>(ids)[lane];
     if constexpr(Mapped) {
      if(v>=0)v=v<experts?mapping[v]:zero+1;
     }
     if(v<0)v=zero;
     if(v<=zero)id=int(v);
     meta[routes+lane]=lane;
     meta[5*routes+1+lane]=0;
     meta[2*routes+lane]=routes;
    }
    if(lane==0) {meta[3*routes]=routes;meta[6*routes+1]=0;meta[6*routes+2]=0;}
    original[lane]=id;
    it.barrier(sycl::access::fence_space::global_and_local);
    if(lane==0) {
     int bad=0;
     for(int j=0;j<routes;++j)bad|=original[j]>zero;
     invalid[0]=bad;meta[6*routes+2]=bad;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if(invalid[0]) {
     if(lane<routes) {
      meta[lane]=-1;meta[routes+lane]=-1;
      meta[2*routes+lane]=lane?routes:0;
      meta[3*routes+1+lane]=0;meta[4*routes+1+lane]=lane;
     }
     if(lane==0){meta[3*routes]=routes;meta[6*routes+1]=1;}
     return;
    }
    int rank=0;
    if(lane<routes) {
     for(int j=0;j<routes;++j)rank+=original[j]<id||(original[j]==id&&j<lane);
     sorted[rank]=id;
     meta[3*routes+1+rank]=lane/topk;
     meta[4*routes+1+rank]=lane;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if(lane<routes) {
     int g=0;
     for(int j=1;j<=rank;++j)g+=sorted[j]!=sorted[j-1];
     meta[lane]=id;
     if(rank==0||sorted[rank]!=sorted[rank-1]) {
      meta[2*routes+g]=rank;
      meta[routes+g]=id;
      meta[5*routes+1+g]=id;
     }
    }
    it.barrier(sycl::access::fence_space::global_and_local);
    if(lane==0) {
     int groups=1,clip=routes;
     for(int j=0;j<routes;++j) {
      if(j&&sorted[j]!=sorted[j-1])++groups;
      if(sorted[j]==zero&&clip==routes)clip=j;
     }
     for(int j=0;j<=routes;++j)meta[2*routes+j]=sycl::min(meta[2*routes+j],clip);
     meta[6*routes+1]=groups;
    }
   });
 });
}
}
extern "C" int resident_prepare_mapped(void*queue,const void*ids,
 const int32_t*mapping,int32_t*meta,float*gate,float*up,float*down,
 int T,int topk,int zero,int wide,int experts) {
 try {
  if(!queue||!ids||!mapping||!meta||!gate||!up||!down||T<1||T>4||
     topk<1||topk>10||zero<1||zero>=experts||experts!=512||
     (wide!=0&&wide!=1))return 2;
  auto&q=*static_cast<sycl::queue*>(queue);
  if(!q.is_in_order())return 2;
  if(wide)prepare<true,true>(q,ids,meta,gate,up,down,T,topk,zero,mapping,experts);
  else prepare<false,true>(q,ids,meta,gate,up,down,T,topk,zero,mapping,experts);
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" const char* resident_error(){return error.c_str();}
extern "C" int resident_prepare(void*queue,const void*ids,int32_t*meta,
 float*gate,float*up,float*down,int T,int topk,int zero,int wide) {
 try {
  if(!queue||!ids||!meta||!gate||!up||!down||T<1||T>4||topk<1||topk>10||
     zero<1||zero>511||(wide!=0&&wide!=1))return 2;
  auto&q=*static_cast<sycl::queue*>(queue);
  if(!q.is_in_order())return 2;
  if(wide)prepare<true>(q,ids,meta,gate,up,down,T,topk,zero);
  else prepare<false>(q,ids,meta,gate,up,down,T,topk,zero);
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}

extern "C" int resident_prepare_q8(void*queue,const void*ids,const int32_t*mapping,
 int32_t*meta,float*gate,float*up,float*down,const void*input,float*xfp,void*xq,
 int T,int topk,int zero,int wide,int experts) {
 try {
  if(!queue||!ids||!meta||!gate||!up||!down||!input||!xfp||!xq||T<1||T>4||
     topk<1||topk>10||zero<1||zero>=experts||experts!=512||(wide!=0&&wide!=1))return 2;
  auto&q=*static_cast<sycl::queue*>(queue);if(!q.is_in_order())return 2;
#define PREP(W,M) prepare<W,M,true>(q,ids,meta,gate,up,down,T,topk,zero,mapping,experts,static_cast<const bf16*>(input),xfp,static_cast<Q81Block*>(xq))
  if(mapping){if(wide){PREP(true,true);}else{PREP(false,true);}}
  else{if(wide){PREP(true,false);}else{PREP(false,false);}}
#undef PREP
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
