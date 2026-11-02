// SPDX-License-Identifier: Apache-2.0
// Single-GPU down541 packed-word/all-hit method; TP128/256 and native permutation.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../gemm/xpu_tp_fp8_codec.hpp"
#include <cstdint>
#include <string>
#pragma clang fp contract(off) reassociate(off)
namespace tp_expert_down {
thread_local std::string error;
inline float bf(uint16_t x){return sycl::bit_cast<float>(uint32_t(x)<<16);}
inline uint16_t rne(float x){uint32_t u=sycl::bit_cast<uint32_t>(x);if((u&0x7fffffffU)>0x7f800000U)return uint16_t((u>>16)|64);return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);}
template<int C>inline float sum(float(&v)[C]){
#pragma unroll
 for(int s=1;s<C;s*=2){
#pragma unroll
  for(int j=0;j<C;j+=2*s)v[j]+=v[j+s];}
 return v[0];}
class DownRouteSerial512VLoad;
void run_serial512(sycl::queue&q,const uint16_t*a,const uint8_t*w,const float*s,const int*ids,const int*map,const float*weights,uint16_t*y){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.parallel_for<DownRouteSerial512VLoad>(sycl::nd_range<1>(2048*32,256),props,[=](sycl::nd_item<1>it){
  constexpr int I=512,C=16;auto sg=it.get_sub_group();int lane=it.get_local_linear_id()%32,row=it.get_global_linear_id()/32,col=lane*C;float weighted[8]{};
  sycl::vec<int,8>local_ids,sorted_rows;sycl::vec<float,8>coefficients;
  local_ids.load(0,ids);sorted_rows.load(0,map);coefficients.load(0,weights);
#pragma unroll
  for(int route=0;route<8;++route){int expert=local_ids[route];
   if(expert>=0){
    int sorted=sorted_rows[route];sycl::vec<uint16_t,C>xv;xv.load(0,a+sorted*I+col);sycl::vec<uint32_t,C/4>wv;wv.load(0,reinterpret_cast<const uint32_t*>(w+expert*(2048*I)+row*I+col));
    float scale=s[expert*16*(I/128)+(row/128)*(I/128)+col/128]*256.f;float p[C];
#pragma unroll
    for(int word=0;word<C/4;++word){auto values=word540::expert(wv[word],scale);
#pragma unroll
     for(int j=0;j<4;++j)p[word*4+j]=values[j]*bf(xv[word*4+j]);}
    float value=sycl::reduce_over_group(sg,sum(p),sycl::plus<float>());
    weighted[route]=bf(rne(value))*coefficients[route];
   }
  }
  if(lane==0)y[row]=rne(sum(weighted));
 });
}
template<int I>class Down;
template<int I>void run(sycl::queue&q,const uint16_t*a,const uint8_t*w,const float*s,const int*ids,const int*map,const float*weights,uint16_t*y){
 if constexpr(I==512){run_serial512(q,a,w,s,ids,map,weights,y);return;}
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.submit([&](sycl::handler&h){sycl::local_accessor<float,1>partial(8,h);
  h.parallel_for<Down<I>>(sycl::nd_range<1>(2048*256,256),props,[=](sycl::nd_item<1>it){
   constexpr int C=I/32;auto sg=it.get_sub_group();int tid=it.get_local_linear_id(),lane=tid%32,route=tid/32,row=it.get_group_linear_id(),col=lane*C,expert=ids[route],sorted=map[route];
   float value=0.f;
   if(expert>=0){
   sycl::vec<uint16_t,C>xv;xv.load(0,a+sorted*I+col);sycl::vec<uint32_t,C/4>wv;wv.load(0,reinterpret_cast<const uint32_t*>(w+expert*(2048*I)+row*I+col));
   float scale=s[expert*16*(I/128)+(row/128)*(I/128)+col/128]*256.f;float p[C];
#pragma unroll
   for(int word=0;word<C/4;++word){auto values=word540::expert(wv[word],scale);
#pragma unroll
    for(int j=0;j<4;++j)p[word*4+j]=values[j]*bf(xv[word*4+j]);}
   value=sycl::reduce_over_group(sg,sum(p),sycl::plus<float>());
   }
   // The group barrier below must include all eight route subgroups.
   // Invalid subgroups skip memory reads but still write their zero partial.
   if(lane==0)partial[route]=expert>=0?bf(rne(value))*weights[route]:0.f;
   it.barrier(sycl::access::fence_space::local_space);
   if(tid==0){float v[8];
#pragma unroll
    for(int j=0;j<8;++j)v[j]=partial[j];y[row]=rne(sum(v));}
  });
 });
}
}
extern "C" const char* tp_expert_down_error(){return tp_expert_down::error.c_str();}
extern "C" int tp_expert_down_run(void*queue,const uint16_t*a,const uint8_t*w,const float*s,const int*ids,const int*map,const float*weights,uint16_t*y,int intermediate){
 if(!queue||!a||!w||!s||!ids||!map||!weights||!y||(intermediate!=128&&intermediate!=256&&intermediate!=512))return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(intermediate==128)tp_expert_down::run<128>(q,a,w,s,ids,map,weights,y);else if(intermediate==256)tp_expert_down::run<256>(q,a,w,s,ids,map,weights,y);else tp_expert_down::run<512>(q,a,w,s,ids,map,weights,y);return 0;}catch(const std::exception&e){tp_expert_down::error=e.what();return 1;}}
