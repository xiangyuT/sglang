// SPDX-License-Identifier: MIT
// Original Strata finiteQ8_1 arithmetic, fused with the existing SGLang MoE producers.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>
namespace sglang_moe_quant1323 {
using bf16=sycl::ext::oneapi::bfloat16;
struct Q81Block{sycl::half2 ds;int8_t qs[32];};static_assert(sizeof(Q81Block)==36);
thread_local std::string error;
inline float finite_half(float x){return sycl::fabs(x)>65504.f?sycl::copysign(65504.f,x):x;}
inline float warp_sum(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x+=sycl::permute_group_by_xor(sg,x,o);return x;}
inline float warp_max(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x=sycl::fmax(x,sycl::permute_group_by_xor(sg,x,o));return x;}
template<bool Act>class ProduceQuantize;
template<bool Act>void launch(sycl::queue&q,const void*src,const float*up,float*value,void*output,int count){
 auto*y=static_cast<Q81Block*>(output);
 q.parallel_for<ProduceQuantize<Act>>(sycl::nd_range<1>(size_t((count+255)/256)*256,256),[=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(32)]]{
  const int i=it.get_global_linear_id();if(i>=count)return;auto sg=it.get_sub_group();float xi;
  if constexpr(Act){const float g=static_cast<const float*>(src)[i];xi=(g/(1.f+sycl::exp(-g)))*up[i];}
  else xi=float(static_cast<const bf16*>(src)[i]);
  value[i]=xi;
  const float amax=warp_max(sg,sycl::fabs(xi)),sum=warp_sum(sg,xi),d=finite_half(amax/127.f);
  float code=amax==0.f?0.f:sycl::round(xi/d);code=code>127.f?127.f:code< -127.f?-127.f:code;
  y[i/32].qs[i%32]=static_cast<int8_t>(code);
  if(i%32==0)y[i/32].ds=sycl::vec<float,2>(finite_half(d),finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
 });
}
}
extern "C" const char*xpu_moe_quant_error(){return sglang_moe_quant1323::error.c_str();}
extern "C" int xpu_moe_input_quantize(void*queue,const void*src,float*value,void*out,int K,int T){
 if(!queue||!src||!value||!out||K!=2560||T<1||T>32)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;sglang_moe_quant1323::launch<false>(q,src,nullptr,value,out,K*T);return 0;}catch(const std::exception&e){sglang_moe_quant1323::error=e.what();return 1;}
}
extern "C" int xpu_moe_activate_quantize(void*queue,const float*g,const float*up,float*value,void*out,int routes){
 if(!queue||!g||!up||!value||!out||routes<1||routes>320)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;sglang_moe_quant1323::launch<true>(q,g,up,value,out,routes*640);return 0;}catch(const std::exception&e){sglang_moe_quant1323::error=e.what();return 1;}
}
