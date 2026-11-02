// SPDX-License-Identifier: MIT
// Strata Q8_1 finite quantization; see STRATA-LICENSE and GGML-LICENSE.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
namespace sglang_expert_quantize {
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81Block)==36 && alignof(Q81Block)==4 && offsetof(Q81Block,qs)==4);
thread_local std::string error;
class QuantizeQ81;
inline float finite_half(float x){return sycl::fabs(x)>65504.0f?sycl::copysign(65504.0f,x):x;}
inline float warp_sum(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x+=sycl::permute_group_by_xor(sg,x,o);return x;}
inline float warp_max(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x=sycl::fmax(x,sycl::permute_group_by_xor(sg,x,o));return x;}
}
extern "C" int xpu_expert_quantize(void* queue,const float* x,void* output,int K,int T){
 if(!queue||!x||!output||(K!=640&&K!=2560&&K!=6144)||T<1||T>2048)return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);auto* y=static_cast<sglang_expert_quantize::Q81Block*>(output);
  const int count=K*T;
  q.parallel_for<sglang_expert_quantize::QuantizeQ81>(sycl::nd_range<1>(size_t((count+255)/256)*256,256),
   [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
    int i=it.get_global_linear_id();if(i>=count)return;auto sg=it.get_sub_group();float xi=x[i];
    float amax=sglang_expert_quantize::warp_max(sg,sycl::fabs(xi));float sum=sglang_expert_quantize::warp_sum(sg,xi);
    float d=sglang_expert_quantize::finite_half(amax/127.0f);
    float quantized=amax==0.0f?0.0f:sycl::round(xi/d);
    quantized=quantized>127.0f?127.0f:quantized< -127.0f? -127.0f:quantized;
    y[i/32].qs[i%32]=static_cast<int8_t>(quantized);
    if(i%32==0)y[i/32].ds=sycl::vec<float,2>(sglang_expert_quantize::finite_half(d),sglang_expert_quantize::finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
   });
  return 0;
 }catch(const std::exception& e){sglang_expert_quantize::error=e.what();return 1;}
}

extern "C" const char* xpu_expert_quantize_error(){return sglang_expert_quantize::error.c_str();}
