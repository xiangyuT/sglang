// SPDX-License-Identifier: Apache-2.0
// Extend the existing SG32 B70 router selection to Float32 E512/top10.
// Existing Torch softmax and selected-weight normalization remain outside.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <string>
namespace sglang_gguf_top10 {
thread_local std::string error;class Select;
void launch(sycl::queue&q,const float*scores,float*weights,int64_t*ids,int M){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.parallel_for<Select>(sycl::nd_range<1>(size_t(M)*32,32),props,[=](sycl::nd_item<1>it){
  const int row=it.get_group_linear_id(),lane=it.get_local_linear_id();auto sg=it.get_sub_group();
  sycl::vec<float,16>v;v.load(0,scores+row*512+lane*16);uint32_t keys[16];
#pragma unroll
  for(int e=0;e<16;++e){const uint32_t b=sycl::bit_cast<uint32_t>(v[e]);keys[e]=(b&0x7fffffffu)>0x7f800000u?0xffffffffu:(b&0x80000000u)?~b:b^0x80000000u;}
#pragma unroll
  for(int k=0;k<10;++k){
   uint32_t best=0;
#pragma unroll
   for(int e=0;e<16;++e)best=sycl::max(best,keys[e]);
   best=sycl::reduce_over_group(sg,best,sycl::maximum<uint32_t>());
   int expert=512;
#pragma unroll
   for(int e=0;e<16;++e)if(keys[e]==best)expert=sycl::min(expert,lane*16+e);
   expert=sycl::reduce_over_group(sg,expert,sycl::minimum<int>());
   if(lane==0){ids[row*10+k]=expert;weights[row*10+k]=scores[row*512+expert];}
   if(lane==expert/16)keys[expert%16]=0;
  }
 });
}
}
extern "C" const char*xpu_gguf_top10_error(){return sglang_gguf_top10::error.c_str();}
extern "C" int xpu_gguf_top10(void*queue,const float*x,float*w,int64_t*ids,int M){
 if(!queue||!x||!w||!ids||M<1||M>8)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;sglang_gguf_top10::launch(q,x,w,ids,M);return 0;}catch(const std::exception&e){sglang_gguf_top10::error=e.what();return 1;}
}
