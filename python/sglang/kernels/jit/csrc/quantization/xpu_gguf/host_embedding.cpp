// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
namespace sglang_host_embedding {
class Gather;
thread_local std::string error;
}
extern "C" const char* xpu_gguf_host_embedding_error(){return sglang_host_embedding::error.c_str();}
extern "C" int xpu_gguf_host_embedding(void* stream,const uint8_t* weight,const int64_t* ids,uint8_t* out,int T,int rows,int bytes){
 if(!stream||!weight||!ids||!out||T<1||T>4096||rows<1||rows>262144||bytes!=2720)return 2;
 try{auto& q=*static_cast<sycl::queue*>(stream);if(!q.has_property<sycl::property::queue::in_order>())return 2;
  size_t n=size_t(T)*bytes/16;
  q.parallel_for<sglang_host_embedding::Gather>(sycl::nd_range<1>((n+127)/128*128,128),[=](sycl::nd_item<1> it){
   size_t u=it.get_global_linear_id();if(u>=n)return;int t=u/(bytes/16),col=u%(bytes/16);int64_t row=ids[t];
   sycl::uint4 value(0);if(row>=0&&row<rows)value=*reinterpret_cast<const sycl::uint4*>(weight+size_t(row)*bytes+col*16);
   *reinterpret_cast<sycl::uint4*>(out+u*16)=value;
  });return 0;
 }catch(const std::exception& e){sglang_host_embedding::error=e.what();return 1;}
}
