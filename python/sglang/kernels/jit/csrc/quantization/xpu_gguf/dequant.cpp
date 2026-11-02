// SPDX-License-Identifier: Apache-2.0
#include "dequant.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <string>
#pragma clang fp contract(off) reassociate(off)
namespace sglang_gguf {
thread_local std::string error;
template<typename D>class Decode;
template<typename D>void run(sycl::queue&q,const void*packed,D*output,int type,int64_t count){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 const int64_t blocks=(count+255)/256,groups=(blocks+3)/4;
 q.parallel_for<Decode<D>>(sycl::nd_range<1>(groups*128,128),props,[=](sycl::nd_item<1>it){
  const int64_t block=it.get_global_linear_id()/32;int lane=it.get_local_linear_id()%32;
  // The 32/64-element formats write one contiguous eight-value run per lane.
  // A partial final 256-element superblock must not read beyond those blocks.
  if(block*256+lane*8>=count)return;
  dq_dispatch<D>(type,packed,block,output+block*256,lane);
 });
}
}
extern "C" const char* xpu_gguf_error(){return sglang_gguf::error.c_str();}
extern "C" int xpu_gguf_dequant(void*queue,const void*packed,void*output,int type,int64_t count,int output_type){
 int block=256;
 switch(type){case 8:case 20:block=32;break;case 42:block=64;break;
  case 2:case 3:case 6:case 7:block=32;break;
  case 11:case 12:case 13:case 14:case 16:case 17:case 18:case 21:case 22:case 23:case 29:break;
  default:return 2;}
 if(!queue||!packed||!output||count<=0||count%block||output_type<0||output_type>1)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);
  if(output_type==0)sglang_gguf::run(q,packed,static_cast<sycl::half*>(output),type,count);
  else sglang_gguf::run(q,packed,static_cast<float*>(output),type,count);
  return 0;
 }catch(const std::exception&e){sglang_gguf::error=e.what();return 1;}
}
