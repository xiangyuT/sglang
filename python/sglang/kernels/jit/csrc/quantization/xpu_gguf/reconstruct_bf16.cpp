// SPDX-License-Identifier: Apache-2.0
// Fused reconstruction of the original BF16 dense matrix, without an FP32 image.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include "dequant.hpp"
#include <string>
using BF=sycl::ext::oneapi::bfloat16;
namespace sglang_gguf {
template<> inline BF cvt<BF>(float v){return BF(v);}
template<> inline void store_run<BF,4>(BF* y,const float* v){sycl::vec<uint16_t,4> b;for(int i=0;i<4;++i)b[i]=sycl::bit_cast<uint16_t>(BF(v[i]));*reinterpret_cast<sycl::vec<uint16_t,4>*>(y)=b;}
template<> inline void store_run<BF,8>(BF* y,const float* v){sycl::vec<uint16_t,8> b;for(int i=0;i<8;++i)b[i]=sycl::bit_cast<uint16_t>(BF(v[i]));*reinterpret_cast<sycl::vec<uint16_t,8>*>(y)=b;}
}
namespace sglang_dense_reconstruct {
thread_local std::string error;
class Original;
void original(sycl::queue& q,const void* packed,BF* out,int type,int64_t count){
 size_t groups=(size_t(count)+1023)/1024;
 q.parallel_for<Original>(sycl::nd_range<1>(groups*128,128),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
  int64_t block=it.get_global_linear_id()/32;int lane=it.get_local_linear_id()%32;
  if(block*256+lane*8>=count)return;sglang_gguf::dq_dispatch<BF>(type,packed,block,out+block*256,lane);
 });
}
template<int K>class Q8Soa;
template<int K>void q8soa(sycl::queue& q,const uint8_t* packed,BF* out,int N){
 size_t units=size_t(N)*K/8;
 q.parallel_for<Q8Soa<K>>(sycl::nd_range<1>((units+127)/128*128,128),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
  size_t u=it.get_global_linear_id();if(u>=units)return;
  int row=u/(K/8),k=(u%(K/8))*8,lane=row%16,kb=k/32;
  const auto* b=packed+size_t(row/16)*(K/32*34)*16+kb*544;
  float d=float(*reinterpret_cast<const sycl::half*>(b+lane*2));sycl::vec<uint16_t,8> result;
  for(int j=0;j<8;++j){int c=k%32+j;auto v=*reinterpret_cast<const int8_t*>(b+32+(c/4)*64+lane*4+c%4);result[j]=sycl::bit_cast<uint16_t>(BF(float(v)*d));}
  *reinterpret_cast<sycl::vec<uint16_t,8>*>(out+u*8)=result;
 });
}
}
extern "C" const char* xpu_gguf_reconstruct_bf16_error(){return sglang_dense_reconstruct::error.c_str();}
extern "C" int xpu_gguf_reconstruct_bf16(void* stream,const void* packed,void* out,int type,int K,int N,int soa){
 if(!stream||!packed||!out||packed==out||(K!=640&&K!=2560&&K!=6144)||N<1||N>248320||(soa!=0&&soa!=1))return 2;
 if(type!=8&&type!=14&&type!=20&&type!=21&&type!=23&&type!=42)return 2;
 int block=(type==8||type==20)?32:type==42?64:256;if(K%block)return 2;
 if(soa&&(type!=8||N%16))return 2;
 try{auto& q=*static_cast<sycl::queue*>(stream);if(!q.has_property<sycl::property::queue::in_order>())return 2;
  if(!soa)sglang_dense_reconstruct::original(q,packed,static_cast<BF*>(out),type,int64_t(N)*K);
  else if(K==640)sglang_dense_reconstruct::q8soa<640>(q,static_cast<const uint8_t*>(packed),static_cast<BF*>(out),N);
  else if(K==2560)sglang_dense_reconstruct::q8soa<2560>(q,static_cast<const uint8_t*>(packed),static_cast<BF*>(out),N);
  else sglang_dense_reconstruct::q8soa<6144>(q,static_cast<const uint8_t*>(packed),static_cast<BF*>(out),N);
  return 0;
 }catch(const std::exception& e){sglang_dense_reconstruct::error=e.what();return 1;}
}
