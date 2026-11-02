// SPDX-License-Identifier: Apache-2.0
// B70 AR small GEMV: retained FP16/FP32 dequantization and reduction trees.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int,int,float,unsigned int) noexcept;
namespace ar_small_b70_abi1 {
thread_local std::string error;
template<int K,int R> class Gemv;
inline float fp8(uint8_t v) {
 uint16_t mag=v&127,sign=uint16_t(v&128)<<8;
 uint16_t bits=mag==127?sign|0x7e00:sign|(mag<<7);
 return float(sycl::bit_cast<sycl::half>(bits))*256.f;
}
inline uint16_t bf16(float x) {
 uint32_t u=sycl::bit_cast<uint32_t>(x);
 if((u&0x7fffffffU)>0x7f800000U)return uint16_t((u>>16)|64);
 return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);
}
inline float sum16(float (&p)[16]) {
#pragma unroll
 for(int s=1;s<16;s*=2) {
#pragma unroll
  for(int i=0;i<16;i+=2*s)p[i]=p[i]+p[i+s];
 }
 return p[0];
}
template<int K,int R>
void run(sycl::queue& q,const uint8_t* x,const uint8_t* w,const float* xs,const float* ws,
 uint16_t* y,int n,int ws0,int ws1) {
 constexpr int WG=K/16,WARP=WG/32;
 namespace one=sycl::ext::oneapi::experimental;
 namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<256>};
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<float,1> partial(R*WARP,h);
  h.parallel_for<Gemv<K,R>>(sycl::nd_range<1>((n/R)*WG,WG),props,[=](sycl::nd_item<1> it){
   auto sg=it.get_sub_group();int tid=it.get_local_linear_id(),col=tid*16,group=col/128;
   int row0=it.get_group_linear_id()*R;
   sycl::vec<uint32_t,4> xv;xv.load(0,reinterpret_cast<const uint32_t*>(x+col));
   float scale=xs[group];if constexpr(K==512)scale=float(sycl::half(scale));
   float xf[16];
#pragma unroll
   for(int j=0;j<16;++j){float v=fp8(uint8_t(xv[j/4]>>((j%4)*8)))*scale;
    if constexpr(K==512)v=float(sycl::half(v));xf[j]=v;}
#pragma unroll
   for(int r=0;r<R;++r){
    int row=row0+r;sycl::vec<uint32_t,4> v;
    v.load(0,reinterpret_cast<const uint32_t*>(w+row*K+col));
    float sw=ws[(row/128)*ws0+group*ws1];if constexpr(K==512)sw=float(sycl::half(sw));
    float p[16],zf[16];
#pragma unroll
    for(int j=0;j<16;++j){float z=fp8(uint8_t(v[j/4]>>((j%4)*8)))*sw;
     if constexpr(K==512)z=float(sycl::half(z));zf[j]=z;p[j]=z*xf[j];}
    if constexpr(K==2048){
#pragma unroll
     for(int j=0;j<16;j+=2){p[j]=sycl::fma(zf[j],xf[j],p[j+1]);p[j+1]=0.f;}
    }
    float value=sycl::reduce_over_group(sg,sum16(p),sycl::plus<float>());
    if constexpr(WARP==1){if(tid==0)y[row]=bf16(value);}
    else {if(tid%32==0)partial[r*WARP+tid/32]=value;}
   }
   if constexpr(WARP>1){
    it.barrier(sycl::access::fence_space::local_space);
    float part=tid<R*4?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
    float value=__spirv_GroupNonUniformFAdd(3,3,part,4);
#else
    float value=part;
#endif
    if(tid<R*4 && tid%4==0)y[row0+tid/4]=bf16(value);
   }
  });
 });
}
}
extern "C" int ar_small_abi(){return 1;}
extern "C" const char* ar_small_error(){return ar_small_b70_abi1::error.c_str();}
extern "C" int ar_small_run(void* queue,const uint8_t* x,const uint8_t* w,const float* xs,
 const float* ws,uint16_t* y,int n,int k,int ws0,int ws1,int rows){
 try{
  if(!queue||!x||!w||!xs||!ws||!y||ws0<1||ws1<1)return 2;
  auto& q=*static_cast<sycl::queue*>(queue);
  if(n==1024 && k==2048 && rows==4)
   ar_small_b70_abi1::run<2048,4>(q,x,w,xs,ws,y,n,ws0,ws1);
  else if(n==2048 && k==512 && rows==2)
   ar_small_b70_abi1::run<512,2>(q,x,w,xs,ws,y,n,ws0,ws1);
  else return 2;
  return 0;
 }catch(const std::exception& e){ar_small_b70_abi1::error=e.what();return 1;}
}
