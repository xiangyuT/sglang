// SPDX-License-Identifier: Apache-2.0
// Single-GPU packed-word dense method adapted to TP shared/output shapes.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "xpu_tp_fp8_codec.hpp"
#include <cstdint>
#include <string>
#pragma clang fp contract(off) reassociate(off)
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int,int,float,unsigned int) noexcept;
namespace tp_aux_fp8 {
thread_local std::string error;
inline uint16_t rne(float x){uint32_t u=sycl::bit_cast<uint32_t>(x);if((u&0x7fffffffU)>0x7f800000U)return uint16_t((u>>16)|64);return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);}
template<int C>inline float sum(float(&v)[C]){
#pragma unroll
 for(int s=1;s<C;s*=2){
#pragma unroll
  for(int j=0;j<C;j+=2*s)v[j]+=v[j+s];}
 return v[0];}
template<int K,int R,bool Half>class Projection;
template<int K,int R,bool Half>void run(sycl::queue&q,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,uint16_t*y,int n,int ws0,int ws1){
 constexpr int WARP=K>=512?K/512:1,C=K/(WARP*32);namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<256>};
 q.submit([&](sycl::handler&h){sycl::local_accessor<float,1>partial(R*WARP,h);
  h.parallel_for<Projection<K,R,Half>>(sycl::nd_range<1>((n/R)*(WARP*32),WARP*32),props,[=](sycl::nd_item<1>it){
   auto sg=it.get_sub_group();int tid=it.get_local_linear_id(),col=tid*C,block=col/128,row0=it.get_group_linear_id()*R;
   sycl::vec<uint32_t,C/4>xv;xv.load(0,reinterpret_cast<const uint32_t*>(x+col));float xf[C],scale=(Half?float(sycl::half(xs[block])):xs[block])*256.f;
#pragma unroll
   for(int word=0;word<C/4;++word){auto v=word540::div256(xv[word]);
#pragma unroll
    for(int j=0;j<4;++j)xf[word*4+j]=Half?float(sycl::half(v[j]*scale)):v[j]*scale;}
   float raw_scale=ws[(row0/128)*ws0+block*ws1];float sw=(Half?float(sycl::half(raw_scale)):raw_scale)*256.f;
#pragma unroll
   for(int r=0;r<R;++r){sycl::vec<uint32_t,C/4>wv;wv.load(0,reinterpret_cast<const uint32_t*>(w+(row0+r)*K+col));float p[C],zf[C];
#pragma unroll
    for(int word=0;word<C/4;++word){auto v=word540::div256(wv[word]);
#pragma unroll
     for(int j=0;j<4;++j){int c=word*4+j;zf[c]=Half?float(sycl::half(v[j]*sw)):v[j]*sw;p[c]=zf[c]*xf[c];}}
    if constexpr(!Half){
#pragma unroll
     for(int c=0;c<C;c+=2){p[c]=sycl::fma(zf[c],xf[c],p[c+1]);p[c+1]=0.f;}
    }
    float value=sycl::reduce_over_group(sg,sum(p),sycl::plus<float>());
    if constexpr(WARP==1){if(tid==0)y[row0+r]=rne(value);}else{if(tid%32==0)partial[r*WARP+tid/32]=value;}
   }
   if constexpr(WARP>1){it.barrier(sycl::access::fence_space::local_space);float part=tid<R*WARP?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
    float value=__spirv_GroupNonUniformFAdd(3,3,part,WARP);
#else
    float value=part;
#endif
    if(tid<R*WARP&&tid%WARP==0)y[row0+tid/WARP]=rne(value);
   }
  });
 });
}
template<int K,bool Half>void select(sycl::queue&q,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,uint16_t*y,int n,int ws0,int ws1,int mode){
 switch(mode){case 0:run<K,1,Half>(q,x,w,xs,ws,y,n,ws0,ws1);break;case 1:run<K,2,Half>(q,x,w,xs,ws,y,n,ws0,ws1);break;case 2:run<K,4,Half>(q,x,w,xs,ws,y,n,ws0,ws1);break;case 3:run<K,8,Half>(q,x,w,xs,ws,y,n,ws0,ws1);break;}
}
}
extern "C" const char* tp_aux_fp8_error(){return tp_aux_fp8::error.c_str();}
extern "C" int tp_aux_fp8_run(void*queue,const uint8_t*x,const uint8_t*w,const float*xs,const float*ws,uint16_t*y,int n,int k,int ws0,int ws1,int mode){
 if(!queue||!x||!w||!xs||!ws||!y||ws0<1||ws1<1||mode<0||mode>7||!((k==2048&&(n==2048||n==512||n==256))||(n==2048&&(k==1024||k==256||k==128))))return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);
if(mode<4){switch(k){
case 2048:tp_aux_fp8::select<2048,true>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 1024:tp_aux_fp8::select<1024,true>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 256:tp_aux_fp8::select<256,true>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 128:tp_aux_fp8::select<128,true>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
}}
else{switch(k){
case 2048:tp_aux_fp8::select<2048,false>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 1024:tp_aux_fp8::select<1024,false>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 256:tp_aux_fp8::select<256,false>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
case 128:tp_aux_fp8::select<128,false>(q,x,w,xs,ws,y,n,ws0,ws1,mode%4);break;
}}return 0;}catch(const std::exception&e){tp_aux_fp8::error=e.what();return 1;}}
