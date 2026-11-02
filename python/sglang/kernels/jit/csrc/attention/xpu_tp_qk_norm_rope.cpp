// SPDX-License-Identifier: Apache-2.0
// TP2/B70 M1: Gemma QK norm256, NeoX rotary64 with BF16 products, gate copy.
// Reduction follows the installed Triton blocked[2]/SG32/four-subgroup recipe.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <string>
#pragma clang fp contract(off) reassociate(off)
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int,int,float,unsigned) noexcept;
namespace tp_qk_rope {
thread_local std::string error;
using BF=sycl::ext::oneapi::bfloat16;
inline float bf(uint16_t x){return sycl::bit_cast<float>(uint32_t(x)<<16);}
inline uint16_t rne(float x){return sycl::bit_cast<uint16_t>(BF(x));}
class QK;
void run(sycl::queue&q,const uint16_t*qg,const uint16_t*k,const uint16_t*qw,const uint16_t*kw,const uint16_t*cos,const int64_t*position,uint16_t*qo,uint16_t*ko,uint16_t*gate,int rows,float eps){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.submit([&](sycl::handler&h){sycl::local_accessor<float,1>partial(4,h);
  h.parallel_for<QK>(sycl::nd_range<1>(9*128,128),props,[=](sycl::nd_item<1>it){
   int head=it.get_group_linear_id(),tid=it.get_local_linear_id(),lane=tid%32,warp=tid/32,col=tid*2;
   bool key=head==8;const uint16_t*in=key?k:qg+head*512;const uint16_t*w=key?kw:qw;uint16_t*out=key?ko:qo+head*256;
   sycl::vec<uint16_t,2>raw;raw.load(0,in+col);float x0=bf(raw[0]),x1=bf(raw[1]);
   float sum=sycl::reduce_over_group(it.get_sub_group(),x0*x0+x1*x1,sycl::plus<float>());
   if(lane==0)partial[warp]=sum;
   it.barrier(sycl::access::fence_space::local_space);
   float a=tid<4?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
   float total=__spirv_GroupNonUniformFAdd(3,3,a,4u);
#else
   float total=a;
#endif
   if(tid==0)partial[0]=sycl::rsqrt(total*0x1p-8f+eps);
   it.barrier(sycl::access::fence_space::local_space);
   float inv=partial[0];int64_t p=*position;
   sycl::vec<uint16_t,2>output;
#pragma unroll
   for(int j=0;j<2;++j){int c=col+j;float value=bf(rne((bf(raw[j])*inv)*(bf(w[c])+1.f)));
    if(c<64){int half=c/32,l=c%32,other=c+(half?-32:32);float partner=bf(rne((bf(in[other])*inv)*(bf(w[other])+1.f)));
     if(p>=0&&p<rows){float cc=bf(cos[p*64+l]),ss=bf(cos[p*64+32+l]);float left=bf(rne(value*cc)),right=bf(rne(partner*ss));value=half?left+right:left-right;}else value=sycl::nan(0u);
    }
    output[j]=rne(value);
   }
   output.store(0,out+col);
   if(!key){sycl::vec<uint16_t,2>g;g.load(0,in+256+col);g.store(0,gate+head*256+col);}
  });
 });
}
}
extern "C" const char*tp_qk_rope_error(){return tp_qk_rope::error.c_str();}
extern "C" int tp_qk_rope_run(void*queue,const uint16_t*qg,const uint16_t*k,const uint16_t*qw,const uint16_t*kw,const uint16_t*cos,const int64_t*pos,uint16_t*qo,uint16_t*ko,uint16_t*gate,int rows,float eps){
 if(!queue||!qg||!k||!qw||!kw||!cos||!pos||!qo||!ko||!gate||rows<1||eps<=0)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);tp_qk_rope::run(q,qg,k,qw,kw,cos,pos,qo,ko,gate,rows,eps);return 0;}catch(const std::exception&e){tp_qk_rope::error=e.what();return 1;}
}
