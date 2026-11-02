// SPDX-License-Identifier: Apache-2.0
// B70 QSA index scores: page16, BF16 Q/K, four query heads, D128.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <limits>
#include <string>
namespace qsa_index_score {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
class Score;
SYCL_ESIMD_FUNCTION inline es::simd<float,16> positive(es::simd<float,16> x){auto y=es::max(x,0.f);y.merge(x,x!=x);return y;}
void launch(sycl::queue&q,const uint16_t*query,const uint16_t*cache,const int32_t*table,
 const int32_t*lengths,float*out,int rows,int width,int table_width,int pages,float invscale){
 const int tiles=(width+15)/16;
 q.parallel_for<Score>(sycl::nd_range<1>((size_t(rows)*tiles+3)/4*4,4),[=](sycl::nd_item<1>it) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
  if(it.get_global_linear_id()>=size_t(rows)*tiles)return;
  const int token=int(it.get_global_linear_id())/tiles,block=int(it.get_global_linear_id())%tiles;
  const int first=block*16,len=lengths[token];es::simd<uint32_t,16>cols(first,1);auto valid=(cols<uint32_t(width))&(cols<uint32_t(len));
  es::simd<float,16>value=-std::numeric_limits<float>::infinity();
  if(first<len&&block<table_width){
   int page=table[token*table_width+block];page=page<0?0:page;
   if(page<pages){
    const uint16_t*w=cache+size_t(page)*16*128;const uint16_t*x=query+size_t(token)*4*128;
    es::simd<float,64>sum=0.f;
#pragma unroll
    for(int k=0;k<128;k+=16){
     auto bw=es::load_2d<uint32_t,8,16,1,true,false>(reinterpret_cast<const uint32_t*>(w),255,15,255,k/2,0);
     auto aw=es::load_2d<uint32_t,8,4,1,false,false>(reinterpret_cast<const uint32_t*>(x),255,3,255,k/2,0);
     auto b=bw.template bit_cast_view<uint16_t>().read();auto a=aw.template bit_cast_view<uint16_t>().read();
     using P=es::xmx::dpas_argument_type;
     sum=es::xmx::dpas<8,4,float,float,uint16_t,uint16_t,P::bf16,P::bf16>(sum,b,a);
    }
    es::simd<float,16>s0=sum.template select<16,1>(0),s1=sum.template select<16,1>(16),s2=sum.template select<16,1>(32),s3=sum.template select<16,1>(48);
    s0=positive(s0);s1=positive(s1);s2=positive(s2);s3=positive(s3);
    value=((s0+s1)+(s2+s3))*invscale;
   }
  }
  value.merge(-std::numeric_limits<float>::infinity(),!valid);
  es::scatter<float,16>(out+size_t(token)*width,cols*4u,value,cols<uint32_t(width));
 });
}
}
extern "C" const char*xpu_qsa_index_score_error(){return qsa_index_score::error.c_str();}
extern "C" int xpu_qsa_index_score(void*queue,const uint16_t*query,const uint16_t*cache,const int32_t*table,
 const int32_t*lengths,float*out,int rows,int width,int table_width,int pages,float invscale){
 if(!queue||!query||!cache||!table||!lengths||!out||rows<1||rows>64||width<1||table_width<1||pages<1||!(invscale>0))return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
 qsa_index_score::launch(q,query,cache,table,lengths,out,rows,width,table_width,pages,invscale);return 0;
 }catch(const std::exception&e){qsa_index_score::error=e.what();return 1;}
}
