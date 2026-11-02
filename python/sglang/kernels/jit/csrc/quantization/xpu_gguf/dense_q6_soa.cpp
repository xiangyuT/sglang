// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Strata dot_q6/dot_iq4nl/dot_q8 formula port; exact32FP32chains/tree.
// SGLang ESIMD DP4A row packets, original encoded row bytes and outputstride.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <string>
namespace sglang_dense_q6_soa {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
template<int L>SYCL_ESIMD_FUNCTION inline es::simd<uint32_t,L> iq4_e_lookup(es::simd<uint32_t,L> q){
    constexpr uint32_t table[]={0x81818181u,0x98989898u,0xadadadadu,0xbfbfbfbfu,
        0xcfcfcfcfu,0xddddddddu,0xeaeaeaeau,0xf6f6f6f6u,
        0x01010101u,0x0d0d0d0du,0x19191919u,0x26262626u,
        0x35353535u,0x45454545u,0x59595959u,0x71717171u};
    constexpr auto select=(es::bfn_t::x&~es::bfn_t::z)|(es::bfn_t::y&es::bfn_t::z);
    es::simd<uint32_t,L> mask[4],a[8],b[4],c[2];
#pragma unroll
    for(int i=0;i<4;++i){es::simd<uint32_t,L> bit=(q>>i)&0x01010101u;mask[i]=(bit<<8)-bit;}
#pragma unroll
    for(int i=0;i<8;++i)a[i]=es::bfn<select>(es::simd<uint32_t,L>(table[2*i]),es::simd<uint32_t,L>(table[2*i+1]),mask[0]);
#pragma unroll
    for(int i=0;i<4;++i)b[i]=es::bfn<select>(a[2*i],a[2*i+1],mask[1]);
#pragma unroll
    for(int i=0;i<2;++i)c[i]=es::bfn<select>(b[2*i],b[2*i+1],mask[2]);
    return es::bfn<select>(c[0],c[1],mask[3]);
}
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t,16> word(const uint8_t*w,es::simd<uint32_t,16>off,es::simd_mask<16>valid){
 auto a=es::gather<uint16_t,16>(reinterpret_cast<const uint16_t*>(w),off,valid);
 auto b=es::gather<uint16_t,16>(reinterpret_cast<const uint16_t*>(w),off+2u,valid);
 es::simd<uint32_t,16> lo=a,hi=b;return lo|(hi<<16);
}
SYCL_ESIMD_FUNCTION inline int32_t actword(const block_q8_1*x,int block,int word){
 es::simd<uint32_t,1>off=uint32_t(block*36+4+word*4);
 auto v=es::gather<int32_t,1>(reinterpret_cast<const int32_t*>(x),off);return v[0];
}
SYCL_ESIMD_FUNCTION inline float actscale(const block_q8_1*x,int block){
 es::simd<uint32_t,1>off=uint32_t(block*36);
 auto v=es::gather<uint16_t,1>(reinterpret_cast<const uint16_t*>(x),off);
 es::simd<float,1> ds=v.template bit_cast_view<sycl::half>().read();return ds[0];
}
SYCL_ESIMD_FUNCTION inline es::simd<int32_t,16> dot(es::simd<uint32_t,16>w,int32_t a,es::simd<int32_t,16>sum){
 es::simd<int32_t,16>sw=w,sa=a;return es::dp4a<int32_t>(sum,sw,sa);
}
template<int Type,int NT>class DensePacket;
template<int Type,int NT>
void launch(sycl::queue&q,const uint8_t*w,const block_q8_1*x,float*y,int K,int N,int RB,int stride){
 constexpr int QK=Type==14?256:32,IPB=Type==14?32:Type==8?4:2,STEP=Type==14?1:2,WG=16,NR=16;
 q.parallel_for<DensePacket<Type,NT>>(sycl::nd_range<1>(size_t((N+15)/16)*WG,WG),[=](sycl::nd_item<1>it)[[intel::sycl_explicit_simd]]{
#pragma clang fp contract(off)
  es::slm_init<WG*NT*NR*4>();
  int sid=it.get_local_id(0),base=it.get_group(0)*16;
  es::simd<uint32_t,16>rows(base,1);auto valid=rows<uint32_t(N);es::simd<uint32_t,16>rowoff=rows*uint32_t(RB);
  es::simd<float,16>acc[2][NT];
#pragma unroll
  for(int branch=0;branch<2;++branch){
#pragma unroll
   for(int t=0;t<NT;++t)acc[branch][t]=0.f;
   for(int index=sid+branch*16;index<K/QK*IPB;index+=32){
    const int kb=index/IPB,part=STEP*(index%IPB);
    if constexpr(Type==14){
     const auto* block=w+size_t(base)*RB+kb*3360;
     auto vl=es::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(block+part*64));
     auto vh=es::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(block+2048+64*(8*(part/16)+part%8)))>>(2*((part%16)/8));
     auto db=es::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block+3328));
     es::simd<float,16>dw=db.template bit_cast_view<sycl::half>().read();
     es::simd<uint32_t,16>codes[2];es::simd<int32_t,16>scales[2];
#pragma unroll
     for(int i=0;i<2;++i){
      auto c=((vl>>(4*i))&0x0f0f0f0fu)|(((vh>>(4*i))<<4)&0x30303030u);
      codes[i]=((c|0x80808080u)-0x20202020u)^0x80808080u;
      auto s=es::block_load<int8_t,16>(reinterpret_cast<const int8_t*>(block+3072+16*(8*(part/16)+(part%16)/4+4*i)));scales[i]=s;
     }
// Batch the identical-per-output-row activation requests across tokens.
     // Keep Q8 bytes, FP32 terms and all 32 accumulation chains unchanged.
     constexpr int NA=NT<=1?1:NT<=2?2:4;
     es::simd<uint32_t,NA> token_offsets(0,uint32_t((K/32)*36));
     es::simd<uint32_t,NA> token_ids(0,1);
     auto token_mask=token_ids<uint32_t(NT);
     es::simd<int32_t,NA> aw[2];
     es::simd<float,NA> ads[2];
#pragma unroll
     for(int i=0;i<2;++i){
      const int ab=kb*8+4*(part/16)+(part%16)/8+2*i;
      auto off=token_offsets+uint32_t(ab*36);
      aw[i]=es::gather<int32_t,NA>(reinterpret_cast<const int32_t*>(x),
                                  off+uint32_t(4+(part%8)*4),token_mask);
      auto bits=es::gather<uint16_t,NA>(reinterpret_cast<const uint16_t*>(x),off,token_mask);
      ads[i]=bits.template bit_cast_view<sycl::half>().read();
     }
#pragma unroll
     for(int t=0;t<NT;++t){
      es::simd<float,16>inner=0.f;
#pragma unroll
      for(int i=0;i<2;++i){
       es::simd<int32_t,16>z=0;auto d=dot(codes[i],aw[i][t],z)*scales[i];
       es::simd<float,16>v=d,term=v*float(ads[i][t]);term.merge(0.f,!valid);inner=inner+term;
      }
      es::simd<float,16>term=dw*inner;term.merge(0.f,!valid);acc[branch][t]=acc[branch][t]+term;
     }
    }else{
     constexpr int BS=Type==8?34:18;
     const auto off=rowoff+uint32_t(kb*BS);
     auto db=es::gather<uint16_t,16>(reinterpret_cast<const uint16_t*>(w),off,valid);
     es::simd<float,16>dw=db.template bit_cast_view<sycl::half>().read();
     es::simd<uint32_t,16>lo[2],hi[2];
#pragma unroll
     for(int j=0;j<2;++j){
      auto c=word(w,off+uint32_t(2+(part+j)*4),valid);
      if constexpr(Type==20){lo[j]=iq4_e_lookup<16>(c&0x0f0f0f0fu);hi[j]=iq4_e_lookup<16>((c>>4)&0x0f0f0f0fu);}
      else lo[j]=c;
     }
#pragma unroll
     for(int t=0;t<NT;++t){
      const int ab=t*(K/32)+kb;es::simd<int32_t,16>d=0;
#pragma unroll
      for(int j=0;j<2;++j){d=dot(lo[j],actword(x,ab,part+j),d);if constexpr(Type==20)d=dot(hi[j],actword(x,ab,part+j+4),d);}
      es::simd<float,16>fv=d,factor=dw*actscale(x,ab),term=factor*fv;
      term.merge(0.f,!valid);acc[branch][t]=acc[branch][t]+term;
     }
    }
   }
  }
#pragma unroll
  for(int t=0;t<NT;++t)es::slm_block_store<float,16>(sid*NT*64+t*64,acc[0][t]+acc[1][t]);
  es::barrier();
  if(sid<NT){
   es::simd<float,16>v[16];
#pragma unroll
   for(int i=0;i<16;++i)v[i]=es::slm_block_load<float,16>(i*NT*64+sid*64);
#pragma unroll
   for(int o=8;o;o/=2){
#pragma unroll
    for(int i=0;i<o;++i)v[i]=v[i]+v[i+o];
   }
   es::scatter<float,16>(y,(rows+uint32_t(sid)*uint32_t(stride))*4u,v[0],valid);
  }
 });
}
template<int Type>void dispatch(int T,sycl::queue&q,const uint8_t*w,const block_q8_1*x,float*y,int K,int N,int RB,int stride){
#define RUN(E) case E:launch<Type,E>(q,w,x,y,K,N,RB,stride);break
 switch(T){RUN(1);RUN(2);RUN(3);RUN(4);}
#undef RUN
}
}
extern "C" const char*xpu_gguf_dense_q6_soa_error(){return sglang_dense_q6_soa::error.c_str();}
extern "C" int xpu_gguf_dense_q6_soa(void*queue,const void*w,const void*x,float*y,int Type,int K,int N,int T,int RB,int stride){
 const int QK=Type==14?256:32,BS=Type==14?210:Type==8?34:18;
 if(!queue||!w||!x||!y||Type!=14||K!=2560||K%QK||N<16||N%16||N>248320||T<1||T>4||RB!=K/QK*BS||stride<N)return 2;
 try{
  auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define RUN(KIND) case KIND:sglang_dense_q6_soa::dispatch<KIND>(T,q,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),y,K,N,RB,stride);break
  switch(Type){RUN(14);}
#undef RUN
  return 0;
 }catch(const std::exception&e){sglang_dense_q6_soa::error=e.what();return 1;}
}

namespace sglang_dense_q6_soa {class Permute;}
extern "C" int xpu_gguf_q6_field_permute(void* queue,const uint8_t* input,uint8_t* output,int N,int inverse) {
 if(!queue||!input||!output||input==output||N<16||N>248320||(inverse!=0&&inverse!=1))return 2;
 try {
  auto& q=*static_cast<sycl::queue*>(queue);
  if(!q.has_property<sycl::property::queue::in_order>())return 2;
  const size_t bytes=size_t(N)*2100,full=N/16*16;
  q.parallel_for<sglang_dense_q6_soa::Permute>(sycl::range<1>((bytes+255)/256*256),[=](sycl::id<1> id){
   const size_t i=id[0];if(i>=bytes)return;
   const size_t row=i/2100,off=i%2100;const int kb=off/210,b=off%210;
   if(row>=full){output[i]=input[i];return;}
   const size_t p=(row/16)*33600+kb*3360+
      (b<128?(b/4)*64+(row%16)*4+b%4:
       b<192?2048+((b-128)/4)*64+(row%16)*4+(b-128)%4:
       b<208?3072+(b-192)*16+row%16:3328+(row%16)*2+b-208);
   if(inverse)output[i]=input[p];else output[p]=input[i];
  });return 0;
 }catch(const std::exception& e){sglang_dense_q6_soa::error=e.what();return 1;}
}
