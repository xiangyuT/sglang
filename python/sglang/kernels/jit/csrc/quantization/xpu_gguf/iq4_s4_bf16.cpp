// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd rawIQ4 signednibblebasis.
#include "dequant.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <string>
namespace sglang_iq4_s4_bf16 {
namespace i4e=sycl::ext::intel::esimd;
template<int L>SYCL_ESIMD_FUNCTION inline i4e::simd<uint16_t,L> bf16_output(i4e::simd<float,L>v){i4e::simd<sycl::ext::oneapi::bfloat16,L>b=v;return b.template bit_cast_view<uint16_t>().read();}

using Q81Block=block_q8_1;
thread_local std::string error;
template<int L> SYCL_ESIMD_FUNCTION inline i4e::simd<uint32_t,L> iq4_int4_coarse(i4e::simd<uint32_t,L> q){
 auto q1=q>>1,q2=q>>2,q3=q>>3;
 auto plus=i4e::bfn<static_cast<i4e::bfn_t>(0x7e)>(q,q1,q2)&~q3&0x11111111u;
 auto minus=i4e::bfn<static_cast<i4e::bfn_t>(0x38)>(q,q1,q2)&q3&0x11111111u;
 return (q+plus-minus)^0x88888888u;
}
template<int L> SYCL_ESIMD_FUNCTION inline i4e::simd<uint32_t,L> iq4_int4_residual(i4e::simd<uint32_t,L> q){
 auto q1=q>>1,q2=q>>2,q3=q>>3;
 i4e::simd<uint32_t,L> result=0;
 {auto lo=i4e::bfn<static_cast<i4e::bfn_t>(61)>(q,q1,q2);
   auto hi=i4e::bfn<static_cast<i4e::bfn_t>(247)>(q,q1,q2);
   result|=(i4e::bfn<static_cast<i4e::bfn_t>(0xca)>(lo,hi,q3)&0x11111111u)<<0;}
 {auto lo=i4e::bfn<static_cast<i4e::bfn_t>(216)>(q,q1,q2);
   auto hi=i4e::bfn<static_cast<i4e::bfn_t>(8)>(q,q1,q2);
   result|=(i4e::bfn<static_cast<i4e::bfn_t>(0xca)>(lo,hi,q3)&0x11111111u)<<1;}
 {auto lo=i4e::bfn<static_cast<i4e::bfn_t>(188)>(q,q1,q2);
   auto hi=i4e::bfn<static_cast<i4e::bfn_t>(58)>(q,q1,q2);
   result|=(i4e::bfn<static_cast<i4e::bfn_t>(0xca)>(lo,hi,q3)&0x11111111u)<<2;}
 {auto lo=i4e::bfn<static_cast<i4e::bfn_t>(126)>(q,q1,q2);
   auto hi=i4e::bfn<static_cast<i4e::bfn_t>(70)>(q,q1,q2);
   result|=(i4e::bfn<static_cast<i4e::bfn_t>(0xca)>(lo,hi,q3)&0x11111111u)<<3;}
 return result;
}
template<int L> SYCL_ESIMD_FUNCTION inline i4e::simd<uint16_t,L*2> iq4_int4_vnni(i4e::simd<uint32_t,L> q){
 // INT4 Src1 packs eight K elements per dword; keep the 16-row dword order.
 // Repacking as four-element halfwords mixed adjacent output rows in source01.
 return q.template bit_cast_view<uint16_t>().read();
}
template<int NT,int KB>class IQ4PairAllTInt4;
template<int NT,int KB>
void iq4_pair_all_t_launch_int4(const uint8_t* weights,const Q81Block* acts,float* y,int N,int stride,sycl::queue* q){
    constexpr int NR=16,XB=KB*8,RB=KB*136;
    q->parallel_for<IQ4PairAllTInt4<NT,KB>>(sycl::nd_range<1>(size_t((N+15)/16)*16,16),
        [=](sycl::nd_item<1> it)[[intel::sycl_explicit_simd]]{
            i4e::slm_init<32*NT*16*4>();
            const int sid=it.get_local_linear_id(),base=it.get_group_linear_id()*16;
            i4e::simd<uint32_t,16> row(base,1);auto valid=row<uint32_t(N);
            i4e::simd<float,16> acc[2][NT];
#pragma unroll
            for(int p=0;p<2;++p){
#pragma unroll
                for(int mr=0;mr<NT;++mr)acc[p][mr]=0.f;
            }
            for(int block=sid/4;block<KB;block+=4){
                auto header=i4e::load_2d<uint32_t,2,16,1,true,false>(reinterpret_cast<const uint32_t*>(weights),
                    RB-1,N-1,RB-1,block*34,base);
                i4e::simd<uint32_t,16> dh=header.template select<16,1>(0),sl=header.template select<16,1>(16);
                i4e::simd<uint16_t,16> bits=dh&0xffffu;
                i4e::simd<float,16> d=bits.template bit_cast_view<sycl::half>().read();
                {const int quarter=sid%4;
                    auto packed8=i4e::load_2d<uint32_t,8,16,1,true,false>(reinterpret_cast<const uint32_t*>(weights),
                        RB-1,N-1,RB-1,block*34+2+quarter*8,base);
#pragma unroll
                    for(int half=0;half<2;++half){
                        const int part=quarter*2+half,seg=block*8+part;
                        i4e::simd<uint32_t,64> packed=packed8.template select<64,1>(half*64);
                        i4e::simd<uint32_t,16> ls=((sl>>(4*part))&15u)|(((dh>>(16+2*part))&3u)<<4);
                        i4e::simd<int32_t,16> scale=ls;scale-=32;
                        i4e::simd<float,16> fscale=scale;const i4e::simd<float,16> dw=d*fscale;
                        auto Bc=iq4_int4_vnni<64>(iq4_int4_coarse<64>(packed));
                        auto Br=iq4_int4_vnni<64>(iq4_int4_residual<64>(packed));
                        auto a_words=i4e::load_2d<uint32_t,8,NT,1,false,false>(reinterpret_cast<const uint32_t*>(acts),
                            XB*36-1,NT-1,XB*36-1,seg*9+1,0);
                        i4e::simd<int8_t,NT*32> A=a_words.template bit_cast_view<int8_t>();
                        i4e::simd<int32_t,NT*16> zero=0;
                        i4e::simd<int8_t,NT*32> interleaved;
#pragma unroll
                        for(int mr=0;mr<NT;++mr){
                            interleaved.template select<16,2>(mr*32)=A.template select<16,1>(mr*32);
                            interleaved.template select<16,2>(mr*32+1)=A.template select<16,1>(mr*32+16);
                        }
                        using P=i4e::xmx::dpas_argument_type;
                        auto Cc=i4e::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Bc,interleaved);
                        auto Cr=i4e::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Br,interleaved);
                        auto C=Cc*16+Cr;
#pragma unroll
                        for(int mr=0;mr<NT;++mr){
                            i4e::simd<uint16_t,1> dbits=reinterpret_cast<const uint16_t*>(acts)[(mr*XB+seg)*18];
                            i4e::simd<float,1> dvalue=dbits.template bit_cast_view<sycl::half>().read();
                            const float ds=dvalue[0];
                            i4e::simd<int32_t,16> ci=C.template select<16,1>(mr*16);
                            i4e::simd<float,16> dot=ci;const i4e::simd<float,16> factor=dw*ds;
                            i4e::simd<float,16> term=factor*dot;
                            term.merge(0.f,!valid);
                            const i4e::simd<float,16> next=acc[half][mr]+term;acc[half][mr].merge(next,valid);
                        }
                    }
                }
            }
#pragma unroll
            for(int part=0;part<2;++part){
#pragma unroll
                for(int mr=0;mr<NT;++mr)i4e::slm_block_store<float,16>((sid*2+part)*NT*64+mr*64,acc[part][mr]);
            }
            i4e::barrier();
            if(sid<NT){
            i4e::simd<float,16> p[16];
#pragma unroll
            for(int v=0;v<16;++v)p[v]=i4e::slm_block_load<float,16>(v*NT*64+sid*64)+
                i4e::slm_block_load<float,16>((v+16)*NT*64+sid*64);
#pragma unroll
            for(int off=8;off;off/=2){
#pragma unroll
                for(int v=0;v<off;++v)p[v]+=p[v+off];
            }
            i4e::simd<uint32_t,16> offsets=(row+uint32_t(sid)*uint32_t(stride))*4u;
            i4e::scatter<uint16_t,16>(reinterpret_cast<uint16_t*>(y),offsets/2u,bf16_output(p[0]),valid);
            }
        });
}

template<int NT,int KB>class IQ4Q8SplitInt4;
template<int NT,int KB>
void iq4_q8_split_launch_int4(const uint8_t* weights,const Q81Block* acts,float* y,int N,int stride,sycl::queue* q){
    constexpr int NR=16,XB=KB*8,RB=KB*136;
    q->parallel_for<IQ4Q8SplitInt4<NT,KB>>(sycl::nd_range<1>(size_t((N+15)/16)*16,16),
        [=](sycl::nd_item<1> it)[[intel::sycl_explicit_simd]]{
            i4e::slm_init<32*NT*16*4>();
            const int sid=it.get_local_linear_id(),base=it.get_group_linear_id()*16;
            i4e::simd<uint32_t,16> row(base,1);auto valid=row<uint32_t(N);
            i4e::simd<float,16> acc[2][NT];
#pragma unroll
            for(int p=0;p<2;++p){
#pragma unroll
                for(int mr=0;mr<NT;++mr)acc[p][mr]=0.f;
            }
            for(int block=sid/4;block<KB;block+=4){
                auto header=i4e::load_2d<uint32_t,2,16,1,true,false>(reinterpret_cast<const uint32_t*>(weights),
                    RB-1,N-1,RB-1,block*34,base);
                i4e::simd<uint32_t,16> dh=header.template select<16,1>(0),sl=header.template select<16,1>(16);
                i4e::simd<uint16_t,16> bits=dh&0xffffu;
                i4e::simd<float,16> d=bits.template bit_cast_view<sycl::half>().read();
                {const int quarter=sid%4;
                    const int first_segment=block*8+quarter*2;
                    auto aq=i4e::load_2d<uint32_t,16,NT,1,false,false>(reinterpret_cast<const uint32_t*>(acts),
                        XB*36-1,NT-1,XB*36-1,first_segment*9,0);
                    auto tail=i4e::load_2d<uint32_t,2,NT,1,false,false>(reinterpret_cast<const uint32_t*>(acts),
                        XB*36-1,NT-1,XB*36-1,first_segment*9+16,0);
                    constexpr int AP=decltype(aq)::length/NT,TP=decltype(tail)::length/NT;
                    static_assert(AP==16 && TP==2);
                    auto packed8=i4e::load_2d<uint32_t,8,16,1,true,false>(reinterpret_cast<const uint32_t*>(weights),
                        RB-1,N-1,RB-1,block*34+2+quarter*8,base);
#pragma unroll
                    for(int half=0;half<2;++half){
                        const int part=quarter*2+half,seg=block*8+part;
                        i4e::simd<uint32_t,64> packed=packed8.template select<64,1>(half*64);
                        i4e::simd<uint32_t,16> ls=((sl>>(4*part))&15u)|(((dh>>(16+2*part))&3u)<<4);
                        i4e::simd<int32_t,16> scale=ls;scale-=32;
                        i4e::simd<float,16> fscale=scale;const i4e::simd<float,16> dw=d*fscale;
                        auto Bc=iq4_int4_vnni<64>(iq4_int4_coarse<64>(packed));
                        auto Br=iq4_int4_vnni<64>(iq4_int4_residual<64>(packed));
                        i4e::simd<uint32_t,NT*8> a_words;
#pragma unroll
                        for(int mr=0;mr<NT;++mr){
                            if(half==0)a_words.template select<8,1>(mr*8)=aq.template select<8,1>(mr*AP+1);
                            else{
                                a_words.template select<6,1>(mr*8)=aq.template select<6,1>(mr*AP+10);
                                a_words.template select<2,1>(mr*8+6)=tail.template select<2,1>(mr*TP);
                            }
                        }
                        i4e::simd<int8_t,NT*32> A=a_words.template bit_cast_view<int8_t>();
                        i4e::simd<int32_t,NT*16> zero=0;
                        i4e::simd<int8_t,NT*32> interleaved;
#pragma unroll
                        for(int mr=0;mr<NT;++mr){
                            interleaved.template select<16,2>(mr*32)=A.template select<16,1>(mr*32);
                            interleaved.template select<16,2>(mr*32+1)=A.template select<16,1>(mr*32+16);
                        }
                        using P=i4e::xmx::dpas_argument_type;
                        auto Cc=i4e::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Bc,interleaved);
                        auto Cr=i4e::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Br,interleaved);
                        auto C=Cc*16+Cr;
#pragma unroll
                        for(int mr=0;mr<NT;++mr){
                            i4e::simd<uint32_t,1> dhdr=half==0?aq.template select<1,1>(mr*AP).read():aq.template select<1,1>(mr*AP+9).read();
                            i4e::simd<uint16_t,1> dbits=dhdr&0xffffu;
                            i4e::simd<float,1> dvalue=dbits.template bit_cast_view<sycl::half>().read();
                            const float ds=dvalue[0];
                            i4e::simd<int32_t,16> ci=C.template select<16,1>(mr*16);
                            i4e::simd<float,16> dot=ci;const i4e::simd<float,16> factor=dw*ds;
                            i4e::simd<float,16> term=factor*dot;
                            term.merge(0.f,!valid);
                            const i4e::simd<float,16> next=acc[half][mr]+term;acc[half][mr].merge(next,valid);
                        }
                    }
                }
            }
#pragma unroll
            for(int part=0;part<2;++part){
#pragma unroll
                for(int mr=0;mr<NT;++mr)i4e::slm_block_store<float,16>((sid*2+part)*NT*64+mr*64,acc[part][mr]);
            }
            i4e::barrier();
            if(sid<NT){
            i4e::simd<float,16> p[16];
#pragma unroll
            for(int v=0;v<16;++v)p[v]=i4e::slm_block_load<float,16>(v*NT*64+sid*64)+
                i4e::slm_block_load<float,16>((v+16)*NT*64+sid*64);
#pragma unroll
            for(int off=8;off;off/=2){
#pragma unroll
                for(int v=0;v<off;++v)p[v]+=p[v+off];
            }
            i4e::simd<uint32_t,16> offsets=(row+uint32_t(sid)*uint32_t(stride))*4u;
            i4e::scatter<uint16_t,16>(reinterpret_cast<uint16_t*>(y),offsets/2u,bf16_output(p[0]),valid);
            }
        });
}


template<int KB>void dispatch(int T,int mode,const uint8_t*w,const Q81Block*x,float*y,int N,int stride,sycl::queue*q){
#define RUN(E) case E:if(mode==0)iq4_pair_all_t_launch_int4<E,KB>(w,x,y,N,stride,q);else iq4_q8_split_launch_int4<E,KB>(w,x,y,N,stride,q);break
 switch(T){RUN(1);RUN(2);RUN(3);RUN(4);}
#undef RUN
}
}
extern "C" const char* xpu_iq4_s4_bf16_error(){return sglang_iq4_s4_bf16::error.c_str();}
extern "C" int xpu_iq4_s4_bf16(void*queue,const void*w,const void*x,float*y,int K,int N,int T,int row_bytes,int stride,int mode){
 if(!queue||!w||!x||!y||(K!=2560&&K!=6144)||N<1||N>248320||T<1||T>4||row_bytes!=K/256*136||stride<N||mode<0||mode>1)return 2;
 try{
  auto*q=static_cast<sycl::queue*>(queue);if(!q->has_property<sycl::property::queue::in_order>())return 2;
  if(K==2560)sglang_iq4_s4_bf16::dispatch<10>(T,mode,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),y,N,stride,q);
  else sglang_iq4_s4_bf16::dispatch<24>(T,mode,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),y,N,stride,q);
  return 0;
 }catch(const std::exception&e){sglang_iq4_s4_bf16::error=e.what();return 1;}
}
