// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Ported from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd:
// native_mmvq.dp.cpp MtpQ8SoAInt8/MtpQ8MultiSoAInt8, and original Q8_1 quantizer.
// See STRATA-LICENSE and GGML-LICENSE. Byte-preserving SoA16; direct BF16
// output and NT16 are the adapter extensions; scalar dot reduction retained.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <string>
namespace mtp_q8_soa {
namespace i4e=sycl::ext::intel::esimd;
struct Q81Block {sycl::half2 ds; int8_t qs[32];};
static_assert(sizeof(Q81Block)==36);
thread_local std::string error;
SYCL_ESIMD_FUNCTION inline i4e::simd<uint16_t,16> round_bf16(i4e::simd<float,16> v){
 auto b=v.template bit_cast_view<uint32_t>().read();auto r=(b+0x7fffu+((b>>16)&1u))>>16;
 r.merge((b>>16)|0x40u,(b&0x7fffffffu)>0x7f800000u);return i4e::convert<uint16_t>(r);
}
namespace {
template<int KB>class MtpQ8SoAInt8;
template<int KB>
void mtp_q8_soa_launch(const uint8_t* weights,const Q81Block* acts,uint16_t* y,int N,sycl::queue* q){
#pragma clang fp contract(off)
    constexpr int WG=8,RB=KB*34;
    q->parallel_for<MtpQ8SoAInt8<KB>>(sycl::nd_range<1>(size_t((N+15)/16)*WG,WG),
        [=](sycl::nd_item<1> it)[[intel::sycl_explicit_simd]]{
            i4e::slm_init<16*16*4>();
            const int sid=it.get_local_linear_id(),base=it.get_group_linear_id()*16;
            i4e::simd<uint32_t,16> row(base,1);auto valid=row<uint32_t(N);
            i4e::simd<float,16> acc[2][2];
#pragma unroll
            for(int branch=0;branch<2;++branch){
#pragma unroll
                for(int half=0;half<2;++half)acc[branch][half]=0.f;
            }
#pragma unroll
            for(int branch=0;branch<2;++branch){
                const int sub=sid+branch*8;
                for(int kb=sub;kb<KB;kb+=16){
                    const uint8_t* block=weights+size_t(base)*RB+kb*544;
                    auto dh=i4e::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block));
                    i4e::simd<float,16> d=dh.template bit_cast_view<sycl::half>().read();
                    auto bytes=i4e::block_load<uint64_t,64>(reinterpret_cast<const uint64_t*>(block+32));
                    i4e::simd<int8_t,512> B=bytes.template bit_cast_view<int8_t>();
                    const uint8_t* raw=reinterpret_cast<const uint8_t*>(acts+kb);
                    i4e::simd<uint32_t,8> q_offsets(4,4);
                    auto u=i4e::gather<int32_t,8>(reinterpret_cast<const int32_t*>(raw),q_offsets);
                    i4e::simd<int32_t,16> aw=0;
                    aw.template select<4,1>(0)=u.template select<4,1>(0);
                    aw.template select<4,1>(12)=u.template select<4,1>(4);
                    i4e::simd<int8_t,64> A=aw.template bit_cast_view<int8_t>();
                    i4e::simd<int32_t,32> zero=0;
                    auto dots=i4e::xmx::dpas<8,2,int32_t>(zero,B,A);
                    i4e::simd<uint16_t,1> dsbits=reinterpret_cast<const uint16_t*>(raw)[0];
                    i4e::simd<float,1> dsvalue=dsbits.template bit_cast_view<sycl::half>().read();
                    const float ds=dsvalue[0];
                    const i4e::simd<float,16> factor=d*ds;
#pragma unroll
                    for(int half=0;half<2;++half){
                        i4e::simd<int32_t,16> ci=dots.template select<16,1>(half*16);
                        i4e::simd<float,16> dot=ci;
                        i4e::simd<float,16> term=factor*dot;
                        term.merge(0.f,!valid);
                        const i4e::simd<float,16> next=acc[branch][half]+term;
                        acc[branch][half].merge(next,valid);
                    }
                }
            }
#pragma unroll
            for(int half=0;half<2;++half)
                i4e::slm_block_store<float,16>((sid*2+half)*64,acc[0][half]+acc[1][half]);
            i4e::barrier();
            if(sid==0){
                i4e::simd<float,16> partial[16];
#pragma unroll
                for(int v=0;v<16;++v)partial[v]=i4e::slm_block_load<float,16>(v*64);
#pragma unroll
                for(int off=8;off;off/=2){
#pragma unroll
                    for(int v=0;v<off;++v)partial[v]+=partial[v+off];
                }
                i4e::scatter<uint16_t,16>(y,row*2u,round_bf16(partial[0]),valid);
            }
        });
}
}
namespace {
template<int NT,int KB>class MtpQ8MultiSoAInt8;
template<int NT,int KB>
void mtp_q8_multi_soa_launch(const uint8_t* weights,const Q81Block* acts,uint16_t* y,int N,sycl::queue* q){
#pragma clang fp contract(off)
    static_assert(NT==4||NT==8||NT==16);constexpr int WG=8,RB=KB*34;
    q->parallel_for<MtpQ8MultiSoAInt8<NT,KB>>(sycl::nd_range<1>(size_t((N+15)/16)*WG,WG),
        [=](sycl::nd_item<1> it)[[intel::sycl_explicit_simd]]{
            i4e::slm_init<16*NT*16*4>();
            const int sid=it.get_local_linear_id(),base=it.get_group_linear_id()*16;
            i4e::simd<uint32_t,16> row(base,1);auto valid=row<uint32_t(N);
            i4e::simd<float,16> acc[2][2][NT];
#pragma unroll
            for(int branch=0;branch<2;++branch){
#pragma unroll
                for(int half=0;half<2;++half){
#pragma unroll
                    for(int e=0;e<NT;++e)acc[branch][half][e]=0.f;
                }
            }
#pragma unroll
            for(int branch=0;branch<2;++branch){
                const int sub=sid+branch*8;
                for(int kb=sub;kb<KB;kb+=16){
                    const uint8_t* block=weights+size_t(base)*RB+kb*544;
                    auto dh=i4e::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block));
                    i4e::simd<float,16> d=dh.template bit_cast_view<sycl::half>().read();
                    auto bytes=i4e::block_load<uint64_t,64>(reinterpret_cast<const uint64_t*>(block+32));
                    i4e::simd<int8_t,512> B=bytes.template bit_cast_view<int8_t>();
#pragma unroll
                    for(int e0=0;e0<NT;e0+=4){
                        i4e::simd<int32_t,64> aw=0;i4e::simd<float,4> activation_scale;
#pragma unroll
                        for(int e=0;e<4;++e){
                            const uint8_t* raw=reinterpret_cast<const uint8_t*>(acts+(e0+e)*KB+kb);
                            i4e::simd<uint32_t,8> offsets(4,4);
                            auto u=i4e::gather<int32_t,8>(reinterpret_cast<const int32_t*>(raw),offsets);
                            aw.template select<4,1>(e*16)=u.template select<4,1>(0);
                            aw.template select<4,1>(e*16+12)=u.template select<4,1>(4);
                            i4e::simd<uint16_t,1> dsbits=reinterpret_cast<const uint16_t*>(raw)[0];
                            i4e::simd<float,1> dsvalue=dsbits.template bit_cast_view<sycl::half>().read();activation_scale[e]=dsvalue[0];
                        }
                        i4e::simd<int8_t,256> A=aw.template bit_cast_view<int8_t>();
                        i4e::simd<int32_t,128> zero=0;
                        auto dots=i4e::xmx::dpas<8,8,int32_t>(zero,B,A);
#pragma unroll
                        for(int e=0;e<4;++e){
                            const float ds=activation_scale[e];const i4e::simd<float,16> factor=d*ds;
#pragma unroll
                            for(int half=0;half<2;++half){
                                i4e::simd<int32_t,16> ci=dots.template select<16,1>(e*32+half*16);
                                i4e::simd<float,16> dot=ci;i4e::simd<float,16> term=factor*dot;
                                term.merge(0.f,!valid);
                                const i4e::simd<float,16> next=acc[branch][half][e0+e]+term;
                                acc[branch][half][e0+e].merge(next,valid);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for(int half=0;half<2;++half){
#pragma unroll
                for(int e=0;e<NT;++e)i4e::slm_block_store<float,16>(((sid*2+half)*NT+e)*64,acc[0][half][e]+acc[1][half][e]);
            }
            i4e::barrier();
            for(int token=sid;token<NT;token+=WG){
                i4e::simd<float,16> partial[16];
#pragma unroll
                for(int v=0;v<16;++v)partial[v]=i4e::slm_block_load<float,16>((v*NT+token)*64);
#pragma unroll
                for(int off=8;off;off/=2){
#pragma unroll
                    for(int v=0;v<off;++v)partial[v]+=partial[v+off];
                }
                i4e::scatter<uint16_t,16>(y,(row+uint32_t(token)*uint32_t(N))*2u,round_bf16(partial[0]),valid);
            }
        });
}
}


inline float finite_half(float v){return sycl::fabs(v)>65504.f?sycl::copysign(65504.f,v):v;}
class Quant;
void quant(sycl::queue&q,const uint16_t*x,Q81Block*y,int K,int T){
 q.parallel_for<Quant>(sycl::nd_range<1>(size_t((K*T+255)/256)*256,256),[=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(32)]]{
  const int i=it.get_global_linear_id();if(i>=K*T)return;
  float xi=sycl::bit_cast<float>(uint32_t(x[i])<<16),am=sycl::fabs(xi),sum=xi;auto sg=it.get_sub_group();
  for(int off=16;off;off>>=1){am=sycl::fmax(am,sycl::permute_group_by_xor(sg,am,off));sum+=sycl::permute_group_by_xor(sg,sum,off);}
  float d=finite_half(am/127.f),code=am==0.f?0.f:sycl::round(xi/d);code=sycl::clamp(code,-127.f,127.f);
  y[i/32].qs[i%32]=int8_t(code);
  if(i%32==0)y[i/32].ds=sycl::vec<float,2>(finite_half(d),finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
 });
}
template<int KB>void dispatch(sycl::queue*q,const uint8_t*w,const Q81Block*a,uint16_t*y,int N,int T){
 if(T==1)mtp_q8_soa_launch<KB>(w,a,y,N,q);
 else if(T==4)mtp_q8_multi_soa_launch<4,KB>(w,a,y,N,q);
 else if(T==8)mtp_q8_multi_soa_launch<8,KB>(w,a,y,N,q);
 else mtp_q8_multi_soa_launch<16,KB>(w,a,y,N,q);
}
}
extern "C" const char* xpu_mtp_q8_soa_error(){return mtp_q8_soa::error.c_str();}
extern "C" int xpu_mtp_q8_quant(void*queue,const uint16_t*x,void*y,int K,int T){
 if(!queue||!x||!y||(K!=640&&K!=2560&&K!=6144)||T<1||T>16)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;mtp_q8_soa::quant(q,x,static_cast<mtp_q8_soa::Q81Block*>(y),K,T);return 0;}catch(const std::exception&e){mtp_q8_soa::error=e.what();return 1;}
}
extern "C" int xpu_mtp_q8_soa(void*queue,const uint8_t*w,const void*a,uint16_t*y,int K,int N,int T){
 if(!queue||!w||!a||!y||(K!=640&&K!=2560&&K!=6144)||N<16||N%16||(T!=1&&T!=4&&T!=8&&T!=16))return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
 auto*acts=static_cast<const mtp_q8_soa::Q81Block*>(a);
 if(K==640)mtp_q8_soa::dispatch<20>(&q,w,acts,y,N,T);else if(K==2560)mtp_q8_soa::dispatch<80>(&q,w,acts,y,N,T);else mtp_q8_soa::dispatch<192>(&q,w,acts,y,N,T);return 0;
 }catch(const std::exception&e){mtp_q8_soa::error=e.what();return 1;}
}
