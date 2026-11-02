// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Original quantization formulas: Copyright (c) 2023-2026 the ggml authors.
// Ported from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd:
// sycl/src/kernels/cuda/native_mmvq.dp.cpp (IQ4PairAllT, IQ4 lookup, Q8_1)
// and sycl/include/strata/kernels/q8_1_finite.hpp.
// See STRATA-LICENSE and GGML-LICENSE.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
namespace sglang_iq4 {
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81Block)==36 && alignof(Q81Block)==4 && offsetof(Q81Block,qs)==4);
thread_local std::string error;
class QuantizeQ81;
inline float finite_half(float x){return sycl::fabs(x)>65504.0f?sycl::copysign(65504.0f,x):x;}
inline float warp_sum(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x+=sycl::permute_group_by_xor(sg,x,o);return x;}
inline float warp_max(sycl::sub_group sg,float x){for(int o=16;o;o>>=1)x=sycl::fmax(x,sycl::permute_group_by_xor(sg,x,o));return x;}
namespace {
namespace i4e=sycl::ext::intel::esimd;
template<int L>SYCL_ESIMD_FUNCTION inline i4e::simd<uint32_t,L> iq4_e_lookup(i4e::simd<uint32_t,L> q){
    constexpr uint32_t table[]={0x81818181u,0x98989898u,0xadadadadu,0xbfbfbfbfu,
        0xcfcfcfcfu,0xddddddddu,0xeaeaeaeau,0xf6f6f6f6u,
        0x01010101u,0x0d0d0d0du,0x19191919u,0x26262626u,
        0x35353535u,0x45454545u,0x59595959u,0x71717171u};
    constexpr auto select=(i4e::bfn_t::x&~i4e::bfn_t::z)|(i4e::bfn_t::y&i4e::bfn_t::z);
    i4e::simd<uint32_t,L> mask[4],a[8],b[4],c[2];
#pragma unroll
    for(int i=0;i<4;++i){i4e::simd<uint32_t,L> bit=(q>>i)&0x01010101u;mask[i]=(bit<<8)-bit;}
#pragma unroll
    for(int i=0;i<8;++i)a[i]=i4e::bfn<select>(i4e::simd<uint32_t,L>(table[2*i]),i4e::simd<uint32_t,L>(table[2*i+1]),mask[0]);
#pragma unroll
    for(int i=0;i<4;++i)b[i]=i4e::bfn<select>(a[2*i],a[2*i+1],mask[1]);
#pragma unroll
    for(int i=0;i<2;++i)c[i]=i4e::bfn<select>(b[2*i],b[2*i+1],mask[2]);
    return i4e::bfn<select>(c[0],c[1],mask[3]);
}
}
namespace {
template<int NT,int KB>class IQ4PairAllT;
template<int NT,int KB>
void iq4_pair_all_t_launch(const uint8_t* weights,const Q81Block* acts,float* y,int N,sycl::queue* q,int output_stride){
    constexpr int NR=16,XB=KB*8,RB=KB*136;
    q->parallel_for<IQ4PairAllT<NT,KB>>(sycl::nd_range<1>(size_t((N+15)/16)*16,16),
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
                        const auto lo=iq4_e_lookup<64>(packed&0x0f0f0f0fu);
                        const auto hi=iq4_e_lookup<64>((packed>>4)&0x0f0f0f0fu);
                        i4e::simd<uint32_t,128> words;
                        words.template select<64,1>(0)=lo;words.template select<64,1>(64)=hi;
                        i4e::simd<int8_t,512> B=words.template bit_cast_view<int8_t>();
                        auto a_words=i4e::load_2d<uint32_t,8,NT,1,false,false>(reinterpret_cast<const uint32_t*>(acts),
                            XB*36-1,NT-1,XB*36-1,seg*9+1,0);
                        i4e::simd<int8_t,NT*32> A=a_words.template bit_cast_view<int8_t>();
                        i4e::simd<int32_t,NT*16> zero=0;
                        auto C=i4e::xmx::dpas<8,NT,int32_t>(zero,B,A);
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
            i4e::simd<uint32_t,16> offsets=(row+uint32_t(sid)*uint32_t(output_stride))*4u;
            i4e::scatter<float,16>(y,offsets,p[0],valid);
            }
        });
}
template<int KB>
void iq4_pair_all_t_dispatch(int T,const uint8_t* w,const Q81Block* x,float* y,int N,sycl::queue* q,int output_stride){
    switch(T){
    case 1:iq4_pair_all_t_launch<1,KB>(w,x,y,N,q,output_stride);break;
    case 2:iq4_pair_all_t_launch<2,KB>(w,x,y,N,q,output_stride);break;
    case 3:iq4_pair_all_t_launch<3,KB>(w,x,y,N,q,output_stride);break;
    case 4:iq4_pair_all_t_launch<4,KB>(w,x,y,N,q,output_stride);break;
    }
}
}

} // namespace sglang_iq4
extern "C" const char* xpu_iq4_error(){return sglang_iq4::error.c_str();}
extern "C" int xpu_iq4_quantize(void* queue,const float* x,void* output,int K,int T){
 if(!queue||!x||!output||(K!=2560&&K!=6144)||T<1||T>4)return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);auto* y=static_cast<sglang_iq4::Q81Block*>(output);
  const int count=K*T;
  q.parallel_for<sglang_iq4::QuantizeQ81>(sycl::nd_range<1>(size_t(count),256),
   [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
    int i=it.get_global_linear_id();auto sg=it.get_sub_group();float xi=x[i];
    float amax=sglang_iq4::warp_max(sg,sycl::fabs(xi));float sum=sglang_iq4::warp_sum(sg,xi);
    float d=sglang_iq4::finite_half(amax/127.0f);
    float quantized=amax==0.0f?0.0f:sycl::round(xi/d);
    quantized=quantized>127.0f?127.0f:quantized< -127.0f? -127.0f:quantized;
    y[i/32].qs[i%32]=static_cast<int8_t>(quantized);
    if(i%32==0)y[i/32].ds=sycl::vec<float,2>(sglang_iq4::finite_half(d),sglang_iq4::finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
   });
  return 0;
 }catch(const std::exception& e){sglang_iq4::error=e.what();return 1;}
}
extern "C" int xpu_iq4_mmvq(void* queue,const void* weights,const void* acts,float* y,int K,int N,int T){
 if(!queue||!weights||!acts||!y||N<1||(K!=2560&&K!=6144)||T<1||T>4||
  (reinterpret_cast<uintptr_t>(weights)&63)||(reinterpret_cast<uintptr_t>(acts)&63))return 2;
 try{
  auto* q=static_cast<sycl::queue*>(queue);auto* w=static_cast<const uint8_t*>(weights);auto* x=static_cast<const sglang_iq4::Q81Block*>(acts);
  if(K==2560)sglang_iq4::iq4_pair_all_t_dispatch<10>(T,w,x,y,N,q,N);
  else sglang_iq4::iq4_pair_all_t_dispatch<24>(T,w,x,y,N,q,N);
  return 0;
 }catch(const std::exception& e){sglang_iq4::error=e.what();return 1;}
}

extern "C" int xpu_iq4_mmvq_output_stride(void* queue,const void* weights,const void* acts,
    float* y,int K,int N,int T,int output_stride){
 if(!queue||!weights||!acts||!y||N<1||(K!=2560&&K!=6144)||T<1||T>4||output_stride<N||
  (reinterpret_cast<uintptr_t>(weights)&63)||(reinterpret_cast<uintptr_t>(acts)&63))return 2;
 try{
  auto* q=static_cast<sycl::queue*>(queue);auto* w=static_cast<const uint8_t*>(weights);auto* x=static_cast<const sglang_iq4::Q81Block*>(acts);
  if(K==2560)sglang_iq4::iq4_pair_all_t_dispatch<10>(T,w,x,y,N,q,output_stride);
  else sglang_iq4::iq4_pair_all_t_dispatch<24>(T,w,x,y,N,q,output_stride);
  return 0;
 }catch(const std::exception& e){sglang_iq4::error=e.what();return 1;}
}

namespace sglang_iq4 {
template<int DType,bool GDN> class QuantizeLayout;
template<int DType,bool GDN>
void quantize_layout(sycl::queue& q,const void* input,Q81Block* output,int K,int T) {
 const int count=K*T;
 q.parallel_for<QuantizeLayout<DType,GDN>>(sycl::nd_range<1>(size_t((count+255)/256)*256,256),
  [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]] {
   const int i=it.get_global_linear_id();
   if(i>=count)return; // Supported widths contain complete 32-lane groups.
   int offset=i;
   if constexpr(GDN) {
    const int row=i/6144,j=i%6144;
    const int group=j/(16*128),head=(j/128)%16,dim=j%128;
    offset=row*6144+(head*3+group)*128+dim;
   }
   float xi;
   if constexpr(DType==0)xi=static_cast<const float*>(input)[offset];
   else if constexpr(DType==1)xi=sycl::bit_cast<float>(uint32_t(static_cast<const uint16_t*>(input)[offset])<<16);
   else xi=float(static_cast<const sycl::half*>(input)[offset]);
   auto sg=it.get_sub_group();
   const float amax=warp_max(sg,sycl::fabs(xi)),sum=warp_sum(sg,xi);
   const float d=finite_half(amax/127.0f);
   float quantized=amax==0.0f?0.0f:sycl::round(xi/d);
   quantized=quantized>127.0f?127.0f:quantized< -127.0f? -127.0f:quantized;
   output[i/32].qs[i%32]=static_cast<int8_t>(quantized);
   if(i%32==0)output[i/32].ds=sycl::vec<float,2>(finite_half(d),finite_half(sum)).convert<sycl::half,sycl::rounding_mode::automatic>();
  });
}
}
extern "C" int xpu_iq4_quantize_layout(void* queue,const void* x,void* output,int K,int T,int dtype,int gdn){
 if(!queue||!x||!output||(K!=640&&K!=2560&&K!=6144)||T<1||T>4||dtype<0||dtype>2||(gdn!=0&&gdn!=1)||(gdn&&K!=6144))return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);auto* y=static_cast<sglang_iq4::Q81Block*>(output);
#define SUBMIT(D,G) sglang_iq4::quantize_layout<D,G>(q,x,y,K,T)
  if(gdn){
   if(dtype==0){SUBMIT(0,true);}else if(dtype==1){SUBMIT(1,true);}else{SUBMIT(2,true);}
  }else{
   if(dtype==0){SUBMIT(0,false);}else if(dtype==1){SUBMIT(1,false);}else{SUBMIT(2,false);}
  }
#undef SUBMIT
  return 0;
 }catch(const std::exception& e){sglang_iq4::error=e.what();return 1;}
}

// Every 32-lane group owns one independent Q8_1 block. Combining token rows
// changes only the grid, preserving the scale, rounding and reduction order.
// Keep the existing four-row ABI unchanged for older callers.
extern "C" int xpu_iq4_quantize_layout_wide(void* queue,const void* x,void* output,int K,int T,int dtype,int gdn){
 if(!queue||!x||!output||(K!=640&&K!=2560&&K!=6144)||T<5||T>32||dtype<0||dtype>2||(gdn!=0&&gdn!=1)||(gdn&&K!=6144))return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);auto* y=static_cast<sglang_iq4::Q81Block*>(output);
  if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define SUBMIT(D,G) sglang_iq4::quantize_layout<D,G>(q,x,y,K,T)
  if(gdn){
   if(dtype==0){SUBMIT(0,true);}else if(dtype==1){SUBMIT(1,true);}else{SUBMIT(2,true);}
  }else{
   if(dtype==0){SUBMIT(0,false);}else if(dtype==1){SUBMIT(1,false);}else{SUBMIT(2,false);}
  }
#undef SUBMIT
  return 0;
 }catch(const std::exception& e){sglang_iq4::error=e.what();return 1;}
}
