// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Ported Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd IQ3PublicXmxKernel.
// Original IQ3_S rows and Q8_1 images, exact32-chain FP32 summation tree.
// SGLang adaptations: direct GGML table, no dpct dependency, output stride.
#include "dequant.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <string>
namespace sglang_iq3_native {
thread_local std::string error;
inline int word(const void* p,int i) {
 const auto* q=static_cast<const uint16_t*>(p);return int(uint32_t(q[2*i])|(uint32_t(q[2*i+1])<<16));
}
inline int sign_word(uint32_t positive,uint32_t signs) {
 uint32_t bit=((signs&15u)*0x00204081u)&0x01010101u;
 return int((positive^(bit*255u))+bit);
}
struct SplitWeight {int g[8];int ls;float dw;};
struct IQ3XmxWeightPrep {
 static SplitWeight load(const void* vbq,int block,int part,const uint32_t* grid) {
  const auto* w=static_cast<const block_iq3_s*>(vbq)+block;
  const uint32_t qs0=uint32_t(word(w->qs,part)),qs1=uint32_t(word(w->qs,part+1));
  const uint32_t signs=uint32_t(word(w->signs,part/2));const int qh=w->qh[part/2];
  SplitWeight result{};
#pragma unroll
  for(int j=0;j<8;++j) {
   const uint32_t index=(((j<4?qs0:qs1)>>(8*(j%4)))&255u)|((uint32_t(qh)<<(8-j))&256u);
   result.g[j]=sign_word(grid[index],signs>>(4*j));
  }
  result.ls=1+2*((w->scales[part/4]>>((part<<1)&4))&15);
  result.dw=float(w->d);return result;
 }
};
namespace iq3jm=sycl::ext::oneapi::experimental::matrix;
inline float iq3_xmx_float_finish(const SplitWeight& w,const block_q8_1* activation,int sum) {
 sum*=w.ls;const float factor=w.dw*float(activation->ds[0]);return factor*sum;
}
template<int NT,int KB>class IQ3NativeXmxKernel;
template<int NT,int KB>
void launch(const uint8_t* w,const block_q8_1* x,float* y,int N,int stride,sycl::queue* q){
    constexpr int NR=16,NS=32,XB=KB*8,RB=KB*sizeof(block_iq3_s);
    q->parallel_for<IQ3NativeXmxKernel<NT,KB>>(sycl::nd_range<1>((N+NR-1)/NR*512,512),
      [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
        auto group=it.get_group();auto sg=it.get_sub_group();int lid=it.get_local_linear_id(),sid=lid/16,lane=lid%16,base=it.get_group_linear_id()*NR,row=base+lane;
        auto& table=*sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[512]>(group);


        auto& partial=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[NS*NT*NR]>(group);
        table[lid]=iq3s_grid[lid];it.barrier(sycl::access::fence_space::local_space);
        float acc[NT]={};
        for(int seg=sid;seg<XB;seg+=NS){
            SplitWeight pw{};if(row<N)pw=IQ3XmxWeightPrep::load(w+uint32_t(row)*uint32_t(RB),seg/8,2*(seg%8),table);
            iq3jm::joint_matrix<sycl::sub_group,int8_t,iq3jm::use::a,NT,32,iq3jm::layout::row_major> A;
            iq3jm::joint_matrix<sycl::sub_group,int8_t,iq3jm::use::b,32,NR,iq3jm::layout::ext_intel_packed> B;
            iq3jm::joint_matrix<sycl::sub_group,int32_t,iq3jm::use::accumulator,NT,NR> C;

                auto pa=sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(x[seg].qs);
                iq3jm::joint_matrix_load(sg,A,pa,XB*sizeof(block_q8_1));
                iq3jm::joint_matrix_fill(sg,B,0);

                    int kr=0;
                    iq3jm::joint_matrix_apply(sg,B,[&](auto& element){
                        const int word=kr<4?pw.g[0]:kr<8?pw.g[1]:kr<12?pw.g[2]:kr<16?pw.g[3]:kr<20?pw.g[4]:kr<24?pw.g[5]:kr<28?pw.g[6]:pw.g[7];
                        element=int8_t(uint32_t(word)>>(8*(kr%4)));++kr;
                    });


            iq3jm::joint_matrix_fill(sg,C,0);iq3jm::joint_matrix_mad(sg,C,A,B,C);

                int mr=0;
                iq3jm::joint_matrix_apply(sg,C,[&](auto& element){
                    acc[mr]+=iq3_xmx_float_finish(pw,x+mr*XB+seg,int(element));++mr;
                });


        }
#pragma unroll
        for(int j=0;j<NT;++j)partial[sid*NT*NR+j*NR+lane]=acc[j];
        it.barrier(sycl::access::fence_space::local_space);
        const int rr=lid/32,vl=lid%32;
#pragma unroll
        for(int j=0;j<NT;++j){float v=partial[vl*NT*NR+j*NR+rr];v+=partial[(vl^16)*NT*NR+j*NR+rr];
#pragma unroll
            for(int off=8;off;off>>=1)v+=sycl::permute_group_by_xor(sg,v,off);
            if(vl==0&&base+rr<N)y[size_t(j)*stride+uint32_t(base+rr)]=v;
        }
    });
}
template<int KB>void dispatch(int T,const uint8_t* w,const block_q8_1* x,float* y,int N,int stride,sycl::queue* q) {
#define RUN(E) case E:launch<E,KB>(w,x,y,N,stride,q);break
 switch(T){RUN(1);RUN(2);RUN(3);RUN(4);}
#undef RUN
}
}
extern "C" const char* xpu_iq3_xmx_error(){return sglang_iq3_native::error.c_str();}
extern "C" int xpu_iq3_xmx(void* queue,const void* w,const void* x,float* out,
 int K,int N,int T,int row_bytes,int stride) {
 if(!queue||!w||!x||!out||(K!=2560&&K!=6144)||N<1||N>16384||T<1||T>4||row_bytes!=K/256*110||stride<N)return 2;
 try {
  auto* q=static_cast<sycl::queue*>(queue);
  if(!q->has_property<sycl::property::queue::in_order>())return 2;
  if(K==2560)sglang_iq3_native::dispatch<10>(T,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),out,N,stride,q);
  else sglang_iq3_native::dispatch<24>(T,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),out,N,stride,q);
  return 0;
 }catch(const std::exception& e){sglang_iq3_native::error=e.what();return 1;}
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd IQ3PairedS4 port.
// Exact32 FP32 chains; SGLang output stride and lossless field permutation.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <string>
namespace sglang_iq3_packed {
namespace p3=sycl::ext::intel::esimd;
thread_local std::string error;
template<int NT,int KB>class IQ3PairedS4;
template<int NT,int KB>
void iq3_paired_s4_launch(const uint8_t* weights,const block_q8_1* acts,float* y,int N,int stride,sycl::queue* q){
 constexpr int NR=16,NS=32,WG=16,XB=KB*8,Partial=2048;
 q->parallel_for<IQ3PairedS4<NT,KB>>(
  sycl::nd_range<1>(size_t(N/16)*WG,WG),
  [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
#pragma clang fp contract(off)
   p3::slm_init<2048+WG*NT*NR*4>();
   const int sid=int(it.get_local_id(0)),base=int(it.get_group(0))*16;
   p3::simd<uint32_t,16> rows(base,1);auto valid=rows<uint32_t(N);
   p3::simd<uint32_t,32> table_offsets(sid*128,4);
   auto table=p3::gather<uint32_t,32>(iq3s_grid,table_offsets);
   p3::slm_block_store<uint32_t,32>(sid*128,table);
   p3::barrier();
   p3::simd<float,16> acc[2][NT];
#pragma unroll
   for(int branch=0;branch<2;++branch){
#pragma unroll
    for(int e=0;e<NT;++e)acc[branch][e]=0.f;
   }
   const auto* tile=weights+size_t(base)*KB*110;
#pragma unroll
   for(int branch=0;branch<2;++branch){
   for(int seg=sid+branch*16;seg<XB;seg+=NS){
    const int part=seg%8;const auto* block=tile+(seg/8)*1760;
    const auto* plane=block+part*208;
    auto q0=p3::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(plane));
    auto q1=p3::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(plane+64));
    auto signs=p3::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(plane+128));
    auto qh8=p3::block_load<uint8_t,16>(plane+192);
    p3::simd<uint32_t,16> qh=qh8;
    auto scales=p3::block_load<uint32_t,16>(reinterpret_cast<const uint32_t*>(block+1664));
    p3::simd<int32_t,16> ls=1+2*((scales>>(4*part))&15u);
    auto dbits=p3::block_load<uint16_t,16>(reinterpret_cast<const uint16_t*>(block+1728));
    p3::simd<float,16> d=dbits.template bit_cast_view<sycl::half>().read();
    p3::simd<uint32_t,64> coarse_words,sign_words;
#pragma unroll
    for(int group=0;group<2;++group){
     p3::simd<uint32_t,64> index,sign;
#pragma unroll
     for(int j=0;j<4;++j){
      p3::simd<uint32_t,16> qs=group==0?q0:q1;
      index.template select<16,1>(j*16)=((qs>>(8*j))&255u)|(((qh>>(group*4+j))&1u)<<8);
      sign.template select<16,1>(j*16)=(signs>>(4*(group*4+j)))&15u;
     }
     auto grid=p3::slm_gather<uint32_t,64>(index*4u);
     // Each positive codebook coefficient is odd in1..15.
     p3::simd<uint32_t,64> packed=(grid&15u)|((grid>>4)&240u)|((grid>>8)&3840u)|((grid>>12)&61440u);
     p3::simd<uint32_t,64> positive=(packed>>1)&0x7777u;
     // Guarded nibble subtraction also handles signed zero without carries.
     p3::simd<uint32_t,64> negative=((0x8888u-positive)^0x8888u)&65535u;
     p3::simd<uint32_t,64> bit=(sign&1u)|((sign&2u)<<3)|((sign&4u)<<6)|((sign&8u)<<9);
     p3::simd<uint32_t,64> mask=bit*15u;
     p3::simd<uint32_t,64> c=positive^((positive^negative)&mask),sgn=0x1111u+bit*14u;
#pragma unroll
     for(int pair=0;pair<2;++pair){
      coarse_words.template select<16,1>((group*2+pair)*16)=c.template select<16,1>(pair*32)|(c.template select<16,1>(pair*32+16)<<16);
      sign_words.template select<16,1>((group*2+pair)*16)=sgn.template select<16,1>(pair*32)|(sgn.template select<16,1>(pair*32+16)<<16);
     }
    }
    p3::simd<uint16_t,128> Bc=coarse_words.template bit_cast_view<uint16_t>().read();
    p3::simd<uint16_t,128> Bs=sign_words.template bit_cast_view<uint16_t>().read();
    auto aw=p3::load_2d<uint32_t,8,NT,1,false,false>(reinterpret_cast<const uint32_t*>(acts),XB*36-1,NT-1,XB*36-1,seg*9+1,0);
    p3::simd<int8_t,NT*32> A=aw.template bit_cast_view<int8_t>();
    p3::simd<int32_t,NT*16> zero=0;
    using P=p3::xmx::dpas_argument_type;
    // Natural K0..31, as in the qualified Q2 S4 basis probe; no IQ4 interleave.
    auto Cc=p3::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Bc,A);
    auto Cs=p3::xmx::dpas<8,NT,int32_t,int32_t,uint16_t,int8_t,P::s4,P::s8>(zero,Bs,A);
#pragma unroll
    for(int e=0;e<NT;++e){
     p3::simd<int32_t,16> c=Cc.template select<16,1>(e*16),sgn=Cs.template select<16,1>(e*16);
     p3::simd<int32_t,16> scaled=(c*2+sgn)*ls;
     p3::simd<float,16> dot=scaled;
     p3::simd<uint32_t,1> off=uint32_t((e*XB+seg)*36);
     auto bits=p3::gather<uint16_t,1>(reinterpret_cast<const uint16_t*>(acts),off);
     p3::simd<float,1> ds=bits.template bit_cast_view<sycl::half>().read();
     p3::simd<float,16> factor=d*float(ds[0]);
     p3::simd<float,16> term=factor*dot;
     term.merge(0.f,!valid);acc[branch][e]=acc[branch][e]+term;
    }
   }
   }
#pragma unroll
   for(int e=0;e<NT;++e)p3::slm_block_store<float,16>(Partial+sid*NT*64+e*64,acc[0][e]+acc[1][e]);
   p3::barrier();
   if(sid<NT){
    p3::simd<float,16> v[16];
#pragma unroll
    for(int r=0;r<16;++r)v[r]=p3::slm_block_load<float,16>(Partial+r*NT*64+sid*64);
#pragma unroll
    for(int off=8;off;off/=2){
#pragma unroll
     for(int r=0;r<off;++r)v[r]=v[r]+v[r+off];
    }
    p3::block_store<float,16>(y+size_t(sid)*stride+base,v[0]);
   }
  });
}
template<int KB>void dispatch(int T,const uint8_t*w,const block_q8_1*x,float*y,int N,int stride,sycl::queue*q){
#define RUN(E) case E:iq3_paired_s4_launch<E,KB>(w,x,y,N,stride,q);break
 switch(T){RUN(1);RUN(2);RUN(3);RUN(4);}
#undef RUN
}
class IQ3FieldPermutation;
}
extern "C" const char* xpu_iq3_packed_error(){return sglang_iq3_packed::error.c_str();}
extern "C" int xpu_iq3_field_permute(void*queue,const uint8_t*src,uint8_t*dst,int K,int N,int inverse){
 if(!queue||!src||!dst||src==dst||(K!=2560&&K!=6144)||N<16||N>16384||N%16||(inverse!=0&&inverse!=1))return 2;
 try{
  auto*q=static_cast<sycl::queue*>(queue);if(!q->has_property<sycl::property::queue::in_order>())return 2;
  const int KB=K/256;const size_t bytes=size_t(N)*KB*110;
  q->parallel_for<sglang_iq3_packed::IQ3FieldPermutation>(sycl::range<1>(bytes),[=](sycl::id<1>id){
   const size_t b=id[0],tile=b/(KB*1760),kb=(b/1760)%KB;const int j=b%1760;int row,raw;
   if(j<1664){const int seg=j/208,off=j%208;
    if(off<64){row=off/4;raw=2+seg*8+off%4;}
    else if(off<128){row=(off-64)/4;raw=6+seg*8+(off-64)%4;}
    else if(off<192){row=(off-128)/4;raw=74+seg*4+(off-128)%4;}
    else{row=off-192;raw=66+seg;}
   }else if(j<1728){row=(j-1664)/4;raw=106+(j-1664)%4;}
   else{row=(j-1728)/2;raw=(j-1728)%2;}
   const size_t original=((tile*16+row)*KB+kb)*110+raw;
   if(inverse)dst[original]=src[b];else dst[b]=src[original];
  });return 0;
 }catch(const std::exception&e){sglang_iq3_packed::error=e.what();return 1;}
}
extern "C" int xpu_iq3_packed(void*queue,const void*w,const void*x,float*out,int K,int N,int T,int row_bytes,int stride){
 if(!queue||!w||!x||!out||(K!=2560&&K!=6144)||N<16||N>16384||N%16||T<1||T>4||row_bytes!=K/256*110||stride<N)return 2;
 try{
  auto*q=static_cast<sycl::queue*>(queue);if(!q->has_property<sycl::property::queue::in_order>())return 2;
  if(K==2560)sglang_iq3_packed::dispatch<10>(T,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),out,N,stride,q);
  else sglang_iq3_packed::dispatch<24>(T,static_cast<const uint8_t*>(w),static_cast<const block_q8_1*>(x),out,N,stride,q);
  return 0;
 }catch(const std::exception&e){sglang_iq3_packed::error=e.what();return 1;}
}
