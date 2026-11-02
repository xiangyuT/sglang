// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Original compressed dense dot formulas from Strata
// c3ed1d047768d94f43d7384ccf740a214f5a53dd, iq_kernels.dp.cpp.
// Adapted to heterogeneous SGLang projections and explicit output strides.
// See STRATA-LICENSE and GGML-LICENSE.
#include "dequant.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <string>

namespace sglang_dense_q8_bf16 {
thread_local std::string error;
inline int get_int_b2(const void* x,int i) {
    const auto* p=static_cast<const uint16_t*>(x);
    return int(uint32_t(p[2*i])|(uint32_t(p[2*i+1])<<16));
}
inline int get_int_b4(const void* x,int i) {return static_cast<const int*>(x)[i];}
inline int dp4a(int a,int b,int c) {
#pragma unroll
    for(int i=0;i<4;++i)c+=int(int8_t(uint32_t(a)>>(8*i)))*int(int8_t(uint32_t(b)>>(8*i)));
    return c;
}
inline int swar_ne4(unsigned x) {
    x=(x|(x>>4))&0x0f0f0f0fu;x=(x|(x>>2))&0x03030303u;
    x=(x|(x>>1))&0x01010101u;return int(x*255u);
}
inline int swar_sub4(unsigned a,unsigned s) {
    return int(((a&0x7f7f7f7fu)+(s&0x01010101u))^(a&0x80808080u));
}
inline float dot_iq3(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_iq3_s*>(row)+block;
    const sycl::int2 packed(get_int_b2(w->qs,part),get_int_b2(w->qs,part+1));
    const auto* qs=reinterpret_cast<const uint8_t*>(&packed);
    const int qh=w->qh[part/2],signs32=get_int_b2(w->signs,part/2);
    const auto* signs=reinterpret_cast<const uint8_t*>(&signs32);
    int sum=0;
#pragma unroll
    for(int l=0;l<8;l+=2) {
        const int grid0=iq3s_grid[qs[l]|((qh<<(8-l))&256)];
        const int grid1=iq3s_grid[qs[l+1]|((qh<<(7-l))&256)];
        const int s0=swar_ne4(((signs[l/2]&3)<<7)|((signs[l/2]&12)<<21));
        const int s1=swar_ne4(((signs[l/2]&48)<<3)|((signs[l/2]&192)<<17));
        sum=dp4a(swar_sub4(grid0^s0,s0),get_int_b4(a[part/2].qs,l),sum);
        sum=dp4a(swar_sub4(grid1^s1,s1),get_int_b4(a[part/2].qs,l+1),sum);
    }
    sum*=1+2*((w->scales[part/4]>>((part<<1)&4))&15);
    const float d=float(w->d)*float(a[part/2].ds[0]);
    return d*sum;
}
inline float dot_q6(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_q6_K*>(row)+block;
    const int ao=4*(part/16)+(part%16)/8,so=8*(part/16)+(part%16)/4;
    const int vl=get_int_b2(w->ql,part);
    const int vh=get_int_b2(w->qh,8*(part/16)+part%8)>>(2*((part%16)/8));
    float sum=0;
#pragma unroll
    for(int i=0;i<2;++i) {
        const uint32_t codes=((uint32_t(vl)>>(4*i))&0x0f0f0f0fu)|
            (((uint32_t(vh)>>(4*i))<<4)&0x30303030u);
        // All input bytes are in [0,63], so subtraction cannot borrow across
        // bytes after setting the high bit. Preserve the signed -32 offset.
        const int signed_codes=int(((codes|0x80808080u)-0x20202020u)^0x80808080u);
        const int dot=dp4a(signed_codes,get_int_b4(a[ao+2*i].qs,part%8),0);
        sum+=float(a[ao+2*i].ds[0])*(dot*int(w->scales[so+4*i]));
    }
    return float(w->d)*sum;
}
inline float dot_iq4nl(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_iq4_nl*>(row)+block;
    int sum=0;
#pragma unroll
    for(int l=0;l<2;++l) {
        const uint32_t codes=get_int_b2(w->qs,part+l);
        uint32_t lo=0,hi=0;
#pragma unroll
        for(int j=0;j<4;++j) {
            lo|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*j))&15]))<<(8*j);
            hi|=uint32_t(uint8_t(kvalues_iq4nl[(codes>>(8*j+4))&15]))<<(8*j);
        }
        sum=dp4a(int(lo),get_int_b4(a->qs,part+l),sum);
        sum=dp4a(int(hi),get_int_b4(a->qs,part+l+4),sum);
    }
    const float d=float(w->d)*float(a->ds[0]);
    return d*sum;
}
inline float dot_q8(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_q8_0*>(row)+block;
    int sum=0;
#pragma unroll
    for(int i=0;i<2;++i)sum=dp4a(get_int_b2(w->qs,part+i),get_int_b4(a->qs,part+i),sum);
    return float(w->d)*float(a->ds[0])*float(sum);
}
inline uint32_t q2_decode4(uint32_t codes) {
    codes&=255;codes=(codes|(codes<<12))&0x000f000fu;
    codes=(codes|(codes<<6))&0x03030303u;
    return (codes+0x7f7f7f7fu)^0x80808080u;
}
inline float dot_q2(const void* row,const block_q8_1* a,int block,int part) {
    const auto* w=static_cast<const block_q2_0*>(row)+block;
    const auto* codes=reinterpret_cast<const uint16_t*>(w->qs)+part*4;
    int sum=0;
#pragma unroll
    for(int j=0;j<4;++j) {
        sum=dp4a(get_int_b4(a[part].qs,2*j),int(q2_decode4(codes[j])),sum);
        sum=dp4a(get_int_b4(a[part].qs,2*j+1),int(q2_decode4(codes[j]>>8)),sum);
    }
    return float(w->d)*float(a[part].ds[0])*sum;
}
template<int Type,int T> class Dense;
template<int Type,int T>
void submit(sycl::queue& q,const uint8_t* w,const block_q8_1* x,float* out,
    int K,int N,int row_bytes,int output_stride) {
    constexpr int QK=Type==14||Type==21?256:Type==42?64:32;
    constexpr int IPB=Type==14?32:Type==21?8:Type==8?4:2;
    constexpr int STEP=Type==14||Type==42?1:2;
    q.parallel_for<Dense<Type,T>>(sycl::nd_range<1>(size_t((N+3)/4)*128,128),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int row=it.get_group_linear_id()*4+it.get_local_linear_id()/32;
            const int lane=it.get_local_linear_id()%32;
            const auto* weight=w+int64_t(sycl::min(row,N-1))*row_bytes;
            float sum[T]={};
            for(int index=lane;index<(K/QK)*IPB;index+=32) {
                const int block=index/IPB,part=STEP*(index%IPB);
#pragma unroll
                for(int t=0;t<T;++t) {
                    const auto* a=x+int64_t(t)*(K/32)+block*(QK/32);
                    if constexpr(Type==21)sum[t]+=dot_iq3(weight,a,block,part);
                    else if constexpr(Type==14)sum[t]+=dot_q6(weight,a,block,part);
                    else if constexpr(Type==20)sum[t]+=dot_iq4nl(weight,a,block,part);
                    else if constexpr(Type==8)sum[t]+=dot_q8(weight,a,block,part);
                    else sum[t]+=dot_q2(weight,a,block,part);
                }
            }
#pragma unroll
            for(int t=0;t<T;++t) {
#pragma unroll
                for(int offset=16;offset;offset>>=1)
                    sum[t]+=sycl::permute_group_by_xor(it.get_sub_group(),sum[t],offset);
                if(lane==0&&row<N)reinterpret_cast<sycl::ext::oneapi::bfloat16*>(out)[int64_t(t)*output_stride+row]=sycl::ext::oneapi::bfloat16(sum[t]);
            }
        });
}
template<int Type>
void dispatch(int T,sycl::queue& q,const uint8_t* w,const block_q8_1* x,float* out,
    int K,int N,int row_bytes,int stride) {
#define RUN(NT) case NT:submit<Type,NT>(q,w,x,out,K,N,row_bytes,stride);break
    switch(T){RUN(1);RUN(2);RUN(3);RUN(4);}
#undef RUN
}
}
extern "C" const char* xpu_gguf_dense_q8_bf16_error(){return sglang_dense_q8_bf16::error.c_str();}
extern "C" int xpu_gguf_dense_q8_bf16(void* queue,const void* weight,const void* acts,float* out,
    int type,int K,int N,int T,int row_bytes,int output_stride) {
    const int qk=type==14||type==21?256:type==42?64:32;
    const int bytes=type==14?210:type==21?110:type==8?34:18;
    if(!queue||!weight||!acts||!out||K<=0||K%qk||N<=0||T<1||T>4||
        row_bytes!=K/qk*bytes||output_stride<N)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        const auto* w=static_cast<const uint8_t*>(weight);
        const auto* x=static_cast<const block_q8_1*>(acts);
#define RUN(TYPE) case TYPE:sglang_dense_q8_bf16::dispatch<TYPE>(T,q,w,x,out,K,N,row_bytes,output_stride);break
        switch(type){RUN(8);RUN(14);RUN(20);RUN(21);RUN(42);default:return 2;}
#undef RUN
        return 0;
    }catch(const std::exception& e){sglang_dense_q8_bf16::error=e.what();return 1;}
}
