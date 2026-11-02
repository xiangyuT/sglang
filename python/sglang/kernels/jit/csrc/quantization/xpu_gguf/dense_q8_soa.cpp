// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Strata/SGLang MTP Q8 SoA16 loads and DPAS, adapted to the existing dense
// Q8_0 operator's four 8-value partials and exact32 FP32 accumulation tree.
#include "dequant.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <string>
namespace sglang_q8_dense_soa {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
template<int NT, int KB, bool BF16> class Dense;
template<int NT, int KB, bool BF16>
void launch(sycl::queue& q, const uint8_t* weights, const block_q8_1* acts,
            void* output, int N, int stride) {
    constexpr int WG=8, NR=16, MX=NT<=2?NT:NT<=4?4:8;
    q.parallel_for<Dense<NT,KB,BF16>>(
        sycl::nd_range<1>(size_t(N/NR)*WG,WG),
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
#pragma clang fp contract(off)
            es::slm_init<WG*4*NT*NR*4>();
            const int sid=it.get_local_linear_id(),base=it.get_group_linear_id()*NR;
            es::simd<uint32_t,NR> rows(base,1);
            auto valid=rows<uint32_t(N);
            es::simd<float,NR> acc[4][NT];
#pragma unroll
            for(int part=0;part<4;++part) {
#pragma unroll
                for(int t=0;t<NT;++t) acc[part][t]=0.f;
            }
            // Four partial 8-element integer dots per Q8 block reproduce the
            // existing DP4A kernel. A single 32-element dot would round at
            // different boundaries and cannot replace these four chains.
            for(int kb=sid;kb<KB;kb+=WG) {
                const auto* block=weights+size_t(base)*(KB*34)+kb*544;
                auto dh=es::block_load<uint16_t,NR>(reinterpret_cast<const uint16_t*>(block));
                es::simd<float,NR> dw=dh.template bit_cast_view<sycl::half>().read();
                auto data=es::block_load<uint64_t,64>(reinterpret_cast<const uint64_t*>(block+32));
                es::simd<int8_t,512> B=data.template bit_cast_view<int8_t>().read();
                // One vector gather replaces token-by-token word/scale sends.
                // Preserve the exact four partial products and 32 FP32 chains.
                constexpr int NA=NT<=1?1:NT<=2?2:NT<=4?4:8;
                es::simd<uint32_t,NA*8> lanes(0,1);
                auto offsets=((lanes>>3)*uint32_t(KB)+uint32_t(kb))*36u+
                             4u+(lanes&7u)*4u;
                auto packed=es::gather<uint32_t,NA*8>(
                    reinterpret_cast<const uint32_t*>(acts),offsets,
                    lanes<uint32_t(NT*8));
                es::simd<uint32_t,NT*8> words=packed.template select<NT*8,1>(0);
                es::simd<uint32_t,NA> tokens(0,1);
                auto db=es::gather<uint16_t,NA>(reinterpret_cast<const uint16_t*>(acts),
                    (tokens*uint32_t(KB)+uint32_t(kb))*36u,tokens<uint32_t(NT));
                es::simd<float,NA> scales=db.template bit_cast_view<sycl::half>().read();
                es::simd<float,NT> scale=scales.template select<NT,1>(0);
#pragma unroll
                for(int part=0;part<4;++part) {
                    es::simd<uint32_t,MX*8> aw=0;
#pragma unroll
                    for(int t=0;t<NT;++t)
                        aw.template select<2,1>(t*8+part*2)=words.template select<2,1>(t*8+part*2);
                    es::simd<int8_t,MX*32> A=aw.template bit_cast_view<int8_t>().read();
                    es::simd<int32_t,MX*NR> zero=0;
                    auto product=es::xmx::dpas<8,MX,int32_t>(zero,B,A);
#pragma unroll
                    for(int t=0;t<NT;++t) {
                        es::simd<int32_t,NR> integer=product.template select<NR,1>(t*NR);
                        const float ds=scale[t];
                        es::simd<float,NR> dot=integer,factor=dw*ds,term=factor*dot;
                        term.merge(0.f,!valid);
                        acc[part][t]=acc[part][t]+term;
                    }
                }
            }
#pragma unroll
            for(int part=0;part<4;++part) {
#pragma unroll
                for(int t=0;t<NT;++t)
                    es::slm_block_store<float,NR>(((sid*4+part)*NT+t)*64,acc[part][t]);
            }
            es::barrier();
            if(sid<NT) {
                es::simd<float,NR> partial[4];
#pragma unroll
                for(int part=0;part<4;++part) {
                    es::simd<float,NR> v[WG];
#pragma unroll
                    for(int s=0;s<WG;++s)
                        v[s]=es::slm_block_load<float,NR>(((s*4+part)*NT+sid)*64);
#pragma unroll
                    for(int off=4;off;off/=2) {
#pragma unroll
                        for(int s=0;s<off;++s) v[s]=v[s]+v[s+off];
                    }
                    partial[part]=v[0];
                }
                auto value=(partial[0]+partial[2])+(partial[1]+partial[3]);
                if constexpr(BF16) {
                    es::simd<sycl::ext::oneapi::bfloat16,NR> b=value;
                    es::simd<uint16_t,NR> bits=b.template bit_cast_view<uint16_t>().read();
                    es::scatter<uint16_t,NR>(reinterpret_cast<uint16_t*>(output),
                        (rows+uint32_t(sid)*uint32_t(stride))*2u,bits,valid);
                } else {
                    es::scatter<float,NR>(reinterpret_cast<float*>(output),
                        (rows+uint32_t(sid)*uint32_t(stride))*4u,value,valid);
                }
            }
        });
}
template<int KB,bool BF16>
void dispatch(sycl::queue& q,const uint8_t* w,const block_q8_1* x,
              void* out,int N,int T,int stride) {
#define RUN(NT) case NT:launch<NT,KB,BF16>(q,w,x,out,N,stride);break
    switch(T) { RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8); }
#undef RUN
}
}
extern "C" const char* xpu_gguf_dense_q8_soa_error(){return sglang_q8_dense_soa::error.c_str();}
extern "C" int xpu_gguf_dense_q8_soa(void* queue,const void* w,const void* x,void* out,
    int K,int N,int T,int stride,int bf16) {
    if(!queue||!w||!x||!out||(K!=640&&K!=2560&&K!=6144)||N<16||N%16||
       N>248320||T<1||T>4||stride<N||(bf16!=0&&bf16!=1))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define RUN(KB) case KB*32: if(bf16)sglang_q8_dense_soa::dispatch<KB,true>(q,(const uint8_t*)w,(const block_q8_1*)x,out,N,T,stride);else sglang_q8_dense_soa::dispatch<KB,false>(q,(const uint8_t*)w,(const block_q8_1*)x,out,N,T,stride);break
        switch(K) { RUN(20);RUN(80);RUN(192); }
#undef RUN
        return 0;
    }catch(const std::exception& e){sglang_q8_dense_soa::error=e.what();return 1;}
}

// Separate ABI keeps the original four-row contract intact. Each extra token
// has the same four integer products, 32 FP32 chains and final reduction tree;
// only the encoded weight load is shared across more token rows.
extern "C" int xpu_gguf_dense_q8_soa_wide(void* queue,const void* w,const void* x,void* out,
    int K,int N,int T,int stride,int bf16) {
    if(!queue||!w||!x||!out||(K!=640&&K!=2560&&K!=6144)||N<16||N%16||
       N>248320||T<5||T>8||stride<N||(bf16!=0&&bf16!=1))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
#define RUN(KB) case KB*32: if(bf16)sglang_q8_dense_soa::dispatch<KB,true>(q,(const uint8_t*)w,(const block_q8_1*)x,out,N,T,stride);else sglang_q8_dense_soa::dispatch<KB,false>(q,(const uint8_t*)w,(const block_q8_1*)x,out,N,T,stride);break
        switch(K) { RUN(20);RUN(80);RUN(192); }
#undef RUN
        return 0;
    }catch(const std::exception& e){sglang_q8_dense_soa::error=e.what();return 1;}
}

namespace sglang_q8_dense_soa { class Permute; }
extern "C" int xpu_gguf_q8_field_permute(void* queue,const uint8_t* input,uint8_t* output,
    int K,int N,int inverse) {
    if(!queue||!input||!output||input==output||(K!=640&&K!=2560&&K!=6144)||
       N<16||N>248320||N%16||(inverse!=0&&inverse!=1))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        const int blocks=K/32,row_bytes=blocks*34;
        const size_t bytes=size_t(N)*row_bytes;
        q.parallel_for<sglang_q8_dense_soa::Permute>(
            sycl::range<1>((bytes+255)/256*256),[=](sycl::id<1> id) {
                const size_t i=id[0];if(i>=bytes)return;
                const size_t row=i/row_bytes,offset=i%row_bytes;
                const int block=offset/34,byte=offset%34;
                const size_t packed=(row/16)*size_t(row_bytes)*16+block*544+
                    (byte<2?(row%16)*2+byte:32+((byte-2)/4)*64+(row%16)*4+(byte-2)%4);
                if(inverse)output[i]=input[packed];else output[packed]=input[i];
            });
        return 0;
    }catch(const std::exception& e){sglang_q8_dense_soa::error=e.what();return 1;}
}
