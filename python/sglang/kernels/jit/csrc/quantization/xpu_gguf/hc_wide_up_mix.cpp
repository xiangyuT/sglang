// SPDX-License-Identifier: Apache-2.0
// Original BF16 HC up weights, FP32 DPAS and BF16 high rounding.
// The ABI supports 8..32 rows; Python selects the faster 8..24-row cases.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <string>
namespace sglang_hc_wide_up {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
template<int WG,int E> class Mix;
template<int N> SYCL_ESIMD_FUNCTION inline es::simd<float,N> bf16(es::simd<uint16_t,N> v) {
    es::simd<uint32_t,N> bits=es::convert<uint32_t>(v)<<16;
    return bits.template bit_cast_view<float>().read();
}
template<int N> SYCL_ESIMD_FUNCTION inline es::simd<uint16_t,N> rounded(es::simd<float,N> v) {
    auto bits=v.template bit_cast_view<uint32_t>().read();
    es::simd<uint32_t,N> out=(bits+0x7fffu+((bits>>16)&1u))>>16;
    out.merge((bits>>16)|0x40u,(bits&0x7fffffffu)>0x7f800000u);
    return es::convert<uint16_t>(out);
}
template<int WG,int E> void launch(sycl::queue&q,const uint16_t*low,const uint16_t*w,const uint16_t*norm,uint16_t*out,int tokens) {
    q.parallel_for<Mix<WG,E>>(sycl::nd_range<1>(160*((tokens+E-1)/E),WG),[=](sycl::nd_item<1>it) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
        const int row=int(it.get_global_linear_id()%160)*16;const int first_token=int(it.get_global_linear_id()/160)*E;
        es::simd<float,E*16> sums[4];
#pragma unroll
        for(int branch=0;branch<4;++branch)sums[branch]=0.f;
        for(int k=0;k<320;k+=16) {
            auto aa=es::load_2d<uint32_t,8,E,1,false,false>(reinterpret_cast<const uint32_t*>(low+first_token*320),639,tokens-first_token-1,639,k/2,0);
            auto a=aa.template bit_cast_view<uint16_t>().read();
#pragma unroll
            for(int branch=0;branch<4;++branch) {
                auto bb=es::load_2d<uint32_t,8,16,1,true,false>(reinterpret_cast<const uint32_t*>(w),639,10239,639,k/2,row+branch*2560);
                auto b=bb.template bit_cast_view<uint16_t>().read();
                using P=es::xmx::dpas_argument_type;
                sums[branch]=es::xmx::dpas<8,E,float,float,uint16_t,uint16_t,P::bf16,P::bf16>(sums[branch],b,a);
            }
        }
#pragma unroll
        for(int token=0;token<E;++token) if(first_token+token<tokens) {
            es::simd<float,16> parts[4];
#pragma unroll
            for(int branch=0;branch<4;++branch) {
                auto high=bf16<16>(rounded<16>(sums[branch].template select<16,1>(token*16)));
                auto gate=es::inv<float>(1.f+es::exp<float>(-high));
                auto n=bf16<16>(es::block_load<uint16_t,16>(norm+(first_token+token)*10240+branch*2560+row));
                parts[branch]=gate*n;
            }
            auto value=(((parts[0]+parts[1])+parts[2])+parts[3])*.25f;
            es::block_store<uint16_t,16>(out+(first_token+token)*2560+row,rounded<16>(value));
        }
    });
}
}
extern "C" const char*xpu_hc_wide_up_mix_error(){return sglang_hc_wide_up::error.c_str();}
extern "C" int xpu_hc_wide_up_mix(void*queue,const uint16_t*low,const uint16_t*w,const uint16_t*norm,uint16_t*out,int tokens) {
    if(!queue||!low||!w||!norm||!out||tokens<8||tokens>32||tokens%4)return 2;
    try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
        sglang_hc_wide_up::launch<1,8>(q,low,w,norm,out,tokens);
        return 0;
    }catch(const std::exception&e){sglang_hc_wide_up::error=e.what();return 1;}
}
