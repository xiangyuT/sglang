// SPDX-License-Identifier: Apache-2.0
// Prototype: original BF16 HC up weights, FP32 DPAS then BF16 high rounding.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <string>
namespace hc_up_sequential {
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
template<int WG,int E> void launch(sycl::queue&q,const uint16_t*low,const uint16_t*w,const uint16_t*norm,uint16_t*out) {
    q.parallel_for<Mix<WG,E>>(sycl::nd_range<1>(160*4/E,WG),[=](sycl::nd_item<1>it) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
        const int row=int(it.get_global_linear_id()%160)*16;const int first_token=int(it.get_global_linear_id()/160)*E;
        es::simd<float,E*16> sums[4];
#pragma unroll
        for(int branch=0;branch<4;++branch)sums[branch]=0.f;
        for(int k=0;k<320;k+=16) {
            auto aa=es::load_2d<uint32_t,8,E,1,false,false>(reinterpret_cast<const uint32_t*>(low+first_token*320),639,E-1,639,k/2,0);
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
        for(int token=0;token<E;++token) {
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
extern "C" const char*xpu_hc_up_mix_error(){return hc_up_sequential::error.c_str();}
extern "C" int xpu_hc_up_mix(void*queue,const uint16_t*low,const uint16_t*w,const uint16_t*norm,uint16_t*out,int tile,int wg) {
    if(!queue||!low||!w||!norm||!out||wg!=1||(tile!=1&&tile!=2&&tile!=4))return 2;
    try{auto&q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
        if(tile==4)hc_up_sequential::launch<1,4>(q,low,w,norm,out);
        else if(tile==2)hc_up_sequential::launch<1,2>(q,low,w,norm,out);
        else hc_up_sequential::launch<1,1>(q,low,w,norm,out);
        return 0;
    }catch(const std::exception&e){hc_up_sequential::error=e.what();return 1;}
}
