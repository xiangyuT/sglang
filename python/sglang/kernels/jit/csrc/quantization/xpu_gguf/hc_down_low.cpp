// SPDX-License-Identifier: Apache-2.0
// B70 bounded-row BF16 GEMM. Lossless original row-major operands; FP32 DPAS.
// Direct 2D-transposed dword loads supply the VNNI BF16 matrix operand.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>
#include <string>
namespace hc_down_low_fused {
namespace es=sycl::ext::intel::esimd;
thread_local std::string error;
template<int E,bool Direct>class Gemm;
class ReduceLow;
SYCL_ESIMD_FUNCTION inline es::simd<uint16_t,16> round_bf16(es::simd<float,16> v) {
    auto bits=v.template bit_cast_view<uint32_t>().read();
    es::simd<uint32_t,16> rounded=(bits+0x7fffu+((bits>>16)&1u))>>16;
    rounded.merge((bits>>16)|0x40u,(bits&0x7fffffffu)>0x7f800000u);
    return es::convert<uint16_t>(rounded);
}
template<int E,bool Direct>
void launch(sycl::queue& q,const uint16_t* x,const uint16_t* weight,uint16_t* out,
    float* partial,float* raw,int m,int n,int k,int splits) {
    const int tiles=(n+15)/16,tasks=tiles*splits;
    q.parallel_for<Gemm<E,Direct>>(sycl::nd_range<1>((tasks+3)/4*4,4),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
        const int task=int(item.get_global_linear_id());if(task>=tasks)return;
        const int part=task/tiles,row=(task%tiles)*16;
        const int first=part*(k/splits),last=first+k/splits;
        es::simd<float,E*16> sum=0.f;
        for(int begin=first;begin<last;begin+=16) {
            auto bw=es::load_2d<uint32_t,8,16,1,true,false>(
                reinterpret_cast<const uint32_t*>(weight),k*2-1,n-1,k*2-1,begin/2,row);
            es::simd<uint16_t,256> b=bw.template bit_cast_view<uint16_t>().read();
            auto aw=es::load_2d<uint32_t,8,E,1,false,false>(reinterpret_cast<const uint32_t*>(x),k*2-1,m-1,k*2-1,begin/2,0);
            es::simd<uint16_t,E*16> a=aw.template bit_cast_view<uint16_t>().read();
            using P=es::xmx::dpas_argument_type;
            sum=es::xmx::dpas<8,E,float,float,uint16_t,uint16_t,P::bf16,P::bf16>(sum,b,a);
        }
        es::simd<uint32_t,16> rows(row,1),offsets=rows*4u;
        auto valid=rows<uint32_t(n);
#pragma unroll
        for(int e=0;e<E;++e)if(e<m) {
            es::simd<float,16> value=sum.template select<16,1>(e*16);
            if constexpr(Direct) {
                es::scatter<uint16_t,16>(out+size_t(e)*n,rows*2u,round_bf16(value),valid);
                if(raw)es::scatter<float,16>(raw+size_t(e)*n,offsets,value,valid);
            }else es::scatter<float,16>(partial+(size_t(part)*m+e)*n,offsets,value,valid);
        }
    });
}
void reduce_low(sycl::queue& q,const float* partial,uint16_t* out,uint16_t* low) {
    constexpr int m=4,n=324,splits=10,total=m*n,blocks=(total+15)/16;
    q.parallel_for<ReduceLow>(sycl::range<1>(blocks),[=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
        es::simd<uint32_t,16> index(uint32_t(item[0])*16,1);auto valid=index<uint32_t(total);
        es::simd<float,16> value=0.f;
        for(int p=0;p<splits;++p)value+=es::gather<float,16>(partial+size_t(p)*total,index*4u,valid);
        auto rounded=round_bf16(value);
        es::scatter<uint16_t,16>(out,index*2u,rounded,valid);
        es::simd<uint32_t,16> high=es::convert<uint32_t>(rounded)<<16;
        auto score=high.template bit_cast_view<float>().read();
        auto input=score*0.25f;
        auto activation=input*es::inv(1.f+es::exp(-input));
        auto row=index/324u,col=index%324u;
        auto low_index=row*320u+col;
        es::scatter<uint16_t,16>(low,low_index*2u,round_bf16(activation),valid&(col<320u));
    });
}
}
extern "C" const char* xpu_hc_down_low_error(){return hc_down_low_fused::error.c_str();}
extern "C" int xpu_hc_down_low(void* queue,const uint16_t* x,const uint16_t* weight,
    uint16_t* scores,float* partial,uint16_t* low) {
    if(!queue||!x||!weight||!scores||!partial||!low)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        hc_down_low_fused::launch<4,false>(q,x,weight,scores,partial,nullptr,4,324,10240,10);
        hc_down_low_fused::reduce_low(q,partial,scores,low);
        return 0;
    }catch(const std::exception& e){hc_down_low_fused::error=e.what();return 1;}
}
