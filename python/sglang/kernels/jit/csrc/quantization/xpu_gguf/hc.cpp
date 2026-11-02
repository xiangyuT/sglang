// SPDX-License-Identifier: Apache-2.0
// Reproduce the installed Torch XPU contiguous MeanOps reduction ordering,
// fusing BF16 conversion, square, mean, rsqrt and Gemma normalization.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>
namespace sglang_hc {
using bf16=sycl::ext::oneapi::bfloat16;
thread_local std::string error;
template<int SG,int Vec> class Normalize;
template<int SG,int Vec>
void launch(sycl::queue& q,const bf16* x,const bf16* w,bf16* y,float* means,
            int T,int WG,float eps) {
    const int outputs=T*4;
    int height=1;while(height*2<=outputs)height*=2;
    height=sycl::min(height,WG/32);
    const int width=WG/height,groups=outputs;
    height=1;
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<float,1> local(WG,h);
        h.parallel_for<Normalize<SG,Vec>>(sycl::nd_range<2>(
            sycl::range<2>(groups*height,width),sycl::range<2>(height,width)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int ly=it.get_local_id(0),lx=it.get_local_id(1);
            const int row=it.get_group(0)*height+ly;
            float v[Vec]={};
            if(row<outputs) {
                for(int j=lx*Vec;j<2560;j+=width*Vec) {
                    for(int k=0;k<Vec;++k) {
                        const float a=j+k<2560?float(x[int64_t(row)*2560+j+k]):0;
                        v[k]=v[k]+a*a;
                    }
                }
            }
            float value=v[0];for(int k=1;k<Vec;++k)value=value+v[k];
            local[ly*width+lx]=value;
            for(int offset=width/2;offset>=SG;offset/=2) {
                it.barrier(sycl::access::fence_space::local_space);
                if(lx<offset) {
                    value=value+local[ly*width+lx+offset];
                    local[ly*width+lx]=value;
                }
            }
            for(int offset=1;offset<sycl::min(width,SG);offset*=2)
                value=value+sycl::shift_group_left(it.get_sub_group(),value,offset);
            if(lx==0) {
                const float mean=value*(1.f/2560.f);
                local[ly*width]=mean;
                if(row<outputs)means[row]=mean;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if(row>=outputs)return;
            const float inv=sycl::rsqrt(local[ly*width]+eps);
            const int stream=row%4;
            for(int j=lx;j<2560;j+=width) {
                const float norm=float(x[int64_t(row)*2560+j])*inv;
                const float gain=1.f+float(w[stream*2560+j]);
                y[int64_t(row)*2560+j]=bf16(norm*gain);
            }
        });
    });
}
}
extern "C" int xpu_gguf_hc_norm(void* queue,const void* x,const void* w,void* y,float* means,int T,float eps) {
    if(!queue||!x||!w||!y||!means||T<1||T>4||!(eps>0))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        sglang_hc::launch<32,4>(q,(const sglang_hc::bf16*)x,(const sglang_hc::bf16*)w,(sglang_hc::bf16*)y,means,T,1024,eps);
        return 0;
    }catch(const std::exception& e){sglang_hc::error=e.what();return 1;}
}
extern "C" const char* xpu_gguf_hc_error(){return sglang_hc::error.c_str();}
