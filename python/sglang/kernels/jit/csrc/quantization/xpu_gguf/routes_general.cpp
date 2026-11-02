// SPDX-License-Identifier: Apache-2.0
// Route batching using the separately attributed Strata/ggml decoders.
#include "dequant.hpp"
#include <string>
#include <limits>
namespace sglang_gguf {
extern thread_local std::string error;
template<bool Pair> class ExpertRowsGeneral;
template<bool Pair>
void expert_rows_general(sycl::queue& q, const uint8_t* w, const uint8_t* wu,
                 const float* x, const int32_t* slots, const int32_t* tokens, float* out, float* outu,
                 int type, int k, int n, int routes, int64_t expert_bytes,
                 int row_bytes, int capacity, int input_rows) {
    const int groups = (n + 3) / 4;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> decoded(4*256,h);
        h.parallel_for<ExpertRowsGeneral<Pair>>(
            sycl::nd_range<1>(size_t(groups)*routes*128,128),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int route = it.get_group_linear_id()/groups;
                const int row = (it.get_group_linear_id()%groups)*4+it.get_local_linear_id()/32;
                if(row>=n)return;
                const int lane=it.get_local_linear_id()%32;
                const int slot=slots[route];
                if (slot < 0 || slot >= capacity) {
                    if (lane == 0) {
                        out[int64_t(route)*n+row] = std::numeric_limits<float>::quiet_NaN();
                        if constexpr (Pair) outu[int64_t(route)*n+row] = std::numeric_limits<float>::quiet_NaN();
                    }
                    return;
                }
                const uint8_t* weight=w+int64_t(slot)*expert_bytes+int64_t(row)*row_bytes;
                const uint8_t* weightu=Pair ? wu+int64_t(slot)*expert_bytes+int64_t(row)*row_bytes : nullptr;
                const int input_row = Pair ? tokens[route] : route;
                if (input_row < 0 || input_row >= input_rows) {
                    if (lane == 0) {
                        out[int64_t(route)*n+row] = std::numeric_limits<float>::quiet_NaN();
                        if constexpr (Pair) outu[int64_t(route)*n+row] = std::numeric_limits<float>::quiet_NaN();
                    }
                    return;
                }
                const float* act=x+int64_t(input_row)*k;
                float* tile=decoded.get_multi_ptr<sycl::access::decorated::no>().get()+(it.get_local_linear_id()/32)*256;
                auto sg=it.get_sub_group();float sum=0,sumu=0;
                for(int base=0;base<k;base+=256){
                    int count=sycl::min(256,k-base);
                    if(lane*8<count)dq_dispatch<float>(type,weight,base/256,tile,lane);
                    sycl::group_barrier(sg);
                    for(int j=lane;j<count;j+=32)sum=sycl::fma(tile[j],act[base+j],sum);
                    sycl::group_barrier(sg);
                    if constexpr(Pair){
                        if(lane*8<count)dq_dispatch<float>(type,weightu,base/256,tile,lane);
                        sycl::group_barrier(sg);
                        for(int j=lane;j<count;j+=32)sumu=sycl::fma(tile[j],act[base+j],sumu);
                        sycl::group_barrier(sg);
                    }
                }
                sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());
                if constexpr(Pair)sumu=sycl::reduce_over_group(sg,sumu,sycl::plus<float>());
                if(lane==0){out[int64_t(route)*n+row]=sum;if constexpr(Pair)outu[int64_t(route)*n+row]=sumu;}
            });
    });
}
}
extern "C" int xpu_gguf_expert_rows_general(void* queue,const void* weight,const void* weightu,
                                    const float* x,const int32_t* slots,const int32_t* tokens,float* out,float* outu,
                                    int type,int k,int n,int routes,int64_t expert_bytes,int row_bytes,int pair,int capacity,int input_rows){
    if(!queue||!weight||!x||!slots||!out||k<=0||n<=0||routes<=0||expert_bytes<=0||row_bytes<=0||capacity<=0||input_rows<=0||(pair&&!tokens)||(!pair&&input_rows!=routes)||(pair!=0&&pair!=1)||(pair&&(!weightu||!outu)))return 2;
    try{
        auto& q=*static_cast<sycl::queue*>(queue);
        if(pair)sglang_gguf::expert_rows_general<true>(q,static_cast<const uint8_t*>(weight),static_cast<const uint8_t*>(weightu),x,slots,tokens,out,outu,type,k,n,routes,expert_bytes,row_bytes,capacity,input_rows);
        else sglang_gguf::expert_rows_general<false>(q,static_cast<const uint8_t*>(weight),nullptr,x,slots,tokens,out,nullptr,type,k,n,routes,expert_bytes,row_bytes,capacity,input_rows);
        return 0;
    }catch(const std::exception& e){sglang_gguf::error=e.what();return 1;}
}
