// SPDX-License-Identifier: Apache-2.0
// Bounded MTP argmax + p(max). Uses the block log-sum-exp decomposition also
// used by Strata row_top_prob_split; independent SYCL implementation.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>

namespace mtp_top1_probability {
thread_local std::string error;
class Partial;
class Finish;

inline uint64_t key(float value, uint32_t index) {
    uint32_t bits=sycl::bit_cast<uint32_t>(value);
    // Torch argmax: first NaN wins; signed zero compares equal.
    uint32_t ordered=sycl::isnan(value)?0xffffffffu:
        ((bits&0x7fffffffu)==0?0x80000000u:
        ((bits&0x80000000u)?~bits:(bits^0x80000000u)));
    return (uint64_t(ordered)<<32)|(0xffffffffu-index);
}
inline float unkey(uint64_t value) {
    uint32_t ordered=uint32_t(value>>32);
    if(ordered==0xffffffffu)return sycl::nan(0u);
    uint32_t bits=(ordered&0x80000000u)?(ordered^0x80000000u):~ordered;
    return sycl::bit_cast<float>(bits);
}
void run(sycl::queue& q,const float* logits,float* probability,int64_t* index,
         uint64_t* keys,float* sums,int rows,int vocab,int blocks) {
    q.parallel_for<Partial>(sycl::nd_range<1>(size_t(rows)*blocks*256,256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        int group=int(it.get_group_linear_id()),row=group/blocks,block=group%blocks;
        int first=block*((vocab+blocks-1)/blocks),last=sycl::min(vocab,first+(vocab+blocks-1)/blocks);
        uint64_t best=0;
        for(int i=first+int(it.get_local_linear_id());i<last;i+=256)
            best=sycl::max(best,key(logits[size_t(row)*vocab+i],uint32_t(i)));
        best=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<uint64_t>());
        float maximum=unkey(best),sum=0.f;
        if(!sycl::isinf(maximum)||maximum>0.f)
            for(int i=first+int(it.get_local_linear_id());i<last;i+=256)
                sum+=sycl::exp(logits[size_t(row)*vocab+i]-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        if(it.get_local_linear_id()==0){keys[group]=best;sums[group]=sum;}
    });
    q.parallel_for<Finish>(sycl::nd_range<1>(size_t(rows)*64,64),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        int row=int(it.get_group_linear_id()),lane=int(it.get_local_linear_id());
        uint64_t local=lane<blocks?keys[row*blocks+lane]:0;
        uint64_t best=sycl::reduce_over_group(it.get_group(),local,sycl::maximum<uint64_t>());
        float maximum=unkey(best),sum=0.f;
        if(lane<blocks)sum=sums[row*blocks+lane]*sycl::exp(unkey(local)-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        if(lane==0){index[row]=int64_t(0xffffffffu-uint32_t(best));probability[row]=1.f/sum;}
    });
}
}
extern "C" const char* xpu_mtp_top1_probability_error(){return mtp_top1_probability::error.c_str();}
extern "C" int xpu_mtp_top1_probability(void* queue,const float* logits,float* probability,
    int64_t* index,uint64_t* keys,float* sums,int rows,int vocab,int blocks) {
    if(!queue||!logits||!probability||!index||!keys||!sums||rows<1||rows>4||vocab<64||vocab>262144||blocks<1||blocks>64||blocks>vocab)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        mtp_top1_probability::run(q,logits,probability,index,keys,sums,rows,vocab,blocks);
        return 0;
    }catch(const std::exception& e){mtp_top1_probability::error=e.what();return 1;}
}
