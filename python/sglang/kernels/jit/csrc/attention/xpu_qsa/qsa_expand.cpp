// SPDX-License-Identifier: Apache-2.0
// SYCL implementation of SGLang's torch_expand_qsa_block_indices contract:
// expand blocks, append the visible uncompressed tail, and stably compact.
#include <sycl/sycl.hpp>
#include <cstdint>

namespace sglang_qsa {
class ExpandBlocks;
inline int64_t logical_token(const int32_t* blocks, int column, int tokens,
                             int ratio, int64_t position, int64_t length) {
    if (column < tokens) {
        const int32_t block=blocks[column/ratio];
        const int64_t token=int64_t(block)*ratio+column%ratio;
        return block>=0 && token>=0 && token<length ? token : -1;
    }
    const int64_t visible=position+1;
    int64_t group=visible/ratio;
    if(visible<0 && visible%ratio)--group; // Torch floor division.
    const int64_t start=group*ratio;
    const int offset=column-tokens;
    const int64_t token=start+offset;
    return offset<visible-start && token<length ? token : -1;
}
}

extern "C" int xpu_qsa_expand(void* queue,const int32_t* blocks,
    const int64_t* positions,const int64_t* lengths,int32_t* output,
    int rows,int block_topk,int ratio,int token_topk) {
    if(!queue||!blocks||!positions||!lengths||!output||rows<1||ratio<2||ratio>128||
       token_topk<1||token_topk>8192||block_topk!=(token_topk+ratio-1)/ratio)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        const int width=token_topk+ratio-1;
        q.parallel_for<sglang_qsa::ExpandBlocks>(sycl::nd_range<1>(size_t(rows)*256,256),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int row=item.get_group_linear_id(),lane=item.get_local_linear_id();
                const int span=(width+255)/256;
                const int start=lane*span,end=sycl::min(start+span,width);
                const auto* source=blocks+int64_t(row)*block_topk;
                auto* target=output+int64_t(row)*width;
                const int64_t position=positions[row],length=lengths[row];
                int count=0;
                for(int column=start;column<end;++column)
                    count+=sglang_qsa::logical_token(source,column,token_topk,ratio,position,length)>=0;
                int prefix=sycl::exclusive_scan_over_group(item.get_group(),count,sycl::plus<int>());
                const int total=sycl::reduce_over_group(item.get_group(),count,sycl::plus<int>());
                // Torch stably partitions by nonnegative values; negative
                // tails from padded query positions retain their exact value.
                int invalid=total+start-prefix;
                for(int column=start;column<end;++column){
                    const int64_t token=sglang_qsa::logical_token(source,column,token_topk,ratio,position,length);
                    if(token>=0)target[prefix++]=static_cast<int32_t>(token);
                    else target[invalid++]=static_cast<int32_t>(token);
                }
            });
        return 0;
    } catch(const std::exception&){return 1;}
}
