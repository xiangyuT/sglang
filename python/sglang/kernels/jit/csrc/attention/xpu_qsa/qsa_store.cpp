// SPDX-License-Identifier: Apache-2.0
// Small QSA index-cache writes without a general-purpose sort. Source buffers
// and valid locations belong to the pool allocator. Duplicate rows keep the
// last source, matching deterministic Torch indexed assignment.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <exception>
namespace sglang_qsa_store {
template<class Element, class Index> class Store;
template<class Element, class Index>
void submit(sycl::queue& q, Element* dst, const Element* src, const Index* loc,
            int rows, int cols, int capacity, int64_t row_stride, int64_t col_stride) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int,1> destination(1,h);
        h.parallel_for<Store<Element,Index>>(
            sycl::nd_range<1>(size_t(rows)*256,256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int row=it.get_group_linear_id();
                if (it.get_local_linear_id()==0) {
                    int64_t target=loc[row];
                    if (target<0) target+=capacity;
                    if (target>=0 && target<capacity) {
                        for (int later=row+1;later<rows;++later) {
                            int64_t other=loc[later];
                            if (other<0) other+=capacity;
                            if (other==target) { target=-1;break; }
                        }
                    } else target=-1;
                    destination[0]=int(target);
                }
                it.barrier(sycl::access::fence_space::local_space);
                const int target=destination[0];
                if (target<0) return;
                for (int col=it.get_local_linear_id();col<cols;col+=256)
                    dst[int64_t(target)*cols+col]=src[int64_t(row)*row_stride+int64_t(col)*col_stride];
            });
    });
}
}
extern "C" int xpu_qsa_store_rows(void* queue,void* dst,const void* src,const void* loc,
    int rows,int cols,int capacity,int element_bytes,int index_bytes,
    int64_t row_stride,int64_t col_stride) {
    if (!queue||!dst||!src||!loc||rows<0||rows>8192||cols<=0||capacity<=0||
        (element_bytes!=2&&element_bytes!=8)||(index_bytes!=4&&index_bytes!=8)||
        row_stride<0||col_stride<0) return 2;
    if (!rows) return 0;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
#define RUN(E,I) sglang_qsa_store::submit(q,static_cast<E*>(dst),static_cast<const E*>(src),static_cast<const I*>(loc),rows,cols,capacity,row_stride,col_stride)
        if (element_bytes==2) {
            if (index_bytes==4) { RUN(uint16_t,int32_t); } else { RUN(uint16_t,int64_t); }
        } else {
            if (index_bytes==4) { RUN(uint64_t,int32_t); } else { RUN(uint64_t,int64_t); }
        }
#undef RUN
        return 0;
    } catch (const std::exception&) { return 1; }
}
