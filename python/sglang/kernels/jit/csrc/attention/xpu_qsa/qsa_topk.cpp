// SPDX-License-Identifier: Apache-2.0
// Prototype: stable QSA TopK512 over a zero-start valid prefix.
// The QSA scorer masks the suffix to -inf. Never read that padded suffix.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace qsa_hybrid_prefix_top512 {
thread_local std::string error;
constexpr int K=512, WG=256;
class Streaming;
inline uint64_t key(float score, uint32_t index) {
  uint32_t b=sycl::bit_cast<uint32_t>(score);
  // Match installed Torch XPU KeyTraits<float>: IEEE radix bit order,
  // including signed zero and NaN payload/sign, then stable low-index ties.
  const uint32_t ordered=(b&0x80000000u) ? ~b : (b^0x80000000u);
  return (uint64_t(ordered)<<32)|uint32_t(~index);
}
void short_launch(sycl::queue& q, const float* scores, const void* lengths,
            int32_t* output, int rows, int columns, int64_t stride,
            bool lengths64) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t,1> a(sycl::range<1>(2*K),h);
    h.parallel_for<Streaming>(sycl::nd_range<1>(size_t(rows)*WG,WG),
        [=](sycl::nd_item<1> it) {
      const int row=int(it.get_group_linear_id()), lane=int(it.get_local_linear_id());
      int64_t n=lengths64 ? static_cast<const int64_t*>(lengths)[row] :
                          static_cast<const int32_t*>(lengths)[row];
      n=n<0 ? 0 : (n>columns ? columns : n);
      if(n>2048)return;
      for(int i=lane;i<K;i+=WG) {
        const int64_t pad=n+i;
        a[i]=pad<columns ? key(sycl::bit_cast<float>(uint32_t(0xff800000u)),uint32_t(pad)) : 0;
      }
      it.barrier(sycl::access::fence_space::local_space);
      for(int base=0;base<n;base+=K) {
        for(int i=lane;i<K;i+=WG) {
          int column=base+i;
          a[K+i]=column<n ? key(scores[int64_t(row)*stride+column],uint32_t(column)) : 0;
        }
        it.barrier(sycl::access::fence_space::local_space);
        // Incoming half ascending; retained top half is already descending.
        for(int size=2;size<=K;size*=2) {
          for(int distance=size/2;distance;distance/=2) {
            for(int i=lane;i<K;i+=WG) {
              const int other=i^distance;
              if(i<other) {
                uint64_t x=a[K+i],y=a[K+other];
                bool ascending=(i&size)==0;
                if(ascending ? x>y : x<y) { a[K+i]=y; a[K+other]=x; }
              }
            }
            it.barrier(sycl::access::fence_space::local_space);
          }
        }
        // Merge the descending/ascending bitonic sequence; keep top512.
        for(int distance=K;distance;distance/=2) {
          for(int i=lane;i<2*K;i+=WG) {
            const int other=i^distance;
            if(i<other) {
              uint64_t x=a[i],y=a[other];
              if(x<y) { a[i]=y; a[other]=x; }
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
        }
      }
      for(int i=lane;i<K;i+=WG) {
        uint64_t v=a[i];
        const uint32_t index=~uint32_t(v);
        output[int64_t(row)*K+i]=v && index<uint32_t(n) ? int32_t(index) : -1;
      }
    });
  });
}
constexpr int SHORT_LIMIT=2048;
class Tile;
template<bool Final> class Merge;
inline int valid_count(const void* lengths,int row,int columns,bool lengths64) {
  int64_t n=lengths64?static_cast<const int64_t*>(lengths)[row]:
                      static_cast<const int32_t*>(lengths)[row];
  return n<0?0:(n>columns?columns:int(n));
}
void tile_launch(sycl::queue& q,const float* scores,const void* lengths,
    uint64_t* output,int rows,int columns,int64_t stride,bool lengths64) {
  const int tiles=(columns+1023)/1024;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t,1> a(sycl::range<1>(1024),h);
    h.parallel_for<Tile>(sycl::nd_range<1>(size_t(rows)*tiles*WG,WG),
        [=](sycl::nd_item<1> it) {
      int group=int(it.get_group_linear_id()),row=group/tiles,tile=group%tiles;
      int lane=int(it.get_local_linear_id()),n=valid_count(lengths,row,columns,lengths64);
      int effective=sycl::min(columns,n+K),active=(effective+1023)/1024;
      if(n<=SHORT_LIMIT||tile>=active)return;
      for(int i=lane;i<1024;i+=WG) {
        int column=tile*1024+i;
        a[i]=column<n?key(scores[int64_t(row)*stride+column],uint32_t(column)):
          (column<columns?key(sycl::bit_cast<float>(uint32_t(0xff800000u)),uint32_t(column)):0);
      }
      it.barrier(sycl::access::fence_space::local_space);
      for(int size=2;size<=1024;size*=2) {
        for(int distance=size/2;distance;distance/=2) {
          for(int i=lane;i<1024;i+=WG) {
            int other=i^distance;
            if(i<other) {
              uint64_t x=a[i],y=a[other];
              bool descending=(i&size)==0;
              if(descending?x<y:x>y){a[i]=y;a[other]=x;}
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
        }
      }
      for(int i=lane;i<K;i+=WG)output[(int64_t(row)*tiles+tile)*K+i]=a[i];
    });
  });
}
template<bool Final>
void merge_launch(sycl::queue& q,const uint64_t* input,uint64_t* output,
    int32_t* indices,const void* lengths,int rows,int columns,
    int lists_in,int span,bool lengths64) {
  const int lists_out=(lists_in+1)/2;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t,1> a(sycl::range<1>(1024),h);
    h.parallel_for<Merge<Final>>(sycl::nd_range<1>(size_t(rows)*lists_out*WG,WG),
        [=](sycl::nd_item<1> it) {
      int group=int(it.get_group_linear_id()),row=group/lists_out,pair=group%lists_out;
      int lane=int(it.get_local_linear_id()),n=valid_count(lengths,row,columns,lengths64);
      int effective=sycl::min(columns,n+K),initial=(effective+1023)/1024;
      int active=(initial+span-1)/span;
      if(n<=SHORT_LIMIT||pair>=(active+1)/2)return;
      int left=pair*2,right=left+1;
      for(int i=lane;i<K;i+=WG) {
        a[i]=input[(int64_t(row)*lists_in+left)*K+i];
        a[K+i]=right<active?input[(int64_t(row)*lists_in+right)*K+(K-1-i)]:0;
      }
      it.barrier(sycl::access::fence_space::local_space);
      for(int distance=K;distance;distance/=2) {
        for(int i=lane;i<1024;i+=WG) {
          int other=i^distance;
          if(i<other) {
            uint64_t x=a[i],y=a[other];
            if(x<y){a[i]=y;a[other]=x;}
          }
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      for(int i=lane;i<K;i+=WG) {
        if constexpr(Final) {
          uint64_t v=a[i];uint32_t index=~uint32_t(v);
          indices[int64_t(row)*K+i]=v&&index<uint32_t(n)?int32_t(index):-1;
        } else output[(int64_t(row)*lists_out+pair)*K+i]=a[i];
      }
    });
  });
}
void hybrid_launch(sycl::queue& q,const float* scores,const void* lengths,
    int32_t* output,uint64_t* work0,uint64_t* work1,int rows,int columns,
    int64_t stride,bool lengths64) {
  short_launch(q,scores,lengths,output,rows,columns,stride,lengths64);
  if(columns<=SHORT_LIMIT)return;
  tile_launch(q,scores,lengths,work0,rows,columns,stride,lengths64);
  int lists=(columns+1023)/1024,span=1;
  const uint64_t* input=work0;uint64_t* next=work1;
  while(lists>1) {
    if((lists+1)/2==1)
      merge_launch<true>(q,input,next,output,lengths,rows,columns,lists,span,lengths64);
    else merge_launch<false>(q,input,next,output,lengths,rows,columns,lists,span,lengths64);
    lists=(lists+1)/2;span*=2;
    const uint64_t* previous=input;input=next;next=const_cast<uint64_t*>(previous);
  }
}
}
extern "C" const char* xpu_qsa_hybrid_top512_error(){
  return qsa_hybrid_prefix_top512::error.c_str();
}
extern "C" int xpu_qsa_hybrid_top512(void* queue,const float* scores,
    const void* lengths,int32_t* output,uint64_t* work0,uint64_t* work1,
    int rows,int columns,int64_t row_stride,int length_bytes) {
  if(!queue||!scores||!lengths||!output||rows<1||rows>64||columns<1||
      columns>65536||row_stride<columns||(length_bytes!=4&&length_bytes!=8)||
      (columns>2048&&(!work0||!work1)))return 2;
  try {
    auto& q=*static_cast<sycl::queue*>(queue);
    if(!q.has_property<sycl::property::queue::in_order>())return 2;
    qsa_hybrid_prefix_top512::hybrid_launch(q,scores,lengths,output,work0,work1,
                                         rows,columns,row_stride,length_bytes==8);
    return 0;
  }catch(const std::exception& e){qsa_hybrid_prefix_top512::error=e.what();return 1;}
}
