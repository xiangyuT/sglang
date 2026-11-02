// SPDX-License-Identifier: Apache-2.0
// TP shape adaptation of single-GPU resident124 GU and down541 all-hit kernels.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../gemm/xpu_tp_fp8_codec.hpp"
#include <cstdint>
#include <string>
#pragma clang fp contract(off) reassociate(off)
namespace tp_expert_gu {
thread_local std::string error;
inline float bf(uint16_t x){return sycl::bit_cast<float>(uint32_t(x)<<16);}
inline uint16_t rne(float x){uint32_t u=sycl::bit_cast<uint32_t>(x);if((u&0x7fffffffU)>0x7f800000U)return uint16_t((u>>16)|64);return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);}
template<int C>inline float sum(float(&v)[C]){
#pragma unroll
 for(int s=1;s<C;s*=2){
#pragma unroll
  for(int j=0;j<C;j+=2*s)v[j]+=v[j+s];}
 return v[0];}
inline int balanced_local(sycl::sub_group sg,const int*ids,const int*table,int rank,int lane){
 int e=lane<8?ids[lane]:-1;sycl::vec<int,2>pair(-1);
 if(e>=0&&e<256)pair.load(0,table+e*2);
 int left=pair[0],right=pair[1];bool forced0=left>=0&&right<0,forced1=right>=0&&left<0,flexible=left>=0&&right>=0;
 // Pack two counts (each <= 8) and eight distinct flexible-route bits into
 // one sum. The fields cannot carry into one another. This replaces two
 // reductions plus a prefix scan while preserving the same ownership rule.
 unsigned bit=lane<8?(1U<<(8+lane)):0U;
 unsigned packed=sycl::reduce_over_group(sg,forced0?1U:(forced1?16U:(flexible?bit:0U)),sycl::plus<unsigned>());
 int count0=packed&15U,count1=(packed>>4)&15U;
 int position=sycl::popcount((packed>>8)&((1U<<lane)-1));
 int difference=count0-count1,imbalance=difference<0?-difference:difference;
 int owner=forced0?0:(forced1?1:-1);
 if(flexible)owner=position<imbalance?(difference>0?1:0):((position-imbalance)&1);
 return owner==rank?(rank==0?left:right):-1;
}
template<int I,int R,bool Fuse,bool Replica>class GateUpTilePacked;
template<int I,bool Fuse=false,bool Replica=false>void run(sycl::queue&q,const uint16_t*x,const uint8_t*w13,const float*s13,const int*ids,uint16_t*raw,uint16_t*activated=nullptr,const int*replicas=nullptr,int*local_ids=nullptr,int rank=0){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.submit([&](sycl::handler&h){sycl::local_accessor<float,1>partial((I==512?4:2)*8,h);
  h.parallel_for<GateUpTilePacked<I,(I==512?4:2),Fuse,Replica>>(sycl::nd_range<1>((8*I/(I==512?4:2))*128,128),props,[=](sycl::nd_item<1>it){
   constexpr int R=I==512?4:2,H=2048;int group=it.get_group_linear_id(),tid=it.get_local_linear_id(),lane=tid%32,warp=tid/32;
   int route=group/(I/R),row0=(group%(I/R))*R,expert=ids[route],col=tid*16;
   auto sg=it.get_sub_group();
   if constexpr(Replica){
    int local=balanced_local(sg,ids,replicas,rank,lane);
    expert=sycl::select_from_group(sg,local,route);
    // One work-group writes each route ID. The subsequent down kernel sees
    // all eight writes through the normal in-order stream dependency.
    if(tid==0&&row0==0)local_ids[route]=expert;
   }
   // Every work-item in the group sees the same route. Nonlocal experts
   // contribute zero without reading any expert weights or activation rows.
   if(expert<0){
    if(tid==0){for(int r=0;r<R;++r){raw[route*(2*I)+row0+r]=0;raw[route*(2*I)+I+row0+r]=0;if constexpr(Fuse)activated[route*I+row0+r]=0;}}
    return;
   }
   sycl::vec<uint32_t,8>xv;xv.load(0,reinterpret_cast<const uint32_t*>(x+col));float xf[16];
#pragma unroll
   for(int j=0;j<16;++j)xf[j]=bf(uint16_t(xv[j/2]>>((j%2)*16)));
   float gs=s13[expert*(2*I/128)*16+(row0/128)*16+col/128]*256.f;
   float us=s13[expert*(2*I/128)*16+((row0+I)/128)*16+col/128]*256.f;
#pragma unroll
   for(int r=0;r<R;++r){int row=row0+r;sycl::vec<uint32_t,4>gv,uv;
    gv.load(0,reinterpret_cast<const uint32_t*>(w13+expert*(2*I*H)+row*H+col));
    uv.load(0,reinterpret_cast<const uint32_t*>(w13+expert*(2*I*H)+(row+I)*H+col));
    float gp[16],up[16];
#pragma unroll
    for(int word=0;word<4;++word){auto gword=word540::expert(gv[word],gs),uword=word540::expert(uv[word],us);
#pragma unroll
     for(int j=0;j<4;++j){int k=word*4+j;gp[k]=gword[j]*xf[k];up[k]=uword[j]*xf[k];}}
    float g=sycl::reduce_over_group(sg,sum(gp),sycl::plus<float>()),u=sycl::reduce_over_group(sg,sum(up),sycl::plus<float>());
    if(lane==0){partial[r*8+warp]=g;partial[r*8+4+warp]=u;}
   }
   it.barrier(sycl::access::fence_space::local_space);
   if constexpr(Fuse){
    if(tid<R){int r=tid;float g=(partial[r*8]+partial[r*8+1])+(partial[r*8+2]+partial[r*8+3]);float u=(partial[r*8+4]+partial[r*8+5])+(partial[r*8+6]+partial[r*8+7]);
     uint16_t gb=rne(g),ub=rne(u);raw[route*(2*I)+row0+r]=gb;raw[route*(2*I)+I+row0+r]=ub;
     float gate=bf(gb),up=bf(ub);float silu=gate/(1.f+sycl::exp(-gate));activated[route*I+row0+r]=rne(bf(rne(silu))*up);
    }
   }else{
    if(tid==0){
#pragma unroll
     for(int r=0;r<R;++r){float g=(partial[r*8]+partial[r*8+1])+(partial[r*8+2]+partial[r*8+3]);float u=(partial[r*8+4]+partial[r*8+5])+(partial[r*8+6]+partial[r*8+7]);
      raw[route*(2*I)+row0+r]=rne(g);raw[route*(2*I)+I+row0+r]=rne(u);}
    }
   }
  });
 });
}
}
extern "C" const char* tp_expert_gu_error(){return tp_expert_gu::error.c_str();}
extern "C" int tp_expert_gu_run(void*queue,const uint16_t*x,const uint8_t*w,const float*s,const int*ids,uint16_t*raw,int intermediate){
 if(!queue||!x||!w||!s||!ids||!raw||(intermediate!=128&&intermediate!=256&&intermediate!=512))return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);if(intermediate==128)tp_expert_gu::run<128>(q,x,w,s,ids,raw);else if(intermediate==256)tp_expert_gu::run<256>(q,x,w,s,ids,raw);else tp_expert_gu::run<512>(q,x,w,s,ids,raw);return 0;}catch(const std::exception&e){tp_expert_gu::error=e.what();return 1;}}

namespace tp_expert_gu {
class LocalExpertMap;
void map_local(sycl::queue&q,const int*ids,const int*mapping,int*out){
 q.parallel_for<LocalExpertMap>(sycl::range<1>(8),[=](sycl::id<1>i){
  int e=ids[i[0]];out[i[0]]=e>=0&&e<256?mapping[e]:-1;
 });
}
}
extern "C" int tp_expert_map_run(void*queue,const int*ids,const int*mapping,int*out){
 if(!queue||!ids||!mapping||!out)return 2;
 try{tp_expert_gu::map_local(*static_cast<sycl::queue*>(queue),ids,mapping,out);return 0;}
 catch(const std::exception&e){tp_expert_gu::error=e.what();return 1;}
}

extern "C" int tp_expert_gu_silu_run(void*queue,const uint16_t*x,const uint8_t*w,const float*s,const int*ids,uint16_t*raw,uint16_t*activated,int intermediate){
 if(!queue||!x||!w||!s||!ids||!raw||!activated||intermediate!=512)return 2;
 try{auto&q=*static_cast<sycl::queue*>(queue);tp_expert_gu::run<512,true>(q,x,w,s,ids,raw,activated);return 0;}
 catch(const std::exception&e){tp_expert_gu::error=e.what();return 1;}
}

extern "C" int tp_expert_replica_gu_silu_run(void*queue,const uint16_t*x,const uint8_t*w,const float*s,const int*ids,const int*replicas,uint16_t*raw,uint16_t*activated,int*local_ids,int rank){
 if(!queue||!x||!w||!s||!ids||!replicas||!raw||!activated||!local_ids||(rank!=0&&rank!=1))return 2;
 try{tp_expert_gu::run<512,true,true>(*static_cast<sycl::queue*>(queue),x,w,s,ids,raw,activated,replicas,local_ids,rank);return 0;}
 catch(const std::exception&e){tp_expert_gu::error=e.what();return 1;}
}

namespace tp_expert_gu {
class ReplicaBalanceMapPacked;
void map_replicas(sycl::queue&q,const int*ids,const int*local_by_rank,int*out,int tokens,int rank){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<128>};
 q.submit([&](sycl::handler&h){
  h.parallel_for<ReplicaBalanceMapPacked>(sycl::nd_range<1>(uint64_t(tokens)*32,32),props,[=](sycl::nd_item<1>it){
   int token=it.get_group_linear_id(),lane=it.get_local_linear_id();auto sg=it.get_sub_group();
   int local=balanced_local(sg,ids+token*8,local_by_rank,rank,lane);
   if(lane<8)out[token*8+lane]=local;
  });
 });
}
}
extern "C" int tp_expert_replica_map_run(void*queue,const int*ids,const int*local_by_rank,int*out,int tokens,int rank){
 if(!queue||!ids||!local_by_rank||!out||tokens<=0||tokens>=(1<<28)||(rank!=0&&rank!=1))return 2;
 try{tp_expert_gu::map_replicas(*static_cast<sycl::queue*>(queue),ids,local_by_rank,out,tokens,rank);return 0;}
 catch(const std::exception&e){tp_expert_gu::error=e.what();return 1;}
}
