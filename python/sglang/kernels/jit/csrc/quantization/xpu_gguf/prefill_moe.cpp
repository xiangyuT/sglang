// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Port of Strata 46c152a1310f77cd2f0ce2c7cd213962df14fe83
// sycl/src/prefill/xmx_moe.dp.cpp: grouped FP16 XMX GEMM.
// SGLang adaptation: contiguous weight batches, 8x64 sparse-row tiles,
// checked C ABI, no global queue or library-owned USM allocations.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <string>
namespace sglang_prefill {
namespace jm=sycl::ext::oneapi::experimental::matrix;
using half=sycl::half;
thread_local std::string error;
constexpr int EMAX=32;
struct Group { int off[EMAX+1]; int tiles[EMAX+1]; int n; };
class GroupedF16;
void launch(sycl::queue& q,const half* X,const half* W,float* Y,Group gp,int total,int N,int K) {
 constexpr int BM=8,BN=64,KS=32,LD=40,WG=64;
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<half,1> a(BM*LD,h),b(BN*LD,h);
  auto props=sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<128>};
  h.parallel_for<GroupedF16>(sycl::nd_range<2>({size_t(total),size_t(N/BN)*WG},{1,WG}),props,
   [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
    const int t=it.get_group(0),lid=it.get_local_id(1),sid=lid/16;
    int e=0;while(e+1<gp.n&&t>=gp.tiles[e+1])++e;
    const int m0=(t-gp.tiles[e])*BM,rows=sycl::min(BM,gp.off[e+1]-gp.off[e]-m0);
    const int n0=it.get_group(1)*BN;
    const half* xe=X+size_t(gp.off[e]+m0)*K;
    const half* we=W+size_t(e)*N*K;
    float* ye=Y+size_t(gp.off[e]+m0)*N;
    auto sg=it.get_sub_group();
    jm::joint_matrix<sycl::sub_group,float,jm::use::accumulator,8,16> c;
    jm::joint_matrix_fill(sg,c,0.f);
    half* pa=a.template get_multi_ptr<sycl::access::decorated::no>().get();
    half* pb=b.template get_multi_ptr<sycl::access::decorated::no>().get();
    for(int k0=0;k0<K;k0+=KS){
     for(int v=lid;v<BM*KS/8;v+=WG){
      int r=v/(KS/8),k=v%(KS/8)*8;sycl::vec<half,8> z(0);
      if(r<rows)z=*reinterpret_cast<const sycl::vec<half,8>*>(xe+size_t(r)*K+k0+k);
      *reinterpret_cast<sycl::vec<half,8>*>(pa+r*LD+k)=z;
     }
     for(int v=lid;v<BN*KS/8;v+=WG){
      int r=v/(KS/8),k=v%(KS/8)*8;
      *reinterpret_cast<sycl::vec<half,8>*>(pb+r*LD+k)=*reinterpret_cast<const sycl::vec<half,8>*>(we+size_t(n0+r)*K+k0+k);
     }
     it.barrier(sycl::access::fence_space::local_space);
     for(int kk=0;kk<KS;kk+=16){
      jm::joint_matrix<sycl::sub_group,half,jm::use::a,8,16,jm::layout::row_major> av;
      jm::joint_matrix<sycl::sub_group,half,jm::use::b,16,16,jm::layout::col_major> bv;
      jm::joint_matrix_load(sg,av,a.template get_multi_ptr<sycl::access::decorated::no>()+kk,LD);
      jm::joint_matrix_load(sg,bv,b.template get_multi_ptr<sycl::access::decorated::no>()+sid*16*LD+kk,LD);
      jm::joint_matrix_mad(sg,c,av,bv,c);
     }
     it.barrier(sycl::access::fence_space::local_space);
    }
    if(rows==8)jm::joint_matrix_store(sg,c,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(ye+n0+sid*16),N,jm::layout::row_major);
    else{
     // Each subgroup owns a distinct slice after all shared loads finish.
     float* temp=reinterpret_cast<float*>(pb)+sid*8*16;
     jm::joint_matrix_store(sg,c,sycl::address_space_cast<sycl::access::address_space::local_space,sycl::access::decorated::no>(temp),16,jm::layout::row_major);
     sycl::group_barrier(sg);
     int lane=sg.get_local_linear_id();for(int r=0;r<rows;++r)ye[size_t(r)*N+n0+sid*16+lane]=temp[r*16+lane];
    }
   });
 });
}
}
extern "C" const char* xpu_gguf_prefill_error(){return sglang_prefill::error.c_str();}
extern "C" int xpu_gguf_prefill_f16(void* queue,const void* X,const void* W,float* Y,const int32_t* counts,int E,int N,int K){
 using namespace sglang_prefill;
 if(!queue||!X||!W||!Y||!counts||E<1||E>EMAX||N<64||N%64||K<32||K%32)return 2;
 Group gp{};gp.n=E;int rows=0,tiles=0;
 for(int e=0;e<E;++e){if(counts[e]<0||counts[e]>32768)return 2;gp.off[e]=rows;gp.tiles[e]=tiles;rows+=counts[e];tiles+=(counts[e]+7)/8;}
 gp.off[E]=rows;gp.tiles[E]=tiles;if(rows==0)return 0;
 try{auto& q=*static_cast<sycl::queue*>(queue);if(!q.has_property<sycl::property::queue::in_order>())return 2;
 launch(q,static_cast<const half*>(X),static_cast<const half*>(W),Y,gp,tiles,N,K);return 0;
 }catch(const std::exception& e){error=e.what();return 1;}
}
