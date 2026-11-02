// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Adapted from Strata c3ed1d047768d94f43d7384ccf740a214f5a53dd,
// sycl/src/kernels/cuda/kv_q8.dp.cpp. See STRATA-LICENSE.
// NHD inline records: 256 signed bytes + four FP16 block scales. The
// allocator-owned owner table resolves repeated padded slots without sorting.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <exception>
#include "strata_fp16_bits.hpp"
namespace sglang_qsa_int8_store {
template<class I> class Reset;
template<class I> class Owner;
template<class I> class Store;
template<class I>
void submit(sycl::queue& q, const uint16_t* k, const uint16_t* v,
 const I* loc, int8_t* kd, int8_t* vd, int32_t* owner, uint32_t* error,
 int n, int heads, int capacity, int64_t kr, int64_t kh, int64_t vr, int64_t vh) {
 if(n>64){
 q.parallel_for<Reset<I>>(sycl::range<1>(n), [=](sycl::id<1> i) {
  const int64_t dst=loc[i]; if(dst>=0 && dst<capacity) {
   sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
    sycl::access::address_space::global_space> ref(owner[dst]); ref.store(-1);
  }
 });
 q.parallel_for<Owner<I>>(sycl::range<1>(n), [=](sycl::id<1> i) {
  const int64_t dst=loc[i]; if(dst>=0 && dst<capacity) {
   sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
    sycl::access::address_space::global_space> ref(owner[dst]); ref.fetch_max(int(i[0]));
  }
 });
 }
 q.submit([&](sycl::handler& h) {
  sycl::local_accessor<float,1> maxima(2,h);
  h.parallel_for<Store<I>>(sycl::nd_range<1>(size_t(n)*heads*4*2*64,64),
   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
    int g=it.get_group_linear_id(); const int block=g%4;g/=4;
    const int head=g%heads;g/=heads;const int row=g%n;const bool value=g/n;
    const int64_t dst=loc[row];if(dst<0||dst>=capacity)return;
    const int lane=it.get_local_linear_id();
    if(n<=64){
     // Graph-sized writes resolve duplicates in this workgroup. No owner
     // reset/atomic pass, and no host synchronization or replay-time decision.
     const bool duplicate=lane>row && lane<n && loc[lane]==dst;
     if(sycl::any_of_group(it.get_group(),duplicate))return;
    }else if(owner[dst]!=row)return;
    const uint16_t raw=(value?v:k)[int64_t(row)*(value?vr:kr)+head*(value?vh:kh)+block*64+lane];
    const float x=sycl::bit_cast<float>(uint32_t(raw)<<16);
    const auto sg=it.get_sub_group();
    float a=sycl::fabs(x);
    // Propagate invalid inputs to an explicit sticky error flag, never an
    // undefined float-to-int conversion. A NaN scale poisons the affected KV.
    const bool bad=sycl::any_of_group(sg,!sycl::isfinite(x));
    a=bad?sycl::bit_cast<float>(uint32_t(0x7f800000)):a;
    for(int d=16;d;d>>=1)a=sycl::fmax(a,sycl::permute_group_by_xor(sg,a,d));
    if((lane&31)==0)maxima[lane>>5]=a;
    it.barrier(sycl::access::fence_space::local_space);
    const float amax=sycl::fmax(maxima[0],maxima[1]);
    uint16_t bits=strata::kernels::f16_from_f32(amax/127.0f);
    float scale=strata::kernels::f32_from_f16(bits);
    const bool invalid=!sycl::isfinite(scale);
    if(invalid){bits=0x7e00;scale=0.0f;}
    int quant=scale>0.0f?sycl::vec<float,1>{x/scale}.convert<int,sycl::rounding_mode::rte>()[0]:0;
    quant=sycl::clamp(quant,-127,127);
    int8_t* output=(value?vd:kd)+(int64_t(dst)*heads+head)*264;
    output[block*64+lane]=int8_t(quant);
    if(lane==0){
     reinterpret_cast<uint16_t*>(output+256)[block]=bits;
     if(invalid){sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
      sycl::memory_scope::device,sycl::access::address_space::global_space> flag(*error);flag.fetch_or(1);}
    }
   });
 });
}
}
extern "C" int xpu_qsa_int8_store(void* queue,const uint16_t* k,const uint16_t* v,
 const void* loc,int8_t* kd,int8_t* vd,int32_t* owner,uint32_t* error,
 int n,int heads,int capacity,int index_bytes,int64_t kr,int64_t kh,int64_t vr,int64_t vh){
 if(!queue||!k||!v||!loc||!kd||!vd||!owner||!error||n<=0||n>8192||
  (heads!=1&&heads!=2)||capacity<=0||(index_bytes!=4&&index_bytes!=8)||
  kh<256||vh<256||kr<0||vr<0)return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(queue);
  if(!q.is_in_order())return 3;
#define RUN(I) sglang_qsa_int8_store::submit(q,k,v,static_cast<const I*>(loc),kd,vd,owner,error,n,heads,capacity,kr,kh,vr,vh)
  if(index_bytes==4){RUN(int32_t);}else{RUN(int64_t);}
#undef RUN
  return 0;
 }catch(const std::exception&){return 1;}
}
