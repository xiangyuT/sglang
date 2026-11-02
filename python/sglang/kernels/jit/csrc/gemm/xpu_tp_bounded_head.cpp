// SPDX-License-Identifier: Apache-2.0
// TP2 Q4 conservative filter and original BF16 exact rows.
// Phase 0 produces a local lower threshold; native MAX exchanges it before phase 1.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int, int, float, unsigned int) noexcept;
namespace xpu_tp_bounded_head {
thread_local std::string error;
struct Args {
 const uint16_t *x,*w;
 const uint8_t *low,*high;
 const float *scale,*rho;
 float *lower,*upper,*blockmax,*threshold,*norm;
 uint32_t *control;
 int32_t *candidates;
 uint16_t *y;
 float *approx;
 const uint32_t *eligible;
 int n,capacity,exact_groups;
};
template<int Bits,bool Seed>class Prepare;template<int Bits,bool Seed>class Approx;template<int Bits,bool Seed>class Threshold;template<int Bits,bool Seed>class Collect;template<int Bits,bool Seed>class Exact;
inline float bf(uint16_t v){return sycl::bit_cast<float>(uint32_t(v)<<16);}
inline uint16_t to_bf(float f){
 uint32_t v=sycl::bit_cast<uint32_t>(f);
 if((v&0x7fffffffU)>0x7f800000U)return uint16_t((v>>16)|64);
 return uint16_t((v+0x7fffU+((v>>16)&1U))>>16);
}
inline float sum16(float (&p)[16]){
#pragma unroll
 for(int s=1;s<16;s*=2){
#pragma unroll
  for(int i=0;i<16;i+=2*s)p[i]+=p[i+s];
 }
 return p[0];
}
// Four even high bits -> bit4 of four byte lanes. Each intermediate
// field is isolated, so no carry/borrow can cross the byte lanes below.
inline uint32_t expand_even(uint32_t high){
 uint32_t v=high&0x55u;
 v=(v|(v<<12))&0x00050005u;
 v=(v|(v<<6))&0x01010101u;
 return v<<4;
}
inline sycl::vec<float,8> decode8_5(uint32_t low,uint32_t high){
 uint32_t ev=(low&0x0f0f0f0fu)|expand_even(high);
 uint32_t od=((low>>4)&0x0f0f0f0fu)|expand_even(high>>1);
 ev=((ev|0x80808080u)-0x0f0f0f0fu)^0x80808080u;
 od=((od|0x80808080u)-0x0f0f0f0fu)^0x80808080u;
 auto a=sycl::bit_cast<sycl::vec<int8_t,4>>(ev).convert<float>();
 auto b=sycl::bit_cast<sycl::vec<int8_t,4>>(od).convert<float>();
 return sycl::vec<float,8>(a[0],b[0],a[1],b[1],a[2],b[2],a[3],b[3]);
}
inline uint32_t expand2_even(uint32_t low){
 uint32_t v=low&0x3333u;v=(v|(v<<8))&0x00330033u;return (v|(v<<4))&0x03030303u;
}
template<int Bits>inline sycl::vec<float,8> decode8(uint32_t low,uint32_t high){
 if constexpr(Bits==5)return decode8_5(low,high);
 else{
  uint32_t ev,od;
  if constexpr(Bits==4){ev=low&0x0f0f0f0fu;od=(low>>4)&0x0f0f0f0fu;}
  else{ev=expand2_even(low)|(expand_even(high)>>2);od=expand2_even(low>>2)|(expand_even(high>>1)>>2);}
  constexpr uint32_t bias=Bits==4?0x07070707u:0x03030303u;
  ev=((ev|0x80808080u)-bias)^0x80808080u;od=((od|0x80808080u)-bias)^0x80808080u;
  auto a=sycl::bit_cast<sycl::vec<int8_t,4>>(ev).convert<float>();
  auto b=sycl::bit_cast<sycl::vec<int8_t,4>>(od).convert<float>();
  return sycl::vec<float,8>(a[0],b[0],a[1],b[1],a[2],b[2],a[3],b[3]);
 }
}
template<int Bits,bool Seed>void run(sycl::queue& q,Args a,int phase){
 namespace one=sycl::ext::oneapi::experimental;
 namespace intel=sycl::ext::intel::experimental;
 const float inf=std::numeric_limits<float>::infinity();
 one::properties p128{one::sub_group_size<32>,intel::grf_size<128>};
 one::properties p256{one::sub_group_size<32>,intel::grf_size<256>};
 if(phase==0){
 q.submit([&](sycl::handler& h){
  h.parallel_for<Prepare<Bits,Seed>>(sycl::nd_range<1>(128,128),p128,[=](sycl::nd_item<1> it){
   int tid=it.get_local_linear_id();float xf[16],hi=0.f,lo=inf;uint32_t bad=0;
#pragma unroll
   for(int j=0;j<16;++j){uint16_t bits=a.x[tid*16+j],mag=bits&0x7fff;
    xf[j]=bf(bits);float v=sycl::fabs(xf[j]);
    if(mag>=0x7f80||(mag>0&&mag<0x80))bad=1;
    hi=sycl::max(hi,v);if(mag)lo=sycl::min(lo,v);
   }
   auto g=it.get_group();
   hi=sycl::reduce_over_group(g,hi,sycl::maximum<float>());
   lo=sycl::reduce_over_group(g,lo,sycl::minimum<float>());
   bad=sycl::reduce_over_group(g,bad,sycl::bit_or<uint32_t>());
   bool safe=!bad&&*a.eligible;
   if(hi>0.f)safe=safe&&hi<=0x1p60f&&lo>=0x1p-60f&&lo/hi>=0x1p-60f;
   float sum=0.f;
#pragma unroll
   for(int j=0;j<16;++j){float v=safe&&hi>0.f?xf[j]/hi:0.f;sum+=v*v;}
   sum=sycl::reduce_over_group(g,sum,sycl::plus<float>());
   if(tid==0){
    float norm=hi>0.f&&safe?sycl::sqrt(sum)*hi*(1.f+0x1p-10f):0.f;
    *a.norm=sycl::nextafter(norm,inf);
    a.control[0]=safe?1:0;a.control[1]=safe?0:a.n;a.control[2]=safe?0:1;
    a.control[3]=0; // word3=seedrow+1(0 means none); uint64 final-Exact work counters words4..15 persist
   }
  });
 });
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<float,1> partial(8*4,h);
  h.parallel_for<Approx<Bits,Seed>>(sycl::nd_range<1>(size_t(a.n/8)*128,128),p128,[=](sycl::nd_item<1> it){
   int tid=it.get_local_linear_id(),col=tid*16,row0=it.get_group_linear_id()*8;
   if(!a.control[0]){
    if(tid<8){a.lower[row0+tid]=-inf;a.upper[row0+tid]=inf;if(a.approx)a.approx[row0+tid]=0.f;}
    if(tid==0)a.blockmax[it.get_group_linear_id()]=-inf;
    return;
   }
   float xf[16];
#pragma unroll
   for(int j=0;j<16;++j)xf[j]=bf(a.x[col+j]);
#pragma unroll
   for(int r=0;r<8;++r){
    int row=row0+r;uint32_t lo0,lo1;uint16_t highs=0;
    if constexpr(Bits==3){uint32_t packed=*reinterpret_cast<const uint32_t*>(a.low+size_t(row)*512+col/4);lo0=packed&65535u;lo1=packed>>16;}
    else{sycl::vec<uint32_t,2> lows;lows.load(0,reinterpret_cast<const uint32_t*>(a.low+size_t(row)*1024+col/2));lo0=lows[0];lo1=lows[1];}
    if constexpr(Bits!=4)highs=*reinterpret_cast<const uint16_t*>(a.high+size_t(row)*256+col/8);
    float scale=a.scale[size_t(row)*16+col/128],p[16];
    auto first=decode8<Bits>(lo0,uint32_t(highs)&255u);
    auto second=decode8<Bits>(lo1,uint32_t(highs)>>8);
#pragma unroll
    for(int j=0;j<8;++j){
     float w=first[j]*scale;p[j]=w*xf[j];
     float w2=second[j]*scale;p[j+8]=w2*xf[j+8];
    }
    float sum=sycl::reduce_over_group(it.get_sub_group(),sum16(p),sycl::plus<float>());
    if(tid%32==0)partial[r*4+tid/32]=sum;
   }
   it.barrier(sycl::access::fence_space::local_space);
   float part=tid<32?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
   float value=__spirv_GroupNonUniformFAdd(3,3,part,4);
#else
   float value=part;
#endif
   float lower=-inf;
   if(tid<32&&tid%4==0){
    int row=row0+tid/4;
    float radius=sycl::nextafter((*a.norm*a.rho[row])*(1.f+0x1p-20f)+0x1p-112f,inf);
    float upper=inf;
    if(sycl::isfinite(value)&&sycl::isfinite(radius)){
     lower=bf(to_bf(sycl::nextafter(value-radius,-inf)));
     upper=bf(to_bf(sycl::nextafter(value+radius,inf)));
    }
    a.lower[row]=lower;a.upper[row]=upper;if(a.approx)a.approx[row]=value;
   }
   lower=sycl::reduce_over_group(it.get_group(),lower,sycl::maximum<float>());
   if(tid==0)a.blockmax[it.get_group_linear_id()]=lower;
  });
 });
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<float,1> seed_partial(4,h);
  h.parallel_for<Threshold<Bits,Seed>>(sycl::nd_range<1>(256,256),p128,[=](sycl::nd_item<1> it){
   int tid=it.get_local_linear_id();float value=-inf;
   // Existing head938 mode6 contiguous-vector maximum; no full lower[] scan.
   float accum[16];
#pragma unroll
   for(int j=0;j<16;++j)accum[j]=-inf;
   for(int base=tid*16;base<a.n/8;base+=256*16){
    if(base+16<=a.n/8){sycl::vec<float,16> v;v.load(0,a.blockmax+base);
#pragma unroll
     for(int j=0;j<16;++j)accum[j]=sycl::max(accum[j],v[j]);
    }else{
#pragma unroll
     for(int j=0;j<16;++j)if(base+j<a.n/8)accum[j]=sycl::max(accum[j],a.blockmax[base+j]);
    }
   }
#pragma unroll
   for(int j=0;j<16;++j)value=sycl::max(value,accum[j]);
   value=sycl::reduce_over_group(it.get_group(),value,sycl::maximum<float>());
   if(value==0.f){value=-inf;
    for(int i=tid;i<a.n/8;i+=256)value=sycl::max(value,a.blockmax[i]);
    value=sycl::reduce_over_group(it.get_group(),value,sycl::maximum<float>());
   }
   if constexpr(Seed){
    if(!a.control[0]){if(tid==0)*a.threshold=value;return;}
    int block=a.n/8;
    for(int base=tid*16;base<a.n/8;base+=256*16){
     if(base+16<=a.n/8){sycl::vec<float,16> v;v.load(0,a.blockmax+base);
#pragma unroll
      for(int j=0;j<16;++j)if(v[j]==value)block=sycl::min(block,base+j);
     }else{
#pragma unroll
      for(int j=0;j<16;++j)if(base+j<a.n/8&&a.blockmax[base+j]==value)block=sycl::min(block,base+j);
     }
    }
    block=sycl::reduce_over_group(it.get_group(),block,sycl::minimum<int>());
    int row=a.n;
    if(tid<8&&block<a.n/8&&a.lower[block*8+tid]==value)row=block*8+tid;
    row=sycl::reduce_over_group(it.get_group(),row,sycl::minimum<int>());
    if(row==a.n){if(tid==0)*a.threshold=value;return;}
    // Reuse the original BF16 Exact row tree:128threads,16products/thread,
    // SG32 reduce then the original four-part clustered reduction.
    if(tid<128){
     int col=tid*16;float products[16];sycl::vec<uint32_t,8> raw;
     raw.load(0,reinterpret_cast<const uint32_t*>(a.w+size_t(row)*2048+col));
#pragma unroll
     for(int j=0;j<16;++j)products[j]=bf(uint16_t(raw[j/2]>>((j%2)*16)))*bf(a.x[col+j]);
     float sum=sycl::reduce_over_group(it.get_sub_group(),sum16(products),sycl::plus<float>());
     if(tid%32==0)seed_partial[tid/32]=sum;
    }
    it.barrier(sycl::access::fence_space::local_space);
    float part=tid<4?seed_partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
    float exact=__spirv_GroupNonUniformFAdd(3,3,part,4);
#else
    float exact=part;
#endif
    if(tid==0){*a.threshold=sycl::max(value,bf(to_bf(exact)));a.control[3]=uint32_t(row+1);reinterpret_cast<uint64_t*>(a.control+16)[0]+=1;}
   }else if(tid==0)*a.threshold=value;
  });
 });
 }
 if(phase==1){
 q.submit([&](sycl::handler& h){
  h.parallel_for<Collect<Bits,Seed>>(sycl::nd_range<1>(size_t((a.n+255)/256)*256,256),p128,[=](sycl::nd_item<1> it){
   int row=it.get_global_linear_id();if(row>=a.n)return;
   a.y[row]=0xff80;
   if(a.control[0]&&a.upper[row]>=*a.threshold){
    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> count(a.control[1]);
    uint32_t index=count.fetch_add(1);if(index<uint32_t(a.capacity))a.candidates[index]=row;
   }
  });
 });
 q.submit([&](sycl::handler& h){
  sycl::local_accessor<float,1> partial(8*4,h);
  h.parallel_for<Exact<Bits,Seed>>(sycl::nd_range<1>(size_t(a.exact_groups)*128,128),p256,[=](sycl::nd_item<1> it){
   int tid=it.get_local_linear_id(),col=tid*16;
   bool full=!a.control[0]||a.control[1]>uint32_t(a.capacity);
   int total=full?a.n:int(a.control[1]);
   if(it.get_group_linear_id()==0&&tid==0){
    a.control[2]=!a.control[0]?1:full?2:0;
    auto* work=reinterpret_cast<uint64_t*>(a.control+4);
    work[0]+=1;work[1]+=uint64_t(total);work[2]+=full?1:0;
    work[3]+=!a.control[0]?1:0;work[4]+=uint64_t(a.control[1]);
    work[5]+=(a.control[0]&&a.control[1]>uint32_t(a.capacity))?1:0;
   }
   float xf[16];
#pragma unroll
   for(int j=0;j<16;++j)xf[j]=bf(a.x[col+j]);
   for(int start=it.get_group_linear_id()*8;start<total;start+=a.exact_groups*8){
#pragma unroll
    for(int r=0;r<8;++r){
     int index=start+r,row=index<total?(full?index:a.candidates[index]):-1;
     float p[16];sycl::vec<uint32_t,8> raw;
     if(row>=0)raw.load(0,reinterpret_cast<const uint32_t*>(a.w+size_t(row)*2048+col));
#pragma unroll
     for(int j=0;j<16;++j){float w=row>=0?bf(uint16_t(raw[j/2]>>((j%2)*16))):0.f;p[j]=w*xf[j];}
     float sum=sycl::reduce_over_group(it.get_sub_group(),sum16(p),sycl::plus<float>());
     if(tid%32==0)partial[r*4+tid/32]=sum;
    }
    it.barrier(sycl::access::fence_space::local_space);
    float part=tid<32?partial[tid]:0.f;
#ifdef __SYCL_DEVICE_ONLY__
    float value=__spirv_GroupNonUniformFAdd(3,3,part,4);
#else
    float value=part;
#endif
    if(tid<32&&tid%4==0){int index=start+tid/4;if(index<total){int row=full?index:a.candidates[index];a.y[row]=to_bf(value);}}
    it.barrier(sycl::access::fence_space::local_space);
   }
  });
 });
}
}
}
extern "C" int xpu_tp_bounded_head_abi_size(){return sizeof(xpu_tp_bounded_head::Args);}
extern "C" int xpu_tp_bounded_head_control_words(){return 18;}
extern "C" const char* xpu_tp_bounded_head_error(){return xpu_tp_bounded_head::error.c_str();}
extern "C" int xpu_tp_bounded_head_run(void* queue,const xpu_tp_bounded_head::Args* args,int phase){
 if(!queue||!args||phase<0||phase>1)return 2;const auto&a=*args;
 if(reinterpret_cast<uintptr_t>(a.control)%8||!a.x||!a.w||!a.low||!a.high||!a.scale||!a.rho||!a.lower||!a.upper||!a.blockmax||!a.threshold||!a.norm||!a.control||!a.candidates||!a.y||!a.eligible||a.n<8||a.n>248320||a.n%8||a.capacity<1||a.capacity>a.n||a.exact_groups<1||a.exact_groups>4096)return 2;
 try{xpu_tp_bounded_head::run<4,true>(*static_cast<sycl::queue*>(queue),a,phase);return 0;}
 catch(const std::exception&e){xpu_tp_bounded_head::error=e.what();return 1;}
}
