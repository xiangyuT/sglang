// B70 TP2 plain-greedy head sideband over the existing BF16 2048-element AR.
// Rank-disjoint bytes 0..255 are exact BF16 values; the AR only adds zero.
// This file defines local codecs/reductions, with no IPC or ready-flag protocol.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace {
using BF=sycl::ext::oneapi::bfloat16;
thread_local std::string error;
template<class F> int safe(F f){try{f();return 0;}catch(const std::exception& e){error=e.what();return -1;}}
class HeadMaxPack;class HeadMaxFinish;class HeadLocalMax;class HeadLocalFinish;class HeadGreedyScatter;
uint64_t winner_key(BF value,uint32_t id) {
  uint16_t raw=sycl::bit_cast<uint16_t>(value);
  const bool nan=(raw&0x7f80u)==0x7f80u && (raw&0x007fu);
  if((raw&0x7fffu)==0)raw=0; // +0/-0 compare equal; earliest token wins.
  uint16_t ordered=nan?uint16_t(0xffff):
    ((raw&0x8000u)?uint16_t(~raw):uint16_t(raw^0x8000u));
  return (uint64_t(ordered)<<32)|uint64_t(0xffffffffu-id);
}
uint64_t decode_key(const BF* c,int rank,int bytes){
  uint64_t value=0;
#pragma unroll
  for(int k=0;k<8;k++)if(k<bytes)value|=uint64_t(uint32_t(float(c[rank*bytes+k])))<<(8*k);
  return value;
}
}
extern "C" const char* head_codec_error(){return error.c_str();}
extern "C" int head_codec_max_pack(void* queue,void* threshold,void* carrier,int rank){
 return safe([&]{
  if(rank<0 || rank>1)throw std::runtime_error("TP2 only");
  auto& q=*static_cast<sycl::queue*>(queue);auto input=static_cast<float*>(threshold);auto c=static_cast<BF*>(carrier);
  q.parallel_for<HeadMaxPack>(sycl::range<1>(2048),[=](sycl::id<1> id){
    int i=id[0];uint32_t bits=sycl::bit_cast<uint32_t>(*input);
    c[i]=BF(i>=rank*4 && i<rank*4+4?float((bits>>(8*(i-rank*4)))&255u):0.f);
  });
 });
}
extern "C" int head_codec_max_finish(void* queue,void* carrier,void* threshold){
 return safe([&]{
  auto& q=*static_cast<sycl::queue*>(queue);auto c=static_cast<BF*>(carrier);auto out=static_cast<float*>(threshold);
  q.single_task<HeadMaxFinish>([=]{
    float a=sycl::bit_cast<float>(uint32_t(decode_key(c,0,4))),b=sycl::bit_cast<float>(uint32_t(decode_key(c,1,4)));
    *out=sycl::isnan(a)?a:(sycl::isnan(b)?b:(b>a?b:a));
  });
 });
}
extern "C" int head_codec_prepare(void* queue,void* logits,void* partial,void* carrier,int n,int rank){
 return safe([&]{
  if(n!=124160 || rank<0 || rank>1)throw std::runtime_error("Requires TP2 BF16 shards with N124160");
  auto& q=*static_cast<sycl::queue*>(queue);auto values=static_cast<BF*>(logits);auto parts=static_cast<uint64_t*>(partial);auto c=static_cast<BF*>(carrier);int groups=(n+1023)/1024;
  q.parallel_for<HeadLocalMax>(sycl::nd_range<1>(groups*256,256),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
    int lid=it.get_local_id(0),g=it.get_group(0);uint64_t best=0;
#pragma unroll
    for(int k=0;k<4;k++){int i=g*1024+lid+k*256;if(i<n){uint64_t key=winner_key(values[i],uint32_t(i+rank*n));best=key>best?key:best;}}
    best=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<uint64_t>());
    if(lid==0)parts[g]=best;
  });
  q.parallel_for<HeadLocalFinish>(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
    int lid=it.get_local_id(0);uint64_t best=lid<groups?parts[lid]:0;
    best=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<uint64_t>());
#pragma unroll
    for(int k=0;k<8;k++){int i=lid+k*256;c[i]=BF(i>=rank*8 && i<rank*8+8?float((best>>(8*(i-rank*8)))&255u):0.f);}
  });
 });
}
extern "C" int head_codec_scatter(void* queue,void* carrier,void* logits,int n){
 return safe([&]{
  if(n!=248320)throw std::runtime_error("Requires full vocab248320");
  auto& q=*static_cast<sycl::queue*>(queue);auto c=static_cast<BF*>(carrier);auto out=static_cast<BF*>(logits);
  q.parallel_for<HeadGreedyScatter>(sycl::range<1>(n),[=](sycl::id<1> i){
    uint64_t a=decode_key(c,0,8),b=decode_key(c,1,8),best=a>b?a:b;uint32_t token=0xffffffffu-uint32_t(best);uint16_t score=uint16_t(best>>32);
    uint16_t raw=score==0xffffu?uint16_t(0x7fc0u):((score&0x8000u)?uint16_t(score^0x8000u):uint16_t(~score));
    out[i]=uint32_t(i[0])==token?sycl::bit_cast<BF>(raw):sycl::bit_cast<BF>(uint16_t(0xff80u));
  });
 });
}
