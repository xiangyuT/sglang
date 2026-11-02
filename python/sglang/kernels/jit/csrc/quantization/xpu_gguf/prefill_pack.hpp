// SPDX-License-Identifier: Apache-2.0
// Bounded pinned-host staging, read-only with respect to expert ownership.
#pragma once
#include "prefill_copy.hpp"
#include <map>
#include <tuple>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cstdlib>
namespace sglang_pack {
class Pool {
 std::mutex mu;std::condition_variable wake,finished;bool stop=false;uint64_t generation=0;
 int pending=0,jobs=0;std::atomic<int> index{0};std::function<void(int)> task;
 std::vector<std::thread> threads;
 void work(){for(int i;(i=index.fetch_add(1,std::memory_order_relaxed))<jobs;)task(i);}
public:
 Pool(){for(int i=0;i<7;++i)threads.emplace_back([this]{uint64_t seen=0;for(;;){
  std::unique_lock<std::mutex> lk(mu);wake.wait(lk,[&]{return stop||seen!=generation;});if(stop)return;
  seen=generation;lk.unlock();work();lk.lock();if(--pending==0)finished.notify_one();
 }});}
 ~Pool(){{std::lock_guard<std::mutex> lk(mu);stop=true;}wake.notify_all();for(auto& t:threads)t.join();}
 void run(int n,std::function<void(int)> fn){
  {std::lock_guard<std::mutex> lk(mu);jobs=n;task=std::move(fn);index.store(0);pending=threads.size();++generation;}
  wake.notify_all();work();std::unique_lock<std::mutex> lk(mu);finished.wait(lk,[&]{return pending==0;});task={};
 }
};
inline Pool& pool(){static Pool p;return p;}
class Resolve;class Stage;
struct State {
 sycl::queue q;sycl::queue copy;sycl::context context;int G,D,group;uint64_t* addresses=nullptr;
 uint8_t* host=nullptr;uint8_t* device=nullptr;sycl::event copied[2];bool had_copy[2]={false,false};
 size_t stride;bool resident_direct=false;bool stage_uses_ring=false;
 State(sycl::queue& stream,int g,int d,int gr):q(stream),copy(sglang_gguf_copy::get(stream)),context(stream.get_context()),G(g),D(d),group(gr),stride(size_t(2)*G+D){
  addresses=sycl::malloc_host<uint64_t>(512*4,q);
  if(!addresses)throw std::bad_alloc();
 }
 ~State(){try{q.wait_and_throw();copy.wait_and_throw();}catch(...){}
  sycl::free(addresses,context);sycl::free(host,context);sycl::free(device,context);}
 void resolve(const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,int E,int experts,bool direct_fields=false){
  auto* out=addresses;const char* setting=std::getenv("SGLANG_XPU_GGUF_RESIDENT_EXPERT_AOS");resident_direct=direct_fields||(setting&&setting[0]=='1'&&setting[1]==0);
  q.parallel_for<Resolve>(sycl::range<1>(size_t(E)),[=](sycl::id<1> index){int e=index[0],id=ids[e],slot=-1;
   if(id>=0&&id<experts)slot=slots[id];bool hit=slot>=0&&slot<experts;
   for(int f=0;f<3;++f)out[e*4+f]=(id>=0&&id<experts)?(hit?resident[size_t(slot)*3+f]:mirrors[size_t(id)*3+f]):0;
   out[e*4+3]=hit?UINT64_MAX:0;
  }).wait_and_throw();
  for(int first=0;first<E;first+=group){uint64_t cold=0;for(int e=first;e<E&&e<first+group;++e){
   if(!out[e*4]||!out[e*4+1]||!out[e*4+2])throw std::runtime_error("Missing packed-prefill owner");
   if(out[e*4+3]!=UINT64_MAX)out[e*4+3]=cold++;
   else if(resident_direct&&!direct_fields&&(out[e*4+1]!=out[e*4]+G||out[e*4+2]!=out[e*4]+size_t(2)*G))throw std::runtime_error("Direct resident products require one owned AoS blob");
  }
  uint64_t next=cold;for(int e=first;e<E&&e<first+group;++e)if(out[e*4+3]==UINT64_MAX)out[e*4+3]=(1ull<<63)|next++;
  }
 }
 uint8_t* stage(int first,int n,int bank,const sycl::event& previous,bool used){
  auto* a=addresses+first*4;
  const int g=G,d=D;const size_t bytes=stride;
  const char* setting=std::getenv("SGLANG_XPU_GGUF_HOST_EXPERT_AOS");const char* mode=std::getenv("SGLANG_XPU_GGUF_PREFILL_DIRECT_MIRRORS");
  bool direct=setting&&setting[0]=='1'&&setting[1]==0&&!(mode&&mode[0]=='0'&&mode[1]==0);
  int cold=0;for(int i=0;i<n;++i)if(!(a[i*4+3]>>63)){
   ++cold;if(direct&&(a[i*4+1]!=a[i*4]+g||a[i*4+2]!=a[i*4]+size_t(2)*g))throw std::runtime_error("Direct expert copy requires one owned AoS blob");
  }
  stage_uses_ring=cold>0||!resident_direct;
  if(!stage_uses_ring)return nullptr;
  if(!device){
   host=sycl::malloc_host<uint8_t>(2*group*stride,q);device=sycl::malloc_device<uint8_t>(2*group*stride,q);
   if(!host||!device){if(host)sycl::free(host,context);if(device)sycl::free(device,context);host=device=nullptr;throw std::bad_alloc();}
  }
  auto* h=host+size_t(bank)*group*stride;auto* p=device+size_t(bank)*group*stride;
  if(!direct&&cold){
   if(had_copy[bank])copied[bank].wait_and_throw();
   pool().run(n,[=](int i){if(!(a[i*4+3]>>63)){auto* b=h+a[i*4+3]*bytes;
    std::memcpy(b,reinterpret_cast<const void*>(a[i*4]),g);
    std::memcpy(b+g,reinterpret_cast<const void*>(a[i*4+1]),g);
    std::memcpy(b+2*g,reinterpret_cast<const void*>(a[i*4+2]),d);
   }});
  }
  if(used&&cold)copy.ext_oneapi_submit_barrier({previous});
  if(cold&&direct){
   uint64_t source=0;uint8_t* dest=nullptr;size_t length=0;
   for(int i=0;i<n;++i)if(!(a[i*4+3]>>63)){
    auto src=a[i*4];auto* dst=p+a[i*4+3]*bytes;
    if(length&&src==source+length&&dst==dest+length)length+=bytes;
    else{if(length)copied[bank]=copy.memcpy(dest,reinterpret_cast<const void*>(source),length);source=src;dest=dst;length=bytes;}
   }
   if(length)copied[bank]=copy.memcpy(dest,reinterpret_cast<const void*>(source),length);
   had_copy[bank]=true;
  }else if(cold){copied[bank]=copy.memcpy(p,h,size_t(cold)*stride);had_copy[bank]=true;}
  const size_t units=bytes/16;
  if(!resident_direct)q.parallel_for<Stage>(sycl::nd_range<2>({size_t(n),(units+127)/128*128},{1,128}),[=](sycl::nd_item<2> it){
   size_t j=it.get_global_id(1);if(j>=units)return;int i=it.get_global_id(0);uint64_t tag=a[i*4+3];if(!(tag>>63))return;
   size_t b=j*16;int f=b<size_t(g)?0:b<size_t(2)*g?1:2;size_t off=b-(f==0?0:f==1?g:size_t(2)*g);
   auto* dst=p+(tag&~(1ull<<63))*bytes+b;auto* src=reinterpret_cast<const uint8_t*>(a[i*4+f])+off;
   *reinterpret_cast<sycl::uint4*>(dst)=*reinterpret_cast<const sycl::uint4*>(src);
  });
  if(cold)q.ext_oneapi_submit_barrier({copied[bank]});
  return p;
 }
};
inline State& get(sycl::queue& q,int G,int D,int group){
 using Key=std::tuple<void*,int,int,int>;thread_local std::map<Key,std::unique_ptr<State>> states;
 auto& p=states[{&q,G,D,group}];if(!p)p=std::make_unique<State>(q,G,D,group);return *p;
}
}
