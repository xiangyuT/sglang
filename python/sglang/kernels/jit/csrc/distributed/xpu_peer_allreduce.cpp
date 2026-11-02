// Experimental B70 graphable small-message all-reduce.
// Protocol adapted from the user-provided llm-scaler custom_ar patch, originally
// wyc55069407/vllm-xpu-kernels 894c4337ba2519929c9cd7952f082272895f7ee6.
// This prototype uses SCM_RIGHTS FD transfer, no ptrace/pidfd permissions.
// Cross-device fence/flag behavior must be verified on the selected hardware.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
thread_local std::string error;
constexpr size_t ctrl_bytes = 4096;
constexpr size_t payload_bytes = 4194304;
constexpr int max_elements = 2097152;
constexpr int flag_stride = 16;
constexpr int epoch_word = 256;
constexpr int error_word = 257;
constexpr int cookie_offset = 4080;
constexpr uint32_t spin_limit = 1u << 27;
using BF = sycl::ext::oneapi::bfloat16;
struct alignas(16) Pack { BF v[8]; };
struct Pointers { char* p[4]; };
struct State {
  ze_context_handle_t ctx;
  ze_device_handle_t device;
  char* own = nullptr;
  uint32_t* host_error = nullptr;
  Pointers peers{};
  std::array<int, 4> fds{-1,-1,-1,-1};
  ze_ipc_mem_handle_t ipc{};
  bool exported = false;
  uint64_t cookie = 0;
  int rank, world;
};
void zecheck(ze_result_t r, const char* name) {
  if (r != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::string(name)+": ze status "+std::to_string(uint32_t(r)));
}
template<class F> int safe(F f) {
  try { f(); return 0; }
  catch (const std::exception& e) { error = e.what(); return -1; }
}
}

extern "C" const char* peer_ar_error() { return error.c_str(); }
extern "C" int peer_ar_create(void* queue, int rank, int world, void** out) {
  return safe([&] {
    if ((world!=2 && world!=4) || rank<0 || rank>=world)
      throw std::runtime_error("only 2/4 local ranks supported");
    auto& q=*static_cast<sycl::queue*>(queue);
    auto s = new State(); s->rank=rank; s->world=world;
    s->ctx=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    s->device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,nullptr,0,0};
    void* p=nullptr;
    zecheck(zeMemAllocDevice(s->ctx,&desc,ctrl_bytes+2*world*payload_bytes,4096,s->device,&p),"allocate");
    s->own=static_cast<char*>(p);s->peers.p[rank]=s->own;
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,nullptr,0};
    void* hp=nullptr;
    zecheck(zeMemAllocHost(s->ctx,&hd,4096,4096,&hp),"allocate host status");
    s->host_error=static_cast<uint32_t*>(hp);*s->host_error=0;
    q.memset(p,0,ctrl_bytes+2*world*payload_bytes).wait();
    s->cookie=(uint64_t(getpid())<<32)^uint64_t(uintptr_t(p))^0x9e3779b97f4a7c15ull;
    q.memcpy(s->own+cookie_offset,&s->cookie,8).wait();
    *out=s;
  });
}
extern "C" int peer_ar_export(void* h, uint32_t* raw, int* fd, uint64_t* cookie) {
  return safe([&] {
    auto s=static_cast<State*>(h);
    zecheck(zeMemGetIpcHandle(s->ctx,s->own,&s->ipc),"export");s->exported=true;
    std::memcpy(raw,s->ipc.data,ZE_MAX_IPC_HANDLE_SIZE);
    // Same driver-specific fallback as the reference patch. Imported mapping
    // is verified using a per-allocation cookie before any kernel is launched.
    std::memcpy(fd,s->ipc.data,sizeof(int));*cookie=s->cookie;
  });
}
extern "C" int peer_ar_open(void* h, void* queue, int peer, int fd,
                            uint32_t* raw, int peer_pid, uint64_t cookie) {
  return safe([&] {
    auto s=static_cast<State*>(h);auto& q=*static_cast<sycl::queue*>(queue);
    if(peer<0 || peer>=s->world || peer==s->rank || s->peers.p[peer])
      throw std::runtime_error("invalid/duplicate peer");
    ze_ipc_mem_handle_t handle{};
    std::memcpy(handle.data,raw,ZE_MAX_IPC_HANDLE_SIZE);
    std::memcpy(handle.data,&fd,sizeof(fd));
    auto words=reinterpret_cast<uint32_t*>(handle.data);
    for(int k=1;k<ZE_MAX_IPC_HANDLE_SIZE/4;k++)
      if(words[k]==uint32_t(peer_pid))words[k]=uint32_t(getpid());
    void* p=nullptr;
    zecheck(zeMemOpenIpcHandle(s->ctx,s->device,handle,0,&p),"open");
    uint64_t actual=0;q.memcpy(&actual,static_cast<char*>(p)+cookie_offset,8).wait();
    if(actual!=cookie){zeMemCloseIpcHandle(s->ctx,p);throw std::runtime_error("IPC cookie mismatch");}
    s->peers.p[peer]=static_cast<char*>(p);s->fds[peer]=fd;
  });
}

// Large payloads use ordered send / rendezvous / consume kernels.
// Four ranks read peer snapshots to avoid all-to-all write contention; two
// ranks retain the faster push path. Both use exactly the same byte layout. Small
// messages retain the single-workgroup protocol. The two banks and shared epoch
// are identical, so mixed AR/AG/large/small graph sequences use one ordering.
// Kernel completion provides the global workgroup boundary before publishing.
template<bool Gather> void large_collective(State* s, sycl::queue& q,
    const void* input, void* output, int bytes) {
  if(!q.is_in_order())throw std::runtime_error("Large peer collectives require an in-order queue");
  const auto peers=s->peers;const int rank=s->rank,world=s->world;
  auto p=peers;auto host_error=s->host_error;
  const auto* x=static_cast<const Pack*>(input);
  auto* out=static_cast<Pack*>(output);const int packs=bytes/16;
  const size_t work=((size_t(packs)+255)/256)*256;
  q.parallel_for(sycl::nd_range<1>(work,256),[=](sycl::nd_item<1> it) {
    const int pack=it.get_global_id(0);if(pack>=packs)return;
    auto own=reinterpret_cast<uint32_t*>(p.p[rank]);const uint32_t e=own[epoch_word]+1;
    const size_t off=ctrl_bytes+(e&1)*world*payload_bytes+rank*payload_bytes+pack*16;
    const Pack mine=x[pack];
    if(world==4)*reinterpret_cast<Pack*>(p.p[rank]+off)=mine;
    else for(int peer=0;peer<world;peer++)if(peer!=rank)
      *reinterpret_cast<Pack*>(p.p[peer]+off)=mine;
  });
  q.parallel_for(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it) {
    if(it.get_local_id(0)!=0)return;
    auto own=reinterpret_cast<uint32_t*>(p.p[rank]);const uint32_t e=own[epoch_word]+1;
    sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::system);
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> ready(
              *reinterpret_cast<uint32_t*>(p.p[peer]+rank*flag_stride*4));
          ready.store(e,sycl::memory_order::release);
        }
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> f(own[peer*flag_stride]);
          uint32_t spins=0;
          while(int32_t(f.load(sycl::memory_order::acquire)-e)<0)
            if(++spins==spin_limit){
              own[error_word]=1;
              sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                               sycl::access::address_space::global_space> failed(*host_error);
              failed.store(1,sycl::memory_order::release);break;
            }
        }
    own[epoch_word]=e;
  });
  q.parallel_for(sycl::nd_range<1>(work,256),[=](sycl::nd_item<1> it) {
    const int pack=it.get_global_id(0);if(pack>=packs)return;
    auto own=reinterpret_cast<uint32_t*>(p.p[rank]);const uint32_t e=own[epoch_word];
    sycl::atomic_fence(sycl::memory_order::acquire,sycl::memory_scope::system);
    if constexpr(Gather) {
      for(int peer=0;peer<world;peer++)
        out[size_t(peer)*packs+pack]=peer==rank?x[pack]:*reinterpret_cast<Pack*>(p.p[world==4?peer:rank]+ctrl_bytes+(e&1)*world*payload_bytes+peer*payload_bytes+pack*16);
    } else {
      Pack values[4];
      for(int peer=0;peer<world;peer++)
        values[peer]=peer==rank?x[pack]:*reinterpret_cast<Pack*>(p.p[world==4?peer:rank]+ctrl_bytes+(e&1)*world*payload_bytes+peer*payload_bytes+pack*16);
      Pack result;
      for(int k=0;k<8;k++) {
        float sum=float(values[0].v[k]);for(int peer=1;peer<world;peer++)sum+=float(values[peer].v[k]);
        result.v[k]=own[error_word]?BF(sycl::nan(0u)):BF(sum);
      }
      out[pack]=result;
    }
  });
}

extern "C" int peer_ar_max_elements(){return max_elements;}
extern "C" int peer_ag_max_bytes(){return 2097152;}
extern "C" int peer_ar_run_n(void* h, void* queue, void* input, int n) {
  return safe([&] {
    if(n<8 || n>max_elements || n%8)throw std::runtime_error("BF16 count must be8..2097152 divisible8");
    auto s=static_cast<State*>(h);auto& q=*static_cast<sycl::queue*>(queue);
    for(int i=0;i<s->world;i++)if(!s->peers.p[i])throw std::runtime_error("missing peer");
    // TP4 decode's 20 KiB message benefits from multiple send/consume groups.
    // Keep the single-workgroup path for TP2 and smaller TP4 messages.
    if(n>65536||(s->world==4&&n>=8192)){large_collective<false>(s,q,input,input,n*2);return;}
    auto p=s->peers;int rank=s->rank,world=s->world;
    auto host_error=s->host_error;
    auto x=static_cast<Pack*>(input);
    int wg=256;
    if(world==2){const char* v=std::getenv("SGLANG_XPU_PEER_AR_WG");if(v&&*v)wg=std::atoi(v);}
    if(wg<16||wg>1024||(wg&(wg-1)))throw std::runtime_error("Peer AR WG must be power of two16..1024");
    q.parallel_for(sycl::nd_range<1>(wg,wg),[=](sycl::nd_item<1> it)
        [[sycl::reqd_sub_group_size(16)]] {
      const int lid=it.get_local_id(0);
      auto own=reinterpret_cast<uint32_t*>(p.p[rank]);
      const uint32_t e=own[epoch_word]+1;
      for(int pack=lid;pack<n/8;pack+=wg) {
        const size_t dataoff=ctrl_bytes+(e&1)*world*payload_bytes+rank*payload_bytes+pack*16;
        Pack mine=x[pack];
        for(int peer=0;peer<world;peer++)if(peer!=rank)
          *reinterpret_cast<Pack*>(p.p[peer]+dataoff)=mine;
      }
      sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::system);
      sycl::group_barrier(it.get_group());
      if(lid==0) {
        sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::system);
        // Match the receiver's acquire load. A volatile store is not an
        // atomic release publication and provides no synchronizes-with edge
        // for the payload writes performed by the other work-group lanes.
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> ready(
              *reinterpret_cast<uint32_t*>(p.p[peer]+rank*flag_stride*4));
          ready.store(e,sycl::memory_order::release);
        }
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> f(own[peer*flag_stride]);
          uint32_t spins=0;
          while(int32_t(f.load(sycl::memory_order::acquire)-e)<0)
            if(++spins==spin_limit){
              own[error_word]=1;
              sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                               sycl::access::address_space::global_space> failed(*host_error);
              failed.store(1,sycl::memory_order::release);break;
            }
        }
      }
      sycl::group_barrier(it.get_group());
      sycl::atomic_fence(sycl::memory_order::acquire,sycl::memory_scope::system);
      for(int pack=lid;pack<n/8;pack+=wg) {
        Pack values[4];
        for(int peer=0;peer<world;peer++)
          values[peer]=peer==rank?x[pack]:*reinterpret_cast<Pack*>(p.p[rank]+ctrl_bytes+(e&1)*world*payload_bytes+peer*payload_bytes+pack*16);
        Pack result;
        for(int k=0;k<8;k++) {
          float sum=float(values[0].v[k]);
          for(int peer=1;peer<world;peer++)sum+=float(values[peer].v[k]);
          result.v[k]=own[error_word]?BF(sycl::nan(0u)):BF(sum);
        }
        x[pack]=result;
      }
      sycl::group_barrier(it.get_group());
      if(lid==0)own[epoch_word]=e;
    });
  });
}
extern "C" int peer_ag_run(void* h, void* queue, const void* input, void* output, int bytes) {
  return safe([&] {
    if(!input || !output || bytes<4 || bytes>2097152 || bytes%4)throw std::runtime_error("Allgather bytecount must be4..2097152 divisible4");
    auto s=static_cast<State*>(h);auto& q=*static_cast<sycl::queue*>(queue);
    for(int i=0;i<s->world;i++)if(!s->peers.p[i])throw std::runtime_error("missing peer");
    if(bytes>131072 && bytes%16==0){large_collective<true>(s,q,input,output,bytes);return;}
    auto p=s->peers;int rank=s->rank,world=s->world;
    auto host_error=s->host_error;
    auto x=static_cast<const uint32_t*>(input);auto out=static_cast<uint32_t*>(output);
    q.parallel_for(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it)
        [[sycl::reqd_sub_group_size(16)]] {
      const int lid=it.get_local_id(0);
      auto own=reinterpret_cast<uint32_t*>(p.p[rank]);
      const uint32_t e=own[epoch_word]+1;
      for(int word=lid;word<bytes/4;word+=256) {
        const size_t dataoff=ctrl_bytes+(e&1)*world*payload_bytes+rank*payload_bytes+word*4;
        uint32_t mine=x[word];
        for(int peer=0;peer<world;peer++)if(peer!=rank)
          *reinterpret_cast<uint32_t*>(p.p[peer]+dataoff)=mine;
      }
      sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::system);
      sycl::group_barrier(it.get_group());
      if(lid==0) {
        sycl::atomic_fence(sycl::memory_order::release,sycl::memory_scope::system);
        // Match the receiver's acquire load. A volatile store is not an
        // atomic release publication and provides no synchronizes-with edge
        // for the payload writes performed by the other work-group lanes.
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> ready(
              *reinterpret_cast<uint32_t*>(p.p[peer]+rank*flag_stride*4));
          ready.store(e,sycl::memory_order::release);
        }
        for(int peer=0;peer<world;peer++)if(peer!=rank) {
          sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                           sycl::access::address_space::global_space> f(own[peer*flag_stride]);
          uint32_t spins=0;
          while(int32_t(f.load(sycl::memory_order::acquire)-e)<0)
            if(++spins==spin_limit){
              own[error_word]=1;
              sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::system,
                               sycl::access::address_space::global_space> failed(*host_error);
              failed.store(1,sycl::memory_order::release);break;
            }
        }
      }
      sycl::group_barrier(it.get_group());
      sycl::atomic_fence(sycl::memory_order::acquire,sycl::memory_scope::system);
      for(int peer=0;peer<world;peer++) {
        const auto* values=peer==rank?x:reinterpret_cast<const uint32_t*>(p.p[rank]+ctrl_bytes+(e&1)*world*payload_bytes+peer*payload_bytes);
        for(int word=lid;word<bytes/4;word+=256)out[size_t(peer)*(bytes/4)+word]=values[word];
      }
      sycl::group_barrier(it.get_group());
      if(lid==0)own[epoch_word]=e;
    });
  });
}
extern "C" int peer_ar_run(void* h,void* queue,void* input){return peer_ar_run_n(h,queue,input,2048);}
extern "C" int peer_ar_status(void* h, void* queue, uint32_t* out) {
  return safe([&] {
    auto s=static_cast<State*>(h);auto& q=*static_cast<sycl::queue*>(queue);
    q.memcpy(out,s->own+error_word*4,4).wait();
  });
}
extern "C" int peer_ar_poll(void* h) {
  auto s=static_cast<State*>(h);
  return std::atomic_ref<uint32_t>(*s->host_error).load(std::memory_order_acquire);
}
extern "C" int peer_ar_destroy(void* h, void* queue) {
  return safe([&] {
    auto s=static_cast<State*>(h);static_cast<sycl::queue*>(queue)->wait();
    for(int i=0;i<s->world;i++)if(i!=s->rank && s->peers.p[i]) {
      zecheck(zeMemCloseIpcHandle(s->ctx,s->peers.p[i]),"close");
      if(s->fds[i]>=0)close(s->fds[i]);
    }
    if(s->exported)zecheck(zeMemPutIpcHandle(s->ctx,s->ipc),"put");
    zecheck(zeMemFree(s->ctx,s->host_error),"free host status");
    zecheck(zeMemFree(s->ctx,s->own),"free");delete s;
  });
}
