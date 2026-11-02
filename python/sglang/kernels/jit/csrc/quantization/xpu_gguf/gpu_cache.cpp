// SPDX-License-Identifier: MIT
// GPU expert routing and byte-preserving LRU mirror exchange. Source lineage:
// Copyright (c) 2026 Niko1221 and the Strata contributors; ggml authors.
// Mirrors retain the same encoded layout as GPU slots. See STRATA-LICENSE
// and GGML-LICENSE. Logical IDs and original quantization bytes are preserved.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <type_traits>

namespace sglang_gpu_cache1323 {
thread_local std::string error;
template<bool Wide> class Group;
template<bool Wide, int GroupSize> class WideGroup;
template<bool Wide, int GroupSize = 64>
void group(sycl::queue& q,const void* ids,int32_t* meta,int T,int topk,int experts) {
    const int routes=T*topk;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int,1> original(GroupSize,h),sorted(GroupSize,h),invalid(1,h);
        using Kernel = std::conditional_t<GroupSize == 64, Group<Wide>, WideGroup<Wide,GroupSize>>;
        h.parallel_for<Kernel>(sycl::nd_range<1>(GroupSize,GroupSize),[=](sycl::nd_item<1> it) {
            const int lane=it.get_local_linear_id();
            int id=experts+lane;
            if(lane<routes) {
                const int64_t v=Wide?static_cast<const int64_t*>(ids)[lane]:static_cast<const int32_t*>(ids)[lane];
                if(v>=0&&v<experts)id=int(v);
                meta[routes+lane]=lane;
                meta[5*routes+1+lane]=0;
                meta[2*routes+lane]=routes;
            }
            if(lane==0) {
                meta[3*routes]=routes;
                meta[6*routes+1]=0;
                meta[6*routes+2]=0;
            }
            original[lane]=id;
            it.barrier(sycl::access::fence_space::global_and_local);
            if(lane==0) {
                int bad=0;
                for(int j=0;j<routes;++j)bad|=original[j]>=experts;
                invalid[0]=bad;meta[6*routes+2]=bad;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if(invalid[0])return;
            int rank=0;
            if(lane<routes) {
                for(int j=0;j<routes;++j)
                    rank+=original[j]<id||(original[j]==id&&j<lane);
                sorted[rank]=id;
                meta[3*routes+1+rank]=lane/topk;
                meta[4*routes+1+rank]=lane;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if(lane<routes) {
                int g=0;
                for(int j=1;j<=rank;++j)g+=sorted[j]!=sorted[j-1];
                meta[lane]=g;
                if(rank==0||sorted[rank]!=sorted[rank-1]) {
                    meta[2*routes+g]=rank;
                    meta[5*routes+1+g]=id;
                }
            }
            if(lane==0) {
                int groups=1;
                for(int j=1;j<routes;++j)groups+=sorted[j]!=sorted[j-1];
                meta[6*routes+1]=groups;
            }
        });
    });
}
}
namespace sglang_gpu_cache1323 {
inline void reject(int32_t* meta,int routes,int error) {
    for(int r=0;r<routes;++r) {
        meta[r]=-1;meta[routes+r]=-1;
        meta[2*routes+r]=r?routes:0;
        meta[3*routes+1+r]=0;
        meta[4*routes+1+r]=r;
    }
    meta[3*routes]=routes;
    meta[6*routes+1]=1;
    meta[6*routes+2]=error;
}
template<bool Overflow> class Plan;
template<bool Overflow = false>
void plan(sycl::queue& q,uint64_t* mirror,int32_t* expert_slot,int32_t* slot_expert,
    uint64_t* last_used,uint64_t* clocks,const int32_t* capacity,
    const uint64_t* slot_addresses,int32_t* meta,uint64_t* swaps,int routes,int experts,int max_slots) {
    q.single_task<Plan<Overflow>>([=] {
        if constexpr(Overflow) if(meta[6*routes+2] || meta[6*routes+1]<=capacity[0])return;
        clocks[1]=0;
        if(meta[6*routes+2]){reject(meta,routes,meta[6*routes+2]);return;}
        const int count=meta[6*routes+1],cap=capacity[0];
        if(cap<1||cap>max_slots||(!Overflow&&count>cap)||(Overflow&&(experts!=257||cap>256))){reject(meta,routes,2);return;}
        uint32_t selected[16]={};
        for(int g=0;g<count;++g) {
            const int e=meta[5*routes+1+g],slot=expert_slot[e];
            selected[e/32]|=uint32_t(1)<<(e%32);
            if(slot>=cap||slot< -1){reject(meta,routes,3);return;}
            if(slot<0&&(!mirror[e*3]||!mirror[e*3+1]||!mirror[e*3+2])) {
                reject(meta,routes,4);return;
            }
        }
        uint64_t clock=clocks[0];
        // Match the existing sorted-hits-first, then sorted-misses policy.
        for(int g=0;g<count;++g) {
            const int e=meta[5*routes+1+g],slot=expert_slot[e];
            if constexpr(Overflow) if(e==256){meta[routes+g]=0;continue;}
            if(slot>=0) {
                last_used[e]=++clock;
                meta[routes+g]=slot;
            }
        }
        int cold=0;
        for(int g=0;g<count;++g) {
            const int e=meta[5*routes+1+g];
            if(expert_slot[e]>=0 || (Overflow&&e==256))continue;
            uint64_t oldest=~uint64_t(0);
            int victim=-1,outgoing=-1;
            for(int s=0;s<cap;++s) {
                const int v=slot_expert[s];
                if(v<0||v>=experts)continue;
                if((selected[v/32]>>(v%32))&1u)continue;
                if(last_used[v]<oldest){oldest=last_used[v];victim=s;outgoing=v;}
            }
            if(victim<0) {
                if constexpr(Overflow) {
                    meta[routes+g]=256+e;
                    for(int field=0;field<3;++field)
                        const_cast<uint64_t*>(slot_addresses)[int64_t(256+e)*3+field]=mirror[int64_t(e)*3+field];
                    last_used[e]=++clock;
                    continue;
                }
                reject(meta,routes,5);return;
            }
            for(int field=0;field<3;++field) {
                const uint64_t host=mirror[e*3+field];
                if(!slot_addresses[victim*3+field]){reject(meta,routes,6);return;}
                swaps[cold*4+field]=host;
                mirror[outgoing*3+field]=host;
                mirror[e*3+field]=0;
            }
            swaps[cold*4+3]=victim;
            expert_slot[outgoing]=-1;
            expert_slot[e]=victim;
            slot_expert[victim]=e;
            last_used[e]=++clock;
            meta[routes+g]=victim;
            ++cold;
        }
        for(int r=0;r<routes;++r)meta[r]=meta[routes+meta[r]];
        clocks[0]=clock;clocks[1]=cold;
    });
}
class Exchange;
template<int WG> class ParallelPlan;
template<int WG,int MaxGroups,bool Overflow> class WideParallelPlan;
template<int WG, int MaxGroups = 40, bool Overflow = false>
void plan_parallel(sycl::queue& q,uint64_t* mirror,int32_t* expert_slot,int32_t* slot_expert,
    uint64_t* last_used,uint64_t* clocks,const int32_t* capacity,
    const uint64_t* slot_addresses,int32_t* meta,uint64_t* swaps,int routes,int experts,int max_slots) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int,1> selected(512,h),eligible(512,h),outgoing(512,h);
        sycl::local_accessor<uint64_t,1> age(512,h),clock(1,h);
        sycl::local_accessor<int,1> group_slot(MaxGroups,h),group_error(MaxGroups,h),miss(MaxGroups,h),state(5,h);
        using Kernel = std::conditional_t<MaxGroups == 40, ParallelPlan<WG>, WideParallelPlan<WG,MaxGroups,Overflow>>;
        h.parallel_for<Kernel>(sycl::nd_range<1>(WG,WG),[=](sycl::nd_item<1> it) {
            const int lane=it.get_local_linear_id();
            if(lane==0) {
                state[0]=capacity[0];state[1]=meta[6*routes+1];state[2]=meta[6*routes+2];
                state[3]=0;state[4]=0;clock[0]=clocks[0];clocks[1]=0;
                if(!state[2]&&(state[0]<1||state[0]>max_slots||(!Overflow&&state[1]>state[0])))state[2]=2;
            }
            for(int e=lane;e<512;e+=WG)selected[e]=0;
            it.barrier(sycl::access::fence_space::global_and_local);
            if(state[2]){if(lane==0)reject(meta,routes,state[2]);return;}
            const int count=state[1],cap=state[0];
            if constexpr(Overflow) if(count>cap)return;
            for(int g=lane;g<count;g+=WG) {
                const int e=meta[5*routes+1+g],slot=expert_slot[e];
                group_slot[g]=slot;selected[e]=1;
                group_error[g]=(slot>=cap||slot< -1)?3:
                    slot<0&&(!mirror[e*3]||!mirror[e*3+1]||!mirror[e*3+2])?4:0;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if(lane==0) {
                for(int g=0;g<count;++g) {
                    if(group_error[g]){state[2]=group_error[g];break;}
                    if(group_slot[g]<0)miss[state[3]++]=g;
                }
                state[4]=count-state[3];
            }
            it.barrier(sycl::access::fence_space::local_space);
            if(state[2]){if(lane==0)reject(meta,routes,state[2]);return;}
            // Retain sorted-hits-first stamps, independently of lane order.
            for(int g=lane;g<count;g+=WG)if(group_slot[g]>=0) {
                int rank=0;for(int j=0;j<g;++j)rank+=group_slot[j]>=0;
                const int e=meta[5*routes+1+g];
                last_used[e]=clock[0]+uint64_t(rank)+1;meta[routes+g]=group_slot[g];
            }
            for(int s=lane;state[3]&&s<cap;s+=WG) {
                const int e=slot_expert[s];outgoing[s]=e;
                const bool valid=e>=0&&e<experts&&!selected[e];
                eligible[s]=valid;age[s]=valid?last_used[e]:~uint64_t(0);
            }
            it.barrier(sycl::access::fence_space::global_and_local);
            for(int cold=0;cold<state[3];++cold) {
                uint64_t oldest=~uint64_t(0);int victim=2147483647;
                for(int s=lane;s<cap;s+=WG)if(eligible[s]&&
                    (age[s]<oldest||(age[s]==oldest&&s<victim))) {
                    oldest=age[s];victim=s;
                }
                const uint64_t min_age=sycl::reduce_over_group(it.get_group(),oldest,sycl::minimum<uint64_t>());
                if(min_age==~uint64_t(0)){if(lane==0)reject(meta,routes,5);return;}
                victim=sycl::reduce_over_group(it.get_group(),oldest==min_age?victim:2147483647,sycl::minimum<int>());
                if(victim==2147483647){if(lane==0)reject(meta,routes,5);return;}
                if(lane==0) {
                    const int g=miss[cold],e=meta[5*routes+1+g],v=outgoing[victim];
                    for(int field=0;field<3;++field) {
                        if(!slot_addresses[victim*3+field]){state[2]=6;reject(meta,routes,6);break;}
                        const uint64_t host=mirror[e*3+field];
                        swaps[cold*4+field]=host;mirror[v*3+field]=host;mirror[e*3+field]=0;
                    }
                    if(!state[2]) {
                        swaps[cold*4+3]=victim;expert_slot[v]=-1;expert_slot[e]=victim;
                        slot_expert[victim]=e;last_used[e]=clock[0]+uint64_t(state[4]+cold)+1;
                        meta[routes+g]=victim;eligible[victim]=0;
                    }
                }
                it.barrier(sycl::access::fence_space::global_and_local);
                if(state[2])return;
            }
            for(int r=lane;r<routes;r+=WG)meta[r]=meta[routes+meta[r]];
            if(lane==0){clocks[0]=clock[0]+uint64_t(count);clocks[1]=state[3];}
        });
    });
}
void exchange(sycl::queue& q,const uint64_t* slot_addresses,const uint64_t* clocks,
    const int32_t* meta,const uint64_t* swaps,int routes,int gu) {
    const int bytes=2*gu+460800,units=bytes/16;
    // A bounded persistent grid avoids thousands of empty groups on all-hit
    // steps. Each lane moves one aligned vector in both PCIe directions.
    q.parallel_for<Exchange>(sycl::nd_range<1>(8192,128),[=](sycl::nd_item<1> it) {
        if(meta[6*routes+2])return;
        const int total=int(clocks[1])*units;
        for(int index=it.get_global_linear_id();index<total;index+=8192) {
            const int cold=index/units,byte=(index%units)*16;
            const int field=byte<gu?0:byte<2*gu?1:2;
            const int offset=byte-(field==0?0:field==1?gu:2*gu);
            const int slot=int(swaps[cold*4+3]);
            auto* host=reinterpret_cast<sycl::uint4*>(swaps[cold*4+field]+offset);
            auto* device=reinterpret_cast<sycl::uint4*>(slot_addresses[slot*3+field]+offset);
            const auto incoming=*host,outgoing=*device;
            *device=incoming;*host=outgoing;
        }
    });
}
}
extern "C" int xpu_gguf_gpu_cache_prepare(void* queue,const void* ids,uint64_t* mirror,
    int32_t* expert_slot,int32_t* slot_expert,uint64_t* last_used,uint64_t* clocks,
    const int32_t* capacity,const uint64_t* slot_addresses,int32_t* meta,uint64_t* swaps,
    int type,int T,int topk,int experts,int max_slots,int wide) {
    const int gu=type==22?524800:type==16?422400:type==29?358400:type==42?460800:0;
    if(!queue||!ids||!mirror||!expert_slot||!slot_expert||!last_used||!clocks||!capacity||
        !slot_addresses||!meta||!swaps||!gu||T<1||T>4||topk<1||topk>10||experts<1||experts>512||
        max_slots<1||max_slots>experts||(wide!=0&&wide!=1))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        if(wide)sglang_gpu_cache1323::group<true>(q,ids,meta,T,topk,experts);
        else sglang_gpu_cache1323::group<false>(q,ids,meta,T,topk,experts);
        sglang_gpu_cache1323::plan(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
        sglang_gpu_cache1323::exchange(q,slot_addresses,clocks,meta,swaps,T*topk,gu);
        return 0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return 1;}
}

extern "C" const char* xpu_gguf_gpu_cache_error(){return sglang_gpu_cache1323::error.c_str();}
extern "C" int xpu_gguf_gpu_cache_prepare_parallel(void* queue,const void* ids,uint64_t* mirror,
    int32_t* expert_slot,int32_t* slot_expert,uint64_t* last_used,uint64_t* clocks,
    const int32_t* capacity,const uint64_t* slot_addresses,int32_t* meta,uint64_t* swaps,
    int type,int T,int topk,int experts,int max_slots,int wide,int workgroup) {
    const int gu=type==22?524800:type==16?422400:type==29?358400:type==42?460800:0;
    if(!queue||!ids||!mirror||!expert_slot||!slot_expert||!last_used||!clocks||!capacity||
        !slot_addresses||!meta||!swaps||!gu||T<1||T>4||topk<1||topk>10||experts<1||experts>512||
        max_slots<1||max_slots>experts||(wide!=0&&wide!=1)||
        (workgroup!=64&&workgroup!=128&&workgroup!=256))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        if(wide)sglang_gpu_cache1323::group<true>(q,ids,meta,T,topk,experts);
        else sglang_gpu_cache1323::group<false>(q,ids,meta,T,topk,experts);
        if(workgroup==64)sglang_gpu_cache1323::plan_parallel<64>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
        else if(workgroup==128)sglang_gpu_cache1323::plan_parallel<128>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
        else sglang_gpu_cache1323::plan_parallel<256>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
        sglang_gpu_cache1323::exchange(q,slot_addresses,clocks,meta,swaps,T*topk,gu);
        return 0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return 1;}
}
extern "C" int xpu_gguf_gpu_cache_pointer_kind(void* queue,const void* p) {
    if(!queue||!p)return 0;
    try {
        const auto kind=sycl::get_pointer_type(p,static_cast<sycl::queue*>(queue)->get_context());
        return kind==sycl::usm::alloc::host?1:kind==sycl::usm::alloc::shared?2:kind==sycl::usm::alloc::device?3:0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return -1;}
}

namespace sglang_gpu_cache1323 {
// Map an original encoded byte to the qualified PVR layout, without decoding
// or requantizing it. This lets FP32 prefill use the original row kernels.
inline int pvr_byte(int index,bool down) {
    const int block=down?18:82;
    const int row=index/(10*block),column=(index/block)%10,b=index%block;
    const int lane=row%16,tile=row/16;
    const int base=(tile*10+column)*(down?288:1312);
    if(down) {
        if(b<2)return base+256+lane*2+b;
        const int p=(b-2)/8,k=(b-2)%8;
        return base+p*128+(k<4?0:64)+lane*4+k%4;
    }
    if(b<2)return base+1280+lane*2+b;
    if(b<34)return base+((b-2)/4)*160+lane*4+(b-2)%4;
    if(b<66)return base+((b-34)/4)*160+64+lane*4+(b-34)%4;
    if(b<74)return base+(b-66)*160+128+lane*2;
    return base+(b-74)*160+128+lane*2+1;
}
class StageOriginal;
inline int lowbit_original_byte(int index,int kind) {
    if(kind==16) {
        const int tile=index/5280,k=(index/528)%10,b=index%528;
        int lane,raw;
        if(b<512) {
            const int part=b%256;
            lane=part/32;raw=2+(part%32/4)*8+part%4+(b>=256?4:0);
        }else{lane=(b-512)/2;raw=(b-512)%2;}
        return ((tile*8+lane)*10+k)*66+raw;
    }
    const int tile=index/4480,k=(index/448)%10,b=index%448;
    int lane,raw;
    if(b<256){lane=b/32;raw=b%32;}
    else if(b<384){lane=(b-256)/16;raw=32+(b-256)%16;}
    else{lane=(b-384)/8;raw=48+(b-384)%8;}
    return ((tile*8+lane)*10+k)*56+raw;
}
void stage_original(sycl::queue& q,const int32_t* ids,const int32_t* expert_slots,
    const uint64_t* slot_addresses,const uint64_t* mirrors,
    uint8_t* gate,uint8_t* up,uint8_t* down,int count,int experts,int gu,int packed) {
    const int bytes=2*gu+460800,units=bytes/16;
    const size_t total=size_t(count)*units;
    q.parallel_for<StageOriginal>(sycl::nd_range<1>((total+127)/128*128,128),[=](sycl::nd_item<1> it) {
        const size_t j=it.get_global_linear_id();
        if(j>=total)return;
        const int e=j/units,byte=(j%units)*16,expert=ids[e];
        const int field=byte<gu?0:byte<2*gu?1:2;
        const int size=field==2?460800:gu;
        const int offset=byte-(field==0?0:field==1?gu:2*gu);
        auto* out=(field==0?gate:field==1?up:down)+int64_t(e)*size+offset;
        uint64_t address=0;
        if(expert>=0&&expert<experts) {
            const int slot=expert_slots[expert];
            address=slot>=0&&slot<experts?slot_addresses[int64_t(slot)*3+field]:mirrors[int64_t(expert)*3+field];
        }
        if(!address){*reinterpret_cast<sycl::uint4*>(out)=sycl::uint4(0);return;}
        const auto* src=reinterpret_cast<const uint8_t*>(address);
        if(packed==16||packed==29) {
            const auto word=*reinterpret_cast<const sycl::uint4*>(src+offset);
            if(field==2)*reinterpret_cast<sycl::uint4*>(out)=word;
            else {
                const auto octets=word.as<sycl::uchar16>();
                auto* base=out-offset;
                for(int b=0;b<16;++b)base[lowbit_original_byte(offset+b,packed)]=octets[b];
            }
        } else if(!packed)*reinterpret_cast<sycl::uint4*>(out)=*reinterpret_cast<const sycl::uint4*>(src+offset);
        else for(int b=0;b<16;++b)out[b]=src[pvr_byte(offset+b,field==2)];
    });
}
}
static int stage_from_cache(void* queue,const int32_t* ids,
    const int32_t* expert_slots,const uint64_t* slot_addresses,const uint64_t* mirrors,
    void* gate,void* up,void* down,int type,int count,int experts,bool restore,int lowbit=0) {
    const int gu=type==22?524800:type==16?422400:type==29?358400:type==42?460800:0;
    if(!queue||!ids||!expert_slots||!slot_addresses||!mirrors||!gate||!up||!down||
        !gu||count<1||count>experts||experts>512)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(!q.has_property<sycl::property::queue::in_order>())return 2;
        sglang_gpu_cache1323::stage_original(q,ids,expert_slots,slot_addresses,mirrors,
            static_cast<uint8_t*>(gate),static_cast<uint8_t*>(up),static_cast<uint8_t*>(down),
            count,experts,gu,restore&&type==22?1:lowbit);
        return 0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return 1;}
}
extern "C" int xpu_gguf_gpu_cache_stage(void* queue,const int32_t* ids,
    const int32_t* expert_slots,const uint64_t* slot_addresses,const uint64_t* mirrors,
    void* gate,void* up,void* down,int type,int count,int experts) {
    return stage_from_cache(queue,ids,expert_slots,slot_addresses,mirrors,gate,up,down,type,count,experts,true);
}
extern "C" int xpu_gguf_gpu_cache_stage_encoded(void* queue,const int32_t* ids,
    const int32_t* expert_slots,const uint64_t* slot_addresses,const uint64_t* mirrors,
    void* gate,void* up,void* down,int type,int count,int experts) {
    return stage_from_cache(queue,ids,expert_slots,slot_addresses,mirrors,gate,up,down,type,count,experts,false);
}
extern "C" int xpu_gguf_gpu_cache_stage_lowbit(void* queue,const int32_t* ids,
    const int32_t* expert_slots,const uint64_t* slot_addresses,const uint64_t* mirrors,
    void* gate,void* up,void* down,int type,int count,int experts) {
    if(type!=16&&type!=29)return 2;
    return stage_from_cache(queue,ids,expert_slots,slot_addresses,mirrors,gate,up,down,type,count,experts,false,type);
}

// Original encoded IQ3_S/IQ4_XS gate/up and IQ4_NL/Q8_0 down fields.
// The logical planner is shared with IQ2; only transfer geometry differs.
namespace sglang_gpu_cache1323 {
class MixedExchange;
void exchange_mixed(sycl::queue& q,const uint64_t* slot_addresses,const uint64_t* clocks,
    const int32_t* meta,const uint64_t* swaps,int routes,int gu,int down_bytes) {
    const int bytes=2*gu+down_bytes,units=bytes/16;
    // A bounded persistent grid avoids thousands of empty groups on all-hit
    // steps. Each lane moves one aligned vector in both PCIe directions.
    q.parallel_for<MixedExchange>(sycl::nd_range<1>(8192,128),[=](sycl::nd_item<1> it) {
        if(meta[6*routes+2])return;
        const int total=int(clocks[1])*units;
        for(int index=it.get_global_linear_id();index<total;index+=8192) {
            const int cold=index/units,byte=(index%units)*16;
            const int field=byte<gu?0:byte<2*gu?1:2;
            const int offset=byte-(field==0?0:field==1?gu:2*gu);
            const int slot=int(swaps[cold*4+3]);
            auto* host=reinterpret_cast<sycl::uint4*>(swaps[cold*4+field]+offset);
            auto* device=reinterpret_cast<sycl::uint4*>(slot_addresses[slot*3+field]+offset);
            const auto incoming=*host,outgoing=*device;
            *device=incoming;*host=outgoing;
        }
    });
}
inline bool mixed_sizes(int kind,int down_kind,int& gu,int& down) {
    gu=kind==21?704000:kind==23?870400:0;
    down=down_kind==20?921600:down_kind==8?1740800:down_kind==42?460800:0;
    return gu&&down;
}
class StageMixed;
void stage_mixed(sycl::queue& q,const int32_t* ids,const int32_t* expert_slots,
    const uint64_t* slot_addresses,const uint64_t* mirrors,
    uint8_t* gate,uint8_t* up,uint8_t* down,int count,int experts,int gu,int down_bytes) {
    const int bytes=2*gu+down_bytes,units=bytes/16;
    const size_t total=size_t(count)*units;
    q.parallel_for<StageMixed>(sycl::nd_range<1>((total+127)/128*128,128),[=](sycl::nd_item<1> it) {
        const size_t j=it.get_global_linear_id();if(j>=total)return;
        const int row=j/units,byte=(j%units)*16,expert=ids[row];
        const int field=byte<gu?0:byte<2*gu?1:2;
        const int size=field==2?down_bytes:gu;
        const int offset=byte-(field==0?0:field==1?gu:2*gu);
        auto* out=(field==0?gate:field==1?up:down)+int64_t(row)*size+offset;
        uint64_t address=0;
        if(expert>=0&&expert<experts) {
            const int slot=expert_slots[expert];
            address=slot>=0&&slot<experts?slot_addresses[int64_t(slot)*3+field]:mirrors[int64_t(expert)*3+field];
        }
        *reinterpret_cast<sycl::uint4*>(out)=address?*reinterpret_cast<const sycl::uint4*>(address+offset):sycl::uint4(0);
    });
}
}
extern "C" int xpu_gguf_gpu_cache_prepare_mixed(void* queue,const void* ids,uint64_t* mirror,
    int32_t* expert_slot,int32_t* slot_expert,uint64_t* last_used,uint64_t* clocks,
    const int32_t* capacity,const uint64_t* slot_addresses,int32_t* meta,uint64_t* swaps,
    int type,int down_type,int T,int topk,int experts,int max_slots,int wide,int workgroup) {
    int gu=0,down=0;
    if(!sglang_gpu_cache1323::mixed_sizes(type,down_type,gu,down)||!queue||!ids||!mirror||
       !expert_slot||!slot_expert||!last_used||!clocks||!capacity||!slot_addresses||!meta||!swaps||
       T<1||T>32||topk<1||topk>10||experts<1||experts>512||max_slots<1||max_slots>experts||
       (wide!=0&&wide!=1)||(workgroup!=0&&workgroup!=64&&workgroup!=128&&workgroup!=256))return 2;
    if(T>4 && type!=21 && type!=23)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);if(!q.is_in_order())return 2;
        if(T>16) {
            if(experts!=257 || !workgroup)return 2;
            if(wide)sglang_gpu_cache1323::group<true,512>(q,ids,meta,T,topk,experts);
            else sglang_gpu_cache1323::group<false,512>(q,ids,meta,T,topk,experts);
        }
        else if(T>4) {
            if(wide)sglang_gpu_cache1323::group<true,256>(q,ids,meta,T,topk,experts);
            else sglang_gpu_cache1323::group<false,256>(q,ids,meta,T,topk,experts);
        }
        else if(wide)sglang_gpu_cache1323::group<true>(q,ids,meta,T,topk,experts);
        else sglang_gpu_cache1323::group<false>(q,ids,meta,T,topk,experts);
#define PLAN(WG) do { \
    if(T>16)sglang_gpu_cache1323::plan_parallel<WG,320,true>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots); \
    else if(T>4)sglang_gpu_cache1323::plan_parallel<WG,160>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots); \
    else sglang_gpu_cache1323::plan_parallel<WG>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots); \
} while(false)
        if(workgroup==64)PLAN(64);
        else if(workgroup==128)PLAN(128);
        else if(workgroup==256)PLAN(256);
        else sglang_gpu_cache1323::plan(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
#undef PLAN
        if(T>16)sglang_gpu_cache1323::plan<true>(q,mirror,expert_slot,slot_expert,last_used,clocks,capacity,slot_addresses,meta,swaps,T*topk,experts,max_slots);
        sglang_gpu_cache1323::exchange_mixed(q,slot_addresses,clocks,meta,swaps,T*topk,gu,down);
        return 0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return 1;}
}
extern "C" int xpu_gguf_gpu_cache_stage_mixed(void* queue,const int32_t* ids,
    const int32_t* expert_slots,const uint64_t* slot_addresses,const uint64_t* mirrors,
    void* gate,void* up,void* down,int type,int down_type,int count,int experts) {
    int gu=0,down_bytes=0;
    if(!sglang_gpu_cache1323::mixed_sizes(type,down_type,gu,down_bytes)||!queue||!ids||!expert_slots||
       !slot_addresses||!mirrors||!gate||!up||!down||count<1||count>experts||experts>512)return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);if(!q.is_in_order())return 2;
        sglang_gpu_cache1323::stage_mixed(q,ids,expert_slots,slot_addresses,mirrors,
            static_cast<uint8_t*>(gate),static_cast<uint8_t*>(up),static_cast<uint8_t*>(down),count,experts,gu,down_bytes);
        return 0;
    }catch(const std::exception& e){sglang_gpu_cache1323::error=e.what();return 1;}
}
