// Single-subgroup SYCL LRU, E256/K192/R64. All-hit uses relaxed atomic
// timestamp stores so duplicate routes remain race-free without sorting.
#include "dflash_window.hpp"
#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <string>

namespace dflash_planner {
thread_local std::string error;
class Plan;
void launch(
    sycl::queue& q,
    const int* top,
    const uint32_t* weights,
    int* step,
    int* slot_expert,
    int* expert_slot,
    int* lastuse,
    int* src,
    int* dst,
    int* count,
    int* mapped,
    uint32_t* mapped_weights) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Plan>(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), props, [=](sycl::nd_item<1> item) {
      unsigned lane = item.get_local_linear_id();
      auto group = item.get_sub_group();
      int routed0=top[lane],routed1=lane+32<DFLASH_NATIVE_WINDOW*8?top[lane+32]:-1;
      bool valid0=routed0>=0&&routed0<256,valid1=routed1>=0&&routed1<256;
      int slot0=valid0?expert_slot[routed0]:-1,slot1=valid1?expert_slot[routed1]:-1;
      int now=int32_t(uint32_t(*step)+1u);
#pragma unroll
      for(int route=0;route<2;++route){
        int slot=route==0?slot0:slot1;
        if(slot>=0)sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(lastuse[slot]).store(now);
      }
      bool all_hit=sycl::all_of_group(group,(!valid0||slot0>=0)&&(!valid1||slot1>=0));
      if (all_hit) {
        if (lane == 0) {
          *step = now;
          *count = 0;
        }
        mapped[lane]=valid0?slot0:0; if(lane+32<DFLASH_NATIVE_WINDOW*8) mapped[lane+32]=valid1?slot1:0;
        mapped_weights[lane]=valid0?weights[lane]:0; if(lane+32<DFLASH_NATIVE_WINDOW*8) mapped_weights[lane+32]=valid1?weights[lane+32]:0;
        return;
      }
      item.barrier(sycl::access::fence_space::global_space);
      int experts[6], ages[6], original[6];
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        unsigned s = lane + j * 32;
        experts[j] = slot_expert[s];
        original[j] = experts[j];
        ages[j] = lastuse[s];
      }
      int n = 0;
#pragma unroll 1
      for (unsigned r = 0; r < DFLASH_NATIVE_WINDOW*8; ++r) {
        int e=r<32?sycl::select_from_group(group,routed0,r):sycl::select_from_group(group,routed1,r-32);
        bool local_present = false;
#pragma unroll
        for (int j = 0; j < 6; ++j)
          local_present |= experts[j] == e;
        bool present = sycl::any_of_group(group, local_present);
        if (e >= 0 && e < 256 && !present) {
          int local_oldest = INT32_MAX;
#pragma unroll
          for (int j = 0; j < 6; ++j)
            if (ages[j] != now) local_oldest = sycl::min(local_oldest, ages[j]);
          int oldest = sycl::reduce_over_group(group, local_oldest, sycl::minimum<int>());
          int local_victim = 192;
#pragma unroll
          for (int j = 0; j < 6; ++j)
            if (ages[j] != now && ages[j] == oldest) local_victim = sycl::min(local_victim, int(lane + j * 32));
          int victim = sycl::reduce_over_group(group, local_victim, sycl::minimum<int>());
#pragma unroll
          for (int j = 0; j < 6; ++j)
            if (int(lane + j * 32) == victim) {
              experts[j] = e;
              ages[j] = now;
            }
          if (lane == 0) {
            src[n] = e;
            dst[n] = victim;
          }
          ++n;
        }
      }
#pragma unroll
      for (int j = 0; j < 6; ++j)
        expert_slot[original[j]] = -1;
      item.barrier(sycl::access::fence_space::global_space);
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        int s = lane + j * 32;
        expert_slot[experts[j]] = s;
        slot_expert[s] = experts[j];
        lastuse[s] = ages[j];
      }
      item.barrier(sycl::access::fence_space::global_space);
      if (lane == 0) {
        *step = now;
        *count = n;
      }
      mapped[lane]=valid0?expert_slot[routed0]:0; if(lane+32<DFLASH_NATIVE_WINDOW*8) mapped[lane+32]=valid1?expert_slot[routed1]:0;
      mapped_weights[lane]=valid0?weights[lane]:0; if(lane+32<DFLASH_NATIVE_WINDOW*8) mapped_weights[lane+32]=valid1?weights[lane+32]:0;
    });
  });
}
}  // namespace dflash_planner
extern "C" int dflash_planner_abi() {
  return 1;
}
extern "C" const char* dflash_planner_error() {
  return dflash_planner::error.c_str();
}
extern "C" int dflash_planner_decide(
    void* queue,
    const int* top,
    const void* weights,
    int* step,
    int* slots,
    int* inverse,
    int* age,
    int* src,
    int* dst,
    int* count,
    int* mapped,
    void* mapped_weights) {
  try {
    dflash_planner::launch(
        *static_cast<sycl::queue*>(queue),
        top,
        static_cast<const uint32_t*>(weights),
        step,
        slots,
        inverse,
        age,
        src,
        dst,
        count,
        mapped,
        static_cast<uint32_t*>(mapped_weights));
    return 0;
  } catch (const std::exception& e) {
    dflash_planner::error = e.what();
    return 1;
  }
}
