// SPDX-License-Identifier: Apache-2.0
// B70 AR K192 FP8 gate/up with pinned-host fill. Native down remains separate.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#pragma clang fp contract(off) reassociate(off)
namespace ar_expert_gate_b70_abi1 {
thread_local std::string error;
inline float from_bf(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
inline uint16_t to_bf(float x) {
  uint32_t u = sycl::bit_cast<uint32_t>(x);
  if ((u & 0x7fffffffU) > 0x7f800000U) return uint16_t((u >> 16) | 64);
  return uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
inline float dequant(uint8_t v, float scaled) {
  uint16_t mag = v & 127;
  uint16_t sign = uint16_t(v & 128) << 8;
  uint16_t bits = mag == 127 ? sign | 0x7e00 : sign | (mag << 7);
  float z = float(sycl::bit_cast<sycl::half>(bits)) * scaled;
  return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(z) & 0xffff0000U);
}
inline float sum16(float (&p)[16]) {
#pragma unroll
  for (int stride = 1; stride < 16; stride *= 2) {
#pragma unroll
    for (int i = 0; i < 16; i += 2 * stride)
      p[i] = p[i] + p[i + stride];
  }
  return p[0];
}
inline void lookup(
    const int* ids,
    const int* dst,
    const int* src,
    const int* count,
    int route,
    int& slot,
    int& expert,
    bool& missing,
    bool& owner) {
  slot = ids[route];
  expert = 0;
  missing = false;
  owner = true;
  int n = count[0];
  if (n > 0) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (j < n && dst[j] == slot) {
        expert += src[j];
        missing = true;
      }
    if (missing) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j < route && ids[j] == slot) owner = false;
    }
  }
}

class HybridGate;
void run(
    sycl::queue& q,
    const uint16_t* x,
    uint8_t* w,
    const float* s,
    const int* ids,
    uint16_t* a,
    const uint64_t* hosts,
    const int* dst,
    const int* src,
    const int* count) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<32>, intel::grf_size<256>};
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> partial(32, h);
    h.parallel_for<HybridGate>(sycl::nd_range<1>(8 * 256 * 128, 128), props, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      int tid = it.get_local_linear_id(), lane = tid % 32, warp = tid / 32;
      int group = it.get_group_linear_id();
      int route = group / 256, group_row = group % 256;
      int active_rows = count[0] > 0 ? 4 : 2;
      if (group_row >= 512 / active_rows) return;
      int row0 = group_row * active_rows;
      int slot, expert;
      bool missing, owner;
      lookup(ids, dst, src, count, route, slot, expert, missing, owner);
      const uint8_t* base =
          missing ? reinterpret_cast<const uint8_t*>(hosts[0]) + expert * 2097152LL : w + slot * 2097152LL;
      int col = tid * 16;
      sycl::vec<uint32_t, 8> xv;
      xv.load(0, reinterpret_cast<const uint32_t*>(x + col));
      float xf[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        xf[j] = from_bf(uint16_t(xv[j / 2] >> ((j % 2) * 16)));
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r >= active_rows) break;
        int row = row0 + r;
        sycl::vec<uint32_t, 4> gv, uv;
        gv.load(0, reinterpret_cast<const uint32_t*>(base + row * 2048 + col));
        uv.load(0, reinterpret_cast<const uint32_t*>(base + (row + 512) * 2048 + col));
        if (missing && owner) {
          gv.store(0, reinterpret_cast<uint32_t*>(w + slot * 2097152LL + row * 2048 + col));
          uv.store(0, reinterpret_cast<uint32_t*>(w + slot * 2097152LL + (row + 512) * 2048 + col));
        }
        float gs = s[slot * 128 + (row / 128) * 16 + col / 128] * 256.f;
        float us = s[slot * 128 + ((row + 512) / 128) * 16 + col / 128] * 256.f;
        float gp[16], up[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          gp[j] = dequant(uint8_t(gv[j / 4] >> ((j % 4) * 8)), gs) * xf[j];
          up[j] = dequant(uint8_t(uv[j / 4] >> ((j % 4) * 8)), us) * xf[j];
        }
        float g = sycl::reduce_over_group(sg, sum16(gp), sycl::plus<float>());
        float u = sycl::reduce_over_group(sg, sum16(up), sycl::plus<float>());
        if (lane == 0) {
          partial[r * 8 + warp] = g;
          partial[r * 8 + 4 + warp] = u;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      if (tid == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (r >= active_rows) break;
          float g = (partial[r * 8] + partial[r * 8 + 1]) + (partial[r * 8 + 2] + partial[r * 8 + 3]);
          float u = (partial[r * 8 + 4] + partial[r * 8 + 5]) + (partial[r * 8 + 6] + partial[r * 8 + 7]);
          g = from_bf(to_bf(g));
          u = from_bf(to_bf(u));
          float sigmoid = sycl::native::recip(1.f + sycl::native::exp2(g * -1.4426950408889634f));
          float silu = from_bf(to_bf(g * sigmoid));
          a[route * 512 + row0 + r] = to_bf(silu * u);
        }
      }
    });
  });
}

}  // namespace ar_expert_gate_b70_abi1
extern "C" int ar_expert_gate_abi() {
  return 1;
}
extern "C" const char* ar_expert_gate_error() {
  return ar_expert_gate_b70_abi1::error.c_str();
}
extern "C" int ar_expert_gate_run(
    void* queue,
    const uint16_t* x,
    uint8_t* w,
    const float* s,
    const int* ids,
    uint16_t* a,
    const uint64_t* hosts,
    const int* dst,
    const int* src,
    const int* count) {
  try {
    if (!queue || !x || !w || !s || !ids || !a || !hosts || !dst || !src || !count) return 2;
    ar_expert_gate_b70_abi1::run(*static_cast<sycl::queue*>(queue), x, w, s, ids, a, hosts, dst, src, count);
    return 0;
  } catch (const std::exception& e) {
    ar_expert_gate_b70_abi1::error = e.what();
    return 1;
  }
}
