// SPDX-License-Identifier: Apache-2.0
// Exact FP32 greedy exchange for GGUF target verify: 1..32 independent rows.
// The existing byte all-gather transports only score/ID packets. No IPC ABI
// changes, host scalar reads, approximation or changes to expert selection.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
class LocalMax;
class LocalPacket;
class FinishIds;
class FinishCarrier;

uint64_t key(float value, uint32_t id) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const bool nan = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
  if ((bits & 0x7fffffffu) == 0) bits = 0; // +/-0 tie at the first token.
  uint32_t ordered = nan ? 0xffffffffu :
      ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u));
  return (uint64_t(ordered) << 32) | uint64_t(0xffffffffu - id);
}

uint32_t winner(const float* packet, int rows, int row, int world) {
  uint64_t best = 0;
  for (int rank = 0; rank < world; ++rank) {
    int offset = (rank * rows + row) * 2;
    uint64_t candidate = key(packet[offset], uint32_t(packet[offset + 1]));
    best = sycl::max(best, candidate);
  }
  return 0xffffffffu - uint32_t(best);
}

template<class F> int safe(F f) {
  try { f(); return 0; }
  catch (const std::exception& e) { error = e.what(); return 1; }
}
}

extern "C" const char* xpu_gguf_greedy_error() { return error.c_str(); }

extern "C" int xpu_gguf_greedy_prepare(void* queue, const float* logits,
    uint64_t* partial, float* packet, int rows, int width, int rank, int world) {
  return safe([&] {
    if (!queue || !logits || !partial || !packet || rows < 1 || rows > 32 ||
        width < 1 || width > 248320 || (world != 2 && world != 4) || rank < 0 || rank >= world)
      throw std::runtime_error("Invalid GGUF greedy prepare geometry");
    auto& q = *static_cast<sycl::queue*>(queue);
    if (!q.has_property<sycl::property::queue::in_order>())
      throw std::runtime_error("GGUF greedy exchange requires an in-order queue");
    const int groups = (width + 1023) / 1024;
    q.parallel_for<LocalMax>(sycl::nd_range<2>({size_t(rows), size_t(groups * 256)}, {1, 256}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
      int row = it.get_group(0), g = it.get_group(1), lane = it.get_local_id(1);
      uint64_t best = 0;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        int col = g * 1024 + lane + k * 256;
        if (col < width) best = sycl::max(best, key(logits[row * width + col], uint32_t(rank * width + col)));
      }
      best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
      if (lane == 0) partial[row * groups + g] = best;
    });
    q.parallel_for<LocalPacket>(sycl::nd_range<2>({size_t(rows), 256}, {1, 256}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
      int row = it.get_group(0), lane = it.get_local_id(1);
      uint64_t best = lane < groups ? partial[row * groups + lane] : 0;
      best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<uint64_t>());
      if (lane == 0) {
        uint32_t id = 0xffffffffu - uint32_t(best);
        packet[row * 2] = logits[row * width + id - rank * width];
        packet[row * 2 + 1] = float(id);
      }
    });
  });
}

extern "C" int xpu_gguf_greedy_finish(void* queue, const float* packet,
    int64_t* winners, float* carrier, int rows, int width, int world) {
  return safe([&] {
    if (!queue || !packet || !winners || rows < 1 || rows > 32 || width < 1 ||
        width > 993280 || (world != 2 && world != 4))
      throw std::runtime_error("Invalid GGUF greedy finish geometry");
    auto& q = *static_cast<sycl::queue*>(queue);
    if (carrier) {
      q.parallel_for<FinishCarrier>(sycl::range<2>(rows, width), [=](sycl::id<2> it) {
        int row = it[0], col = it[1];
        uint32_t id = winner(packet, rows, row, world);
        carrier[row * width + col] = uint32_t(col) == id ? 1.f : sycl::bit_cast<float>(0xff800000u);
        if (col == 0) winners[row] = id;
      });
    } else {
      q.parallel_for<FinishIds>(sycl::range<1>(rows), [=](sycl::id<1> it) {
        winners[it[0]] = winner(packet, rows, it[0], world);
      });
    }
  });
}
