// SPDX-License-Identifier: Apache-2.0
// Lossless GGUF expert replacement with a borrowed, owned Torch pinned ring.
// Each ring slot is reusable only after its last in-order DMA has completed.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace sglang_expert_copy {
thread_local std::string error;
struct Ring {
    sycl::queue queue;
    uint8_t* storage;
    uint64_t stride;
    uint64_t next = 0;
    std::vector<sycl::event> events;
    std::vector<bool> pending;
    std::mutex mutex;
    Ring(sycl::queue q, uint8_t* p, uint64_t bytes, int count)
        : queue(q), storage(p), stride(bytes), events(count), pending(count, false) {}
    void wait() {
        for (size_t i = 0; i < events.size(); ++i)
            if (pending[i]) events[i].wait_and_throw();
    }
};
}

extern "C" const char* xpu_gguf_copy_error() {
    return sglang_expert_copy::error.c_str();
}
extern "C" void* xpu_gguf_copy_create(void* queue, uint8_t* storage, uint64_t stride, int slots) {
    try {
        if (!queue || !storage || stride < 1 || slots < 1 || slots > 256) return nullptr;
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) {
            sglang_expert_copy::error = "Expert staging requires an in-order Torch queue";
            return nullptr;
        }
        return new sglang_expert_copy::Ring(q, storage, stride, slots);
    } catch (const std::exception& e) {
        sglang_expert_copy::error = e.what();
        return nullptr;
    }
}
extern "C" int xpu_gguf_copy_expert(void* handle, uint8_t* gate, uint8_t* up, uint8_t* down,
    const uint8_t* source_gate, const uint8_t* source_up, const uint8_t* source_down,
    uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    if (!handle || !gate || !up || !down || !source_gate || !source_up || !source_down ||
        !gate_bytes || !up_bytes || !down_bytes) return 2;
    try {
        auto& r = *static_cast<sglang_expert_copy::Ring*>(handle);
        if (gate_bytes > r.stride || up_bytes > r.stride - gate_bytes ||
            down_bytes > r.stride - gate_bytes - up_bytes) return 2;
        std::lock_guard<std::mutex> lock(r.mutex);
        const size_t slot = r.next++ % r.events.size();
        if (r.pending[slot]) r.events[slot].wait_and_throw();
        uint8_t* staging = r.storage + slot * r.stride;
        std::memcpy(staging, source_gate, gate_bytes);
        std::memcpy(staging + gate_bytes, source_up, up_bytes);
        std::memcpy(staging + gate_bytes + up_bytes, source_down, down_bytes);
        // Save every successful submission so even an exception on the second
        // or third submission leaves a valid lifetime fence for borrowed RAM.
        r.events[slot] = r.queue.memcpy(gate, staging, gate_bytes);
        r.pending[slot] = true;
        r.events[slot] = r.queue.memcpy(up, staging + gate_bytes, up_bytes);
        r.events[slot] = r.queue.memcpy(down, staging + gate_bytes + up_bytes, down_bytes);
        return 0;
    } catch (const std::exception& e) {
        sglang_expert_copy::error = e.what();
        return 1;
    }
}
extern "C" int xpu_gguf_copy_destroy(void* handle) {
    if (!handle) return 0;
    try {
        auto* ring = static_cast<sglang_expert_copy::Ring*>(handle);
        ring->wait();
        delete ring;
        return 0;
    } catch (const std::exception& e) {
        // The caller keeps the ring and its host tensor alive on failure.
        sglang_expert_copy::error = e.what();
        return 1;
    }
}

// Stable logical-expert grouping, matching group_iq2s_routes byte for byte.
// All inputs are bounded CPU metadata. Only the final int32 image crosses H2D.
extern "C" int xpu_gguf_copy_group(void* handle, int32_t* output,
    const int32_t* ids, const int32_t* experts, const int32_t* slots,
    int rows, int topk, int groups, int capacity) {
    if (!handle || !output || !ids || !experts || !slots || rows < 1 || rows > 4 ||
        topk < 1 || topk > 10 || groups < 1 || groups > rows * topk ||
        capacity < groups || capacity > 512) return 2;
    try {
        std::array<int32_t, 512> lookup;
        std::array<bool, 512> occupied{};
        lookup.fill(-1);
        for (int i = 0; i < groups; ++i) {
            if (experts[i] < 0 || experts[i] >= 512 || slots[i] < 0 ||
                slots[i] >= capacity || lookup[experts[i]] != -1 || occupied[slots[i]]) return 2;
            lookup[experts[i]] = slots[i];
            occupied[slots[i]] = true;
        }
        const int routes = rows * topk;
        std::array<int, 40> order;
        for (int i = 0; i < routes; ++i) {
            if (ids[i] < 0 || ids[i] >= 512 || lookup[ids[i]] < 0) return 2;
            order[i] = i;
        }
        std::sort(order.begin(), order.begin() + routes, [&](int a, int b) {
            return ids[a] == ids[b] ? a < b : ids[a] < ids[b];
        });
        auto& r = *static_cast<sglang_expert_copy::Ring*>(handle);
        const uint64_t bytes = uint64_t(3 * routes + 2 * groups + 1) * sizeof(int32_t);
        if (bytes > r.stride) return 2;
        std::lock_guard<std::mutex> lock(r.mutex);
        const size_t slot = r.next++ % r.events.size();
        if (r.pending[slot]) r.events[slot].wait_and_throw();
        auto* host = reinterpret_cast<int32_t*>(r.storage + slot * r.stride);
        auto* group_slots = host + routes;
        auto* starts = group_slots + groups;
        auto* token_rows = starts + groups + 1;
        auto* destinations = token_rows + routes;
        for (int i = 0; i < routes; ++i) host[i] = lookup[ids[i]];
        int group = -1, previous = -1;
        for (int i = 0; i < routes; ++i) {
            const int original = order[i], expert = ids[original];
            if (expert != previous) {
                if (++group >= groups) return 2;
                group_slots[group] = lookup[expert];
                starts[group] = i;
                previous = expert;
            }
            token_rows[i] = original / topk;
            destinations[i] = original;
        }
        if (group + 1 != groups) return 2;
        starts[groups] = routes;
        r.events[slot] = r.queue.memcpy(output, host, bytes);
        r.pending[slot] = true;
        return 0;
    } catch (const std::exception& e) {
        sglang_expert_copy::error = e.what();
        return 1;
    }
}
