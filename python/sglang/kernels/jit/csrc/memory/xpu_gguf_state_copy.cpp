// SPDX-License-Identifier: Apache-2.0
// Accepted-prefix side-state copy. Reuse the queued, byte-preserving SYCL
// copy pattern from xpu_dflash_small_copy.cpp, without a DFlash dependency.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>

namespace sglang_gguf_state_copy {
thread_local std::string error;
class Copy;
class CopyRows;
}

extern "C" const char* xpu_gguf_state_copy_error() {
    return sglang_gguf_state_copy::error.c_str();
}

extern "C" int xpu_gguf_state_copy(
    void* queue, void* destination, const void* source,
    const void* destination_index, const void* step_index,
    int layers, int destination_slots, int source_steps, int row_bytes,
    int wide_destination, int wide_step,
    int64_t destination_layer_stride, int64_t destination_slot_stride,
    int64_t source_layer_stride, int64_t source_step_stride) {
    if (!queue || !destination || !source || !destination_index || !step_index ||
        layers < 1 || layers > 48 || destination_slots < 1 || source_steps < 1 ||
        source_steps > 4 || row_bytes < 16 || row_bytes % 16 ||
        row_bytes > 16 * 1024 * 1024 ||
        (wide_destination != 0 && wide_destination != 1) ||
        (wide_step != 0 && wide_step != 1) ||
        destination_layer_stride < 0 || destination_slot_stride < row_bytes ||
        source_layer_stride < 0 || source_step_stride < row_bytes) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        const int units = row_bytes / 16;
        q.parallel_for<sglang_gguf_state_copy::Copy>(
            sycl::range<1>(size_t(layers) * units), [=](sycl::id<1> item) {
                const int64_t slot = wide_destination
                    ? *static_cast<const int64_t*>(destination_index)
                    : *static_cast<const int32_t*>(destination_index);
                const int64_t step = wide_step
                    ? *static_cast<const int64_t*>(step_index)
                    : *static_cast<const int32_t*>(step_index);
                if (slot < 0 || slot >= destination_slots || step < 0 || step >= source_steps) return;
                const int64_t layer = int64_t(item[0] / units);
                const int64_t byte = int64_t(item[0] % units) * 16;
                const auto* src = static_cast<const uint8_t*>(source)
                    + layer * source_layer_stride + step * source_step_stride + byte;
                auto* dst = static_cast<uint8_t*>(destination)
                    + layer * destination_layer_stride + slot * destination_slot_stride + byte;
                *reinterpret_cast<sycl::uint4*>(dst) = *reinterpret_cast<const sycl::uint4*>(src);
            });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf_state_copy::error = e.what();
        return 1;
    }
}

// Implicit source rows follow the live request order, exactly as arange(B)
// in the generic masked scatter. Keep the single-row ABI above available.
extern "C" int xpu_gguf_state_copy_rows(
    void* queue, void* destination, const void* source,
    const void* destination_index, const void* step_index,
    int layers, int destination_slots, int source_slots, int source_steps,
    int requests, int row_bytes, int wide_destination, int wide_step,
    int64_t destination_layer_stride, int64_t destination_slot_stride,
    int64_t source_layer_stride, int64_t source_slot_stride,
    int64_t source_step_stride) {
    if (!queue || !destination || !source || !destination_index || !step_index ||
        layers < 1 || layers > 48 || destination_slots < 1 || requests < 1 ||
        requests > 8 || source_slots < requests || source_steps < 1 ||
        source_steps > 4 || row_bytes < 16 || row_bytes % 16 ||
        row_bytes > 16 * 1024 * 1024 ||
        (wide_destination != 0 && wide_destination != 1) ||
        (wide_step != 0 && wide_step != 1) ||
        destination_layer_stride < 0 || destination_slot_stride < row_bytes ||
        source_layer_stride < 0 || source_slot_stride < row_bytes ||
        source_step_stride < row_bytes) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        const int units = row_bytes / 16;
        q.parallel_for<sglang_gguf_state_copy::CopyRows>(
            sycl::range<1>(size_t(layers) * requests * units), [=](sycl::id<1> item) {
                const int request = (item[0] / units) % requests;
                const int64_t layer = item[0] / (size_t(units) * requests);
                const int64_t byte = int64_t(item[0] % units) * 16;
                const auto destination_at = [=](int i) -> int64_t {
                    return wide_destination ? static_cast<const int64_t*>(destination_index)[i]
                                            : static_cast<const int32_t*>(destination_index)[i];
                };
                const auto step_at = [=](int i) -> int64_t {
                    return wide_step ? static_cast<const int64_t*>(step_index)[i]
                                     : static_cast<const int32_t*>(step_index)[i];
                };
                const int64_t slot = destination_at(request), step = step_at(request);
                if (slot < 0 || slot >= destination_slots || step < 0 || step >= source_steps) return;
                // Main states have distinct destinations. If callers alias a
                // checkpoint, choose the last valid row deterministically.
                for (int later = request + 1; later < requests; ++later) {
                    const int64_t next_step = step_at(later);
                    if (destination_at(later) == slot && next_step >= 0 && next_step < source_steps) return;
                }
                const auto* src = static_cast<const uint8_t*>(source)
                    + layer * source_layer_stride + request * source_slot_stride
                    + step * source_step_stride + byte;
                auto* dst = static_cast<uint8_t*>(destination)
                    + layer * destination_layer_stride + slot * destination_slot_stride + byte;
                *reinterpret_cast<sycl::uint4*>(dst) = *reinterpret_cast<const sycl::uint4*>(src);
            });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf_state_copy::error = e.what();
        return 1;
    }
}
