// SPDX-License-Identifier: Apache-2.0
// One-request FutureMap relay, using the queued byte-copy pattern of
// xpu_dflash_small_copy.cpp. Publication remains separate from stash.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>

namespace sglang_gguf_relay {
thread_local std::string error;
template<bool Gather> class Relay;
template<bool Gather>
void run(sycl::queue& queue, const void* index,
         int64_t* tokens, void* token, uint32_t* probabilities, uint32_t* probability,
         int64_t* choices, void* choice, sycl::uint4* hidden, sycl::uint4* value,
         int rows, int width, int flags) {
    const int units = width * 2 / 16;
    queue.parallel_for<Relay<Gather>>(sycl::range<1>(units), [=](sycl::id<1> item) {
        int64_t row = (flags & 4) ? *static_cast<const int64_t*>(index)
                                 : *static_cast<const int32_t*>(index);
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) return;
        if constexpr (Gather) value[item[0]] = hidden[row * units + item[0]];
        else hidden[row * units + item[0]] = value[item[0]];
        if (item[0] == 0) {
            if constexpr (Gather) {
                *static_cast<int64_t*>(token) = tokens[row];
                *probability = probabilities[row];
                *static_cast<int64_t*>(choice) = choices[row];
            } else {
                tokens[row] = (flags & 1) ? *static_cast<const int64_t*>(token)
                                         : *static_cast<const int32_t*>(token);
                probabilities[row] = *probability;
                choices[row] = (flags & 2) ? *static_cast<const int64_t*>(choice)
                                          : *static_cast<const int32_t*>(choice);
            }
        }
    });
}
}

extern "C" const char* xpu_gguf_relay_error() { return sglang_gguf_relay::error.c_str(); }
extern "C" int xpu_gguf_relay(void* queue, const void* index, int64_t* tokens, void* token,
    uint32_t* probabilities, uint32_t* probability, int64_t* choices, void* choice,
    void* hidden, void* value, int rows, int width, int gather, int flags) {
    if (!queue || !index || !tokens || !token || !probabilities || !probability ||
        !choices || !choice || !hidden || !value || rows != 2 || width != 10240 ||
        gather < 0 || gather > 1 || flags < 0 || flags > 7) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        if (gather) sglang_gguf_relay::run<true>(q, index, tokens, token, probabilities,
            probability, choices, choice, static_cast<sycl::uint4*>(hidden),
            static_cast<sycl::uint4*>(value), rows, width, flags);
        else sglang_gguf_relay::run<false>(q, index, tokens, token, probabilities,
            probability, choices, choice, static_cast<sycl::uint4*>(hidden),
            static_cast<sycl::uint4*>(value), rows, width, flags);
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf_relay::error = e.what();
        return 1;
    }
}
