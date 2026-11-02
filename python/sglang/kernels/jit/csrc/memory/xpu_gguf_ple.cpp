// SPDX-License-Identifier: Apache-2.0
// IQ4_NL codebook/layout: ggml (MIT), see ../quantization/xpu_gguf/GGML-LICENSE.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>

namespace sglang_gguf_ple {
thread_local std::string error;
class Gather;
}
extern "C" const char* xpu_gguf_ple_error() { return sglang_gguf_ple::error.c_str(); }
extern "C" int xpu_gguf_ple_host_pointer(void* queue, const void* table) {
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        return sycl::get_pointer_type(table, q.get_context()) == sycl::usm::alloc::host;
    } catch (...) { return 0; }
}
extern "C" int xpu_gguf_ple_gather(void* queue, const void* table, const void* indices,
                                  void* output, int64_t rows, int64_t chunk_rows, int count, int wide) {
    if (!queue || !table || !indices || !output || rows < 1 || count < 1 ||
        count > 16384 || chunk_rows < 1 || (wide != 0 && wide != 1)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        q.parallel_for<sglang_gguf_ple::Gather>(sycl::range<1>(size_t(count) * 160),
            [=](sycl::id<1> item) {
                constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
                const int i = item[0] / 160, col = item[0] % 160;
                const int64_t row = wide ? static_cast<const int64_t*>(indices)[i]
                                         : static_cast<const int32_t*>(indices)[i];
                float value = 0;
                if (row >= 0 && row < rows) {
                    const auto address = static_cast<const uint64_t*>(table)[row / chunk_rows];
                    const auto* block = reinterpret_cast<const uint8_t*>(address) + row % chunk_rows * 90 + col / 32 * 18;
                    const float d = float(*reinterpret_cast<const sycl::half*>(block));
                    const int code = (block[2 + col % 16] >> (col % 32 >= 16 ? 4 : 0)) & 15;
                    value = d * float(values[code]);
                }
                static_cast<sycl::ext::oneapi::bfloat16*>(output)[item[0]] = value;
            });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf_ple::error = e.what();
        return 1;
    }
}
