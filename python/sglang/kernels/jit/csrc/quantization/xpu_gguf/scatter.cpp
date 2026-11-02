// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
namespace sglang_gguf {
extern thread_local std::string error;
class ScatterRoutes;
}
extern "C" int xpu_gguf_scatter_routes(void* queue, const float* source,
    const int32_t* indices, float* target, int rows, int total_rows, int width) {
    if (!queue || !source || !indices || !target || rows <= 0 || total_rows <= 0 || width <= 0)
        return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        const size_t count = size_t(rows) * width;
        q.parallel_for<sglang_gguf::ScatterRoutes>(
            sycl::nd_range<1>(((count + 255) / 256) * 256, 256),
            [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_linear_id();
                if (i >= count) return;
                const int destination = indices[i / width];
                if (destination >= 0 && destination < total_rows)
                    target[size_t(destination) * width + i % width] = source[i];
            });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf::error = e.what();
        return 1;
    }
}
