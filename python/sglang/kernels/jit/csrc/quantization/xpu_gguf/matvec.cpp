// SPDX-License-Identifier: Apache-2.0
// Fused FP32 GEMV using the separately attributed Strata/ggml decoders.
#include "dequant.hpp"
#include <string>
namespace sglang_gguf {
extern thread_local std::string error;
class Matvec;
}
extern "C" int xpu_gguf_matvec(void* queue, const void* packed, const float* x,
                               float* y, int type, int k, int n, int tokens,
                               int row_bytes) {
    if (!queue || !packed || !x || !y || k <= 0 || n <= 0 || tokens <= 0 || row_bytes <= 0) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        const int groups = (n + 3) / 4;
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> decoded(4 * 256, h);
            h.parallel_for<sglang_gguf::Matvec>(
                sycl::nd_range<1>(size_t(groups) * tokens * 128, 128),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const int token = it.get_group_linear_id() / groups;
                    const int row = (it.get_group_linear_id() % groups) * 4 + it.get_local_linear_id() / 32;
                    if (row >= n) return;
                    const int lane = it.get_local_linear_id() % 32;
                    auto sg = it.get_sub_group();
                    float* tile = decoded.get_multi_ptr<sycl::access::decorated::no>().get()
                                  + (it.get_local_linear_id() / 32) * 256;
                    const auto* w = static_cast<const uint8_t*>(packed) + size_t(row) * row_bytes;
                    float sum = 0;
                    for (int base = 0; base < k; base += 256) {
                        const int count = sycl::min(256, k - base);
                        if (lane * 8 < count)
                            sglang_gguf::dq_dispatch<float>(type, w, base / 256, tile, lane);
                        sycl::group_barrier(sg);
                        for (int j = lane; j < count; j += 32)
                            sum = sycl::fma(tile[j], x[size_t(token) * k + base + j], sum);
                        sycl::group_barrier(sg);
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    if (lane == 0) y[size_t(token) * n + row] = sum;
                });
        });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf::error = e.what(); return 1;
    }
}
