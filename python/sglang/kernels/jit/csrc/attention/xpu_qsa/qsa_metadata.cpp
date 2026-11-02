// SPDX-License-Identifier: Apache-2.0
// QSA replay metadata, preserving the existing Torch XPU fallback's row layout.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>

namespace sglang_qsa_metadata {
inline int64_t read_integer(const void* p, int64_t i, int bytes) {
    return bytes == 4 ? static_cast<const int32_t*>(p)[i]
                      : static_cast<const int64_t*>(p)[i];
}
class Refresh;
}

extern "C" int xpu_qsa_graph_metadata_causal(
    void* queue, const void* seq, const void* req, const void* ext,
    const void* table, int32_t* lengths, int32_t* requests, int32_t* prefixes,
    int32_t* compressed, int32_t* writes, int32_t* positions, int64_t* states,
    int32_t* rings, int32_t* pages,
    int mode, int bs, int rows, int padding, int extend, int seq_offset,
    int seq_bytes, int req_bytes, int ext_bytes, int table_bytes,
    int table_rows, int row_stride, int max_pages, int ratio, int full_page, int ring_size) {
    if (!queue || !seq || !req || !table || !lengths || !requests || !prefixes ||
        !compressed || !writes || !positions || !states || !rings || !pages ||
        mode < 0 || mode > 2 || bs < 1 || bs > 8 || rows < 1 || rows > 64 ||
        padding < 0 || padding > bs || (mode == 1 && extend < 1) ||
        (mode == 2 && !ext) || ratio != 4 || full_page < ratio ||
        full_page % ratio || ring_size < ratio || ring_size % ratio ||
        row_stride < full_page || max_pages < 1 ||
        table_rows < 1 || (seq_bytes != 4 && seq_bytes != 8) ||
        (req_bytes != 4 && req_bytes != 8) || (ext_bytes != 4 && ext_bytes != 8) ||
        (table_bytes != 4 && table_bytes != 8)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
        if (!q.has_property<sycl::property::queue::in_order>()) return 2;
        q.parallel_for<sglang_qsa_metadata::Refresh>(
            sycl::nd_range<2>({size_t(rows), size_t((max_pages + 127) / 128 * 128)}, {1, 128}),
            [=](sycl::nd_item<2> item) {
                using sglang_qsa_metadata::read_integer;
                const int row = int(item.get_global_id(0));
                const int column = int(item.get_global_id(1));
                int request_row = -1, local_row = 0, count = 0;
                if (mode == 0) request_row = row < bs ? row : -1;
                else if (mode == 1) {
                    const int r = row / extend;
                    if (r < bs - padding) { request_row = r; local_row = row % extend; count = extend; }
                } else {
                    int start = 0;
                    for (int r = 0; r < bs - padding; ++r) {
                        const int size = int(read_integer(ext, r, ext_bytes));
                        if (row >= start && row < start + size) {
                            request_row = r; local_row = row - start; count = size;
                        }
                        start += size;
                    }
                }
                // The existing XPU fallback assigns unused speculative rows
                // to the first request, with length 1 and prefix 0. Retain it.
                int64_t request = read_integer(req, request_row < 0 ? 0 : request_row, req_bytes);
                int length = 1, prefix = 0;
                if (request_row >= 0) {
                    const int base = int(read_integer(seq, request_row, seq_bytes));
                    if (mode == 0) { length = base + seq_offset; prefix = sycl::max(length - 1, 0); }
                    else if (mode == 1) { length = base + local_row + 1; prefix = base; }
                    else { prefix = sycl::max(base - count, 0); length = sycl::min(prefix + 1 + local_row, base); }
                }
                int current = sycl::max(length - 1, 0);
                // Producers own valid request/sequence bounds. Keep invalid
                // metadata memory-safe, as the other native QSA adapters do.
                if (request < 0 || request >= table_rows || current >= row_stride) {
                    request = 0; length = 1; prefix = 0; current = 0;
                }
                const int64_t offset = request * row_stride;
                if (column == 0) {
                    lengths[row] = length; requests[row] = int32_t(request); prefixes[row] = prefix;
                    compressed[row] = length / ratio; positions[row] = current;
                    const int64_t last = read_integer(table, offset + current, table_bytes);
                    writes[row] = length > 0 && length % ratio == 0 ? int32_t(last / ratio) : 0;
                    states[row] = request * ring_size + current % ring_size;
                    for (int k = 0; k < ratio; ++k) {
                        const int member = sycl::max(current - (ratio - 1 - k), 0);
                        rings[row * ratio + k] = int32_t(request * ring_size + member % ring_size);
                    }
                }
                if (column < max_pages && column < row_stride / full_page) {
                    const int64_t loc = read_integer(table, offset + int64_t(column) * full_page, table_bytes);
                    pages[int64_t(row) * max_pages + column] = int32_t(sycl::max(loc / full_page, int64_t(0)));
                }
            });
        return 0;
    } catch (const std::exception&) { return 1; }
}

// Retain the original ABI for existing libraries/callers and provenance runs.
extern "C" int xpu_qsa_graph_metadata(
    void* queue, const void* seq, const void* req, const void* ext,
    const void* table, int32_t* lengths, int32_t* requests, int32_t* prefixes,
    int32_t* compressed, int32_t* writes, int32_t* positions, int64_t* states,
    int32_t* rings, int32_t* pages,
    int mode, int bs, int rows, int padding, int extend, int seq_offset,
    int seq_bytes, int req_bytes, int ext_bytes, int table_bytes,
    int table_rows, int row_stride, int max_pages, int ratio, int full_page) {
    return xpu_qsa_graph_metadata_causal(queue, seq, req, ext, table, lengths,
        requests, prefixes, compressed, writes, positions, states, rings, pages,
        mode, bs, rows, padding, extend, seq_offset, seq_bytes, req_bytes,
        ext_bytes, table_bytes, table_rows, row_stride, max_pages, ratio,
        full_page, ratio);
}
