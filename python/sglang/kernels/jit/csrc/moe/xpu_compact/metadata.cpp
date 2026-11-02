// Single-request DFlash M4 metadata. Consume live translator-selected ids;
// preserve the original buffer tails and token/page/v2p address semantics.
#include "dflash_window.hpp"
#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <string>
namespace dflash_metadata {
thread_local std::string error;
class Fill;
inline void write(void* ptr, int i, int64_t value, bool wide) {
  if (wide)
    static_cast<int64_t*>(ptr)[i] = value;
  else
    static_cast<int32_t*>(ptr)[i] = value;
}
inline int64_t read(const void* ptr, int64_t i, bool wide) {
  return wide ? static_cast<const int64_t*>(ptr)[i] : static_cast<const int32_t*>(ptr)[i];
}
struct Source {
  const void *ids, *rows, *v2p;
  int stride, entry, page, bits;
};
inline int64_t token(Source source, int64_t row, int pos) {
  if (source.v2p) {
    int64_t id = read(source.ids, row * source.stride + (pos / source.page) * source.page, source.bits & 1);
    int64_t virtual_page = id < 0 ? 0 : id / source.page;
    int64_t physical_page = read(source.v2p, virtual_page, source.bits & 4);
    return (physical_page < 0 ? 0 : physical_page) * source.page + pos % source.page;
  }
  int64_t id = read(source.ids, row * source.stride + pos / source.entry, source.bits & 1);
  return source.entry == 1 ? id : id * source.entry + pos % source.entry;
}
}  // namespace dflash_metadata
extern "C" int dflash_native_window() { return DFLASH_NATIVE_WINDOW; }
extern "C" int dflash_metadata_abi() {
  return 1;
}
extern "C" const char* dflash_metadata_error() {
  return dflash_metadata::error.c_str();
}
extern "C" int dflash_metadata_fill(
    void* stream,
    const void* seq,
    const void* rows,
    const void* ids,
    const void* v2p,
    void* qo,
    int32_t* kv,
    int64_t* full_ids,
    void* mask,
    int32_t* window_ptr,
    int64_t* window_ids,
    int32_t* offsets,
    int stride,
    int entry,
    int page,
    int bits,
    int seq_wide,
    int capacity) {
  try {
    auto& queue = *static_cast<sycl::queue*>(stream);
    dflash_metadata::Source source{ids, rows, v2p, stride, entry, page, bits};
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<dflash_metadata::Fill>(
          sycl::nd_range<1>((capacity + 255) / 256 * 256, 256), [=](sycl::nd_item<1> item) {
            int j = item.get_global_linear_id();
            int n = dflash_metadata::read(seq, 0, seq_wide);
            int length = sycl::min(n, 2047), start = n - length;
            int64_t row = dflash_metadata::read(rows, 0, bits & 2);
            if (j == 0) {
              dflash_metadata::write(qo, 0, 0, bits & 8);
              dflash_metadata::write(qo, 1, DFLASH_NATIVE_WINDOW, bits & 8);
              kv[1] = n;
              dflash_metadata::write(mask, 1, DFLASH_NATIVE_WINDOW * (int64_t(n) + DFLASH_NATIVE_WINDOW), bits & 16);
              window_ptr[1] = length;
              offsets[0] = start;
            }
            if (j < n) full_ids[j] = dflash_metadata::token(source, row, j);
            if (j < length) window_ids[j] = dflash_metadata::token(source, row, start + j);
          });
    });
    return 0;
  } catch (const std::exception& e) {
    dflash_metadata::error = e.what();
    return 1;
  }
}
