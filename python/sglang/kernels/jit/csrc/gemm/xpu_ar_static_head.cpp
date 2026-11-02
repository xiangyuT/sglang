// SPDX-License-Identifier: Apache-2.0
// Static BF16 LM-head layout, block256, B70.
// Encoding is lossless. FP32 reduction order may differ from native GEMV.
// Build with -fp-model=precise -ffp-contract=off; SG32/GRF256.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
SYCL_EXTERNAL __attribute__((convergent)) float __spirv_GroupNonUniformFAdd(int, int, float, unsigned int) noexcept;
namespace xpu_ar_static_head {
thread_local std::string error;
class Meta;
class Pack;
class Verify;
template <int R, bool Packed, bool Fma>
class Gemv;
inline float bf(uint16_t v) {
  return sycl::bit_cast<float>(uint32_t(v) << 16);
}
inline uint16_t to_bf(float f) {
  uint32_t v = sycl::bit_cast<uint32_t>(f);
  if ((v & 0x7fffffffU) > 0x7f800000U) return uint16_t((v >> 16) | 64);
  return uint16_t((v + 0x7fffU + ((v >> 16) & 1U)) >> 16);
}
inline float sum16(float (&p)[16]) {
#pragma unroll
  for (int s = 1; s < 16; s *= 2) {
#pragma unroll
    for (int i = 0; i < 16; i += 2 * s)
      p[i] += p[i + s];
  }
  return p[0];
}
void metadata(sycl::queue& q, const uint16_t* w, uint32_t* headers, uint32_t* counter, int blocks) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Meta>(sycl::nd_range<1>(size_t(blocks) * 32, 32), props, [=](sycl::nd_item<1> it) {
      int block = it.get_group_linear_id(), lane = it.get_local_linear_id();
      sycl::vec<uint32_t, 4> raw;
      raw.load(0, reinterpret_cast<const uint32_t*>(w + size_t(block) * 256 + lane * 8));
      uint32_t lo = 127, hi = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t v = (raw[j / 2] >> ((j % 2) * 16 + 8)) & 127;
        lo = sycl::min(lo, v);
        hi = sycl::max(hi, v);
      }
      auto sg = it.get_sub_group();
      lo = sycl::reduce_over_group(sg, lo, sycl::minimum<uint32_t>());
      hi = sycl::reduce_over_group(sg, hi, sycl::maximum<uint32_t>());
      if (lane == 0) {
        if (hi - lo <= 7)
          headers[block] = lo;
        else {
          sycl::atomic_ref<
              uint32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              count(*counter);
          headers[block] = (count.fetch_add(1) << 8) | 128;
        }
      }
    });
  });
}
void pack(
    sycl::queue& q,
    const uint16_t* w,
    uint8_t* low,
    uint8_t* codes,
    const uint32_t* headers,
    uint8_t* fallback,
    int blocks) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Pack>(sycl::nd_range<1>(size_t(blocks) * 32, 32), props, [=](sycl::nd_item<1> it) {
      int block = it.get_group_linear_id(), lane = it.get_local_linear_id();
      size_t offset = size_t(block) * 256 + lane * 8;
      uint32_t header = headers[block], base = header & 127;
      bool bad = header & 128;
      sycl::vec<uint32_t, 4> raw;
      raw.load(0, reinterpret_cast<const uint32_t*>(w + offset));
      sycl::vec<uint32_t, 2> l(0), hi(0);
      uint32_t c = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t b = (raw[j / 2] >> ((j % 2) * 16)) & 65535;
        l[j / 4] |= (b & 255) << ((j % 4) * 8);
        hi[j / 4] |= (b >> 8) << ((j % 4) * 8);
        uint32_t code = bad ? 0 : (((b >> 8) & 127) - base) | ((b >> 15) << 3);
        c |= code << (j * 4);
      }
      l.store(0, reinterpret_cast<uint32_t*>(low + offset));
      reinterpret_cast<uint32_t*>(codes + offset / 2)[0] = c;
      if (bad) hi.store(0, reinterpret_cast<uint32_t*>(fallback + size_t(header >> 8) * 256 + lane * 8));
    });
  });
}
inline uint16_t
decode(const uint8_t* low, const uint8_t* codes, const uint32_t* headers, const uint8_t* fallback, size_t index) {
  uint32_t header = headers[index / 256];
  uint32_t high;
  if (header & 128)
    high = fallback[size_t(header >> 8) * 256 + index % 256];
  else {
    uint32_t c = (codes[index / 2] >> ((index % 2) * 4)) & 15;
    high = ((header & 127) + (c & 7)) | ((c >> 3) << 7);
  }
  return uint16_t((high << 8) | low[index]);
}
void verify(
    sycl::queue& q,
    const uint16_t* w,
    const uint8_t* low,
    const uint8_t* codes,
    const uint32_t* headers,
    const uint8_t* fallback,
    uint32_t* errors,
    int blocks) {
  namespace one = sycl::ext::oneapi::experimental;
  one::properties props{one::sub_group_size<32>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Verify>(sycl::nd_range<1>(size_t(blocks) * 32, 32), props, [=](sycl::nd_item<1> it) {
      size_t start = it.get_group_linear_id() * 256 + it.get_local_linear_id() * 8;
      uint32_t bad = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        bad += w[start + j] != decode(low, codes, headers, fallback, start + j);
      bad = sycl::reduce_over_group(it.get_sub_group(), bad, sycl::plus<uint32_t>());
      if (it.get_local_linear_id() == 0 && bad) {
        sycl::atomic_ref<
            uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            count(*errors);
        count.fetch_add(bad);
      }
    });
  });
}
template <int R, bool Packed, bool Fma>
void gemv(
    sycl::queue& q,
    const uint16_t* x,
    const uint16_t* w,
    const uint8_t* low,
    const uint8_t* codes,
    const uint32_t* headers,
    const uint8_t* fallback,
    uint16_t* y,
    int n) {
  namespace one = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  one::properties props{one::sub_group_size<32>, intel::grf_size<256>};
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> partial(R * 4, h);
    h.parallel_for<Gemv<R, Packed, Fma>>(sycl::nd_range<1>(size_t(n / R) * 128, 128), props, [=](sycl::nd_item<1> it) {
      int tid = it.get_local_linear_id(), col = tid * 16, row0 = it.get_group_linear_id() * R;
      sycl::vec<uint32_t, 8> xv;
      xv.load(0, reinterpret_cast<const uint32_t*>(x + col));
      float xf[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        xf[j] = bf(uint16_t(xv[j / 2] >> ((j % 2) * 16)));
#pragma unroll
      for (int r = 0; r < R; ++r) {
        int row = row0 + r;
        size_t offset = size_t(row) * 2048 + col;
        float wf[16], p[16];
        if constexpr (Packed) {
          uint32_t header = headers[offset / 256];
          bool bad = header & 128;
          sycl::vec<uint32_t, 4> l;
          l.load(0, reinterpret_cast<const uint32_t*>(low + offset));
          sycl::vec<uint32_t, 4> high;
          sycl::vec<uint32_t, 2> c;
          if (bad)
            high.load(0, reinterpret_cast<const uint32_t*>(fallback + size_t(header >> 8) * 256 + col % 256));
          else
            c.load(0, reinterpret_cast<const uint32_t*>(codes + offset / 2));
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            uint32_t hb;
            if (bad)
              hb = (high[j / 4] >> ((j % 4) * 8)) & 255;
            else {
              uint32_t code = (c[j / 8] >> ((j % 8) * 4)) & 15;
              hb = ((header & 127) + (code & 7)) | ((code >> 3) << 7);
            }
            wf[j] = bf(uint16_t((hb << 8) | ((l[j / 4] >> ((j % 4) * 8)) & 255)));
          }
        } else {
          sycl::vec<uint32_t, 8> v;
          v.load(0, reinterpret_cast<const uint32_t*>(w + offset));
#pragma unroll
          for (int j = 0; j < 16; ++j)
            wf[j] = bf(uint16_t(v[j / 2] >> ((j % 2) * 16)));
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
          p[j] = wf[j] * xf[j];
        if constexpr (Fma) {
#pragma unroll
          for (int j = 0; j < 16; j += 2) {
            p[j] = sycl::fma(wf[j], xf[j], p[j + 1]);
            p[j + 1] = 0.f;
          }
        }
        float sum = sycl::reduce_over_group(it.get_sub_group(), sum16(p), sycl::plus<float>());
        if (tid % 32 == 0) partial[r * 4 + tid / 32] = sum;
      }
      it.barrier(sycl::access::fence_space::local_space);
      float part = tid < R * 4 ? partial[tid] : 0.f;
#ifdef __SYCL_DEVICE_ONLY__
      float value = __spirv_GroupNonUniformFAdd(3, 3, part, 4);
#else
   float value=part;
#endif
      if (tid < R * 4 && tid % 4 == 0) y[row0 + tid / 4] = to_bf(value);
    });
  });
}
}  // namespace xpu_ar_static_head
extern "C" const char* ar_static_head_error() {
  return xpu_ar_static_head::error.c_str();
}
extern "C" int
ar_static_head_metadata(void* queue, const uint16_t* w, uint32_t* headers, uint32_t* counter, int blocks) {
  try {
    xpu_ar_static_head::metadata(*static_cast<sycl::queue*>(queue), w, headers, counter, blocks);
    return 0;
  } catch (const std::exception& e) {
    xpu_ar_static_head::error = e.what();
    return 1;
  }
}
extern "C" int ar_static_head_pack(
    void* queue,
    const uint16_t* w,
    uint8_t* low,
    uint8_t* codes,
    const uint32_t* headers,
    uint8_t* fallback,
    int blocks) {
  try {
    xpu_ar_static_head::pack(*static_cast<sycl::queue*>(queue), w, low, codes, headers, fallback, blocks);
    return 0;
  } catch (const std::exception& e) {
    xpu_ar_static_head::error = e.what();
    return 1;
  }
}
extern "C" int ar_static_head_verify(
    void* queue,
    const uint16_t* w,
    const uint8_t* low,
    const uint8_t* codes,
    const uint32_t* headers,
    const uint8_t* fallback,
    uint32_t* errors,
    int blocks) {
  try {
    xpu_ar_static_head::verify(*static_cast<sycl::queue*>(queue), w, low, codes, headers, fallback, errors, blocks);
    return 0;
  } catch (const std::exception& e) {
    xpu_ar_static_head::error = e.what();
    return 1;
  }
}
extern "C" int ar_static_head_gemv(
    void* queue,
    const uint16_t* x,
    const uint16_t* w,
    const uint8_t* low,
    const uint8_t* codes,
    const uint32_t* headers,
    const uint8_t* fallback,
    uint16_t* y,
    int n,
    int rows,
    int packed,
    int fma) {
  try {
    auto& q = *static_cast<sycl::queue*>(queue);
#define CALL(R, P, F) xpu_ar_static_head::gemv<R, P, F>(q, x, w, low, codes, headers, fallback, y, n)
#define SELECT(R)            \
  if (packed) {              \
    if (fma)                 \
      CALL(R, true, true);   \
    else                     \
      CALL(R, true, false);  \
  } else {                   \
    if (fma)                 \
      CALL(R, false, true);  \
    else                     \
      CALL(R, false, false); \
  }
    if (n < 1 || (rows != 1 && rows != 2 && rows != 4 && rows != 8) || n % rows) return 2;
    switch (rows) {
      case 1:
        SELECT(1);
        break;
      case 2:
        SELECT(2);
        break;
      case 4:
        SELECT(4);
        break;
      case 8:
        SELECT(8);
        break;
      default:
        return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    xpu_ar_static_head::error = e.what();
    return 1;
  }
}

extern "C" int ar_static_head_abi() {
  return 1;
}
