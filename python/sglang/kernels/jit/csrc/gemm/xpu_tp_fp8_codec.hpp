// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace word540 {
// Input fields occupy bits0..7 and16..23. Each subfield stays below256,
// so the add/multiply cannot carry into the neighboring16-bit field.
inline uint32_t half_pair(uint32_t raw) {
 uint32_t mag=raw&0x007f007fu;
 uint32_t bits=(mag<<7)|((raw&0x00800080u)<<8);
 uint32_t nan=((mag+0x00010001u)&0x00800080u)>>7;
 return bits^(nan*0x00004180u);
}
inline sycl::vec<float,4> div256(uint32_t raw) {
 auto even=sycl::bit_cast<sycl::vec<sycl::half,2>>(half_pair(raw));
 auto odd=sycl::bit_cast<sycl::vec<sycl::half,2>>(half_pair(raw>>8));
 return sycl::vec<float,4>(float(even[0]),float(odd[0]),float(even[1]),float(odd[1]));
}
inline sycl::vec<float,4> expert(uint32_t raw,float scale) {
 auto value=div256(raw)*scale;
 auto bits=sycl::bit_cast<sycl::vec<uint32_t,4>>(value)&uint32_t(0xffff0000u);
 return sycl::bit_cast<sycl::vec<float,4>>(bits);
}
}
