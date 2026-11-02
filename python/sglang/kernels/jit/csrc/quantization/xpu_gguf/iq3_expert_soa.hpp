// SPDX-License-Identifier: MIT
// Cache-only IQ3_S field mapping and unchanged ggml decoder arithmetic.
#pragma once
#include "dequant.hpp"
namespace sglang_iq3_expert_soa {
inline int64_t offset(int row, int block, int byte) {
    const int64_t tile = int64_t(row / 4) * 4400;
    const int r = row % 4;
    if (byte < 2) return tile + block * 8 + r * 2 + byte;
    if (byte >= 106) return tile + 80 + block * 16 + r * 4 + byte - 106;
    if (byte >= 74) return tile + 560 + block * 128 + r * 32 + byte - 74;
    if (byte >= 66) return tile + 240 + block * 32 + r * 8 + byte - 66;
    const int qs = byte - 2;
    return tile + 1840 + block * 256 + (qs / 4 % 2) * 128 + r * 32 + qs / 8 * 4 + qs % 4;
}

inline float value(const uint8_t* p, int row, int block, int k) {
    const int ib = k % 256 / 32, il = k % 32 / 8, j = k % 8;
    const uint8_t qs = p[offset(row, block, 2 + 8 * ib + 2 * il + j / 4)];
    const uint8_t qh = p[offset(row, block, 66 + ib)];
    const int grid = qs | ((qh << (8 - 2 * il - j / 4)) & 256);
    const int value = (iq3s_grid[grid] >> (8 * (j % 4))) & 255;
    const uint8_t scale = p[offset(row, block, 106 + ib / 2)];
    const float d = float(*reinterpret_cast<const sycl::half*>(p + offset(row, block, 0)))
                    * (1 + 2 * ((scale >> (4 * (ib % 2))) & 15));
    const uint8_t signs = p[offset(row, block, 74 + 4 * ib + il)];
    return d * value * (signs & (1 << j) ? -1.f : 1.f);
}
}
