/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// FP8 E4M3 weight, BF16 activation grouped GEMM for Xe2 (BMG). Activation is
// applied externally between GEMM1 and GEMM2.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/moe/xe20/common/block_2d_copy_d.hpp"
#include "kernels/moe/xe20/common/xe_gemm.hpp"
#include "paged_mainloop.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace moe_w8a16 {
using namespace cute;

template <
    typename TileShape,
    typename SubgroupLayout,
    typename TensorA,
    typename TensorBPacked,
    typename TensorD,
    typename TiledMMA,
    bool WeightScalePerExpert = false,
    bool WeightScaleBlocked = false>
class PagedCompactW8A16Kernel {
 public:
  using ElementA = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyBPacked = decltype(make_block_2d_copy_B(TiledMMA{}, TensorBPacked{}));
  using TiledCopyD = decltype(moe_xe20::make_moe_block_2d_copy_D<void>(TiledMMA{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = moe_w8a16::PagedW8A16MainloopPolicy<Stages>;
  using CollectiveMainloop = PagedW8A16Mainloop<
      MainloopDispatchPolicy,
      TiledCopyA,
      TiledCopyBPacked,
      TiledCopyD,
      TensorA,
      TensorBPacked,
      TensorD,
      TiledMMA,
      WeightScalePerExpert,
      WeightScaleBlocked>;

  struct Params {
    const uint8_t* Activations;    // [M_total, K] bf16 bytes
    const uint8_t* PackedWeights;  // [num_experts, N, K] fp8 e4m3 raw bytes
    const void* WeightScales;      // Per-expert scalar, block scales, or uint8 MXFP8 scales
    const float* Bias;
    ElementD* Outputs;
    const int32_t* M_per_group;
    const int32_t* ExpertIds;
    const int32_t* RowOffsets;
    const uint8_t* HostWeights;
    const int32_t* HostExpertIds;
    const int32_t N;
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    int32_t ld_b;
    int32_t scale_mode;
    bool static_scheduler = false;
    bool tile_major = false;
  };

  auto make_A_tensor(uint8_t* ptr_A, int M, int K) {
    auto* bf16_ptr = reinterpret_cast<cutlass::bfloat16_t*>(ptr_A);
    return make_tensor(make_gmem_ptr(bf16_ptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
  }

  auto make_B_tensors(uint8_t* ptr_B, int N, int K, int ld_b) {
    auto* e4m3_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(ptr_B);
    auto B = make_tensor(make_gmem_ptr(e4m3_ptr), make_layout(make_shape(N, K), make_stride(ld_b, _1{})));
    return B;
  }

  // Per-expert weight-scale pointers + row stride. Mirrors MXFP4's
  // make_scale_ptrs exactly (same gate/up split conventions), just with
  // K_scale = K / FP8_GROUP_SIZE_K instead of K / MXFP4_GROUP_SIZE.
  auto make_D_tensors(ElementD* ptr_D, int pre_rows, int M, int N) {
    return make_tensor(
        make_gmem_ptr<ElementD>(ptr_D + pre_rows * N), make_layout(make_shape(M, N), make_stride(N, _1{})));
  }

  // Adapted: direct compact-expert/N indexing replaces the E-wide search.
  CUTLASS_DEVICE void operator()(Params const& params, sycl::nd_item<3> item, int32_t*) {
    auto mma = params.mma;
    auto wg_tile = mma.tile_mnk();
    int n_tiles = ceil_div(params.N, get<1>(wg_tile));
    int task = item.get_group_linear_id();
    int compact_expert = params.tile_major ? task % params.num_experts : task / n_tiles;
    int n_coord = params.tile_major ? task / params.num_experts : task % n_tiles;
    int M = params.M_per_group[compact_expert];
    if (M <= 0) return;
    int expert_id = params.ExpertIds[compact_expert];
    int pre_rows = params.RowOffsets[compact_expert];
    int N = params.N, K = params.K;
    int64_t scale_n = (N + 127) / 128;
    int64_t scale_k = K / FP8_GROUP_SIZE_K;
    auto* ptr_A = const_cast<uint8_t*>(params.Activations) + int64_t(pre_rows) * K * sizeof(ElementA);
    auto* ptr_B = const_cast<uint8_t*>(params.PackedWeights) + int64_t(expert_id) * N * params.ld_b;
    int host_expert = params.HostExpertIds[compact_expert];
    auto* cache_B = ptr_B;
    if (host_expert >= 0) {
      ptr_B = const_cast<uint8_t*>(params.HostWeights) + int64_t(host_expert) * N * params.ld_b;
    }
    auto* ptr_S = static_cast<const float*>(params.WeightScales) + int64_t(expert_id) * scale_n * scale_k;
    auto A = make_A_tensor(ptr_A, M, K);
    auto B = make_B_tensors(ptr_B, N, K, params.ld_b);
    auto D = make_D_tensors(params.Outputs, pre_rows, M, N);
    int thr_id = item.get_local_linear_id();
    // Usually one tile (one route per token); duplicates remain supported.
    for (int m_coord = 0; m_coord < ceil_div(M, get<0>(wg_tile)); ++m_coord) {
      CollectiveMainloop mainloop;
      mainloop.cache_dest = host_expert >= 0 && m_coord == 0 ? cache_B : nullptr;
      mainloop.cache_source = ptr_B;
      mainloop(A, B, ptr_S, scale_k, D, make_coord(m_coord, n_coord, _, 0), mma, thr_id, nullptr, N, params.scale_mode);
    }
  }
};
}  // namespace moe_w8a16
