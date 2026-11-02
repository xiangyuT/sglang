/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Native causal prefill with Q16/KV64/SG2; preserves the original Q8/subgroup math.
// Original kernel copyright (C) 2026 Intel Corporation. BSD-3-Clause.
#define SYCL_INTEL_TARGET 20
#define CUTLASS_ENABLE_SYCL 1
#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_runner.hpp"
#include <exception>
#include <string>
namespace {
thread_local std::string error;
}
extern "C" int dflash_native_abi() {
  return 1;
}
extern "C" const char* dflash_prefill_error() {
  return error.c_str();
}
extern "C" int dflash_native_prefill(
    void* q, void* k, void* v, int* table, int* lens, int* cuq, void* out, int m, int pages, int tablepages) {
  try {
    prefill::Arguments p{};
    p.q_ptr = q;
    p.k_ptr = k;
    p.v_ptr = v;
    p.o_ptr = out;
    p.q_row_stride = 4096;
    p.q_head_stride = 256;
    p.k_row_stride = 512;
    p.k_head_stride = 256;
    p.v_row_stride = 512;
    p.v_head_stride = 256;
    p.v_dim_stride = 1;
    p.o_row_stride = 4096;
    p.o_head_stride = 256;
    p.b = 1;
    p.b_k = 1;
    p.h = 16;
    p.h_k = 2;
    p.q_group_size = 1;
    p.seqlen_q = m;
    p.seqlen_k = tablepages * 128;
    p.total_q = m;
    p.total_k = pages * 128;
    p.d = p.d_rounded = p.dv = p.dv_rounded = 256;
    p.softmax_scale = 0.0625f;
    p.is_bf16 = true;
    p.is_causal = true;
    p.window_size_left = p.seqlen_k - 1;
    p.window_size_right = 0;
    p.p_dropout = 1.f;
    p.page_table = table;
    p.page_table_batch_stride = tablepages;
    p.max_num_pages_per_seq = tablepages;
    p.page_size = 128;
    p.num_pages = pages;
    p.cu_seqlens_q = cuq;
    p.cu_seqlens_k = lens;
    using QK = cute::Shape<cute::_16, cute::_64, cute::_32>;
    using PV = cute::Shape<cute::_16, cute::_32, cute::_64>;
    using Out = cute::Shape<cute::_16, cute::_256>;
    using SG = cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>, cute::Stride<cute::_1, cute::_1, cute::_1>>;
    return prefill::FMHAConfig<true, false, false, false, false, QK, PV, Out, SG>::run_paged(p);
  } catch (const std::exception& e) {
    error = e.what();
    return 1;
  }
}
