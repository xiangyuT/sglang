// SPDX-License-Identifier: Apache-2.0
// Isolated prefill entry point; decode libraries retain their four-row bound.
#define SGLANG_HC_MAX_TOKENS 4096
#define sglang_hc_bf256g128 sglang_hc_prefill
#define xpu_hc_esimd xpu_hc_prefill_norm
#define xpu_hc_esimd_error xpu_hc_prefill_norm_error
#include "hc_esimd.cpp"
