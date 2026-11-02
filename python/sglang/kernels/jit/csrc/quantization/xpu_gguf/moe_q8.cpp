// SPDX-License-Identifier: Apache-2.0
// Submit the existing, separately attributed Strata Q8_1 dots as one MoE path.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <cstring>

extern "C" int xpu_expert_q8_float_rows(
    void*, const void*, const void*, const float*, const int32_t*,
    const int32_t*, float*, float*, void*, int, int, int, int, int64_t,
    int, int, int, int);
extern "C" const char* xpu_expert_q8_error();

namespace sglang_gguf {
extern thread_local std::string error;
using bf16 = sycl::ext::oneapi::bfloat16;
class MoeInputFloat1323;
class MoeSiluMul1323;
class MoeWeightedOutput1323;
class MoeArenaZero1367;
}

extern "C" int xpu_expert_quantize(void*, const float*, void*, int, int);
extern "C" const char* xpu_expert_quantize_error();
extern "C" int xpu_moe_input_quantize(void*, const void*, float*, void*, int, int);
extern "C" int xpu_moe_activate_quantize(void*, const float*, const float*, float*, void*, int);
extern "C" const char* xpu_moe_quant_error();
extern "C" int xpu_gguf_iq2s_gu(void*, const void*, const void*, const void*,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    float*, float*, int, int, int, int, int);
extern "C" int xpu_gguf_q2_down(void*,const void*,const void*,const int32_t*,
    const int32_t*,const int32_t*,float*,int,int,int,int);
extern "C" int xpu_expert_q8_rows_ptr(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,float*,float*,int,int,int,int,int,int,int,int,int);
extern "C" int xpu_expert_q8_rows_ptr_grouped(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,
    int,int,int,int,int,int,int,int,int,int);
extern "C" int xpu_gguf_mixed_packets(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,
    int,int,int,int,int);
extern "C" int xpu_gguf_mixed_packets_gu_soa(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,
    int,int,int,int,int);
extern "C" const char* xpu_gguf_mixed_packets_error();
extern "C" int xpu_gguf_q8_down_packets(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,float*,int,int,int);
extern "C" int xpu_gguf_mixed_packets_soa(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,
    int,int,int,int,int);
extern "C" int xpu_gguf_iq2s_gu_ptr(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,int,int,int,int,int);
extern "C" int xpu_gguf_iq2s_gu_ptr_split4(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,int,int,int,int,int);
extern "C" int xpu_gguf_q2_down_ptr(void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,float*,int,int,int,int);
extern "C" int xpu_gguf_lowbit_gu(void*,const void*,const void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,const int32_t*,float*,float*,int,int,int,int,int,int);
extern "C" int xpu_raw_q2_down(void*,const void*,const uint64_t*,const void*,
    const int32_t*,const int32_t*,const int32_t*,float*,int,int,int);
extern "C" const char* xpu_raw_q2_down_error();
struct IQ2SGroupedArguments {
    const int32_t* slots;
    const int32_t* starts;
    const int32_t* tokens;
    const int32_t* destinations;
    int groups;
    int workgroup;
};

static int submit_moe_q8(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16, const IQ2SGroupedArguments* grouped = nullptr,
    const uint64_t* slot_addresses = nullptr, bool input_prepared = false,
    int down_kind = 42) {
    if (!queue || !input || !gate || !up || !down || !slots || !tokens ||
        !weights || !output || !xfp || !g || !u || !activation || !y ||
        !scratch || K != 2560 || I != 640 || T < 1 || T > 32 ||
        topk < 1 || topk > 10 || capacity < 1 ||
        (kind != 16 && kind != 21 && kind != 22 && kind != 23 && kind != 29 && kind != 42) ||
        (down_kind != 8 && down_kind != 20 && down_kind != 42) ||
        gu_bytes <= 0 || gu_row_bytes <= 0 || down_bytes <= 0 ||
        down_row_bytes <= 0 || (weights_bf16 != 0 && weights_bf16 != 1)) {
        return 2;
    }
    if (T > 4 && ((kind != 21 && kind != 23) || (down_kind != 8 && down_kind != 20))) return 2;
    try {
        using namespace sglang_gguf;
        auto& q = *static_cast<sycl::queue*>(queue);
        const auto* xb = static_cast<const bf16*>(input);
        const int inputs = T * K, routes = T * topk, intermediates = routes * I;
        const bool arena = g == y;
        if (arena && (!grouped || !grouped->starts || !grouped->destinations ||
            (kind != 21 && kind != 23) ||
            (down_kind != 8 && down_kind != 20) || u != y + intermediates ||
            activation != y + 2 * intermediates || !q.is_in_order())) return 2;
        const char* fused_setting = std::getenv("SGLANG_XPU_GGUF_FUSED_QUANT");
        const bool fused_quant = fused_setting && std::strcmp(fused_setting, "1") == 0;
        const bool fused_input = fused_quant && (grouped || slot_addresses);
        const char* packet_setting = std::getenv("SGLANG_XPU_GGUF_MIXED_PACKETS");
        const bool mixed_packets = packet_setting && std::strcmp(packet_setting,"1")==0
                                 && slot_addresses && grouped && topk==10;
        const char* q8_setting = std::getenv("SGLANG_XPU_GGUF_Q8_DOWN_PACKETS");
        const bool q8_packets = mixed_packets && down_kind == 8 && q8_setting
                                && std::strcmp(q8_setting,"1")==0;
        const char* soa_setting = std::getenv("SGLANG_XPU_GGUF_IQ4_DOWN_SOA");
        const bool down_soa = down_kind == 20 && soa_setting && std::strcmp(soa_setting,"1")==0;
        const char* gu_setting = std::getenv("SGLANG_XPU_GGUF_IQ3_GU_SOA");
        const bool gu_soa = kind == 21 && gu_setting && std::strcmp(gu_setting,"1")==0;
        if ((down_soa || gu_soa) && !mixed_packets) return 2;
        if(input_prepared && (!fused_input || !grouped || !slot_addresses))return 2;
        int status = 0;
        if (fused_input) {
            if(!input_prepared)status = xpu_moe_input_quantize(queue, input, xfp, scratch, K, T);
            if (status) { error = xpu_moe_quant_error(); return status; }
        } else {
            q.parallel_for<MoeInputFloat1323>(sycl::range<1>(inputs), [=](sycl::id<1> id) {
                const int j = id[0];
                xfp[j] = float(xb[j]);
            });
        }
        if (grouped && (kind == 21 || kind == 23)) {
            status = fused_input ? 0 : xpu_expert_quantize(queue, xfp, scratch, K, T);
            if (!status && mixed_packets && kind==21) {
                auto packet = gu_soa ? xpu_gguf_mixed_packets_gu_soa : xpu_gguf_mixed_packets;
                status = packet(
                    queue,slot_addresses,scratch,grouped->slots,grouped->starts,
                    grouped->tokens,grouped->destinations,g,u,
                    21,grouped->groups,routes,capacity,T);
                if (status) error = xpu_gguf_mixed_packets_error();
            } else if (!status) status = xpu_expert_q8_rows_ptr_grouped(
                queue, slot_addresses, scratch, grouped->slots, grouped->starts,
                grouped->tokens, grouped->destinations, g, u,
                kind, K, I, routes, gu_row_bytes, 1, capacity, T, 0, grouped->groups);
            if (status && !(mixed_packets && kind==21)) error = xpu_expert_q8_error();
        } else if (grouped) {
            if ((kind != 22 && kind != 16 && kind != 29) || !grouped->slots || !grouped->starts ||
                !grouped->tokens || !grouped->destinations || grouped->groups < 1)
                return 2;
            status = fused_input ? 0 : xpu_expert_quantize(queue, xfp, scratch, K, T);
            if (status) error = xpu_expert_quantize_error();
            if (!status && kind != 22) status = xpu_gguf_lowbit_gu(
                queue,gate,up,slot_addresses,scratch,grouped->slots,grouped->starts,
                grouped->tokens,grouped->destinations,g,u,kind,grouped->groups,
                routes,T,capacity,grouped->workgroup);
            else if (!status && slot_addresses) {
                const char* split_setting=std::getenv("SGLANG_XPU_GGUF_IQ2S_PLANE_SPLIT");
                const bool split=T==4&&grouped->workgroup==4&&split_setting&&
                    std::strcmp(split_setting,"1")==0;
                auto gu=split?xpu_gguf_iq2s_gu_ptr_split4:xpu_gguf_iq2s_gu_ptr;
                status = gu(
                queue, slot_addresses, scratch, grouped->slots, grouped->starts,
                grouped->tokens, grouped->destinations, g, u,
                grouped->groups, routes, T, capacity, grouped->workgroup);
            }
            else if (!status) status = xpu_gguf_iq2s_gu(
                queue, gate, up, scratch, grouped->slots, grouped->starts,
                grouped->tokens, grouped->destinations, g, u,
                grouped->groups, routes, T, capacity, grouped->workgroup);
        } else if (slot_addresses) {
            status = fused_input ? 0 : xpu_expert_quantize(queue, xfp, scratch, K, T);
            if (status) error = xpu_expert_quantize_error();
            if (!status) status = xpu_expert_q8_rows_ptr(
                queue, slot_addresses, scratch, slots, tokens, g, u,
                kind, K, I, routes, gu_row_bytes, 1, capacity, T, 0);
        } else {
            status = xpu_expert_q8_float_rows(
                queue, gate, up, xfp, slots, tokens, g, u, scratch,
                kind, K, I, routes, gu_bytes, gu_row_bytes, 1, capacity, T);
        }
        if (status) {
            if (!grouped) error = xpu_expert_q8_error();
            return status;
        }
        const char* raw_q2_setting=std::getenv("SGLANG_XPU_GGUF_RAW_Q2_DPAS");
        const bool raw_down=down_kind==42&&kind!=22&&raw_q2_setting&&std::strcmp(raw_q2_setting,"1")==0;
        const bool fused_activation = fused_quant &&
            ((grouped && kind == 22) || raw_down || slot_addresses);
        if (fused_activation) {
            status = xpu_moe_activate_quantize(queue, g, u, activation, scratch, routes);
            if (status) { error = xpu_moe_quant_error(); return status; }
        } else {
            q.parallel_for<MoeSiluMul1323>(sycl::range<1>(intermediates), [=](sycl::id<1> id) {
                const int j = id[0];
                activation[j] = (g[j] / (1.0f + sycl::exp(-g[j]))) * u[j];
            });
        }
        if (down_kind != 42) {
            status = fused_activation ? 0 : xpu_expert_quantize(queue, activation, scratch, I, routes);
            if (!status && mixed_packets && down_kind==20) {
                auto packet = down_soa ? xpu_gguf_mixed_packets_soa : xpu_gguf_mixed_packets;
                status = packet(
                    queue,slot_addresses,scratch,grouped->slots,grouped->starts,
                    grouped->tokens,grouped->destinations,y,nullptr,
                    20,grouped->groups,routes,capacity,T);
                if (status) error = xpu_gguf_mixed_packets_error();
            } else if (!status && q8_packets) {
                status = xpu_gguf_q8_down_packets(
                    queue,slot_addresses,scratch,grouped->slots,grouped->starts,
                    grouped->destinations,y,grouped->groups,routes,capacity);
                if (status) error = xpu_gguf_mixed_packets_error();
            } else if (!status && grouped) status = xpu_expert_q8_rows_ptr_grouped(
                queue, slot_addresses, scratch, grouped->slots, grouped->starts,
                grouped->tokens, grouped->destinations, y, nullptr,
                down_kind, I, K, routes, down_row_bytes, 0, capacity, routes, 2,
                grouped->groups);
            else if (!status) status = xpu_expert_q8_rows_ptr(
                queue, slot_addresses, scratch, slots, nullptr, y, nullptr,
                down_kind, I, K, routes, down_row_bytes, 0, capacity, routes, 2);
            if (status && !(mixed_packets && down_kind==20) && !q8_packets) error = xpu_expert_q8_error();
        } else if (grouped && kind == 22) {
            status = fused_activation ? 0 : xpu_expert_quantize(queue, activation, scratch, I, routes);
            if (status) error = xpu_expert_quantize_error();
            if (!status && slot_addresses) status = xpu_gguf_q2_down_ptr(
                queue, slot_addresses, scratch, grouped->slots, grouped->starts,
                grouped->destinations, y, grouped->groups, routes,
                capacity, grouped->workgroup);
            else if (!status) status = xpu_gguf_q2_down(
                queue, down, scratch, grouped->slots, grouped->starts,
                grouped->destinations, y, grouped->groups, routes,
                capacity, grouped->workgroup);
        } else if (raw_down) {
            status = fused_activation ? 0 : xpu_expert_quantize(queue, activation, scratch, I, routes);
            if (status) error = xpu_expert_quantize_error();
            if (!status) status = xpu_raw_q2_down(
                queue,down,slot_addresses,scratch,
                grouped?grouped->slots:slots,grouped?grouped->starts:nullptr,
                grouped?grouped->destinations:nullptr,y,
                grouped?grouped->groups:routes,routes,capacity);
            if (status) error = xpu_raw_q2_down_error();
        } else if (slot_addresses) {
            status = fused_activation ? 0 : xpu_expert_quantize(queue, activation, scratch, I, routes);
            if (status) error = xpu_expert_quantize_error();
            if (!status) status = xpu_expert_q8_rows_ptr(
                queue, slot_addresses, scratch, slots, nullptr, y, nullptr,
                42, I, K, routes, down_row_bytes, 0, capacity, routes, 2);
        } else {
            status = xpu_expert_q8_float_rows(
                queue, down, nullptr, activation, slots, nullptr, y, nullptr,
                scratch, 42, I, K, routes, down_bytes, down_row_bytes,
                0, capacity, routes);
        }
        if (status) {
            if (!grouped && !raw_down) error = xpu_expert_q8_error();
            return status;
        }
        if (arena) {
            // GU/activation have finished. Restore only the clipped dummy
            // routes that down intentionally skipped, after all aliasing writes.
            const auto* starts = grouped->starts;
            const auto* destinations = grouped->destinations;
            const int groups = grouped->groups;
            q.parallel_for<MoeArenaZero1367>(sycl::range<1>(routes * (K / 4)),
                [=](sycl::id<1> id) {
                    const int i = id[0], position = i / (K / 4);
                    if (position >= starts[groups]) {
                        const int route = destinations[position];
                        reinterpret_cast<sycl::vec<float, 4>*>(y)[
                            int64_t(route) * (K / 4) + i % (K / 4)] =
                            sycl::vec<float, 4>(0.f);
                    }
                });
        }
        auto* out = static_cast<bf16*>(output);
        q.parallel_for<MoeWeightedOutput1323>(sycl::range<1>(inputs), [=](sycl::id<1> id) {
            const int j = id[0], token = j / K, column = j % K;
            // The pinned Torch XPU strided reduction uses vt0=4 independent
            // accumulators, followed by a left-to-right combination. Keep
            // that order before BF16 rounding on the supported small axis.
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int r = 0; r < topk; ++r) {
                const int route = token * topk + r;
                const float w = weights_bf16 ?
                    float(static_cast<const bf16*>(weights)[route]) :
                    static_cast<const float*>(weights)[route];
                const float value = y[int64_t(route) * K + column] * w;
                sums[r % 4] = sums[r % 4] + value;
            }
            const float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
            out[j] = bf16(sum);
        });
        return 0;
    } catch (const std::exception& e) {
        sglang_gguf::error = e.what();
        return 1;
    }
}

extern "C" int xpu_gguf_moe_q8_arena_supported() { return 1; }

extern "C" int xpu_gguf_moe_q8(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16) {
    return submit_moe_q8(queue, input, gate, up, down, slots, tokens, weights, output, xfp, g, u, activation, y, scratch, kind, K, I, T, topk, capacity, gu_bytes, gu_row_bytes, down_bytes, down_row_bytes, weights_bf16);
}

extern "C" int xpu_gguf_moe_q8_pvr(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16,
    const int32_t* grouped_slots, const int32_t* group_starts,
    const int32_t* entry_tokens, const int32_t* destinations,
    int groups, int workgroup) {
    const IQ2SGroupedArguments grouped{grouped_slots, group_starts, entry_tokens, destinations, groups, workgroup};
    return submit_moe_q8(queue, input, gate, up, down, slots, tokens, weights, output, xfp, g, u, activation, y, scratch, kind, K, I, T, topk, capacity, gu_bytes, gu_row_bytes, down_bytes, down_row_bytes, weights_bf16, &grouped);
}

extern "C" int xpu_gguf_moe_q8_addressed(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16,
    const int32_t* grouped_slots, const int32_t* group_starts,
    const int32_t* entry_tokens, const int32_t* destinations,
    int groups, int workgroup, const uint64_t* slot_addresses) {
    if (!slot_addresses || capacity > 512 || (kind == 22 && groups < 1)) return 2;
    const IQ2SGroupedArguments grouped{grouped_slots, group_starts, entry_tokens, destinations, groups, workgroup};
    return submit_moe_q8(queue, input, gate, up, down, slots, tokens, weights,
        output, xfp, g, u, activation, y, scratch, kind, K, I, T, topk,
        capacity, gu_bytes, gu_row_bytes, down_bytes, down_row_bytes,
        weights_bf16, groups ? &grouped : nullptr, slot_addresses);
}

// The resident producer has filled the exact selected stream workspace.
extern "C" int xpu_gguf_moe_q8_addressed_prepared(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16,
    const int32_t* grouped_slots, const int32_t* group_starts,
    const int32_t* entry_tokens, const int32_t* destinations,
    int groups, int workgroup, const uint64_t* slot_addresses) {
    if (!slot_addresses || capacity > 512 || groups < 1 ||
        (kind != 16 && kind != 22 && kind != 29)) return 2;
    const IQ2SGroupedArguments grouped{grouped_slots, group_starts, entry_tokens, destinations, groups, workgroup};
    return submit_moe_q8(queue, input, gate, up, down, slots, tokens, weights,
        output, xfp, g, u, activation, y, scratch, kind, K, I, T, topk,
        capacity, gu_bytes, gu_row_bytes, down_bytes, down_row_bytes,
        weights_bf16, &grouped, slot_addresses, true);
}

// Original mixed-format expert fields; all legacy entry points keep Q2_0 down.
extern "C" int xpu_gguf_moe_q8_mixed_addressed(
    void* queue, const void* input, const void* gate, const void* up,
    const void* down, const int32_t* slots, const int32_t* tokens,
    const void* weights, void* output, float* xfp, float* g, float* u,
    float* activation, float* y, void* scratch,
    int kind, int K, int I, int T, int topk, int capacity,
    int64_t gu_bytes, int gu_row_bytes, int64_t down_bytes,
    int down_row_bytes, int weights_bf16,
    const int32_t* grouped_slots, const int32_t* group_starts,
    const int32_t* entry_tokens, const int32_t* destinations,
    int groups, int workgroup, const uint64_t* slot_addresses,
    int down_kind, int input_prepared) {
    if(!slot_addresses||capacity>512||groups<1||(kind!=21&&kind!=23)||
       (input_prepared!=0&&input_prepared!=1))return 2;
    const IQ2SGroupedArguments grouped{grouped_slots,group_starts,entry_tokens,destinations,groups,workgroup};
    return submit_moe_q8(queue,input,gate,up,down,slots,tokens,weights,output,xfp,g,u,
        activation,y,scratch,kind,K,I,T,topk,capacity,gu_bytes,gu_row_bytes,down_bytes,
        down_row_bytes,weights_bf16,&grouped,slot_addresses,bool(input_prepared),down_kind);
}

extern "C" int xpu_gguf_mixed_max_rows() { return 32; }
