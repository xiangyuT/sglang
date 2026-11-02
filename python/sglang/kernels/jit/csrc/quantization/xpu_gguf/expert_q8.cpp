// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Original Strata lane8 Q8_1 expert dot scheduling, adapted to separate SGLang
// compressed cache tensors. See dot.hpp, STRATA-LICENSE and GGML-LICENSE.
#include "expert_q8_dot.hpp"
namespace sglang_expert_q8 {
inline uint32_t q2_decode4(uint32_t codes) {
    codes&=255;
    codes=(codes|(codes<<12))&0x000f000fu;
    codes=(codes|(codes<<6))&0x03030303u;
    return (codes+0x7f7f7f7fu)^0x80808080u;
}
inline float dot_q2(const void* weight,const block_q8_1* act,int block,int part) {
    const auto* w=(const block_q2_0*)weight+block;
    const auto* packed=(const uint16_t*)w->qs+part*4;
    const auto* a=act+part;
    int dot=0;
#pragma unroll
    for(int j=0;j<4;++j) {
        const uint32_t codes=packed[j];
        dot=dp4a(get_int_b4(a->qs,2*j),int(q2_decode4(codes)),dot);
        dot=dp4a(get_int_b4(a->qs,2*j+1),int(q2_decode4(codes>>8)),dot);
    }
    return float(w->d)*float(a->ds[0])*dot;
}
template<int Type>
inline float row_dot(const uint8_t* row,const block_q8_1* x,int width,int lane,sycl::sub_group sg) {
    constexpr int QK=(Type==8||Type==20)?32:Type==42?64:256;
    constexpr int IPB=Type==8?4:Type==20||Type==42?2:8;
    constexpr int STEP=Type==23?4:(Type==16||Type==20||Type==21||Type==22||Type==8)?2:1;
    float sum=0;
    for(int index=lane;index<(width/QK)*IPB;index+=8) {
        const int block=index/IPB,part=STEP*(index%IPB);
        const auto* act=x+block*(QK/32);
        if constexpr(Type==16)sum+=vec_dot_iq2_xxs_q8_1(row,act,block,part);
        else if constexpr(Type==22)sum+=vec_dot_iq2_s_q8_1(row,act,block,part);
        else if constexpr(Type==29)sum+=vec_dot_iq1_m_q8_1(row,act,block,part);
        else if constexpr(Type==21)sum+=dot_iq3(row,act,block,part);
        else if constexpr(Type==23)sum+=dot_iq4xs(row,act,block,part);
        else if constexpr(Type==20)sum+=dot_iq4nl(row,act,block,part);
        else if constexpr(Type==8)sum+=dot_q8(row,act,block,part);
        else sum+=dot_q2(row,act,block,part);
    }
#pragma unroll
    for(int offset=4;offset;offset>>=1)sum+=sycl::permute_group_by_xor(sg,sum,offset);
    return sum;
}
template<int Type,bool Pair> class Rows;
template<int Type,bool Pair>
void submit(sycl::queue& q,const uint8_t* w,const uint8_t* up,const block_q8_1* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,int width,int rows,
    int routes,int64_t expert_bytes,int row_bytes,int capacity,int input_rows) {
    const int groups=(rows+31)/32;
    q.parallel_for<Rows<Type,Pair>>(sycl::nd_range<1>(size_t(groups)*routes*256,256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int route=it.get_group_linear_id()/groups;
            const int output_row=(it.get_group_linear_id()%groups)*32+it.get_local_linear_id()/8;
            const int lane=it.get_local_linear_id()%8,slot=slots[route];
            const int token=Pair?(tokens?tokens[route]:0):route;
            if(slot<0||slot>=capacity||token<0||token>=input_rows) {
                if(lane==0 && output_row<rows) {
                    out[int64_t(route)*rows+output_row]=std::numeric_limits<float>::quiet_NaN();
                    if constexpr(Pair)outu[int64_t(route)*rows+output_row]=std::numeric_limits<float>::quiet_NaN();
                }
                return;
            }
            // Keep the whole subgroup converged for lane8 XOR reductions even
            // for a partial output tile. Inactive rows read a valid dummy row.
            const int row=sycl::min(output_row,rows-1);
            const auto* act=x+int64_t(token)*(width/32);
            const auto* weight=w+int64_t(slot)*expert_bytes+int64_t(row)*row_bytes;
            const float gate=row_dot<Type>(weight,act,width,lane,it.get_sub_group());
            float u=0;
            if constexpr(Pair)u=row_dot<Type>(up+int64_t(slot)*expert_bytes+int64_t(row)*row_bytes,act,width,lane,it.get_sub_group());
            if(lane==0 && output_row<rows) {
                out[int64_t(route)*rows+output_row]=gate;
                if constexpr(Pair)outu[int64_t(route)*rows+output_row]=u;
            }
        });
}
template<bool Pair>
void dispatch(sycl::queue& q,const uint8_t* w,const uint8_t* up,const block_q8_1* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,int type,int width,int rows,
    int routes,int64_t expert_bytes,int row_bytes,int capacity,int input_rows) {
#define GO(TYPE) case TYPE:submit<TYPE,Pair>(q,w,up,x,slots,tokens,out,outu,width,rows,routes,expert_bytes,row_bytes,capacity,input_rows);break
    switch(type){GO(8);GO(16);GO(20);GO(21);GO(22);GO(23);GO(29);GO(42);default:throw std::runtime_error("Unsupported expert Q8 format");}
#undef GO
}
}
extern "C" const char* xpu_expert_q8_error(){return sglang_expert_q8::error.c_str();}
extern "C" int xpu_expert_q8_rows(void* queue,const void* w,const void* up,const void* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,int type,int width,int rows,
    int routes,int64_t expert_bytes,int row_bytes,int pair,int capacity,int input_rows) {
    if(!queue||!w||!x||!slots||!out||width<=0||rows<=0||routes<=0||expert_bytes<=0||row_bytes<=0||
        capacity<=0||input_rows<=0||(pair!=0&&pair!=1)||(pair&&(!up||!outu))||
        (!pair&&input_rows!=routes)||width%((type==8||type==20)?32:type==42?64:256))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(pair)sglang_expert_q8::dispatch<true>(q,(const uint8_t*)w,(const uint8_t*)up,(const block_q8_1*)x,slots,tokens,out,outu,type,width,rows,routes,expert_bytes,row_bytes,capacity,input_rows);
        else sglang_expert_q8::dispatch<false>(q,(const uint8_t*)w,nullptr,(const block_q8_1*)x,slots,nullptr,out,nullptr,type,width,rows,routes,expert_bytes,row_bytes,capacity,input_rows);
        return 0;
    }catch(const std::exception& e){sglang_expert_q8::error=e.what();return 1;}
}

extern "C" int xpu_expert_quantize(void*,const float*,void*,int,int);
extern "C" const char* xpu_expert_quantize_error();
// Submit the same quantize and row kernels in one host call. The model owns
// the disjoint scratch tensor until all consumers on this stream complete.
extern "C" int xpu_expert_q8_float_rows(void* queue,const void* w,const void* up,const float* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,void* scratch,
    int type,int width,int rows,int routes,int64_t expert_bytes,int row_bytes,
    int pair,int capacity,int input_rows) {
    if(!scratch) return 2;
    int status=xpu_expert_quantize(queue,x,scratch,width,input_rows);
    if(status) {
        sglang_expert_q8::error=xpu_expert_quantize_error();
        return status;
    }
    return xpu_expert_q8_rows(queue,w,up,scratch,slots,tokens,out,outu,type,width,rows,
        routes,expert_bytes,row_bytes,pair,capacity,input_rows);
}

namespace sglang_expert_q8 {
template<int Type,bool Pair> class PointerRows;
template<int Type,bool Pair>
void submit_pointers(sycl::queue& q,const uint64_t* addresses,const block_q8_1* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,int width,int rows,
    int routes,int row_bytes,int capacity,int input_rows,int field) {
    const int groups=(rows+31)/32;
    q.parallel_for<PointerRows<Type,Pair>>(sycl::nd_range<1>(size_t(groups)*routes*256,256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int route=it.get_group_linear_id()/groups;
            const int output_row=(it.get_group_linear_id()%groups)*32+it.get_local_linear_id()/8;
            const int lane=it.get_local_linear_id()%8,slot=slots[route];
            const int token=Pair?(tokens?tokens[route]:0):route;
            const uint64_t base=slot>=0&&slot<capacity?addresses[int64_t(slot)*3+field]:0;
            const uint64_t baseu=Pair&&slot>=0&&slot<capacity?addresses[int64_t(slot)*3+field+1]:0;
            if(!base||(Pair&&!baseu)||token<0||token>=input_rows) {
                if(lane==0 && output_row<rows) {
                    out[int64_t(route)*rows+output_row]=std::numeric_limits<float>::quiet_NaN();
                    if constexpr(Pair)outu[int64_t(route)*rows+output_row]=std::numeric_limits<float>::quiet_NaN();
                }
                return;
            }
            // Keep the whole subgroup converged for lane8 XOR reductions even
            // for a partial output tile. Inactive rows read a valid dummy row.
            const int row=sycl::min(output_row,rows-1);
            const auto* act=x+int64_t(token)*(width/32);
            const auto* weight=reinterpret_cast<const uint8_t*>(base)+int64_t(row)*row_bytes;
            const float gate=row_dot<Type>(weight,act,width,lane,it.get_sub_group());
            float u=0;
            if constexpr(Pair)u=row_dot<Type>(reinterpret_cast<const uint8_t*>(baseu)+int64_t(row)*row_bytes,act,width,lane,it.get_sub_group());
            if(lane==0 && output_row<rows) {
                out[int64_t(route)*rows+output_row]=gate;
                if constexpr(Pair)outu[int64_t(route)*rows+output_row]=u;
            }
        });
}

template<bool Pair>
void dispatch_pointers(sycl::queue& q,const uint64_t* addresses,const block_q8_1* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,int type,
    int width,int rows,int routes,int row_bytes,int capacity,int input_rows,int field) {
#define RUN(TYPE) case TYPE:submit_pointers<TYPE,Pair>(q,addresses,x,slots,tokens,out,outu,width,rows,routes,row_bytes,capacity,input_rows,field);break
    switch(type){RUN(8);RUN(16);RUN(20);RUN(21);RUN(22);RUN(23);RUN(29);RUN(42);default:throw std::runtime_error("Unsupported addressed expert format");}
#undef RUN
}
}
extern "C" int xpu_expert_q8_rows_ptr(void* queue,const uint64_t* addresses,const void* x,
    const int32_t* slots,const int32_t* tokens,float* out,float* outu,
    int type,int width,int rows,int routes,int row_bytes,int pair,int capacity,int input_rows,int field) {
    if(!queue||!addresses||!x||!slots||!out||width<=0||rows<=0||routes<=0||row_bytes<=0||capacity<=0||
        input_rows<=0||(pair!=0&&pair!=1)||(pair&&(!outu||field!=0))||(!pair&&(input_rows!=routes||field!=2))||
        width%((type==8||type==20)?32:type==42?64:256))return 2;
    try {
        auto& q=*static_cast<sycl::queue*>(queue);
        if(pair)sglang_expert_q8::dispatch_pointers<true>(q,addresses,(const block_q8_1*)x,slots,tokens,out,outu,type,width,rows,routes,row_bytes,capacity,input_rows,field);
        else sglang_expert_q8::dispatch_pointers<false>(q,addresses,(const block_q8_1*)x,slots,nullptr,out,nullptr,type,width,rows,routes,row_bytes,capacity,input_rows,field);
        return 0;
    }catch(const std::exception& e){sglang_expert_q8::error=e.what();return 1;}
}

namespace sglang_expert_q8 {
template<int Type, bool Pair> class GroupedPointerRows;
template<int Type, bool Pair>
void submit_grouped_pointers(sycl::queue& q, const uint64_t* addresses,
    const block_q8_1* x, const int32_t* slots, const int32_t* starts,
    const int32_t* tokens, const int32_t* destinations, float* out, float* outu,
    int width, int rows, int routes, int row_bytes, int capacity,
    int input_rows, int field, int groups) {
    const int tiles = (rows + 31) / 32;
    q.parallel_for<GroupedPointerRows<Type, Pair>>(
        sycl::nd_range<1>(size_t(tiles) * routes * 256, 256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int entry = it.get_group_linear_id() / tiles;
            // Preparation clips the final logical zero expert. Every work-item
            // in this workgroup takes the same branch; lane8 reductions remain
            // converged. Skipped output rows were cleared by preparation.
            if (entry >= starts[groups]) return;
            int group = 0;
            while (group + 1 < groups && entry >= starts[group + 1]) ++group;
            const int route = destinations[entry];
            const int token = Pair ? tokens[entry] : route;
            const int slot = slots[group];
            const int output_row = (it.get_group_linear_id() % tiles) * 32
                                 + it.get_local_linear_id() / 8;
            const int lane = it.get_local_linear_id() % 8;
            if (route < 0 || route >= routes) return;
            const uint64_t base = slot >= 0 && slot < capacity
                ? addresses[int64_t(slot) * 3 + field] : 0;
            const uint64_t baseu = Pair && slot >= 0 && slot < capacity
                ? addresses[int64_t(slot) * 3 + field + 1] : 0;
            if (!base || (Pair && !baseu) || token < 0 || token >= input_rows) {
                if (lane == 0 && output_row < rows) {
                    out[int64_t(route) * rows + output_row]
                        = std::numeric_limits<float>::quiet_NaN();
                    if constexpr(Pair) outu[int64_t(route) * rows + output_row]
                        = std::numeric_limits<float>::quiet_NaN();
                }
                return;
            }
            const int row = sycl::min(output_row, rows - 1);
            const auto* act = x + int64_t(token) * (width / 32);
            const auto* weight = reinterpret_cast<const uint8_t*>(base)
                               + int64_t(row) * row_bytes;
            const float gate = row_dot<Type>(weight, act, width, lane,
                                             it.get_sub_group());
            float up = 0;
            if constexpr(Pair) up = row_dot<Type>(
                reinterpret_cast<const uint8_t*>(baseu) + int64_t(row) * row_bytes,
                act, width, lane, it.get_sub_group());
            if (lane == 0 && output_row < rows) {
                out[int64_t(route) * rows + output_row] = gate;
                if constexpr(Pair) outu[int64_t(route) * rows + output_row] = up;
            }
        });
}
}

// Internal mixed-field dispatch; preserve each existing lane8 dot and output
// position while skipping the EP dummy through already prepared group bounds.
extern "C" int xpu_expert_q8_rows_ptr_grouped(void* queue,
    const uint64_t* addresses, const void* x, const int32_t* slots,
    const int32_t* starts, const int32_t* tokens, const int32_t* destinations,
    float* out, float* outu, int type, int width, int rows, int routes,
    int row_bytes, int pair, int capacity, int input_rows, int field, int groups) {
    if (!queue || !addresses || !x || !slots || !starts || !destinations || !out
        || rows <= 0 || routes <= 0 || row_bytes <= 0 || capacity <= 0
        || groups < 1 || groups > routes || input_rows <= 0
        || (pair != 0 && pair != 1)
        || (pair && (!tokens || !outu || field != 0 || (type != 21 && type != 23)))
        || (!pair && (field != 2 || input_rows != routes || (type != 8 && type != 20)))
        || width <= 0 || width % (pair ? 256 : 32)) return 2;
    try {
        auto& q = *static_cast<sycl::queue*>(queue);
#define RUN(TYPE, PAIR) sglang_expert_q8::submit_grouped_pointers<TYPE, PAIR>(q, addresses, static_cast<const block_q8_1*>(x), slots, starts, tokens, destinations, out, outu, width, rows, routes, row_bytes, capacity, input_rows, field, groups)
        if (pair) {
            if (type == 21) RUN(21, true); else RUN(23, true);
        } else {
            if (type == 8) RUN(8, false); else RUN(20, false);
        }
#undef RUN
        return 0;
    } catch (const std::exception& e) {
        sglang_expert_q8::error = e.what();
        return 1;
    }
}
