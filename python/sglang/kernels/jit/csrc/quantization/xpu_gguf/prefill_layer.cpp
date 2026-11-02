#include "prefill_moe.cpp"
#include <algorithm>

namespace sglang_f16_layer {
class Stage;class Gather;class SwiGLU;
void stage(sycl::queue& q,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,uint8_t* gu,uint8_t* down,int E,int experts,int G,int D){
 size_t units=(2*size_t(G)+D)/16;
 q.parallel_for<Stage>(sycl::nd_range<2>({size_t(E),(units+127)/128*128},{1,128}),[=](sycl::nd_item<2> it){
  size_t j=it.get_global_id(1);if(j>=units)return;int e=it.get_global_id(0),b=j*16,id=ids[e];
  int field=b<G?0:b<2*G?1:2,off=b-(field==0?0:field==1?G:2*G);
  auto* dst=field==2?down+size_t(e)*D+off:gu+size_t(e)*2*G+field*G+off;
  uint64_t addr=0;if(id>=0&&id<experts){int slot=slots[id];addr=slot>=0&&slot<experts?resident[size_t(slot)*3+field]:mirrors[size_t(id)*3+field];}
  *reinterpret_cast<sycl::uint4*>(dst)=addr?*reinterpret_cast<const sycl::uint4*>(addr+off):sycl::uint4(0);
 });
}
void gather(sycl::queue& q,const uint16_t* x,const int32_t* order,sycl::half* out,int rows,int T,int K){
 q.parallel_for<Gather>(sycl::range<1>(size_t(rows)*2560),[=](sycl::id<1> id){size_t j=id[0];int pos=order[j/2560];float v=0;
  if(pos>=0&&pos<T*K)v=sycl::bit_cast<float>(uint32_t(x[size_t(pos/K)*2560+j%2560])<<16);out[j]=sycl::half(v);
 });
}
void swiglu(sycl::queue& q,const float* gu,sycl::half* out,int rows){
 q.parallel_for<SwiGLU>(sycl::range<1>(size_t(rows)*640),[=](sycl::id<1> id){size_t j=id[0],r=j/640,c=j%640;float g=gu[r*1280+c],u=gu[r*1280+640+c];out[j]=sycl::half((g/(1.f+sycl::exp(-g)))*u);});
}
}
extern "C" int xpu_gguf_prefill_f16_layer(void* stream,const uint16_t* X,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,const int32_t* order,const int32_t* counts,
 int E,int experts,int topk,int tokens,int caprows,int group,int gu_kind,int down_kind,int gu_bytes,int down_bytes,
 uint8_t* GUW,uint8_t* DW,sycl::half* Xh,sycl::half* Hh,sycl::half* GWh,sycl::half* DWh,float* GU,float* Dout,void* decode_fn){
 using namespace sglang_prefill;
 if(!stream||!X||!ids||!slots||!resident||!mirrors||!order||!counts||!GUW||!DW||!Xh||!Hh||!GWh||!DWh||!GU||!Dout||!decode_fn||E<1||E>experts||experts>512||topk<1||topk>10||tokens<1||tokens>32768||group<1||group>32||gu_bytes%16||down_bytes%16)return 2;
 if(!((((gu_kind==23&&gu_bytes==870400)||(gu_kind==21&&gu_bytes==704000))&&down_kind==8&&down_bytes==1740800)
      ||(gu_kind==42&&down_kind==42&&gu_bytes==460800&&down_bytes==460800)))return 2;
 int all=0;for(int first=0;first<E;first+=group){int rows=0;for(int i=first;i<E&&i<first+group;++i){if(counts[i]<0||counts[i]>tokens*topk)return 2;rows+=counts[i];}if(rows>caprows)return 2;all+=rows;}if(all>tokens*topk)return 2;
 try{auto& q=*static_cast<sycl::queue*>(stream);if(!q.has_property<sycl::property::queue::in_order>())return 2;
 auto decode=reinterpret_cast<int(*)(void*,const void*,void*,int,int64_t,int)>(decode_fn);int offset=0;
 for(int first=0;first<E;first+=group){int n=std::min(group,E-first),rows=0,tiles=0;Group gp{};gp.n=n;
  for(int i=0;i<n;++i){gp.off[i]=rows;gp.tiles[i]=tiles;rows+=counts[first+i];tiles+=(counts[first+i]+7)/8;}
  gp.off[n]=rows;gp.tiles[n]=tiles;if(!rows)continue;
  sglang_f16_layer::stage(q,ids+first,slots,resident,mirrors,GUW,DW,n,experts,gu_bytes,down_bytes);
  sglang_f16_layer::gather(q,X,order+offset,Xh,rows,tokens,topk);
  if(decode(stream,GUW,GWh,gu_kind,int64_t(n)*1280*2560,0))return 3;
  launch(q,Xh,GWh,GU,gp,tiles,1280,2560);
  sglang_f16_layer::swiglu(q,GU,Hh,rows);
  if(decode(stream,DW,DWh,down_kind,int64_t(n)*2560*640,0))return 3;
  launch(q,Hh,DWh,Dout+size_t(offset)*2560,gp,tiles,2560,640);offset+=rows;
 }
 return 0;
 }catch(const std::exception& e){error=e.what();return 1;}
}

#include "prefill_copy.hpp"
extern "C" int xpu_gguf_prefill_f16_layer_copy(void* stream,const uint16_t* X,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,const int32_t* order,const int32_t* counts,
 int E,int experts,int topk,int tokens,int caprows,int group,int gu_kind,int down_kind,int gu_bytes,int down_bytes,
 uint8_t* GUW,uint8_t* DW,sycl::half* Xh,sycl::half* Hh,sycl::half* GWh,sycl::half* DWh,float* GU,float* Dout,void* decode_fn){
 using namespace sglang_prefill;
 if(!stream||!X||!ids||!slots||!resident||!mirrors||!order||!counts||!GUW||!DW||!Xh||!Hh||!GWh||!DWh||!GU||!Dout||!decode_fn||E<1||E>experts||experts>512||topk<1||topk>10||tokens<1||tokens>32768||group<1||group>32||gu_bytes%16||down_bytes%16)return 2;
 if(!((((gu_kind==23&&gu_bytes==870400)||(gu_kind==21&&gu_bytes==704000))&&down_kind==8&&down_bytes==1740800)
      ||(gu_kind==42&&down_kind==42&&gu_bytes==460800&&down_bytes==460800)))return 2;
 int all=0;for(int first=0;first<E;first+=group){int rows=0;for(int i=first;i<E&&i<first+group;++i){if(counts[i]<0||counts[i]>tokens*topk)return 2;rows+=counts[i];}if(rows>caprows)return 2;all+=rows;}if(all>tokens*topk)return 2;
 try{auto& q=*static_cast<sycl::queue*>(stream);if(!q.has_property<sycl::property::queue::in_order>())return 2;
 auto& copy=sglang_gguf_copy::get(q);auto* addresses=sglang_gguf_copy::resolve(q,ids,slots,resident,mirrors,E,experts);sycl::event done[2];bool used[2]={false,false};
 auto decode=reinterpret_cast<int(*)(void*,const void*,void*,int,int64_t,int)>(decode_fn);int offset=0;
 for(int first=0;first<E;first+=group){int n=std::min(group,E-first),rows=0,tiles=0;Group gp{};gp.n=n;
  for(int i=0;i<n;++i){gp.off[i]=rows;gp.tiles[i]=tiles;rows+=counts[first+i];tiles+=(counts[first+i]+7)/8;}
  gp.off[n]=rows;gp.tiles[n]=tiles;if(!rows)continue;
  int bank=(first/group)%2;auto* gb=GUW+size_t(bank)*group*2*gu_bytes;auto* db=DW+size_t(bank)*group*down_bytes;
  if(used[bank])copy.ext_oneapi_submit_barrier({done[bank]});
  for(int i=0;i<n;++i){auto* a=addresses+size_t(first+i)*3;
   copy.memcpy(gb+size_t(i)*2*gu_bytes,reinterpret_cast<const void*>(a[0]),gu_bytes);
   copy.memcpy(gb+size_t(i)*2*gu_bytes+gu_bytes,reinterpret_cast<const void*>(a[1]),gu_bytes);
   copy.memcpy(db+size_t(i)*down_bytes,reinterpret_cast<const void*>(a[2]),down_bytes);
  }
  q.ext_oneapi_submit_barrier({copy.ext_oneapi_submit_barrier()});
  sglang_f16_layer::gather(q,X,order+offset,Xh,rows,tokens,topk);
  if(decode(stream,gb,GWh,gu_kind,int64_t(n)*1280*2560,0))return 3;
  launch(q,Xh,GWh,GU,gp,tiles,1280,2560);
  sglang_f16_layer::swiglu(q,GU,Hh,rows);
  if(decode(stream,db,DWh,down_kind,int64_t(n)*2560*640,0))return 3;
  launch(q,Hh,DWh,Dout+size_t(offset)*2560,gp,tiles,2560,640);done[bank]=q.ext_oneapi_submit_barrier();used[bank]=true;offset+=rows;
 }
 return 0;
 }catch(const std::exception& e){error=e.what();return 1;}
}

#include "prefill_pack.hpp"
#include "iq3_expert_soa.hpp"
#include <array>

namespace sglang_f16_layer {
// Reuse the original decoder functions, output lanes and rounding. Only the
// encoded input address changes: direct resident fields or a cold packed ring.
template<int Type,bool GateUp,bool GuSoA>class DecodeFields;
template<int Type,bool GateUp,bool GuSoA = false>
void decode_fields(sycl::queue& q,std::array<uint64_t,96> sources,sycl::half* output,int experts){
 constexpr size_t field_values=640*2560;
 constexpr size_t values=GateUp?2*field_values:field_values;
 constexpr size_t blocks=values/256;
 auto props=sycl::ext::oneapi::experimental::properties{
  sycl::ext::oneapi::experimental::sub_group_size<32>,
  sycl::ext::intel::experimental::grf_size<128>};
 q.parallel_for<DecodeFields<Type,GateUp,GuSoA>>(sycl::nd_range<2>({size_t(experts),blocks*32},{1,128}),props,[=](sycl::nd_item<2> it){
  size_t e=it.get_global_id(0),block=it.get_global_id(1)/32;
  int lane=it.get_local_id(1)%32,field=GateUp?int(block/(field_values/256)):2;
  size_t source_block=GateUp?block%(field_values/256):block;
  const void* input=reinterpret_cast<const void*>(sources[e*3+field]);
  auto* dst=output+e*values+block*256;
  if constexpr(GuSoA) {
   static_assert(Type==21 && GateUp);
   float v[8];
#pragma unroll
   for(int j=0;j<8;++j) v[j]=sglang_iq3_expert_soa::value(
       static_cast<const uint8_t*>(input),source_block/10,source_block%10,
       lane/4*32+lane%4*8+j);
   sglang_gguf::store_run<sycl::half,8>(dst+lane/4*32+lane%4*8,v);
  }
  else if constexpr(Type==21)sglang_gguf::dq_iq3_s(input,source_block,dst,lane);
  else if constexpr(Type==23)sglang_gguf::dq_iq4_xs(input,source_block,dst,lane);
  else if constexpr(Type==8)sglang_gguf::dq_q8_0(input,source_block,dst,lane);
  else if constexpr(Type==42)sglang_gguf::dq_q2_0(input,source_block,dst,lane);
 });
}
}

int sglang_prefill_f16_layer_packed_impl(void* stream,const uint16_t* X,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,const int32_t* order,const int32_t* counts,
 int E,int experts,int topk,int tokens,int caprows,int group,int gu_kind,int down_kind,int gu_bytes,int down_bytes,
 uint8_t* GUW,uint8_t* DW,sycl::half* Xh,sycl::half* Hh,sycl::half* GWh,sycl::half* DWh,float* GU,float* Dout,void* decode_fn,bool gu_soa){
 using namespace sglang_prefill;
 if(gu_soa && gu_kind!=21)return 2;
 if(!stream||!X||!ids||!slots||!resident||!mirrors||!order||!counts||!Xh||!Hh||!GWh||!DWh||!GU||!Dout||E<1||E>experts||experts>512||topk<1||topk>10||tokens<1||tokens>32768||group<1||group>32)return 2;
 if(!((((gu_kind==23&&gu_bytes==870400)||(gu_kind==21&&gu_bytes==704000))&&down_kind==8&&down_bytes==1740800)
      ||(gu_kind==42&&down_kind==42&&gu_bytes==460800&&down_bytes==460800)))return 2;
 int all=0;for(int first=0;first<E;first+=group){int rows=0;for(int i=first;i<E&&i<first+group;++i){if(counts[i]<0||counts[i]>tokens*topk)return 2;rows+=counts[i];}if(rows>caprows)return 2;all+=rows;}if(all>tokens*topk)return 2;
 try{
  auto& q=*static_cast<sycl::queue*>(stream);if(!q.has_property<sycl::property::queue::in_order>())return 2;
  auto& packed=sglang_pack::get(q,gu_bytes,down_bytes,group);
  packed.resolve(ids,slots,resident,mirrors,E,experts,true);
  sycl::event done[2];bool used[2]={false,false};int offset=0;
  for(int first=0;first<E;first+=group){
   int n=std::min(group,E-first),rows=0,tiles=0;Group gp{};gp.n=n;
   for(int i=0;i<n;++i){gp.off[i]=rows;gp.tiles[i]=tiles;rows+=counts[first+i];tiles+=(counts[first+i]+7)/8;}
   gp.off[n]=rows;gp.tiles[n]=tiles;if(!rows)continue;
   int bank=(first/group)%2;auto* blob=packed.stage(first,n,bank,done[bank],used[bank]);
   std::array<uint64_t,96> sources{};
   for(int i=0;i<n;++i){auto* a=packed.addresses+size_t(first+i)*4;
    for(int f=0;f<3;++f)sources[i*3+f]=(a[3]>>63)?a[f]:reinterpret_cast<uint64_t>(blob+(a[3]*packed.stride)+(f<2?size_t(f)*gu_bytes:size_t(2)*gu_bytes));
   }
   sglang_f16_layer::gather(q,X,order+offset,Xh,rows,tokens,topk);
   if(gu_kind==21) {
    if(gu_soa)sglang_f16_layer::decode_fields<21,true,true>(q,sources,GWh,n);
    else sglang_f16_layer::decode_fields<21,true>(q,sources,GWh,n);
   }
   else if(gu_kind==23)sglang_f16_layer::decode_fields<23,true>(q,sources,GWh,n);
   else sglang_f16_layer::decode_fields<42,true>(q,sources,GWh,n);
   launch(q,Xh,GWh,GU,gp,tiles,1280,2560);
   sglang_f16_layer::swiglu(q,GU,Hh,rows);
   if(down_kind==8)sglang_f16_layer::decode_fields<8,false>(q,sources,DWh,n);
   else sglang_f16_layer::decode_fields<42,false>(q,sources,DWh,n);
   launch(q,Hh,DWh,Dout+size_t(offset)*2560,gp,tiles,2560,640);
   if(packed.stage_uses_ring){done[bank]=q.ext_oneapi_submit_barrier();used[bank]=true;}
   offset+=rows;
  }
  return 0;
 }catch(const std::exception& e){error=e.what();return 1;}
}

extern "C" int xpu_gguf_prefill_f16_layer_packed(void* stream,const uint16_t* X,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,const int32_t* order,const int32_t* counts,
 int E,int experts,int topk,int tokens,int caprows,int group,int gu_kind,int down_kind,int gu_bytes,int down_bytes,
 uint8_t* GUW,uint8_t* DW,sycl::half* Xh,sycl::half* Hh,sycl::half* GWh,sycl::half* DWh,float* GU,float* Dout,void* decode_fn){
 return sglang_prefill_f16_layer_packed_impl(stream,X,ids,slots,resident,mirrors,order,counts,E,experts,topk,tokens,caprows,group,gu_kind,down_kind,gu_bytes,down_bytes,GUW,DW,Xh,Hh,GWh,DWh,GU,Dout,decode_fn,false);
}

extern "C" int xpu_gguf_prefill_f16_layer_packed_gu_soa(void* stream,const uint16_t* X,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,const int32_t* order,const int32_t* counts,
 int E,int experts,int topk,int tokens,int caprows,int group,int gu_kind,int down_kind,int gu_bytes,int down_bytes,
 uint8_t* GUW,uint8_t* DW,sycl::half* Xh,sycl::half* Hh,sycl::half* GWh,sycl::half* DWh,float* GU,float* Dout,void* decode_fn){
 return sglang_prefill_f16_layer_packed_impl(stream,X,ids,slots,resident,mirrors,order,counts,E,experts,topk,tokens,caprows,group,gu_kind,down_kind,gu_bytes,down_bytes,GUW,DW,Xh,Hh,GWh,DWh,GU,Dout,decode_fn,true);
}
