// SPDX-License-Identifier: Apache-2.0
// Opt-in B70 GDN gated RMSNorm128 + block128 FP8 quantization.
// Match installed Triton blocked[1,8]/threads[2,16] reduction and BF16 boundary.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
SYCL_EXTERNAL __attribute__((const, convergent)) float __spirv_GroupNonUniformFAdd(int, int, float, unsigned) noexcept;
SYCL_EXTERNAL __attribute__((const, convergent)) float __spirv_GroupNonUniformFMax(int, int, float, unsigned) noexcept;
namespace tp_gdn_norm_quant_impl {
thread_local std::string error;
inline float bf16(uint16_t x) {
  return sycl::bit_cast<float>(uint32_t(x) << 16);
}
inline uint16_t rounded(float x) {
  auto b = sycl::bit_cast<uint32_t>(x);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
inline float sum16(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFAdd(3, 3, x, 16u);
#else
  return x;
#endif
}
inline float max16(float x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformFMax(3, 3, x, 16u);
#else
  return x;
#endif
}
inline uint8_t encode(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x), mag = bits & 0x7fffffffu;
  if (mag > 0x7f800000u) return 127;
  uint32_t sign = (bits >> 24) & 128u, value;
  if (mag < 0x3c800000u)
    value = uint32_t(sycl::rint(sycl::fabs(x) * 512.f));
  else {
    uint32_t exp = mag >> 23, mant = mag & 0x7fffffu;
    value = ((exp > 120 ? exp - 120 : 0) << 3) + uint32_t(sycl::rint(float(mant) * 0x1p-20f));
  }
  return uint8_t(sign | sycl::min(value, 126u));
}

template<bool Debug>class KernelAR;
template<bool Debug>void run(sycl::queue& stream,const uint16_t*x,const uint16_t*z,const uint16_t*w,
 uint8_t*q,float*s,uint16_t*y,float*rstd,float eps,float limit){
 namespace one=sycl::ext::oneapi::experimental;namespace intel=sycl::ext::intel::experimental;
 one::properties props{one::sub_group_size<32>,intel::grf_size<256>};
 stream.submit([&](sycl::handler&h){h.parallel_for<KernelAR<Debug>>(
  sycl::nd_range<1>(16*32,32),props,[=](sycl::nd_item<1>it){
   unsigned row=it.get_group_linear_id(),lane=it.get_local_linear_id();
   sycl::vec<uint16_t,4>rawx,rawz,raww;rawx.load(0,x+row*128+lane*4);
   rawz.load(0,z+row*128+lane*4);raww.load(0,w+lane*4);
   float a[4],squares[4];
#pragma unroll
   for(int i=0;i<4;++i){a[i]=bf16(rawx[i]);squares[i]=a[i]*a[i];}
   float local=(squares[0]+squares[1])+(squares[2]+squares[3]);
   float reduced=sycl::reduce_over_group(it.get_sub_group(),local,sycl::plus<float>());
   float invstd=sycl::rsqrt(reduced*0x1p-7f+eps);
   float values[4],maximum=0.f;sycl::vec<uint16_t,4>norm;
#pragma unroll
   for(int i=0;i<4;++i){float gate=bf16(rawz[i]);
    float e=sycl::exp2(gate*sycl::bit_cast<float>(0xbfb8aa3bu));
    float sigmoid=1.f/(1.f+e);
    float value=((a[i]*invstd)*bf16(raww[i]))*(gate*sigmoid);
    norm[i]=rounded(value);values[i]=bf16(norm[i]);maximum=sycl::fmax(maximum,sycl::fabs(values[i]));
   }
   float scale=sycl::fmax(sycl::reduce_over_group(it.get_sub_group(),maximum,sycl::maximum<float>()),1e-10f)/limit;
   float inv=1.f/scale;sycl::vec<uint8_t,4>quant;
#pragma unroll
   for(int i=0;i<4;++i)quant[i]=encode(sycl::fmin(sycl::fmax(values[i]*inv,-limit),limit));
   quant.store(0,q+row*128+lane*4);if(lane==0)s[row]=scale;
   if constexpr(Debug){norm.store(0,y+row*128+lane*4);if(lane==0)rstd[row]=invstd;}
  });});
}
}
extern "C" int tp_gdn_norm_quant_abi(){return 1;}
extern "C" const char*tp_gdn_norm_quant_error(){return tp_gdn_norm_quant_impl::error.c_str();}
extern "C" int tp_gdn_norm_quant_run(void*queue,const uint16_t*x,const uint16_t*z,const uint16_t*w,
 uint8_t*q,float*s,uint16_t*y,float*rstd,int rows,float eps,int debug,float limit){
 if(!queue||!x||!z||!w||!q||!s||rows!=16||eps<=0||limit!=448.f||(debug!=0&&debug!=1)||(debug&&(!y||!rstd)))return 2;
 try{auto&stream=*static_cast<sycl::queue*>(queue);
  if(debug)tp_gdn_norm_quant_impl::run<true>(stream,x,z,w,q,s,y,rstd,eps,limit);
  else tp_gdn_norm_quant_impl::run<false>(stream,x,z,w,q,s,y,rstd,eps,limit);
  return 0;
 }catch(const std::exception&e){tp_gdn_norm_quant_impl::error=e.what();return 1;}
}
