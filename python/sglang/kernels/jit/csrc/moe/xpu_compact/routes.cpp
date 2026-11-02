// Experimental B70 DFlash M4 route preparation. SPDX-License-Identifier: BSD-3-Clause
#include "dflash_window.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <exception>
#include <stdexcept>
namespace dflash_routes01 {
constexpr int Routes = DFLASH_NATIVE_WINDOW * 8;
thread_local std::string error;
template<int WG> class Prepare;
template<int WG> void launch(sycl::queue& queue,const int* top,int* sorted,int* cmap,int* ids,int* counts,int* offsets,int* hostids,const int* dst,const int* src,const int* count,const uint64_t* hp,const uint64_t* gp){
 namespace one=sycl::ext::oneapi::experimental;
 one::properties props{one::sub_group_size<16>};
 queue.submit([&](sycl::handler& h){
  sycl::local_accessor<int,1> keys(sycl::range<1>(WG),h);
  h.parallel_for<Prepare<WG>>(sycl::nd_range<1>(sycl::range<1>(WG),sycl::range<1>(WG)),props,[=](sycl::nd_item<1> item){
   int t=item.get_local_linear_id();auto sg=item.get_sub_group();
   int key=t<Routes?(top[t]*Routes+t):(32768+t);
   for(int size=2;size<=Routes;size*=2){
    for(int distance=size/2;distance>0;distance/=2){
     int other;
     if(distance<16)other=sycl::permute_group_by_xor(sg,key,distance);
     else {keys[t]=key;item.barrier(sycl::access::fence_space::local_space);other=keys[t^distance];item.barrier(sycl::access::fence_space::local_space);}
     bool minimum=((t&size)==0)==((t&distance)==0);
     key=minimum?sycl::min(key,other):sycl::max(key,other);
    }
   }
   keys[t]=key;item.barrier(sycl::access::fence_space::local_space);
   int slot=key/Routes,route=key%Routes;
   bool first=t<Routes && (t==0 || slot!=keys[t-1]/Routes);
   int rank=sycl::exclusive_scan_over_group(item.get_group(),int(first),0,sycl::plus<int>());
   int active=sycl::reduce_over_group(item.get_group(),int(first),sycl::plus<int>());
   int misses=count[0],host=-1,zero_host=-1;
   for(int p=0;p<misses;++p){int d=dst[p],s=src[p];if(d==slot)host=s;if(d==0)zero_host=s;}
   if(t<Routes){sorted[t]=route;cmap[route]=t;}
   if(t<Routes && t>=active){ids[t]=0;counts[t]=0;offsets[t]=0;hostids[t]=zero_host;}
   if(first){int length=1;while(t+length<Routes && keys[t+length]/Routes==slot)++length;ids[rank]=slot;counts[rank]=length;offsets[rank]=t;hostids[rank]=host;}
   for(int plane=0;plane<2;++plane){
    int row=plane==0?128:64;auto* input=reinterpret_cast<const uint32_t*>(hp[2+plane]);auto* output=reinterpret_cast<uint32_t*>(gp[2+plane]);
    for(int position=t;position<misses*row;position+=WG){int p=position/row,inside=position%row;output[dst[p]*row+inside]=input[src[p]*row+inside];}
   }
  });
 });
}
}
extern "C" int dflash_routes_abi(){return 1;}
extern "C" const char* dflash_routes_error(){return dflash_routes01::error.c_str();}
extern "C" int dflash_routes_prepare(void* q,const int* top,int* sorted,int* cmap,int* ids,int* counts,int* offsets,int* hostids,const int* dst,const int* src,const int* count,const uint64_t* hp,const uint64_t* gp,int wg){
 try{
  auto& queue=*static_cast<sycl::queue*>(q);
  if(wg<dflash_routes01::Routes)throw std::runtime_error("Route workgroup is smaller than route count");
  switch(wg){
   case 32:dflash_routes01::launch<32>(queue,top,sorted,cmap,ids,counts,offsets,hostids,dst,src,count,hp,gp);break;
   case 64:dflash_routes01::launch<64>(queue,top,sorted,cmap,ids,counts,offsets,hostids,dst,src,count,hp,gp);break;
   case 128:dflash_routes01::launch<128>(queue,top,sorted,cmap,ids,counts,offsets,hostids,dst,src,count,hp,gp);break;
   default:throw std::runtime_error("Unsupported native route workgroup size");
  }return 0;
 }catch(const std::exception& e){dflash_routes01::error=e.what();return 1;}
}
