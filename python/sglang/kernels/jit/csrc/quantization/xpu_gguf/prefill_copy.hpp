// SPDX-License-Identifier: MIT
// Copy-only queue ported from Strata 46c152a1310f77cd2f0ce2c7cd213962df14fe83,
// sycl/src/core/copy_queue.cpp. Copyright (c) 2026 Niko1221 and contributors.
// SGLang: use the caller's context/device and let SYCL own the native list.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <dlfcn.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <cstdio>
#include <stdexcept>
namespace sglang_gguf_copy {
inline sycl::queue& get(sycl::queue& main){
 thread_local std::unordered_map<void*,std::unique_ptr<sycl::queue>> queues;
 auto& ptr=queues[&main];if(ptr)return *ptr;
 void* handle=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_NOLOAD);if(!handle)handle=dlopen("libze_loader.so.1",RTLD_NOW);if(!handle)throw std::runtime_error("Level Zero loader unavailable");
 auto groups=reinterpret_cast<decltype(&zeDeviceGetCommandQueueGroupProperties)>(dlsym(handle,"zeDeviceGetCommandQueueGroupProperties"));
 auto create=reinterpret_cast<decltype(&zeCommandListCreateImmediate)>(dlsym(handle,"zeCommandListCreateImmediate"));
 auto destroy=reinterpret_cast<decltype(&zeCommandListDestroy)>(dlsym(handle,"zeCommandListDestroy"));
 if(!groups||!create||!destroy)throw std::runtime_error("Level Zero copy queue entrypoints unavailable");
 auto dev=main.get_device();auto ctx=main.get_context();
 if(dev.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::runtime_error("Expert copy engine requires Level Zero");
 auto zdev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);auto zctx=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
 uint32_t ng=0;if(groups(zdev,&ng,nullptr)!=ZE_RESULT_SUCCESS)throw std::runtime_error("Level Zero groups query failed");std::vector<ze_command_queue_group_properties_t> gs(ng);
 for(auto& g:gs){std::memset(&g,0,sizeof(g));g.stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;}
 if(groups(zdev,&ng,gs.data())!=ZE_RESULT_SUCCESS)throw std::runtime_error("Level Zero groups query failed");int ordinal=-1;
 for(uint32_t i=0;i<ng;++i)if((gs[i].flags&ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)&&!(gs[i].flags&ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)&&gs[i].numQueues){ordinal=i;break;}
 if(ordinal<0)throw std::runtime_error("No dedicated copy-only Level Zero engine");
 ze_command_queue_desc_t desc{};desc.stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;desc.ordinal=ordinal;desc.index=0;desc.flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER;desc.mode=ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;desc.priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
 ze_command_list_handle_t cl=nullptr;if(create(zctx,zdev,&desc,&cl)!=ZE_RESULT_SUCCESS||!cl)throw std::runtime_error("Copy-only immediate command list creation failed");
 sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero,sycl::queue> in(cl,dev,sycl::ext::oneapi::level_zero::ownership::transfer,sycl::property_list{sycl::property::queue::in_order{}});
 auto import_queue=[&](){try{return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(in,ctx);}catch(...){destroy(cl);throw;}};
 auto owned=import_queue();ptr=std::make_unique<sycl::queue>(std::move(owned));

 std::fprintf(stderr,"SGLang expert prefill: dedicated COPY-only Level Zero group=%d; caller context/device; native list owned by SYCL\n",ordinal);
 return *ptr;
}
struct Metadata {
 sycl::context context;uint64_t* addresses;
 Metadata(sycl::queue& q):context(q.get_context()),addresses(sycl::malloc_host<uint64_t>(512*3,q)){if(!addresses)throw std::bad_alloc();}
 ~Metadata(){sycl::free(addresses,context);}
};
inline Metadata& meta(sycl::queue& q){thread_local std::unordered_map<void*,std::unique_ptr<Metadata>> entries;auto& p=entries[&q];if(!p)p=std::make_unique<Metadata>(q);return *p;}
class Resolve;
inline uint64_t* resolve(sycl::queue& q,const int32_t* ids,const int32_t* slots,const uint64_t* resident,const uint64_t* mirrors,int E,int experts){
 auto* out=meta(q).addresses;
 q.parallel_for<Resolve>(sycl::range<1>(size_t(E)*3),[=](sycl::id<1> i){int e=i[0]/3,f=i[0]%3,id=ids[e];uint64_t addr=0;if(id>=0&&id<experts){int slot=slots[id];addr=slot>=0&&slot<experts?resident[size_t(slot)*3+f]:mirrors[size_t(id)*3+f];}out[i[0]]=addr;}).wait_and_throw();
 for(int i=0;i<E*3;++i)if(!out[i])throw std::runtime_error("Missing original expert owner address");return out;
}
}
