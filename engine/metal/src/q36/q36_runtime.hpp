#pragma once
#include "strata/gpu/metal.hpp"
#include "../poly/launch.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
// dim3: metal_compat's cuda_runtime.h
template<class A> void q36_dispatch(const char* name,const A& a,dim3 grid,dim3 block,void* stream) {
    using namespace strata;
    static const auto source=poly::metal_detail::read_source(STRATA_Q36_METAL_SOURCE);
    std::string error;auto* pipe=gpu::metal::pipeline("q36",source.c_str(),name,error);
    if(!pipe)throw std::runtime_error(error);
    auto rc=gpu::metal::dispatch(static_cast<gpu::Stream>(stream),pipe,&a,sizeof(a),{grid.x,grid.y,grid.z},{block.x,block.y,block.z});
    if(rc!=gpu::kSuccess)throw std::runtime_error(gpu::error_string(rc));
}
