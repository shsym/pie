#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace ops {

inline void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

// Host-side float -> target type converter to safely initialize __half/bf16 buffers
template <typename T>
inline T f2t(float v) { return static_cast<T>(v); }

template <>
inline __half f2t<__half>(float v) { return __float2half(v); }

template <>
inline __nv_bfloat16 f2t<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

} // namespace ops
