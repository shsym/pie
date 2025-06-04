#include "l4ma.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdexcept>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/activation.cuh>

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(status)                                                    \
    do                                                                          \
    {                                                                           \
        cublasStatus_t _status = (status);                                      \
        if (_status != CUBLAS_STATUS_SUCCESS)                                   \
        {                                                                       \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, _status); \
            throw std::runtime_error("cuBLAS error");                           \
        }                                                                       \
    } while (0)

// Helper for SiLU activation (device function)
__device__ __forceinline__ float silu(const float &val) { return val / (1.0f + __expf(-val)); }

// SiLU and elementwise multiply kernel launcher
template <typename T>
void silu_and_mul(
    thrust::device_vector<T> &out,
    const thrust::device_vector<T> &input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl)
{
    T *out_ptr = thrust::raw_pointer_cast(out.data());
    const T *input_ptr = thrust::raw_pointer_cast(input.data());
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    cudaLaunchKernelEx(&config, kernel, out_ptr, input_ptr, d);
}

// Explicit template instantiation for float

template void silu_and_mul<float>(
    thrust::device_vector<float> &out,
    const thrust::device_vector<float> &input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl);

// Remove gemm_bias_cublasLt definition from this file, as it is now in the header.