#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "flashinfer/sampling.cuh"
#include "../ops/ops_common.cuh"

// Test-local multi-dtype OnlineSoftmax wrapper
// Uses the same FlashInfer function but with different dtypes than backend

template<typename T>
cudaError_t online_softmax_test_local(T* logits, T* output, uint32_t batch_size, uint32_t vocab_size,
                                     T* temperature_arr, T temperature_val, void* workspace_buffer,
                                     size_t workspace_buffer_size, bool enable_pdl, cudaStream_t stream) {
    
    // Call FlashInfer OnlineSoftmax with the target dtype
    return flashinfer::sampling::OnlineSoftmax<T>(
        logits,
        output,
        batch_size,
        vocab_size,
        temperature_arr,
        temperature_val,
        workspace_buffer,
        workspace_buffer_size,
        enable_pdl,
        stream
    );
}

// Note: FlashInfer OnlineSoftmax has internal float assumptions
// Only instantiate float for now - __half and __nv_bfloat16 have compilation issues
template cudaError_t online_softmax_test_local<float>(float* logits, float* output, 
    uint32_t batch_size, uint32_t vocab_size, float* temperature_arr, float temperature_val,
    void* workspace_buffer, size_t workspace_buffer_size, bool enable_pdl, cudaStream_t stream);