#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "flashinfer/sampling.cuh"
#include "../ops/ops_common.cuh"

// Test-local multi-dtype TopKMaskLogits wrapper
// Uses the same FlashInfer function but with different dtypes than backend

template<typename T>
cudaError_t topk_mask_logits_test_local(T* logits, T* masked_logits, int32_t* top_k_arr,
                                       uint32_t batch_size, uint32_t top_k_val, uint32_t vocab_size,
                                       cudaStream_t stream) {
    
    // Call FlashInfer TopKMaskLogits with the target dtype
    // Uses the same signature as backend: TopKMaskLogits<T, int32_t>
    return flashinfer::sampling::TopKMaskLogits<T, int32_t>(
        logits,
        masked_logits,
        top_k_arr,
        batch_size,
        top_k_val,
        vocab_size,
        stream
    );
}

// Note: FlashInfer TopKMaskLogits has internal float assumptions
// Only instantiate float for now - __half and __nv_bfloat16 have compilation issues
template cudaError_t topk_mask_logits_test_local<float>(float* logits, float* masked_logits, 
    int32_t* top_k_arr, uint32_t batch_size, uint32_t top_k_val, uint32_t vocab_size, cudaStream_t stream);