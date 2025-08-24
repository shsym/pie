#import "metal_batch_prefill_attention_wrapper.hpp"
#import "metal_batch_prefill_attention.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

namespace MetalBatchPrefillAttention {
    
    static bool initialized_ = false;
    
    bool initialize() {
        if (initialized_) {
            return true;
        }
        
        // Initialize the Metal context first
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context for batch prefill attention" << std::endl;
            return false;
        }
        
        // For now, we don't need additional initialization beyond the context
        // The metal_batch_prefill_attention kernels should be loaded automatically
        
        initialized_ = true;
        std::cout << "Metal batch prefill attention wrapper initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (initialized_) {
            initialized_ = false;
            std::cout << "Metal batch prefill attention wrapper cleaned up" << std::endl;
        }
    }
    
    bool is_initialized() {
        return initialized_;
    }
    
    // Check if initialized before attention operations
    static void ensure_initialized() {
        if (!initialized_) {
            if (!initialize()) {
                throw std::runtime_error("Failed to initialize Metal batch prefill attention");
            }
        }
    }
    
} // namespace MetalBatchPrefillAttention

// Explicit template instantiations for supported types
namespace MetalBatchPrefillAttention {
    
    // Template instantiations for batch_prefill_attention_unified
    template void batch_prefill_attention_unified<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*,
                                                               const int32_t*, const int32_t*, const int32_t*, 
                                                               const int32_t*, bfloat16_t*, int, int, int, int, 
                                                               float, int, id<MTLCommandBuffer>);
    
    template void batch_prefill_attention_unified<float>(const float*, const float*, const float*,
                                                          const int32_t*, const int32_t*, const int32_t*, 
                                                          const int32_t*, float*, int, int, int, int, 
                                                          float, int, id<MTLCommandBuffer>);
    
    // Template instantiations for batch_prefill_attention_tensor
    template void batch_prefill_attention_tensor<bfloat16_t>(const MetalTensor<bfloat16_t>&, 
                                                              const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&,
                                                              const MetalTensor<int32_t>&, const MetalTensor<int32_t>&,
                                                              const MetalTensor<int32_t>&, const MetalTensor<int32_t>&,
                                                              MetalTensor<bfloat16_t>&, int, int, float);
    
    template void batch_prefill_attention_tensor<float>(const MetalTensor<float>&, 
                                                         const MetalTensor<float>&, const MetalTensor<float>&,
                                                         const MetalTensor<int32_t>&, const MetalTensor<int32_t>&,
                                                         const MetalTensor<int32_t>&, const MetalTensor<int32_t>&,
                                                         MetalTensor<float>&, int, int, float);
    
    // Template instantiations for get_workspace_size
    template size_t get_workspace_size<bfloat16_t>(int, int, int);
    template size_t get_workspace_size<float>(int, int, int);
}

// Template specializations are handled in the header

// Convenience function to match FlashInfer API exactly
template<typename T>
void flashinfer_batch_prefill_attention_metal(const T* q_input,
                                               const T* paged_k_cache,
                                               const T* paged_v_cache,
                                               const int32_t* qo_indptr,
                                               const int32_t* kv_page_indptr,
                                               const int32_t* kv_page_indices,
                                               const int32_t* kv_last_page_lens,
                                               T* output,
                                               int num_qo,
                                               int head_dim,
                                               int head_size,
                                               int page_size,
                                               float scale,
                                               int num_kv_pages) {
    MetalBatchPrefillAttention::batch_prefill_attention_unified<T>(
        q_input, paged_k_cache, paged_v_cache,
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        output, num_qo, head_dim, head_size, page_size, scale, num_kv_pages);
}

// Explicit instantiation of the FlashInfer API compatibility function
template void flashinfer_batch_prefill_attention_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*,
                                                                   const int32_t*, const int32_t*, const int32_t*, 
                                                                   const int32_t*, bfloat16_t*, int, int, int, int, 
                                                                   float, int);

template void flashinfer_batch_prefill_attention_metal<float>(const float*, const float*, const float*,
                                                              const int32_t*, const int32_t*, const int32_t*, 
                                                              const int32_t*, float*, int, int, int, int, 
                                                              float, int);