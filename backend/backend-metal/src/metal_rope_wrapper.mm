#import "metal_rope_wrapper.hpp"
#import "metal_rope.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

namespace MetalRoPE {
    
    static bool initialized_ = false;
    
    bool initialize() {
        if (initialized_) {
            return true;
        }
        
        // Initialize the Metal context first
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context for RoPE" << std::endl;
            return false;
        }
        
        // For now, we don't need additional initialization beyond the context
        // The metal_rope kernels should be loaded automatically
        
        initialized_ = true;
        std::cout << "Metal RoPE wrapper initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (initialized_) {
            initialized_ = false;
            std::cout << "Metal RoPE wrapper cleaned up" << std::endl;
        }
    }
    
    bool is_initialized() {
        return initialized_;
    }
    
    // Check if initialized before RoPE operations
    static void ensure_initialized() {
        if (!initialized_) {
            if (!initialize()) {
                throw std::runtime_error("Failed to initialize Metal RoPE");
            }
        }
    }
    
} // namespace MetalRoPE

// Explicit template instantiations for supported types
namespace MetalRoPE {
    
    // Template instantiations for rope_inplace
    template void rope_inplace<bfloat16_t>(bfloat16_t*, const int32_t*, uint32_t, uint32_t, uint32_t, 
                                           float, float, id<MTLCommandBuffer>);
    
    template void rope_inplace<float>(float*, const int32_t*, uint32_t, uint32_t, uint32_t, 
                                      float, float, id<MTLCommandBuffer>);
    
    // Template instantiations for rope_tensor_inplace
    template void rope_tensor_inplace<bfloat16_t>(MetalTensor<bfloat16_t>&, const MetalTensor<int32_t>&, 
                                                  float, float);
    
    template void rope_tensor_inplace<float>(MetalTensor<float>&, const MetalTensor<int32_t>&, 
                                             float, float);
    
    // Template instantiations for rope_batched_inplace
    template void rope_batched_inplace<bfloat16_t>(int, bfloat16_t* const*, const int32_t* const*, 
                                                   const uint32_t*, uint32_t, uint32_t, float, float);
    
    template void rope_batched_inplace<float>(int, float* const*, const int32_t* const*, 
                                              const uint32_t*, uint32_t, uint32_t, float, float);
}

// Template specializations are handled in the header

// Convenience function to match FlashInfer API exactly
template<typename T>
void flashinfer_rope_inplace_metal(T* input_qk,
                                   const int32_t* position_ids,
                                   uint32_t num_tokens,
                                   uint32_t num_heads,
                                   uint32_t head_size,
                                   float rope_theta,
                                   float rope_factor) {
    MetalRoPE::rope_inplace<T>(input_qk, position_ids, num_tokens, num_heads, head_size, rope_theta, rope_factor);
}

// Explicit instantiation of the FlashInfer API compatibility function
template void flashinfer_rope_inplace_metal<bfloat16_t>(bfloat16_t*, const int32_t*, uint32_t, uint32_t, uint32_t, 
                                                        float, float);
template void flashinfer_rope_inplace_metal<float>(float*, const int32_t*, uint32_t, uint32_t, uint32_t, 
                                                   float, float);