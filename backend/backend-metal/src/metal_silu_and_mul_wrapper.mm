#import "metal_silu_and_mul_wrapper.hpp"
#import "metal_silu_and_mul.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

namespace MetalSiLUMul {
    
    static bool initialized_ = false;
    
    bool initialize() {
        if (initialized_) {
            return true;
        }
        
        // Initialize the Metal context first
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context for SiLU and multiply" << std::endl;
            return false;
        }
        
        // For now, we don't need additional initialization beyond the context
        // The metal_silu_and_mul kernels should be loaded automatically
        
        initialized_ = true;
        std::cout << "Metal SiLU and multiply wrapper initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (initialized_) {
            initialized_ = false;
            std::cout << "Metal SiLU and multiply wrapper cleaned up" << std::endl;
        }
    }
    
    bool is_initialized() {
        return initialized_;
    }
    
    // Check if initialized before SiLU and multiply operations
    static void ensure_initialized() {
        if (!initialized_) {
            if (!initialize()) {
                throw std::runtime_error("Failed to initialize Metal SiLU and multiply");
            }
        }
    }
    
} // namespace MetalSiLUMul

// Explicit template instantiations for supported types
namespace MetalSiLUMul {
    
    // Template instantiations for silu_and_mul
    template void silu_and_mul<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, 
                                           uint32_t, uint32_t, id<MTLCommandBuffer>);
    
    template void silu_and_mul<float>(const float*, const float*, float*, 
                                      uint32_t, uint32_t, id<MTLCommandBuffer>);
    
    // Template instantiations for silu_and_mul_tensor
    template void silu_and_mul_tensor<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, 
                                                  MetalTensor<bfloat16_t>&);
    
    template void silu_and_mul_tensor<float>(const MetalTensor<float>&, const MetalTensor<float>&, 
                                             MetalTensor<float>&);
    
    // Template instantiations for silu_and_mul_inplace
    template void silu_and_mul_inplace<bfloat16_t>(MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&);
    template void silu_and_mul_inplace<float>(MetalTensor<float>&, const MetalTensor<float>&);
    
    // Template instantiations for silu_and_mul_batched
    template void silu_and_mul_batched<bfloat16_t>(int, const bfloat16_t* const*, const bfloat16_t* const*, 
                                                   bfloat16_t* const*, const uint32_t*, uint32_t);
    
    template void silu_and_mul_batched<float>(int, const float* const*, const float* const*, 
                                              float* const*, const uint32_t*, uint32_t);
}

// Template specializations are handled in the header

// Convenience function to match common transformer API
template<typename T>
void transformer_silu_mul_metal(const T* gate,
                                 const T* up,
                                 T* output,
                                 uint32_t num_tokens,
                                 uint32_t intermediate_size) {
    MetalSiLUMul::silu_and_mul<T>(gate, up, output, num_tokens, intermediate_size);
}

// Explicit instantiation of the transformer API compatibility function
template void transformer_silu_mul_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, 
                                                     uint32_t, uint32_t);
template void transformer_silu_mul_metal<float>(const float*, const float*, float*, 
                                                uint32_t, uint32_t);