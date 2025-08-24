#import "metal_rmsnorm_wrapper.hpp"
#import "metal_rmsnorm.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

namespace MetalRMSNorm {
    
    static bool initialized_ = false;
    
    bool initialize() {
        if (initialized_) {
            return true;
        }
        
        // Initialize the Metal context first
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context for RMSNorm" << std::endl;
            return false;
        }
        
        // For now, we don't need additional initialization beyond the context
        // The metal_rmsnorm kernels should be loaded automatically
        
        initialized_ = true;
        std::cout << "Metal RMSNorm wrapper initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (initialized_) {
            initialized_ = false;
            std::cout << "Metal RMSNorm wrapper cleaned up" << std::endl;
        }
    }
    
    bool is_initialized() {
        return initialized_;
    }
    
    // Check if initialized before RMSNorm operations
    static void ensure_initialized() {
        if (!initialized_) {
            if (!initialize()) {
                throw std::runtime_error("Failed to initialize Metal RMSNorm");
            }
        }
    }
    
} // namespace MetalRMSNorm

// Explicit template instantiations for supported types
namespace MetalRMSNorm {
    
    // Template instantiations for rmsnorm
    template void rmsnorm<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, 
                                      uint32_t, uint32_t, float, id<MTLCommandBuffer>);
    
    template void rmsnorm<float>(const float*, const float*, float*, 
                                uint32_t, uint32_t, float, id<MTLCommandBuffer>);
    
    // Template instantiations for rmsnorm_tensor
    template void rmsnorm_tensor<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, 
                                             MetalTensor<bfloat16_t>&, float);
    
    template void rmsnorm_tensor<float>(const MetalTensor<float>&, const MetalTensor<float>&, 
                                        MetalTensor<float>&, float);
    
    // Template instantiations for rmsnorm_inplace
    template void rmsnorm_inplace<bfloat16_t>(MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, float);
    template void rmsnorm_inplace<float>(MetalTensor<float>&, const MetalTensor<float>&, float);
    
    // Template instantiations for rmsnorm_batched
    template void rmsnorm_batched<bfloat16_t>(int, const bfloat16_t* const*, const bfloat16_t*, 
                                              bfloat16_t* const*, const uint32_t*, uint32_t, float);
    
    template void rmsnorm_batched<float>(int, const float* const*, const float*, 
                                         float* const*, const uint32_t*, uint32_t, float);
    
    // Template instantiations for get_workspace_size
    template size_t get_workspace_size<bfloat16_t>(uint32_t, uint32_t);
    template size_t get_workspace_size<float>(uint32_t, uint32_t);
}

// Template specializations are handled in the header

// Convenience function to match FlashInfer API exactly
template<typename T>
void flashinfer_rmsnorm_metal(const T* input,
                              const T* weight,
                              T* output,
                              uint32_t num_tokens,
                              uint32_t hidden_size,
                              float eps) {
    MetalRMSNorm::rmsnorm<T>(input, weight, output, num_tokens, hidden_size, eps);
}

// Explicit instantiation of the FlashInfer API compatibility function
template void flashinfer_rmsnorm_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, 
                                                    uint32_t, uint32_t, float);
template void flashinfer_rmsnorm_metal<float>(const float*, const float*, float*, 
                                               uint32_t, uint32_t, float);