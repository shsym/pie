#import "metal_gemm_wrapper.hpp"
#import "metal_gemm.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

namespace MetalGEMM {
    
    static bool initialized_ = false;
    
    bool initialize() {
        if (initialized_) {
            return true;
        }
        
        // Initialize the Metal context first
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context for GEMM" << std::endl;
            return false;
        }
        
        // Initialize the Metal GEMM kernel
        if (!initialize_metal_gemm()) {
            std::cerr << "Failed to initialize Metal GEMM kernels" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "Metal GEMM wrapper initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (initialized_) {
            cleanup_metal_gemm();
            initialized_ = false;
            std::cout << "Metal GEMM wrapper cleaned up" << std::endl;
        }
    }
    
    bool is_initialized() {
        return initialized_;
    }
    
    // Check if initialized before GEMM operations
    static void ensure_initialized() {
        if (!initialized_) {
            if (!initialize()) {
                throw std::runtime_error("Failed to initialize Metal GEMM");
            }
        }
    }
    
} // namespace MetalGEMM

// Explicit template instantiations for supported types
namespace MetalGEMM {
    
    // Template instantiations for gemm
    template void gemm<bfloat16_t>(id<MTLCommandBuffer>, const bfloat16_t*, const bfloat16_t*, 
                                   const bfloat16_t*, bfloat16_t*, int, int, int, void*, size_t, bool, bool);
    
    template void gemm<float>(id<MTLCommandBuffer>, const float*, const float*, 
                             const float*, float*, int, int, int, void*, size_t, bool, bool);
    
    // Template instantiations for gemm_tensor
    template void gemm_tensor<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, 
                                          MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>*, bool, bool);
    
    template void gemm_tensor<float>(const MetalTensor<float>&, const MetalTensor<float>&, 
                                     MetalTensor<float>&, const MetalTensor<float>*, bool, bool);
    
    // Template instantiations for gemm_batched
    template void gemm_batched<bfloat16_t>(int, const bfloat16_t* const*, const bfloat16_t* const*, 
                                           bfloat16_t* const*, const int*, const int*, const int*, bool, bool);
    
    template void gemm_batched<float>(int, const float* const*, const float* const*, 
                                      float* const*, const int*, const int*, const int*, bool, bool);
}

// Implementation is in the header as template specialization

// Convenience function to match CUDA API exactly
template<typename T>
void gemm_cublasLt_metal(id<MTLCommandBuffer> commandBuffer,
                         const T* A, const T* B, const T* bias, T* C,
                         int m, int n, int k,
                         void* workspace, size_t workspace_size,
                         bool transa, bool transb) {
    MetalGEMM::gemm<T>(commandBuffer, A, B, bias, C, m, n, k, workspace, workspace_size, transa, transb);
}

// Explicit instantiation of the CUDA API compatibility function
template void gemm_cublasLt_metal<bfloat16_t>(id<MTLCommandBuffer>, const bfloat16_t*, const bfloat16_t*, 
                                               const bfloat16_t*, bfloat16_t*, int, int, int, void*, size_t, bool, bool);