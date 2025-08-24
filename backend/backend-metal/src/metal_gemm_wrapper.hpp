#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_gemm.hpp"
#include <Metal/Metal.h>

/**
 * @brief C++ wrapper for Metal GEMM operations
 * 
 * Provides a clean C++ interface that matches the CUDA backend API
 * and integrates with the Metal infrastructure components.
 */
namespace MetalGEMM {
    
    /**
     * @brief Initialize Metal GEMM subsystem
     * Uses the shared MetalContext for device management
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup Metal GEMM resources
     */
    void cleanup();
    
    /**
     * @brief Check if GEMM is initialized
     */
    bool is_initialized();
    
    /**
     * @brief Template wrapper for GEMM operation matching CUDA API
     * 
     * Template interface that matches gemm_cublasLt<T> from CUDA backend.
     * Automatically handles type dispatch and Metal buffer management.
     * 
     * @param commandBuffer Metal command buffer for GPU operations
     * @param A Input matrix A
     * @param B Input matrix B  
     * @param bias Optional bias vector (can be nullptr)
     * @param C Output matrix C
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A / rows in B
     * @param workspace Workspace buffer (for API compatibility)
     * @param workspace_size Workspace size (for API compatibility)
     * @param transa Whether matrix A is transposed
     * @param transb Whether matrix B is transposed
     */
    template<typename T>
    void gemm(id<MTLCommandBuffer> commandBuffer,
              const T* A, const T* B, const T* bias, T* C,
              int m, int n, int k,
              void* workspace, size_t workspace_size,
              bool transa = false, bool transb = false);
    
    /**
     * @brief Tensor-based GEMM operation
     * 
     * Higher-level interface using MetalTensor objects.
     * Handles memory management and provides type safety.
     * 
     * @param A Input tensor A
     * @param B Input tensor B
     * @param C Output tensor C (must be pre-allocated)
     * @param bias Optional bias tensor
     * @param transa Whether matrix A is transposed
     * @param transb Whether matrix B is transposed
     */
    template<typename T>
    void gemm_tensor(const MetalTensor<T>& A, 
                     const MetalTensor<T>& B, 
                     MetalTensor<T>& C,
                     const MetalTensor<T>* bias = nullptr,
                     bool transa = false, 
                     bool transb = false);
    
    /**
     * @brief Batched GEMM operation
     * 
     * Performs multiple GEMM operations in a single call.
     * Useful for transformer attention computations.
     * 
     * @param batch_count Number of matrices in batch
     * @param A_array Array of input matrices A
     * @param B_array Array of input matrices B
     * @param C_array Array of output matrices C
     * @param m Array of row counts for each matrix
     * @param n Array of column counts for each matrix
     * @param k Array of inner dimensions for each matrix
     * @param transa Whether matrices A are transposed
     * @param transb Whether matrices B are transposed
     */
    template<typename T>
    void gemm_batched(int batch_count,
                      const T* const* A_array,
                      const T* const* B_array,
                      T* const* C_array,
                      const int* m, const int* n, const int* k,
                      bool transa = false, bool transb = false);
}

// Template implementations

template<typename T>
void MetalGEMM::gemm(id<MTLCommandBuffer> commandBuffer,
                     const T* A, const T* B, const T* bias, T* C,
                     int m, int n, int k,
                     void* workspace, size_t workspace_size,
                     bool transa, bool transb) {
    
    // Ensure initialization
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize Metal GEMM");
    }
    
    // Type dispatch - only support the types we have kernels for
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        auto& context = MetalContext::getInstance();
        metal_gemm_bfloat16(context.getDevice(), 
                           context.getCommandQueue(),
                           A, B, bias, C, m, n, k,
                           workspace, workspace_size, 
                           transa, transb);
    } else if constexpr (std::is_same_v<T, float>) {
        // For float, we could convert to bfloat16 or implement a separate kernel
        throw std::runtime_error("Float32 GEMM not implemented yet - use bfloat16");
    } else {
        throw std::runtime_error("Unsupported type for Metal GEMM");
    }
}

template<typename T>
void MetalGEMM::gemm_tensor(const MetalTensor<T>& A, 
                            const MetalTensor<T>& B, 
                            MetalTensor<T>& C,
                            const MetalTensor<T>* bias,
                            bool transa, bool transb) {
    
    // Validate tensor shapes
    const auto& A_shape = A.shape();
    const auto& B_shape = B.shape();
    const auto& C_shape = C.shape();
    
    if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
        throw std::runtime_error("GEMM tensors must be 2D");
    }
    
    // Extract dimensions
    int m = A_shape[0];
    int k = A_shape[1];
    int n = B_shape[1];
    
    if (B_shape[0] != k) {
        throw std::runtime_error("Matrix dimension mismatch for GEMM");
    }
    
    if (C_shape[0] != m || C_shape[1] != n) {
        throw std::runtime_error("Output tensor shape mismatch for GEMM");
    }
    
    // Validate bias if provided
    const T* bias_ptr = nullptr;
    if (bias != nullptr) {
        const auto& bias_shape = bias->shape();
        if (bias_shape.size() != 1 || bias_shape[0] != n) {
            throw std::runtime_error("Bias tensor must be 1D with size n");
        }
        bias_ptr = bias->data();
    }
    
    // Create command buffer
    auto& context = MetalContext::getInstance();
    id<MTLCommandBuffer> commandBuffer = [context.getCommandQueue() commandBuffer];
    
    // Call the GEMM operation
    gemm<T>(commandBuffer, A.data(), B.data(), bias_ptr, C.data(),
            m, n, k, nullptr, 0, transa, transb);
    
    // Wait for completion
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

template<typename T>
void MetalGEMM::gemm_batched(int batch_count,
                             const T* const* A_array,
                             const T* const* B_array,
                             T* const* C_array,
                             const int* m, const int* n, const int* k,
                             bool transa, bool transb) {
    
    // For now, implement as a loop of individual GEMM operations
    // A more efficient implementation would use a single kernel
    auto& context = MetalContext::getInstance();
    
    for (int i = 0; i < batch_count; ++i) {
        id<MTLCommandBuffer> commandBuffer = [context.getCommandQueue() commandBuffer];
        
        gemm<T>(commandBuffer, A_array[i], B_array[i], nullptr, C_array[i],
                m[i], n[i], k[i], nullptr, 0, transa, transb);
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}