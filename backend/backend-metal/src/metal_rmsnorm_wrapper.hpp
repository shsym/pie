#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_rmsnorm.hpp"
#include <Metal/Metal.h>

/**
 * @brief C++ wrapper for Metal RMSNorm operations
 * 
 * Provides a clean C++ interface that matches the FlashInfer API
 * and integrates with the Metal infrastructure components.
 */
namespace MetalRMSNorm {
    
    /**
     * @brief Initialize Metal RMSNorm subsystem
     * Uses the shared MetalContext for device management
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup Metal RMSNorm resources
     */
    void cleanup();
    
    /**
     * @brief Check if RMSNorm is initialized
     */
    bool is_initialized();
    
    /**
     * @brief Template wrapper for RMSNorm operation matching FlashInfer API
     * 
     * Performs RMS normalization: output = input * rsqrt(mean(input^2) + eps) * weight
     * 
     * @param input Input tensor [num_tokens, hidden_size]
     * @param weight Scale weights [hidden_size]  
     * @param output Output tensor [num_tokens, hidden_size]
     * @param num_tokens Number of tokens (sequence length)
     * @param hidden_size Hidden dimension size
     * @param eps Epsilon for numerical stability
     * @param commandBuffer Metal command buffer for GPU operations
     */
    template<typename T>
    void rmsnorm(const T* input,
                 const T* weight,
                 T* output,
                 uint32_t num_tokens,
                 uint32_t hidden_size,
                 float eps,
                 id<MTLCommandBuffer> commandBuffer = nil);
    
    /**
     * @brief Tensor-based RMSNorm operation
     * 
     * Higher-level interface using MetalTensor objects.
     * Handles memory management and provides type safety.
     * 
     * @param input Input tensor [num_tokens, hidden_size]
     * @param weight Weight tensor [hidden_size]
     * @param output Output tensor [num_tokens, hidden_size] (must be pre-allocated)
     * @param eps Epsilon for numerical stability (default: 1e-5)
     */
    template<typename T>
    void rmsnorm_tensor(const MetalTensor<T>& input,
                        const MetalTensor<T>& weight,
                        MetalTensor<T>& output,
                        float eps = 1e-5f);
    
    /**
     * @brief In-place RMSNorm operation
     * 
     * Applies RMSNorm directly to the input tensor, modifying it in place.
     * 
     * @param input_output Input/output tensor [num_tokens, hidden_size]
     * @param weight Weight tensor [hidden_size]
     * @param eps Epsilon for numerical stability
     */
    template<typename T>
    void rmsnorm_inplace(MetalTensor<T>& input_output,
                         const MetalTensor<T>& weight,
                         float eps = 1e-5f);
    
    /**
     * @brief Batched RMSNorm operation
     * 
     * Performs RMSNorm on multiple sequences in a single call.
     * Useful for batch processing in transformer models.
     * 
     * @param batch_count Number of sequences in batch
     * @param input_array Array of input tensors
     * @param weight Weight tensor (shared across batch)
     * @param output_array Array of output tensors
     * @param num_tokens Array of token counts for each sequence
     * @param hidden_size Hidden dimension size (same for all)
     * @param eps Epsilon for numerical stability
     */
    template<typename T>
    void rmsnorm_batched(int batch_count,
                         const T* const* input_array,
                         const T* weight,
                         T* const* output_array,
                         const uint32_t* num_tokens,
                         uint32_t hidden_size,
                         float eps = 1e-5f);
    
    /**
     * @brief Get workspace size required for RMSNorm operation
     * 
     * @param num_tokens Maximum number of tokens
     * @param hidden_size Hidden dimension size
     * @return Required workspace size in bytes
     */
    template<typename T>
    size_t get_workspace_size(uint32_t num_tokens, uint32_t hidden_size);
}

// Template implementations

template<typename T>
void MetalRMSNorm::rmsnorm(const T* input,
                           const T* weight,
                           T* output,
                           uint32_t num_tokens,
                           uint32_t hidden_size,
                           float eps,
                           id<MTLCommandBuffer> commandBuffer) {
    
    // Ensure initialization
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize Metal RMSNorm");
    }
    
    // Type dispatch to the appropriate Metal kernel
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        int result = metal_rmsnorm_bfloat16(input, weight, output, num_tokens, hidden_size, eps);
        if (result != 0) {
            throw std::runtime_error("Metal RMSNorm bfloat16 operation failed");
        }
    } else if constexpr (std::is_same_v<T, float>) {
        int result = metal_rmsnorm_float32(input, weight, output, num_tokens, hidden_size, eps);
        if (result != 0) {
            throw std::runtime_error("Metal RMSNorm float32 operation failed");
        }
    } else {
        throw std::runtime_error("Unsupported type for Metal RMSNorm");
    }
}

template<typename T>
void MetalRMSNorm::rmsnorm_tensor(const MetalTensor<T>& input,
                                  const MetalTensor<T>& weight,
                                  MetalTensor<T>& output,
                                  float eps) {
    
    // Validate tensor shapes
    const auto& input_shape = input.shape();
    const auto& weight_shape = weight.shape();
    const auto& output_shape = output.shape();
    
    if (input_shape.size() != 2) {
        throw std::runtime_error("Input tensor must be 2D [num_tokens, hidden_size]");
    }
    
    if (weight_shape.size() != 1) {
        throw std::runtime_error("Weight tensor must be 1D [hidden_size]");
    }
    
    if (output_shape != input_shape) {
        throw std::runtime_error("Output tensor shape must match input tensor shape");
    }
    
    uint32_t num_tokens = static_cast<uint32_t>(input_shape[0]);
    uint32_t hidden_size = static_cast<uint32_t>(input_shape[1]);
    
    if (weight_shape[0] != hidden_size) {
        throw std::runtime_error("Weight tensor size must match hidden dimension");
    }
    
    // Call the kernel
    rmsnorm<T>(input.data(), weight.data(), output.data(), 
               num_tokens, hidden_size, eps);
}

template<typename T>
void MetalRMSNorm::rmsnorm_inplace(MetalTensor<T>& input_output,
                                   const MetalTensor<T>& weight,
                                   float eps) {
    
    // Use the tensor-based interface with input_output as both input and output
    rmsnorm_tensor<T>(input_output, weight, input_output, eps);
}

template<typename T>
void MetalRMSNorm::rmsnorm_batched(int batch_count,
                                   const T* const* input_array,
                                   const T* weight,
                                   T* const* output_array,
                                   const uint32_t* num_tokens,
                                   uint32_t hidden_size,
                                   float eps) {
    
    // For now, implement as a loop of individual RMSNorm operations
    // A more efficient implementation would use a single batched kernel
    for (int i = 0; i < batch_count; ++i) {
        rmsnorm<T>(input_array[i], weight, output_array[i], 
                   num_tokens[i], hidden_size, eps);
    }
}

template<typename T>
size_t MetalRMSNorm::get_workspace_size(uint32_t num_tokens, uint32_t hidden_size) {
    // For the current implementation, no additional workspace is needed
    // In a more sophisticated implementation, we might need temporary buffers
    return 0;
}