#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_silu_and_mul.hpp"
#include <Metal/Metal.h>

/**
 * @brief C++ wrapper for Metal SiLU and multiply operations
 * 
 * Provides a clean C++ interface for the SiLU activation and elementwise multiply
 * operation commonly used in transformer feed-forward networks.
 */
namespace MetalSiLUMul {
    
    /**
     * @brief Initialize Metal SiLU and multiply subsystem
     * Uses the shared MetalContext for device management
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup Metal SiLU and multiply resources
     */
    void cleanup();
    
    /**
     * @brief Check if SiLU and multiply is initialized
     */
    bool is_initialized();
    
    /**
     * @brief Template wrapper for SiLU and multiply operation
     * 
     * Performs the operation: output = silu(gate) * up
     * where silu(x) = x / (1 + exp(-x))
     * 
     * This is commonly used in transformer feed-forward networks where:
     * - gate comes from the gate projection
     * - up comes from the up projection
     * - output is the final feed-forward result
     * 
     * @param gate Gate projection tensor [num_tokens, intermediate_size]
     * @param up Up projection tensor [num_tokens, intermediate_size]
     * @param output Output tensor [num_tokens, intermediate_size]
     * @param num_tokens Number of tokens (sequence length)
     * @param intermediate_size Hidden size of feed-forward layer
     * @param commandBuffer Metal command buffer for GPU operations
     */
    template<typename T>
    void silu_and_mul(const T* gate,
                      const T* up,
                      T* output,
                      uint32_t num_tokens,
                      uint32_t intermediate_size,
                      id<MTLCommandBuffer> commandBuffer = nil);
    
    /**
     * @brief Tensor-based SiLU and multiply operation
     * 
     * Higher-level interface using MetalTensor objects.
     * Handles memory management and provides type safety.
     * 
     * @param gate Gate projection tensor [num_tokens, intermediate_size]
     * @param up Up projection tensor [num_tokens, intermediate_size] 
     * @param output Output tensor [num_tokens, intermediate_size] (must be pre-allocated)
     */
    template<typename T>
    void silu_and_mul_tensor(const MetalTensor<T>& gate,
                             const MetalTensor<T>& up,
                             MetalTensor<T>& output);
    
    /**
     * @brief In-place SiLU and multiply operation
     * 
     * Performs the operation using the gate tensor as output:
     * gate = silu(gate) * up
     * 
     * @param gate_output Gate projection tensor, also used as output [num_tokens, intermediate_size]
     * @param up Up projection tensor [num_tokens, intermediate_size]
     */
    template<typename T>
    void silu_and_mul_inplace(MetalTensor<T>& gate_output,
                               const MetalTensor<T>& up);
    
    /**
     * @brief Batched SiLU and multiply operation
     * 
     * Performs SiLU and multiply on multiple sequences in a single call.
     * Useful for batch processing in transformer models.
     * 
     * @param batch_count Number of sequences in batch
     * @param gate_array Array of gate projection tensors
     * @param up_array Array of up projection tensors
     * @param output_array Array of output tensors
     * @param num_tokens Array of token counts for each sequence
     * @param intermediate_size Feed-forward hidden size (same for all)
     */
    template<typename T>
    void silu_and_mul_batched(int batch_count,
                               const T* const* gate_array,
                               const T* const* up_array,
                               T* const* output_array,
                               const uint32_t* num_tokens,
                               uint32_t intermediate_size);
}

// Template implementations

template<typename T>
void MetalSiLUMul::silu_and_mul(const T* gate,
                                 const T* up,
                                 T* output,
                                 uint32_t num_tokens,
                                 uint32_t intermediate_size,
                                 id<MTLCommandBuffer> commandBuffer) {
    
    // Ensure initialization
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize Metal SiLU and multiply");
    }
    
    // Type dispatch to the appropriate Metal kernel
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        int result = metal_silu_and_mul_bfloat16(gate, up, output, num_tokens, intermediate_size);
        if (result != 0) {
            throw std::runtime_error("Metal SiLU and multiply bfloat16 operation failed");
        }
    } else if constexpr (std::is_same_v<T, float>) {
        int result = metal_silu_and_mul_float32(gate, up, output, num_tokens, intermediate_size);
        if (result != 0) {
            throw std::runtime_error("Metal SiLU and multiply float32 operation failed");
        }
    } else {
        throw std::runtime_error("Unsupported type for Metal SiLU and multiply");
    }
}

template<typename T>
void MetalSiLUMul::silu_and_mul_tensor(const MetalTensor<T>& gate,
                                        const MetalTensor<T>& up,
                                        MetalTensor<T>& output) {
    
    // Validate tensor shapes
    const auto& gate_shape = gate.shape();
    const auto& up_shape = up.shape();
    const auto& output_shape = output.shape();
    
    if (gate_shape.size() != 2 || up_shape.size() != 2 || output_shape.size() != 2) {
        throw std::runtime_error("All tensors must be 2D [num_tokens, intermediate_size]");
    }
    
    if (gate_shape != up_shape || gate_shape != output_shape) {
        throw std::runtime_error("All tensors must have the same shape");
    }
    
    uint32_t num_tokens = static_cast<uint32_t>(gate_shape[0]);
    uint32_t intermediate_size = static_cast<uint32_t>(gate_shape[1]);
    
    // Call the kernel
    silu_and_mul<T>(gate.data(), up.data(), output.data(), num_tokens, intermediate_size);
}

template<typename T>
void MetalSiLUMul::silu_and_mul_inplace(MetalTensor<T>& gate_output,
                                         const MetalTensor<T>& up) {
    
    // Use the tensor-based interface with gate_output as both input and output
    silu_and_mul_tensor<T>(gate_output, up, gate_output);
}

template<typename T>
void MetalSiLUMul::silu_and_mul_batched(int batch_count,
                                         const T* const* gate_array,
                                         const T* const* up_array,
                                         T* const* output_array,
                                         const uint32_t* num_tokens,
                                         uint32_t intermediate_size) {
    
    // For now, implement as a loop of individual SiLU and multiply operations
    // A more efficient implementation would use a single batched kernel
    for (int i = 0; i < batch_count; ++i) {
        silu_and_mul<T>(gate_array[i], up_array[i], output_array[i], 
                        num_tokens[i], intermediate_size);
    }
}