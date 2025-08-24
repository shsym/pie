#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp" 
#include "metal_rope.hpp"
#include <Metal/Metal.h>

/**
 * @brief C++ wrapper for Metal RoPE (Rotary Position Embedding) operations
 * 
 * Provides a clean C++ interface that matches the FlashInfer API
 * and integrates with the Metal infrastructure components.
 */
namespace MetalRoPE {
    
    /**
     * @brief Initialize Metal RoPE subsystem
     * Uses the shared MetalContext for device management
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup Metal RoPE resources
     */
    void cleanup();
    
    /**
     * @brief Check if RoPE is initialized
     */
    bool is_initialized();
    
    /**
     * @brief Template wrapper for RoPE operation matching FlashInfer API
     * 
     * Applies rotary position embedding to query and key tensors in-place.
     * Implements the RoPE transformation: input_qk = rope(input_qk, position_ids)
     * 
     * @param input_qk Input/output tensor [num_tokens, num_heads, head_size]
     * @param position_ids Position IDs for each token [num_tokens]  
     * @param num_tokens Number of tokens (sequence length)
     * @param num_heads Number of attention heads
     * @param head_size Size of each attention head (must be even)
     * @param rope_theta Base for rotary frequency computation (default: 10000.0)
     * @param rope_factor Scaling factor for RoPE (default: 1.0)
     * @param commandBuffer Metal command buffer for GPU operations
     */
    template<typename T>
    void rope_inplace(T* input_qk,
                      const int32_t* position_ids,
                      uint32_t num_tokens,
                      uint32_t num_heads,
                      uint32_t head_size,
                      float rope_theta = 10000.0f,
                      float rope_factor = 1.0f,
                      id<MTLCommandBuffer> commandBuffer = nil);
    
    /**
     * @brief Tensor-based RoPE operation
     * 
     * Higher-level interface using MetalTensor objects.
     * Handles memory management and provides type safety.
     * 
     * @param input_qk Input/output tensor [num_tokens, num_heads, head_size]
     * @param position_ids Position tensor [num_tokens]
     * @param rope_theta Base for rotary frequency computation
     * @param rope_factor Scaling factor for RoPE
     */
    template<typename T>
    void rope_tensor_inplace(MetalTensor<T>& input_qk,
                             const MetalTensor<int32_t>& position_ids,
                             float rope_theta = 10000.0f,
                             float rope_factor = 1.0f);
    
    /**
     * @brief Batched RoPE operation
     * 
     * Applies RoPE to multiple sequences in a single call.
     * Useful for batch processing in transformer models.
     * 
     * @param batch_count Number of sequences in batch
     * @param input_qk_array Array of input/output tensors
     * @param position_ids_array Array of position ID tensors
     * @param num_tokens Array of token counts for each sequence
     * @param num_heads Number of attention heads (same for all)
     * @param head_size Head dimension size (same for all)
     * @param rope_theta Base for rotary frequency computation
     * @param rope_factor Scaling factor for RoPE
     */
    template<typename T>
    void rope_batched_inplace(int batch_count,
                              T* const* input_qk_array,
                              const int32_t* const* position_ids_array,
                              const uint32_t* num_tokens,
                              uint32_t num_heads,
                              uint32_t head_size,
                              float rope_theta = 10000.0f,
                              float rope_factor = 1.0f);
}

// Template implementations

template<typename T>
void MetalRoPE::rope_inplace(T* input_qk,
                             const int32_t* position_ids,
                             uint32_t num_tokens,
                             uint32_t num_heads,
                             uint32_t head_size,
                             float rope_theta,
                             float rope_factor,
                             id<MTLCommandBuffer> commandBuffer) {
    
    // Ensure initialization
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize Metal RoPE");
    }
    
    // Validate head_size is even (required for RoPE)
    if (head_size % 2 != 0) {
        throw std::runtime_error("RoPE head_size must be even");
    }
    
    // Type dispatch to the appropriate Metal kernel
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        int result = metal_rope_bfloat16(input_qk, position_ids, num_tokens, 
                                         num_heads, head_size, rope_theta, rope_factor);
        if (result != 0) {
            throw std::runtime_error("Metal RoPE bfloat16 operation failed");
        }
    } else if constexpr (std::is_same_v<T, float>) {
        int result = metal_rope_float32(input_qk, position_ids, num_tokens,
                                        num_heads, head_size, rope_theta, rope_factor);
        if (result != 0) {
            throw std::runtime_error("Metal RoPE float32 operation failed");
        }
    } else {
        throw std::runtime_error("Unsupported type for Metal RoPE");
    }
}

template<typename T>
void MetalRoPE::rope_tensor_inplace(MetalTensor<T>& input_qk,
                                    const MetalTensor<int32_t>& position_ids,
                                    float rope_theta,
                                    float rope_factor) {
    
    // Validate tensor shapes
    const auto& qk_shape = input_qk.shape();
    const auto& pos_shape = position_ids.shape();
    
    if (qk_shape.size() != 3) {
        throw std::runtime_error("Input QK tensor must be 3D [num_tokens, num_heads, head_size]");
    }
    
    if (pos_shape.size() != 1) {
        throw std::runtime_error("Position IDs tensor must be 1D [num_tokens]");
    }
    
    uint32_t num_tokens = static_cast<uint32_t>(qk_shape[0]);
    uint32_t num_heads = static_cast<uint32_t>(qk_shape[1]);
    uint32_t head_size = static_cast<uint32_t>(qk_shape[2]);
    
    if (pos_shape[0] != num_tokens) {
        throw std::runtime_error("Position IDs size must match number of tokens");
    }
    
    // Call the kernel
    rope_inplace<T>(input_qk.data(), position_ids.data(), 
                    num_tokens, num_heads, head_size, rope_theta, rope_factor);
}

template<typename T>
void MetalRoPE::rope_batched_inplace(int batch_count,
                                     T* const* input_qk_array,
                                     const int32_t* const* position_ids_array,
                                     const uint32_t* num_tokens,
                                     uint32_t num_heads,
                                     uint32_t head_size,
                                     float rope_theta,
                                     float rope_factor) {
    
    // For now, implement as a loop of individual RoPE operations
    // A more efficient implementation would use a single batched kernel
    for (int i = 0; i < batch_count; ++i) {
        rope_inplace<T>(input_qk_array[i], position_ids_array[i], 
                        num_tokens[i], num_heads, head_size, rope_theta, rope_factor);
    }
}