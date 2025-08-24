#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_batch_prefill_attention.hpp"
#include <Metal/Metal.h>

/**
 * @brief C++ wrapper for Metal batch prefill attention operations
 * 
 * Provides a clean C++ interface for FlashInfer-style attention with paged KV cache.
 * This implements the core attention mechanism used in transformer models.
 */
namespace MetalBatchPrefillAttention {
    
    /**
     * @brief Initialize Metal batch prefill attention subsystem
     * Uses the shared MetalContext for device management
     * @return true if successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup Metal batch prefill attention resources
     */
    void cleanup();
    
    /**
     * @brief Check if batch prefill attention is initialized
     */
    bool is_initialized();
    
    /**
     * @brief Template wrapper for unified batch prefill attention operation
     * 
     * Implements FlashInfer-style attention with paged memory management:
     * output = softmax(QK^T / sqrt(d_k))V
     * 
     * This is the core attention mechanism where:
     * - Q (queries) come from current input tokens
     * - K,V (keys,values) are stored in paged cache from previous tokens
     * - Attention is computed across all pages for each sequence
     * 
     * @param q_input Query tensor [num_qo, head_dim] where head_dim = num_heads * head_size
     * @param paged_k_cache Paged key cache [num_pages_total, page_size, head_dim]
     * @param paged_v_cache Paged value cache [num_pages_total, page_size, head_dim]
     * @param qo_indptr Query output indices [num_seqs+1] - sequence boundaries in q_input
     * @param kv_page_indptr KV page indices [num_seqs+1] - page boundaries per sequence
     * @param kv_page_indices Page indices array [total_pages_across_seqs]
     * @param kv_last_page_lens Last page lengths [num_seqs] - tokens in final page per seq
     * @param output Output tensor [num_qo, head_dim]
     * @param num_qo Number of query tokens (total across all sequences)
     * @param head_dim Head dimension (num_heads * head_size)
     * @param head_size Size of each attention head
     * @param page_size Number of tokens per page
     * @param scale Attention scaling factor (typically 1/sqrt(head_size))
     * @param num_kv_pages Total number of KV cache pages
     * @param commandBuffer Metal command buffer for GPU operations
     */
    template<typename T>
    void batch_prefill_attention_unified(const T* q_input,
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
                                          int num_kv_pages,
                                          id<MTLCommandBuffer> commandBuffer = nil);
    
    /**
     * @brief Tensor-based batch prefill attention operation
     * 
     * Higher-level interface using MetalTensor objects.
     * Handles memory management and provides type safety.
     * 
     * @param q_input Query tensor [num_qo, head_dim]
     * @param paged_k_cache Paged key cache [num_pages_total, page_size, head_dim]
     * @param paged_v_cache Paged value cache [num_pages_total, page_size, head_dim]
     * @param qo_indptr Query output indices tensor [num_seqs+1]
     * @param kv_page_indptr KV page indices tensor [num_seqs+1]
     * @param kv_page_indices Page indices tensor [total_pages_across_seqs]
     * @param kv_last_page_lens Last page lengths tensor [num_seqs]
     * @param output Output tensor [num_qo, head_dim] (must be pre-allocated)
     * @param head_size Size of each attention head
     * @param page_size Number of tokens per page
     * @param scale Attention scaling factor
     */
    template<typename T>
    void batch_prefill_attention_tensor(const MetalTensor<T>& q_input,
                                         const MetalTensor<T>& paged_k_cache,
                                         const MetalTensor<T>& paged_v_cache,
                                         const MetalTensor<int32_t>& qo_indptr,
                                         const MetalTensor<int32_t>& kv_page_indptr,
                                         const MetalTensor<int32_t>& kv_page_indices,
                                         const MetalTensor<int32_t>& kv_last_page_lens,
                                         MetalTensor<T>& output,
                                         int head_size,
                                         int page_size,
                                         float scale);
    
    /**
     * @brief Calculate required workspace size for attention operation
     * 
     * @param num_qo Number of query tokens
     * @param head_dim Head dimension
     * @param max_seq_len Maximum sequence length
     * @return Required workspace size in bytes
     */
    template<typename T>
    size_t get_workspace_size(int num_qo, int head_dim, int max_seq_len);
}

// Template implementations

template<typename T>
void MetalBatchPrefillAttention::batch_prefill_attention_unified(const T* q_input,
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
                                                                  int num_kv_pages,
                                                                  id<MTLCommandBuffer> commandBuffer) {
    
    // Ensure initialization
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize Metal batch prefill attention");
    }
    
    // Validate dimensions
    if (head_dim % head_size != 0) {
        throw std::runtime_error("head_dim must be divisible by head_size");
    }
    
    if (scale <= 0.0f) {
        throw std::runtime_error("Scale must be positive");
    }
    
    // Type dispatch to the appropriate Metal kernel
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input, paged_k_cache, paged_v_cache,
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
            output, num_qo, head_dim, head_size, page_size, scale, num_kv_pages);
    } else if constexpr (std::is_same_v<T, float>) {
        metal::batch_prefill_attention::batch_prefill_attention_unified_f32(
            q_input, paged_k_cache, paged_v_cache,
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
            output, num_qo, head_dim, head_size, page_size, scale, num_kv_pages);
    } else {
        throw std::runtime_error("Unsupported type for Metal batch prefill attention");
    }
}

template<typename T>
void MetalBatchPrefillAttention::batch_prefill_attention_tensor(const MetalTensor<T>& q_input,
                                                                 const MetalTensor<T>& paged_k_cache,
                                                                 const MetalTensor<T>& paged_v_cache,
                                                                 const MetalTensor<int32_t>& qo_indptr,
                                                                 const MetalTensor<int32_t>& kv_page_indptr,
                                                                 const MetalTensor<int32_t>& kv_page_indices,
                                                                 const MetalTensor<int32_t>& kv_last_page_lens,
                                                                 MetalTensor<T>& output,
                                                                 int head_size,
                                                                 int page_size,
                                                                 float scale) {
    
    // Validate tensor shapes
    const auto& q_shape = q_input.shape();
    const auto& k_cache_shape = paged_k_cache.shape();
    const auto& v_cache_shape = paged_v_cache.shape();
    const auto& output_shape = output.shape();
    
    if (q_shape.size() != 2) {
        throw std::runtime_error("Query tensor must be 2D [num_qo, head_dim]");
    }
    
    if (k_cache_shape.size() != 3 || v_cache_shape.size() != 3) {
        throw std::runtime_error("KV cache tensors must be 3D [num_pages_total, page_size, head_dim]");
    }
    
    if (output_shape != q_shape) {
        throw std::runtime_error("Output tensor shape must match query tensor shape");
    }
    
    if (k_cache_shape != v_cache_shape) {
        throw std::runtime_error("Key and value cache tensors must have the same shape");
    }
    
    int num_qo = static_cast<int>(q_shape[0]);
    int head_dim = static_cast<int>(q_shape[1]);
    int num_kv_pages = static_cast<int>(k_cache_shape[0]);
    
    if (k_cache_shape[1] != page_size) {
        throw std::runtime_error("KV cache page_size dimension must match page_size parameter");
    }
    
    if (k_cache_shape[2] != head_dim) {
        throw std::runtime_error("KV cache head_dim dimension must match query head_dim");
    }
    
    // Call the kernel
    batch_prefill_attention_unified<T>(
        q_input.data(), paged_k_cache.data(), paged_v_cache.data(),
        qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(), kv_last_page_lens.data(),
        output.data(), num_qo, head_dim, head_size, page_size, scale, num_kv_pages);
}

template<typename T>
size_t MetalBatchPrefillAttention::get_workspace_size(int num_qo, int head_dim, int max_seq_len) {
    // For the current implementation, no additional workspace is needed
    // In a more sophisticated implementation, we might need temporary buffers
    // for intermediate attention scores or softmax computation
    return 0;
}