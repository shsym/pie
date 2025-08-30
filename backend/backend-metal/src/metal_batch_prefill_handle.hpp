#pragma once

#include <cstdint>
#include <cmath>
#import <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace metal {
namespace batch_prefill_attention {

/**
 * @brief Handle for Metal batch prefill attention operations
 * Encapsulates all persistent Metal resources needed for attention computation
 */
struct MetalBatchPrefillHandle {
    // Core Metal resources
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // Pipeline states for different kernels
    id<MTLComputePipelineState> pipeline_bf16;
    id<MTLComputePipelineState> pipeline_f32;
    
    // Configuration bounds (for validation)
    int max_batch_size;
    int max_seq_length; 
    int max_heads;
    int max_head_dim;
    
    // Usage statistics
    size_t total_calls;
    size_t total_bytes_processed;
    
    // Internal state
    bool initialized;
};

/**
 * @brief Workspace layout for Metal batch prefill attention
 * Defines memory regions within the user-provided workspace buffer
 */
struct MetalBatchPrefillWorkspace {
    // Total workspace size required
    size_t total_size;
    
    // Buffer offsets within workspace
    size_t q_buffer_offset;        // Query buffer (converted from BF16 to Half)
    size_t q_buffer_size;
    
    size_t k_buffer_offset;        // Key cache buffer (converted)
    size_t k_buffer_size;
    
    size_t v_buffer_offset;        // Value cache buffer (converted)
    size_t v_buffer_size;
    
    size_t output_buffer_offset;   // Output buffer
    size_t output_buffer_size;
    
    size_t index_buffer_offset;    // Combined index arrays
    size_t index_buffer_size;
    
    size_t params_buffer_offset;   // Kernel parameters
    size_t params_buffer_size;
    
    size_t debug_buffer_offset;    // Debug buffer
    size_t debug_buffer_size;
    
    // Alignment padding
    size_t alignment_padding;
};

// ============================================================================
// Handle Management API
// ============================================================================

/**
 * @brief Create a new Metal batch prefill attention handle
 * @param max_batch_size Maximum batch size this handle will process
 * @param max_seq_length Maximum sequence length
 * @param max_heads Maximum number of attention heads
 * @param max_head_dim Maximum head dimension
 * @return Handle pointer or nullptr on failure
 */
MetalBatchPrefillHandle* metal_batch_prefill_create_handle(
    int max_batch_size = 1024,
    int max_seq_length = 8192, 
    int max_heads = 64,
    int max_head_dim = 8192
);

/**
 * @brief Destroy Metal batch prefill attention handle and free resources
 * @param handle Handle to destroy (can be nullptr)
 */
void metal_batch_prefill_destroy_handle(MetalBatchPrefillHandle* handle);

/**
 * @brief Calculate workspace requirements for given parameters
 * @param handle Valid handle pointer
 * @param num_tokens Number of query tokens
 * @param head_dim Query head dimension (num_query_heads * head_size)
 * @param kv_head_dim KV head dimension (num_kv_heads * head_size)
 * @param page_size KV cache page size
 * @param num_kv_pages Total number of KV cache pages
 * @return Workspace layout with size and offset information
 */
MetalBatchPrefillWorkspace metal_batch_prefill_get_workspace(
    MetalBatchPrefillHandle* handle,
    int num_tokens,
    int head_dim,
    int kv_head_dim,
    int page_size,
    int num_kv_pages
);

// ============================================================================
// Attention Computation API (Handle + Workspace)
// ============================================================================

/**
 * @brief FlashInfer-style unified paged KV cache interface (bf16) with handle/workspace
 * 
 * This is the new primary API that uses explicit handle and workspace management.
 * All memory allocations are done by the user and passed as workspace.
 * 
 * @param handle Valid MetalBatchPrefillHandle
 * @param workspace_buffer User-allocated Metal buffer contents (from [buffer contents])
 * @param workspace_size Size of workspace buffer in bytes
 * @param q_input Query input: [num_qo, head_dim] in bfloat16
 * @param paged_k_cache Key cache: [num_pages_total, page_size, kv_head_dim] in bfloat16  
 * @param paged_v_cache Value cache: [num_pages_total, page_size, kv_head_dim] in bfloat16
 * @param qo_indptr Query offset pointers: [num_seqs+1]
 * @param kv_page_indptr KV page offset pointers: [num_seqs+1] 
 * @param kv_page_indices KV page indices: [total_pages_across_seqs]
 * @param kv_last_page_lens Last page lengths: [num_seqs]
 * @param output Output buffer: [num_qo, head_dim] in bfloat16
 * @param num_qo Number of query tokens
 * @param head_dim Query head dimension (num_query_heads * head_size)
 * @param kv_head_dim KV head dimension (num_kv_heads * head_size)
 * @param head_size Size of each attention head
 * @param page_size KV cache page size
 * @param num_query_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for MQA/GQA)
 * @param scale Attention scaling factor (usually 1/sqrt(head_size))
 * @param num_kv_pages Total number of KV cache pages
 */
void batch_prefill_attention_unified_bf16(
    MetalBatchPrefillHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const void* q_input,
    const void* paged_k_cache,
    const void* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    void* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
);

/**
 * @brief Native float32 variant with handle/workspace
 * Same API as bf16 version but operates on float32 data
 */
void batch_prefill_attention_unified_f32(
    MetalBatchPrefillHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
);

// ============================================================================
// Legacy API (for backward compatibility during migration)
// ============================================================================

/**
 * @brief Legacy API without handle - creates temporary handle internally
 * @deprecated Use handle-based API for better performance
 * These functions are provided for backward compatibility during migration.
 * They create a temporary handle internally which adds overhead.
 */
void batch_prefill_attention_unified_bf16_legacy(
    const void* q_input,
    const void* paged_k_cache,
    const void* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    void* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
);

void batch_prefill_attention_unified_f32_legacy(
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
);

} // namespace batch_prefill_attention
} // namespace metal

#ifdef __cplusplus
}
#endif