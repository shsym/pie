#pragma once

#include <cstdint>
#include <cstddef>

/**
 * Naive Attention Implementation - Header
 * 
 * This provides the baseline O(n²) memory attention implementation
 * for performance comparison against the optimized version.
 */

namespace metal {
namespace naive_attention {

/**
 * Handle for naive attention kernels (for fair comparison with optimized version)
 */
struct MetalNaiveAttentionHandle {
    void* device;
    void* commandQueue;
    void* library;
    void* pipeline_bf16;
    void* pipeline_f32;
    
    // Configuration
    int max_batch_size;
    int max_seq_length;
    int max_heads;
    int max_head_dim;
    
    // Statistics
    uint64_t total_calls;
    uint64_t total_bytes_processed;
    bool initialized;
};

/**
 * Create handle for naive attention kernels
 */
MetalNaiveAttentionHandle* metal_naive_attention_create_handle(
    int max_batch_size,
    int max_seq_length, 
    int max_heads,
    int max_head_dim
);

/**
 * Destroy naive attention handle
 */
void metal_naive_attention_destroy_handle(MetalNaiveAttentionHandle* handle);

/**
 * Naive attention with O(n²) memory - BF16 version
 * 
 * This function implements standard attention that stores the full attention matrix,
 * demonstrating the memory complexity that optimized versions avoid.
 * 
 * Algorithm:
 * 1. Compute S = Q @ K^T and store in O(n²) memory
 * 2. Apply softmax to stored matrix
 * 3. Compute O = S @ V by reading from stored matrix
 * 
 * Memory Complexity: O(n²) for attention matrix storage
 * Purpose: Baseline for measuring optimization benefits
 */
int naive_batch_prefill_attention_unified_bf16(
    MetalNaiveAttentionHandle* handle,
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
 * Naive attention with O(n²) memory - F32 version
 */
int naive_batch_prefill_attention_unified_f32(
    MetalNaiveAttentionHandle* handle,
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

/**
 * Get workspace requirements for naive attention
 * (Same interface as optimized version for fair comparison)
 */
struct MetalNaiveAttentionWorkspace {
    size_t q_buffer_offset;
    size_t q_buffer_size;
    size_t k_buffer_offset;
    size_t k_buffer_size;
    size_t v_buffer_offset;
    size_t v_buffer_size;
    size_t output_buffer_offset;
    size_t output_buffer_size;
    size_t index_buffer_offset;
    size_t index_buffer_size;
    size_t params_buffer_offset;
    size_t params_buffer_size;
    size_t debug_buffer_offset;
    size_t debug_buffer_size;
    // *** O(n²) ATTENTION MATRIX STORAGE ***
    size_t attention_matrix_offset;
    size_t attention_matrix_size;
    size_t alignment_padding;
    size_t total_size;
};

MetalNaiveAttentionWorkspace metal_naive_attention_get_workspace(
    MetalNaiveAttentionHandle* handle,
    int num_tokens,
    int head_dim,
    int kv_head_dim,
    int page_size,
    int num_kv_pages
);

} // namespace naive_attention
} // namespace metal