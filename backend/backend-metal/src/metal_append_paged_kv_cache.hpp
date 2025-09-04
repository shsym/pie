#pragma once

#include <Metal/Metal.h>
#include <cstdint>

/**
 * @brief Initialize Metal kernels for append_paged_kv_cache operation
 * @return true if initialization successful, false otherwise
 */
bool initialize_metal_append_paged_kv_cache();

/**
 * @brief Metal implementation of append_paged_kv_cache for bfloat16 data
 *
 * Appends key-value pairs to paged KV cache structures, equivalent to
 * FlashInfer's AppendPagedKVCache CUDA operation.
 *
 * @param device Metal device
 * @param commandQueue Metal command queue for GPU operations
 * @param k_input Key input tensor [num_tokens, num_kv_heads * head_size]
 * @param v_input Value input tensor [num_tokens, num_kv_heads * head_size]
 * @param paged_k_cache Paged key cache [max_num_pages, page_size, num_kv_heads * head_size]
 * @param paged_v_cache Paged value cache [max_num_pages, page_size, num_kv_heads * head_size]
 * @param kv_batch_indices Batch index for each token [num_tokens]
 * @param kv_positions Position within sequence for each token [num_tokens]
 * @param kv_page_indices Mapping from logical to physical page indices [max_num_pages]
 * @param kv_page_indptr Start/end page indices for each batch [batch_size + 1]
 * @param kv_last_page_lens Length of last page for each batch [batch_size]
 * @param num_tokens Number of input tokens
 * @param num_kv_heads Number of key-value attention heads
 * @param head_size Size of each attention head
 * @param page_size Number of tokens per page
 * @param max_num_pages Maximum number of pages in cache
 * @param batch_size Number of batches
 */
void metal_append_paged_kv_cache_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const void* k_input,
    const void* v_input,
    void* paged_k_cache,
    void* paged_v_cache,
    const uint32_t* kv_batch_indices,
    const uint32_t* kv_positions,
    const uint32_t* kv_page_indices,
    const uint32_t* kv_page_indptr,
    const uint32_t* kv_last_page_lens,
    uint32_t num_tokens,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t page_size,
    uint32_t max_num_pages,
    uint32_t batch_size
);

/**
 * @brief Metal implementation of append_paged_kv_cache for float32 data
 *
 * Float32 version of the append_paged_kv_cache operation.
 * Parameters same as bfloat16 version.
 */
void metal_append_paged_kv_cache_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const void* k_input,
    const void* v_input,
    void* paged_k_cache,
    void* paged_v_cache,
    const uint32_t* kv_batch_indices,
    const uint32_t* kv_positions,
    const uint32_t* kv_page_indices,
    const uint32_t* kv_page_indptr,
    const uint32_t* kv_last_page_lens,
    uint32_t num_tokens,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t page_size,
    uint32_t max_num_pages,
    uint32_t batch_size
);