#pragma once

#include <cstdint>
#include <cmath>

// Metal backend for batch prefill attention operation
// This implements FlashInfer-style attention with paged memory management

#ifdef __cplusplus
extern "C" {
#endif

namespace metal {
namespace batch_prefill_attention {

// FlashInfer-style unified paged KV cache interface (bf16)
// q_input: [num_qo, head_dim] (head_dim = num_query_heads * head_size)
// paged_k_cache, paged_v_cache: [num_pages_total, page_size, head_dim]
// qo_indptr: [num_seqs+1], kv_page_indptr: [num_seqs+1]
// kv_page_indices: [total_pages_across_seqs]
// kv_last_page_lens: [num_seqs]
// output: [num_qo, head_dim]
void batch_prefill_attention_unified_bf16(
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
    int head_size,
    int page_size,
    float scale,
    int num_kv_pages
);

// Native float32 variant (no host-side dtype conversions)
void batch_prefill_attention_unified_f32(
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
    int head_size,
    int page_size,
    float scale,
    int num_kv_pages
);

} // namespace batch_prefill_attention
} // namespace metal

#ifdef __cplusplus
}
#endif