#pragma once

#include <cstdint>
#include <cmath>

// Metal backend for batch prefill attention operation
// This implements FlashInfer-style attention with paged memory management

namespace metal {
namespace batch_prefill_attention {

// Batch prefill attention with paged KV cache support
void batch_prefill_attention_bf16(
    const void* Q,                    // Query tensor: [num_tokens, num_query_heads, head_size]
    const void* K,                    // Key tensor: [kv_len, num_kv_heads, head_size]
    const void* V,                    // Value tensor: [kv_len, num_kv_heads, head_size]
    const int32_t* indptr,           // Index pointer for attention lengths per sequence
    const int32_t* indices,          // Token indices for each sequence
    void* O,                         // Output tensor: [num_tokens, num_query_heads, head_size]
    int num_tokens,
    int num_query_heads,
    int num_kv_heads,
    int head_size,
    int kv_len,
    int page_size,
    float scale = 1.0f / sqrtf(64.0f)
);

void batch_prefill_attention_f32(
    const void* Q,                    // Query tensor: [num_tokens, num_query_heads, head_size]
    const void* K,                    // Key tensor: [kv_len, num_kv_heads, head_size]
    const void* V,                    // Value tensor: [kv_len, num_kv_heads, head_size]
    const int32_t* indptr,           // Index pointer for attention lengths per sequence
    const int32_t* indices,          // Token indices for each sequence
    void* O,                         // Output tensor: [num_tokens, num_query_heads, head_size]
    int num_tokens,
    int num_query_heads,
    int num_kv_heads,
    int head_size,
    int kv_len,
    int page_size,
    float scale = 1.0f / sqrtf(64.0f)
);

} // namespace batch_prefill_attention
} // namespace metal