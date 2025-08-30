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
// OLD API REMOVED - Use handle-based API in metal_batch_prefill_handle.hpp
// 
// Migration guide:
// 1. Include "metal_batch_prefill_handle.hpp" 
// 2. Create handle: auto* handle = metal_batch_prefill_create_handle()
// 3. Get workspace: auto workspace = metal_batch_prefill_get_workspace(handle, ...)
// 4. Allocate workspace buffer: auto buffer = [device newBufferWithLength:workspace.total_size ...]
// 5. Call: batch_prefill_attention_unified_bf16(handle, [buffer contents], [buffer length], ...)
// 6. Destroy handle when done: metal_batch_prefill_destroy_handle(handle)

} // namespace batch_prefill_attention
} // namespace metal

#ifdef __cplusplus
}
#endif