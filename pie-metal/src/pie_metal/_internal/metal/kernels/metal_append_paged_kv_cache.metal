#include <metal_stdlib>
using namespace metal;

struct AppendPagedKVCacheParams {
    uint num_tokens;
    uint num_kv_heads;
    uint head_size;
    uint page_size;
    uint max_num_pages;
    uint batch_size;
    uint k_stride_token;  // num_kv_heads * head_size
    uint k_stride_head;   // head_size
    uint v_stride_token;  // num_kv_heads * head_size
    uint v_stride_head;   // head_size
};

/**
 * Metal kernel for appending key-value pairs to paged KV cache
 * Equivalent to FlashInfer's AppendPagedKVCache CUDA operation
 *
 * Each thread processes one token's K or V data for one head
 */
kernel void metal_append_paged_kv_cache_bfloat16(
    device const bfloat *k_input [[buffer(0)]],        // [num_tokens, num_kv_heads * head_size]
    device const bfloat *v_input [[buffer(1)]],        // [num_tokens, num_kv_heads * head_size]
    device bfloat *paged_k_cache [[buffer(2)]],        // [max_num_pages, page_size, num_kv_heads * head_size]
    device bfloat *paged_v_cache [[buffer(3)]],        // [max_num_pages, page_size, num_kv_heads * head_size]
    device const uint *kv_batch_indices [[buffer(4)]], // [num_tokens] - which batch each token belongs to
    device const uint *kv_positions [[buffer(5)]],     // [num_tokens] - position within sequence
    device const uint *kv_page_indices [[buffer(6)]],  // [max_num_pages] - mapping of logical to physical pages
    device const uint *kv_page_indptr [[buffer(7)]],   // [batch_size + 1] - page range per batch
    device const uint *kv_last_page_lens [[buffer(8)]], // [batch_size] - length of last page per batch
    device const float* params_raw [[buffer(9)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint token_idx = thread_position_in_grid.x;
    const uint head_idx = thread_position_in_grid.y;
    const uint head_offset = thread_position_in_grid.z;

    if (token_idx >= ((uint)params_raw[0]) ||
        head_idx >= ((uint)params_raw[1]) ||
        head_offset >= ((uint)params_raw[2])) {
        return;
    }

    const uint batch_idx = kv_batch_indices[token_idx];
    const uint token_position = kv_positions[token_idx];

    // Validate batch index
    if (batch_idx >= ((uint)params_raw[5])) {
        return;
    }

    // Find which page this token belongs to within the batch
    const uint page_start = kv_page_indptr[batch_idx];
    const uint page_end = kv_page_indptr[batch_idx + 1];

    // Calculate which page within the batch sequence to use
    const uint page_offset_in_batch = token_position / ((uint)params_raw[3]);
    const uint position_in_page = token_position % ((uint)params_raw[3]);

    // Get physical page index
    const uint page_logical = page_start + page_offset_in_batch;
    if (page_logical >= page_end) {
        return; // Token position exceeds allocated pages for this batch
    }

    const uint physical_page = kv_page_indices[page_logical];
    if (physical_page >= ((uint)params_raw[4])) {
        return;
    }

    // Calculate input offsets
    const uint input_offset = token_idx * ((uint)params_raw[6]) +
                             head_idx * ((uint)params_raw[7]) +
                             head_offset;

    // Calculate output offsets in paged cache
    // Layout: [page_idx][token_in_page][head_idx * head_size + head_offset]
    const uint cache_offset = physical_page * (((uint)params_raw[3]) * ((uint)params_raw[1]) * ((uint)params_raw[2])) +
                             position_in_page * (((uint)params_raw[1]) * ((uint)params_raw[2])) +
                             head_idx * ((uint)params_raw[2]) +
                             head_offset;

    // Copy K and V data from input to paged cache
    if (input_offset < ((uint)params_raw[0]) * ((uint)params_raw[1]) * ((uint)params_raw[2])) {
        paged_k_cache[cache_offset] = k_input[input_offset];
        paged_v_cache[cache_offset] = v_input[input_offset];
    }
}

/**
 * Metal kernel for float32 version
 */
kernel void metal_append_paged_kv_cache_float32(
    device const float *k_input [[buffer(0)]],
    device const float *v_input [[buffer(1)]],
    device float *paged_k_cache [[buffer(2)]],
    device float *paged_v_cache [[buffer(3)]],
    device const uint *kv_batch_indices [[buffer(4)]],
    device const uint *kv_positions [[buffer(5)]],
    device const uint *kv_page_indices [[buffer(6)]],
    device const uint *kv_page_indptr [[buffer(7)]],
    device const uint *kv_last_page_lens [[buffer(8)]],
    device const float* params_raw [[buffer(9)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint token_idx = thread_position_in_grid.x;
    const uint head_idx = thread_position_in_grid.y;
    const uint head_offset = thread_position_in_grid.z;

    if (token_idx >= ((uint)params_raw[0]) ||
        head_idx >= ((uint)params_raw[1]) ||
        head_offset >= ((uint)params_raw[2])) {
        return;
    }

    const uint batch_idx = kv_batch_indices[token_idx];
    const uint token_position = kv_positions[token_idx];

    if (batch_idx >= ((uint)params_raw[5])) {
        return;
    }

    const uint page_start = kv_page_indptr[batch_idx];
    const uint page_end = kv_page_indptr[batch_idx + 1];

    const uint page_offset_in_batch = token_position / ((uint)params_raw[3]);
    const uint position_in_page = token_position % ((uint)params_raw[3]);

    const uint page_logical = page_start + page_offset_in_batch;
    if (page_logical >= page_end) {
        return;
    }

    const uint physical_page = kv_page_indices[page_logical];
    if (physical_page >= ((uint)params_raw[4])) {
        return;
    }

    const uint input_offset = token_idx * ((uint)params_raw[6]) +
                             head_idx * ((uint)params_raw[7]) +
                             head_offset;

    const uint cache_offset = physical_page * (((uint)params_raw[3]) * ((uint)params_raw[1]) * ((uint)params_raw[2])) +
                             position_in_page * (((uint)params_raw[1]) * ((uint)params_raw[2])) +
                             head_idx * ((uint)params_raw[2]) +
                             head_offset;

    if (input_offset < ((uint)params_raw[0]) * ((uint)params_raw[1]) * ((uint)params_raw[2])) {
        paged_k_cache[cache_offset] = k_input[input_offset];
        paged_v_cache[cache_offset] = v_input[input_offset];
    }
}