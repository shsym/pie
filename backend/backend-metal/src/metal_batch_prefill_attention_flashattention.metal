#include <metal_stdlib>
using namespace metal;

/**
 * FlashAttention Implementation with True Tiling
 * 
 * Goal: Implement proper FlashAttention algorithm
 * - Uses 2D tiling: Q tiles and KV tiles  
 * - Parallel thread execution with proper work distribution
 * - Online softmax with O(1) memory complexity
 * - Memory-efficient with shared memory utilization
 */

constant int TILE_SIZE_Q [[function_constant(0)]];     
constant int TILE_SIZE_KV [[function_constant(1)]];    
constant bool ENABLE_CAUSAL_MASK [[function_constant(2)]];

constant uint TGP_SIZE = 256;
constant int MAX_HEAD_DIM = 256; 

// Use the same parameter structure as unified attention system
struct UnifiedParams {
    int32_t num_qo;           // Total query tokens across all sequences
    int32_t num_sequences;    // Number of sequences in batch
    int32_t head_dim;         // Total head dimension (num_heads * head_size)
    int32_t head_size;        // Size per attention head
    int32_t num_heads;        // Number of attention heads
    int32_t page_size;        // Tokens per KV cache page
    int32_t max_seq_len;      // Maximum sequence length for tile sizing
    float scale;              // Attention scaling factor (1/sqrt(head_size))
};

kernel void unified_batch_prefill_attention_bf16_flashattention(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant UnifiedParams& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one query token (same as original)
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(params.num_qo)) return;
    
    // --- Get Sequence and KV Page Information (same as original) ---
    // Find which sequence this query belongs to
    int seq_id = 0;
    while (seq_id < 100 && qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; }
    
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;
    
    if (num_pages <= 0) {
        // Zero out output for this query 
        for (int d = tid_in_tgp; d < params.head_dim; d += TGP_SIZE) {
            output[qo_idx * params.head_dim + d] = 0.0h;
        }
        return;
    }
    
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * params.page_size + last_page_len;
    
    // Debug output early to verify parameter passing
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = params.scale;
        debug_out[1] = float(params.head_dim);
        debug_out[2] = float(params.page_size);
        debug_out[3] = float(params.num_qo);
        debug_out[4] = float(total_kv_len);
        debug_out[5] = float(num_pages);
        debug_out[6] = float(last_page_len);
    }
    
    // Use simple parallel approach like original but with improved online softmax
    const int BLOCK_SIZE = 16; // Process KV in blocks of 16
    const int max_head_size = min(params.head_size, 128); // Keep reasonable limit
    
    // Shared memory similar to original
    threadgroup half q_s[128];  // Query for this head
    threadgroup half k_block[16][128];  // KV block
    threadgroup half v_block[16][128]; 
    threadgroup float w_block[16];      // Attention weights
    threadgroup float temp_reduce[TGP_SIZE];
    
    // Online softmax state per head
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[128];
    
    // Process each head (like original kernel)
    for (int h = 0; h < params.num_heads; ++h) {
        // Initialize online softmax state per head
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            acc_i[d] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load Query slice for this head into shared memory
        int q_base = int(qo_idx) * params.head_dim + h * params.head_size;
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            q_s[d] = q_input[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Main Loop: Process KV Cache in Parallel Blocks (like original but with online softmax)
        for (int block_start = 0; block_start < total_kv_len; block_start += BLOCK_SIZE) {
            // Load K/V for this block and head slice (same as original)
            if (tid_in_tgp < BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / params.page_size;
                    int in_page_offset = global_key_idx % params.page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    uint base_addr = page_idx * params.page_size * params.head_dim + 
                                   in_page_offset * params.head_dim + h * params.head_size;

                    for (int d = 0; d < max_head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            
            // Compute scores for this block (same as original)
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;
            if (tid_in_tgp < BLOCK_SIZE && global_key_idx_score < total_kv_len) {
                for (int d = 0; d < max_head_size; ++d) {
                    score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                }
                score *= params.scale;
            } else {
                score = -INFINITY;
            }

            // Online softmax update (improved from original)
            // 1) Block max reduction
            temp_reduce[tid_in_tgp] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float m_j = temp_reduce[0];

            // 2) Update global max and rescale accumulators
            threadgroup float m_prev = -INFINITY;
            if (tid_in_tgp == 0) {
                m_prev = m_i;
                m_i = max(m_i, m_j);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale_factor = exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // 3) Compute weights and update accumulators
            float w = (score > -INFINITY) ? exp(score - m_i) : 0.0f;
            if (tid_in_tgp < BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Update l_i (sum of weights)
            temp_reduce[tid_in_tgp] = (tid_in_tgp < BLOCK_SIZE) ? w : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid_in_tgp == 0) {
                l_i += temp_reduce[0];
            }

            // Accumulate V slice (same as original)
            int dims_per_thread = (max_head_size + TGP_SIZE - 1) / TGP_SIZE;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid_in_tgp * dims_per_thread + i;
                if (d < max_head_size) {
                    float sum_wv_d = 0.0f;
                    int num_keys_in_block = min(BLOCK_SIZE, total_kv_len - block_start);
                    for (int j = 0; j < num_keys_in_block; ++j) {
                        sum_wv_d += w_block[j] * float(v_block[j][d]);
                    }
                    acc_i[d] += sum_wv_d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Finalize output for this head (same as original)
        int out_base = int(qo_idx) * params.head_dim + h * params.head_size;
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = half(acc_i[d] / l_i);
            } else {
                output[out_base + d] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// F32 version of FlashAttention kernel for higher precision testing
kernel void unified_batch_prefill_attention_f32_flashattention(
    device const float* q_input [[buffer(0)]],
    device const float* paged_k_cache [[buffer(1)]],
    device const float* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant UnifiedParams& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // 3D dispatch grid: [q_tiles, kv_tiles, batch_heads]
    uint q_tile_idx = tgid.x;
    uint kv_tile_idx = tgid.y;
    uint batch_head_idx = tgid.z;
    
    // Extract sequence and head from batch_head_idx
    uint seq_id = batch_head_idx / params.num_heads;
    uint head_id = batch_head_idx % params.num_heads;
    
    if (seq_id >= uint(params.num_sequences)) return;
    
    // Map tile coordinates to actual query range for this sequence
    int seq_q_start = qo_indptr[seq_id];
    int seq_q_end = qo_indptr[seq_id + 1];
    int seq_q_len = seq_q_end - seq_q_start;
    
    // Calculate query indices for this tile - use fixed tile size of 32
    int tile_q_start = q_tile_idx * 32;
    int tile_q_end = min(tile_q_start + 32, seq_q_len);
    
    // Skip if this tile is beyond the sequence queries
    if (tile_q_start >= seq_q_len) return;
    
    // Process each query in this tile
    for (int local_q_idx = tile_q_start; local_q_idx < tile_q_end; local_q_idx++) {
        uint qo_idx = seq_q_start + local_q_idx;
        if (qo_idx >= uint(params.num_qo)) continue;
        
        // --- Get KV Page Information for current sequence ---
        int kv_start_page_pos = kv_page_indptr[seq_id];
        int kv_end_page_pos = kv_page_indptr[seq_id + 1];
        int num_pages = kv_end_page_pos - kv_start_page_pos;
        
        if (num_pages <= 0) {
            // Zero out output for this query 
            for (int d = tid_in_tgp; d < params.head_size; d += TGP_SIZE) {
                output[qo_idx * params.head_dim + head_id * params.head_size + d] = 0.0f;
            }
            continue;
        }
        
        int last_page_len = kv_last_page_lens[seq_id];
        int total_kv_len = (num_pages - 1) * params.page_size + last_page_len;
        
        // Debug output to trace execution flow
        if (q_tile_idx == 0 && kv_tile_idx == 0 && batch_head_idx == 0 && local_q_idx == tile_q_start && tid_in_tgp == 0 && debug_out != nullptr) {
            debug_out[0] = params.scale;
            debug_out[1] = float(params.head_dim);
            debug_out[2] = float(params.page_size);
            debug_out[3] = float(params.num_qo);
            debug_out[4] = float(total_kv_len);
            debug_out[5] = float(num_pages);
            debug_out[6] = float(last_page_len);
            debug_out[7] = float(q_tile_idx);
            debug_out[8] = float(kv_tile_idx);
            debug_out[9] = float(batch_head_idx);
            debug_out[10] = float(seq_q_len);       // Are we getting valid sequence length?
            debug_out[11] = float(tile_q_start);    // Are tile bounds correct?
            debug_out[12] = float(tile_q_end);
            debug_out[13] = float(local_q_idx);     // Which query are we processing?
            debug_out[14] = float(qo_idx);          // Final query index
            debug_out[15] = -1.0f;   // KV tile range - to be filled after calculation
            debug_out[16] = -1.0f;
        }
        
        // Use simple parallel approach like original but with improved online softmax
        const int BLOCK_SIZE = 16; // Process KV in blocks of 16
        const int max_head_size = min(params.head_size, 128); // Keep reasonable limit
        
        // Shared memory similar to original but with float precision
        threadgroup float q_s[128];  // Query for this head
        threadgroup float k_block[16][128];  // KV block
        threadgroup float v_block[16][128]; 
        threadgroup float w_block[16];      // Attention weights
        threadgroup float temp_reduce[TGP_SIZE];
        
        // Online softmax state for current head
        threadgroup float m_i;
        threadgroup float l_i;
        threadgroup float acc_i[128];
        // Initialize online softmax state per head
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            acc_i[d] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load Query slice for this head into shared memory
        int q_base = int(qo_idx) * params.head_dim + head_id * params.head_size;
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            q_s[d] = q_input[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Calculate KV range for this tile - proper tiled FlashAttention
        int kv_tile_size = 32;
        int tile_kv_start = kv_tile_idx * kv_tile_size;
        int tile_kv_end = min(tile_kv_start + kv_tile_size, total_kv_len);
        
        // Update debug info with KV tile ranges
        if (q_tile_idx == 0 && kv_tile_idx == 0 && batch_head_idx == 0 && local_q_idx == tile_q_start && tid_in_tgp == 0 && debug_out != nullptr) {
            debug_out[15] = float(tile_kv_start);
            debug_out[16] = float(tile_kv_end);
        }
        
        // Skip if this kv_tile is beyond the sequence KV tokens
        if (tile_kv_start >= total_kv_len) {
            // This tile has no KV to process, but don't skip - contribute zero to merge
            tile_kv_end = tile_kv_start; // Empty range
        }
        
        // Each tile computes partial attention for its assigned KV range
        // Main Loop: Process only the KV tokens assigned to this tile
        for (int block_start = tile_kv_start; block_start < tile_kv_end; block_start += BLOCK_SIZE) {
            // Load K/V for this block and head slice (same as bf16 version)
            if (tid_in_tgp < BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / params.page_size;
                    int in_page_offset = global_key_idx % params.page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    uint base_addr = page_idx * params.page_size * params.head_dim + 
                                   in_page_offset * params.head_dim + head_id * params.head_size;

                    for (int d = 0; d < max_head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            
            // Compute scores for this block (same as bf16 version)
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;
            if (tid_in_tgp < BLOCK_SIZE && global_key_idx_score < total_kv_len) {
                for (int d = 0; d < max_head_size; ++d) {
                    score += q_s[d] * k_block[tid_in_tgp][d];
                }
                score *= params.scale;
            } else {
                score = -INFINITY;
            }

            // Online softmax update (improved from original)
            // 1) Block max reduction
            temp_reduce[tid_in_tgp] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float m_j = temp_reduce[0];

            // 2) Update global max and rescale accumulators
            threadgroup float m_prev = -INFINITY;
            if (tid_in_tgp == 0) {
                m_prev = m_i;
                m_i = max(m_i, m_j);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale_factor = exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // 3) Compute weights and update accumulators
            float w = (score > -INFINITY) ? exp(score - m_i) : 0.0f;
            if (tid_in_tgp < BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Update l_i (sum of weights)
            temp_reduce[tid_in_tgp] = (tid_in_tgp < BLOCK_SIZE) ? w : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid_in_tgp == 0) {
                l_i += temp_reduce[0];
            }

            // Accumulate V slice (same as bf16 version)
            int dims_per_thread = (max_head_size + TGP_SIZE - 1) / TGP_SIZE;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid_in_tgp * dims_per_thread + i;
                if (d < max_head_size) {
                    float sum_wv_d = 0.0f;
                    int num_keys_in_block = min(BLOCK_SIZE, total_kv_len - block_start);
                    for (int j = 0; j < num_keys_in_block; ++j) {
                        sum_wv_d += w_block[j] * v_block[j][d];
                    }
                    acc_i[d] += sum_wv_d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Finalize output for this head (same as bf16 version but with f32 output)
        int out_base = int(qo_idx) * params.head_dim + head_id * params.head_size;
        for (int d = tid_in_tgp; d < max_head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = acc_i[d] / l_i;
            } else {
                output[out_base + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // End of query processing loop
}