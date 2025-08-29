#include <metal_stdlib>
using namespace metal;

// FlashAttention tile configuration
constant int TILE_SIZE_Q = 64;    // Query tile size
constant int TILE_SIZE_KV = 64;   // Key/Value tile size
constant int MAX_HEAD_DIM = 256;  // Maximum head dimension
constant int WARP_SIZE = 32;      // Metal SIMD group size

// F16-specific constants for numerical stability
constant half F16_NEG_INFINITY = half(-65504.0f);
constant half F16_POS_INFINITY = half(65504.0f);
constant half F16_EPSILON = half(6.1e-5f);
constant half F16_SAFE_MAX = half(10.0f);

struct FlashAttentionParams {
    uint32_t head_dim;
    uint32_t head_size;
    uint32_t seq_len_q;
    uint32_t seq_len_kv;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t group_size;
    float scale;
    bool causal_mask;
    uint32_t batch_size;
    // Paging parameters
    uint32_t page_size;
    uint32_t tokens_per_page;
};

// Online softmax state for each query row
struct OnlineSoftmaxState {
    half m_i;    // Running maximum
    half l_i;    // Running sum
    half acc[MAX_HEAD_DIM];  // Output accumulator
};

// Load Q tile to shared memory
inline void load_q_tile(
    threadgroup half* Q_smem,
    device const half* Q_global,
    uint q_tile_start,
    uint head_idx,
    uint seq_idx,
    constant FlashAttentionParams& params,
    uint tid
) {
    uint threads_per_tile = TILE_SIZE_Q * params.head_size;
    
    for (uint i = tid; i < threads_per_tile; i += WARP_SIZE * 4) { // 4 warps per threadgroup
        uint q_local = i / params.head_size;
        uint d = i % params.head_size;
        uint q_global_idx = q_tile_start + q_local;
        
        if (q_global_idx < params.seq_len_q && d < params.head_size) {
            uint global_idx = q_global_idx * params.head_dim + head_idx * params.head_size + d;
            Q_smem[q_local * params.head_size + d] = Q_global[global_idx];
        } else {
            Q_smem[q_local * params.head_size + d] = half(0.0f);
        }
    }
}

// Load paged KV tile to shared memory
inline void load_paged_kv_tile(
    threadgroup half* K_smem,
    threadgroup half* V_smem,
    device const half* paged_k_cache,
    device const half* paged_v_cache,
    device const uint32_t* kv_page_indptr,
    device const uint32_t* kv_page_indices,
    device const uint32_t* kv_last_page_lens,
    uint kv_tile_start,
    uint head_idx,
    uint seq_idx,
    constant FlashAttentionParams& params,
    uint tid
) {
    uint page_start = kv_page_indptr[seq_idx];
    uint page_end = kv_page_indptr[seq_idx + 1];
    uint num_pages = page_end - page_start;
    
    if (num_pages == 0) {
        // Zero out shared memory
        for (uint i = tid; i < TILE_SIZE_KV * params.head_size; i += WARP_SIZE * 4) {
            K_smem[i] = half(0.0f);
            V_smem[i] = half(0.0f);
        }
        return;
    }
    
    uint threads_per_tile = TILE_SIZE_KV * params.head_size;
    
    for (uint i = tid; i < threads_per_tile; i += WARP_SIZE * 4) {
        uint kv_local = i / params.head_size;
        uint d = i % params.head_size;
        uint kv_global_idx = kv_tile_start + kv_local;
        
        if (kv_global_idx < params.seq_len_kv && d < params.head_size) {
            // Map global KV index to page and offset
            uint page_offset = kv_global_idx / params.tokens_per_page;
            uint pos_in_page = kv_global_idx % params.tokens_per_page;
            
            if (page_offset < num_pages) {
                // Handle last page length restriction
                bool valid_token = true;
                if (page_offset == num_pages - 1) {
                    uint last_page_len = kv_last_page_lens[seq_idx];
                    if (pos_in_page >= last_page_len) {
                        valid_token = false;
                    }
                }
                
                if (valid_token) {
                    uint physical_page = kv_page_indices[page_start + page_offset];
                    uint global_k_idx = (physical_page * params.tokens_per_page * params.head_dim) +
                                       (pos_in_page * params.head_dim) +
                                       (head_idx * params.head_size) + d;
                    uint global_v_idx = global_k_idx; // Same layout for V
                    
                    K_smem[kv_local * params.head_size + d] = paged_k_cache[global_k_idx];
                    V_smem[kv_local * params.head_size + d] = paged_v_cache[global_v_idx];
                } else {
                    K_smem[kv_local * params.head_size + d] = half(0.0f);
                    V_smem[kv_local * params.head_size + d] = half(0.0f);
                }
            } else {
                K_smem[kv_local * params.head_size + d] = half(0.0f);
                V_smem[kv_local * params.head_size + d] = half(0.0f);
            }
        } else {
            K_smem[kv_local * params.head_size + d] = half(0.0f);
            V_smem[kv_local * params.head_size + d] = half(0.0f);
        }
    }
}

// Compute attention scores for the tile: S = Q @ K^T
inline void compute_attention_scores(
    threadgroup half* S_smem,
    threadgroup const half* Q_smem,
    threadgroup const half* K_smem,
    uint q_tile_size,
    uint kv_tile_size,
    constant FlashAttentionParams& params,
    uint tid
) {
    // Each thread computes one element of S
    uint total_scores = q_tile_size * kv_tile_size;
    
    for (uint idx = tid; idx < total_scores; idx += WARP_SIZE * 4) {
        uint q_idx = idx / kv_tile_size;
        uint kv_idx = idx % kv_tile_size;
        
        half score = half(0.0f);
        for (uint d = 0; d < params.head_size; d++) {
            half q_val = Q_smem[q_idx * params.head_size + d];
            half k_val = K_smem[kv_idx * params.head_size + d];
            score += q_val * k_val;
        }
        
        score *= half(params.scale);
        
        // Apply causal mask if needed
        if (params.causal_mask && kv_idx > q_idx) {
            score = F16_NEG_INFINITY;
        }
        
        // Clamp to safe range
        score = clamp(score, -F16_SAFE_MAX, F16_SAFE_MAX);
        S_smem[q_idx * kv_tile_size + kv_idx] = score;
    }
}

// Compute row-wise max using Metal SIMD operations
inline half compute_row_max(
    threadgroup const half* S_smem,
    uint row,
    uint kv_tile_size,
    uint tid_in_simd,
    uint simd_id
) {
    half local_max = F16_NEG_INFINITY;
    
    // Each SIMD group processes elements of the row
    for (uint k = tid_in_simd; k < kv_tile_size; k += WARP_SIZE) {
        local_max = max(local_max, S_smem[row * kv_tile_size + k]);
    }
    
    // Reduce within SIMD group
    local_max = simd_max(local_max);
    
    return local_max;
}

// Compute row-wise sum using Metal SIMD operations
inline half compute_row_sum(
    threadgroup const half* S_smem,
    uint row,
    uint kv_tile_size,
    half row_max,
    uint tid_in_simd,
    uint simd_id
) {
    half local_sum = half(0.0f);
    
    // Each SIMD group processes elements of the row
    for (uint k = tid_in_simd; k < kv_tile_size; k += WARP_SIZE) {
        half val = S_smem[row * kv_tile_size + k];
        if (val > F16_NEG_INFINITY + half(1.0f)) {
            local_sum += exp(val - row_max);
        }
    }
    
    // Reduce within SIMD group
    local_sum = simd_sum(local_sum);
    
    return local_sum;
}

// Update online softmax statistics
inline void update_online_softmax(
    thread OnlineSoftmaxState& state,
    half m_j,
    half l_j,
    constant FlashAttentionParams& params
) {
    half m_new = max(state.m_i, m_j);
    half alpha = exp(state.m_i - m_new);
    half beta = exp(m_j - m_new);
    
    // Rescale previous accumulator
    for (uint d = 0; d < params.head_size; d++) {
        state.acc[d] *= alpha;
    }
    
    // Update running statistics
    state.l_i = alpha * state.l_i + beta * l_j;
    state.m_i = m_new;
}

// Accumulate weighted values: acc += P @ V
inline void accumulate_weighted_values(
    thread OnlineSoftmaxState& state,
    threadgroup const half* S_smem,
    threadgroup const half* V_smem,
    uint q_idx,
    uint kv_tile_size,
    constant FlashAttentionParams& params,
    half row_max,
    uint tid
) {
    // Compute softmax weights and accumulate
    for (uint d = 0; d < params.head_size; d++) {
        half weighted_sum = half(0.0f);
        
        for (uint kv_idx = 0; kv_idx < kv_tile_size; kv_idx++) {
            half score = S_smem[q_idx * kv_tile_size + kv_idx];
            half weight = half(0.0f);
            
            if (score > F16_NEG_INFINITY + half(1.0f)) {
                weight = exp(score - row_max);
            }
            
            half v_val = V_smem[kv_idx * params.head_size + d];
            weighted_sum += weight * v_val;
        }
        
        state.acc[d] += weighted_sum;
    }
}

kernel void flash_attention_f16_tiled(
    device const half* Q [[buffer(0)]],                     // [batch_size, seq_len_q, num_heads, head_size]
    device const half* paged_k_cache [[buffer(1)]],         // Paged K cache
    device const half* paged_v_cache [[buffer(2)]],         // Paged V cache
    device const uint32_t* qo_indptr [[buffer(3)]],         // Query sequence boundaries
    device const uint32_t* kv_indptr [[buffer(4)]],         // KV sequence boundaries
    device const uint32_t* kv_page_indptr [[buffer(5)]],    // KV page boundaries
    device const uint32_t* kv_page_indices [[buffer(6)]],   // Physical page indices
    device const uint32_t* kv_last_page_lens [[buffer(7)]], // Last page lengths
    constant FlashAttentionParams& params [[buffer(8)]],    // Parameters
    device half* O [[buffer(9)]],                           // Output [batch_size, seq_len_q, num_heads, head_size]
    threadgroup half* shared_memory [[threadgroup(0)]],     // Shared memory
    uint3 tgid [[threadgroup_position_in_grid]],            // Threadgroup ID
    uint tid [[thread_index_in_threadgroup]],               // Thread ID
    uint simd_id [[simdgroup_index_in_threadgroup]],        // SIMD group ID
    uint tid_in_simd [[thread_index_in_simdgroup]]          // Thread ID within SIMD group
) {
    // Grid layout: [batch_size, num_heads, num_q_tiles]
    uint batch_idx = tgid.x;
    uint head_idx = tgid.y;
    uint q_tile_idx = tgid.z;
    
    if (batch_idx >= params.batch_size || head_idx >= params.num_heads) return;
    
    // Get sequence boundaries
    uint q_start = qo_indptr[batch_idx];
    uint q_end = qo_indptr[batch_idx + 1];
    uint seq_len_q = q_end - q_start;
    
    uint kv_start = kv_indptr[batch_idx];
    uint kv_end = kv_indptr[batch_idx + 1];
    uint seq_len_kv = kv_end - kv_start;
    
    if (seq_len_q == 0 || seq_len_kv == 0) return;
    
    // Calculate tile boundaries
    uint q_tile_start = q_tile_idx * TILE_SIZE_Q;
    uint q_tile_end = min(q_tile_start + TILE_SIZE_Q, seq_len_q);
    uint q_tile_size = q_tile_end - q_tile_start;
    
    if (q_tile_size == 0) return;
    
    // Shared memory layout
    threadgroup half* Q_smem = shared_memory;                                              // [TILE_Q, head_size]
    threadgroup half* K_smem = Q_smem + TILE_SIZE_Q * params.head_size;                   // [TILE_KV, head_size]
    threadgroup half* V_smem = K_smem + TILE_SIZE_KV * params.head_size;                  // [TILE_KV, head_size]
    threadgroup half* S_smem = V_smem + TILE_SIZE_KV * params.head_size;                  // [TILE_Q, TILE_KV]
    
    // Load Q tile once
    load_q_tile(Q_smem, Q, q_tile_start + q_start, head_idx, batch_idx, params, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Initialize online softmax state for each query in the tile
    OnlineSoftmaxState states[TILE_SIZE_Q];
    for (uint q_local = 0; q_local < q_tile_size; q_local++) {
        states[q_local].m_i = F16_NEG_INFINITY;
        states[q_local].l_i = half(0.0f);
        for (uint d = 0; d < params.head_size; d++) {
            states[q_local].acc[d] = half(0.0f);
        }
    }
    
    // Process KV tiles
    uint num_kv_tiles = (seq_len_kv + TILE_SIZE_KV - 1) / TILE_SIZE_KV;
    
    for (uint kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        uint kv_tile_start = kv_tile_idx * TILE_SIZE_KV;
        uint kv_tile_end = min(kv_tile_start + TILE_SIZE_KV, seq_len_kv);
        uint kv_tile_size = kv_tile_end - kv_tile_start;
        
        if (kv_tile_size == 0) break;
        
        // Load KV tile
        load_paged_kv_tile(K_smem, V_smem, paged_k_cache, paged_v_cache,
                          kv_page_indptr, kv_page_indices, kv_last_page_lens,
                          kv_tile_start + kv_start, head_idx, batch_idx, params, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores: S = Q @ K^T
        compute_attention_scores(S_smem, Q_smem, K_smem, q_tile_size, kv_tile_size, params, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Online softmax update for each query row
        for (uint q_local = 0; q_local < q_tile_size; q_local++) {
            // Compute row statistics
            half m_j = compute_row_max(S_smem, q_local, kv_tile_size, tid_in_simd, simd_id);
            half l_j = compute_row_sum(S_smem, q_local, kv_tile_size, m_j, tid_in_simd, simd_id);
            
            // Update online softmax state
            update_online_softmax(states[q_local], m_j, l_j, params);
            
            // Accumulate weighted values
            accumulate_weighted_values(states[q_local], S_smem, V_smem, q_local, kv_tile_size, params, m_j, tid);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final output with normalization
    for (uint q_local = 0; q_local < q_tile_size; q_local++) {
        uint q_global = q_tile_start + q_local + q_start;
        
        for (uint d = tid; d < params.head_size; d += WARP_SIZE * 4) {
            half final_val = half(0.0f);
            if (states[q_local].l_i > F16_EPSILON) {
                final_val = states[q_local].acc[d] / states[q_local].l_i;
            }
            
            uint global_out_idx = q_global * params.head_dim + head_idx * params.head_size + d;
            O[global_out_idx] = final_val;
        }
    }
}