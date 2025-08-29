#include <metal_stdlib>
using namespace metal;

// FlashInfer-style constants and configuration
constant int FLASHINFER_TILE_SIZE_Q = 64;
constant int FLASHINFER_TILE_SIZE_KV = 64;
constant int WARP_SIZE = 32;
constant int MAX_HEAD_DIM = 256;
constant int SHARED_MEM_SIZE = 32768;

// F16 numerical constants for stability
constant half F16_NEG_INFINITY = half(-65504.0f);
constant half F16_POS_INFINITY = half(65504.0f);
constant half F16_EPSILON = half(6.1e-5f);
constant half F16_SAFE_MAX = half(10.0f);

struct FlashInferParams {
    uint32_t head_dim;
    uint32_t head_size;
    uint32_t q_stride_seq;
    uint32_t q_stride_head;
    uint32_t k_stride_head;
    uint32_t v_stride_head;
    uint32_t o_stride_seq;
    uint32_t o_stride_head;
    float scale;
    uint32_t num_layers;
    uint32_t layer_idx;
    uint32_t causal;
    uint32_t num_kv_heads;
    uint32_t group_size;
    float logit_cap;
    
    // FlashInfer-specific scheduling parameters
    uint32_t num_blocks_per_seq;
    uint32_t max_seq_len;
    uint32_t block_size;
    uint32_t load_balance_factor;
};

// Block scheduling information for dynamic load balancing
struct BlockSchedule {
    uint32_t block_idx;
    uint32_t seq_idx;
    uint32_t q_start;
    uint32_t q_end;
    uint32_t kv_start;
    uint32_t kv_end;
    uint32_t workload_estimate;
};

// Online softmax state for FlashAttention algorithm
struct OnlineSoftmaxState {
    half m_i;                           // Running maximum
    half l_i;                           // Running sum
    half acc[MAX_HEAD_DIM];            // Output accumulator
};

// FlashInfer-style block masking utilities
inline bool is_causal_masked(uint32_t q_pos, uint32_t kv_pos, bool causal) {
    return causal && (kv_pos > q_pos);
}

inline bool is_block_masked(
    uint32_t q_block_start,
    uint32_t q_block_end,
    uint32_t kv_block_start, 
    uint32_t kv_block_end,
    bool causal
) {
    if (!causal) return false;
    
    // Block is masked if ALL positions would be masked
    // This happens when the minimum KV position > maximum Q position
    return (kv_block_start > q_block_end - 1);
}

// Load balancing scheduler - estimates workload for dynamic scheduling
inline uint32_t estimate_block_workload(
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t head_dim,
    bool causal
) {
    uint32_t base_workload = q_len * kv_len * head_dim;
    
    if (causal) {
        // Triangular masking reduces workload
        base_workload = base_workload / 2;
    }
    
    return base_workload;
}

// FlashInfer-style dynamic block scheduler
inline BlockSchedule create_block_schedule(
    uint32_t block_id,
    uint32_t seq_idx,
    uint32_t total_q_len,
    uint32_t total_kv_len,
    constant FlashInferParams& params
) {
    BlockSchedule schedule;
    
    // Calculate block boundaries with load balancing
    uint32_t blocks_per_seq = (total_q_len + FLASHINFER_TILE_SIZE_Q - 1) / FLASHINFER_TILE_SIZE_Q;
    uint32_t q_block_idx = block_id % blocks_per_seq;
    // uint32_t kv_blocks = (total_kv_len + FLASHINFER_TILE_SIZE_KV - 1) / FLASHINFER_TILE_SIZE_KV;
    
    schedule.block_idx = block_id;
    schedule.seq_idx = seq_idx;
    schedule.q_start = q_block_idx * FLASHINFER_TILE_SIZE_Q;
    schedule.q_end = min(schedule.q_start + FLASHINFER_TILE_SIZE_Q, total_q_len);
    
    // Dynamic KV range based on causal masking
    if (params.causal) {
        schedule.kv_start = 0;
        schedule.kv_end = min(schedule.q_end, total_kv_len); // Causal constraint
    } else {
        schedule.kv_start = 0;
        schedule.kv_end = total_kv_len;
    }
    
    schedule.workload_estimate = estimate_block_workload(
        schedule.q_end - schedule.q_start,
        schedule.kv_end - schedule.kv_start,
        params.head_dim,
        params.causal
    );
    
    return schedule;
}

// Load Q tile with bounds checking and load balancing
inline void load_q_tile_flashinfer(
    threadgroup half* Q_smem,
    device const half* Q_global,
    BlockSchedule schedule,
    uint32_t head_idx,
    constant FlashInferParams& params,
    uint32_t tid
) {
    uint32_t q_tile_len = schedule.q_end - schedule.q_start;
    uint32_t total_elements = q_tile_len * params.head_size;
    
    // Coalesced loading with multiple elements per thread
    for (uint32_t i = tid; i < total_elements; i += WARP_SIZE * 4) {
        uint32_t q_local = i / params.head_size;
        uint32_t dim_idx = i % params.head_size;
        
        if (q_local < q_tile_len && dim_idx < params.head_size) {
            uint32_t q_global_idx = (schedule.q_start + q_local) * params.q_stride_seq + 
                                   head_idx * params.q_stride_head + dim_idx;
            Q_smem[q_local * params.head_size + dim_idx] = Q_global[q_global_idx];
        } else {
            Q_smem[i] = half(0.0f);
        }
    }
}

// Load paged KV tile with FlashInfer-style block scheduling
inline void load_paged_kv_tile_flashinfer(
    threadgroup half* K_smem,
    threadgroup half* V_smem,
    device const half* paged_k_cache,
    device const half* paged_v_cache,
    device const uint32_t* kv_page_indices,
    device const uint32_t* kv_page_indptr,
    device const uint32_t* kv_last_page_lens,
    BlockSchedule schedule,
    uint32_t kv_tile_start,
    uint32_t kv_tile_size,
    uint32_t head_idx,
    constant FlashInferParams& params,
    uint32_t tid
) {
    uint32_t tokens_per_page = 16;
    uint32_t page_start = kv_page_indptr[schedule.seq_idx];
    uint32_t page_end = kv_page_indptr[schedule.seq_idx + 1];
    uint32_t num_pages = page_end - page_start;
    
    if (num_pages == 0) return;
    
    // Load KV tiles with page boundary handling
    for (uint32_t i = tid; i < kv_tile_size * params.head_size; i += WARP_SIZE * 4) {
        uint32_t kv_local = i / params.head_size;
        uint32_t dim_idx = i % params.head_size;
        uint32_t kv_pos = kv_tile_start + kv_local;
        
        if (kv_local < kv_tile_size && dim_idx < params.head_size && kv_pos < schedule.kv_end) {
            uint32_t page_offset = kv_pos / tokens_per_page;
            uint32_t pos_in_page = kv_pos % tokens_per_page;
            
            if (page_offset < num_pages) {
                // Check last page length constraint
                if (page_offset == num_pages - 1) {
                    uint32_t last_page_len = kv_last_page_lens[schedule.seq_idx];
                    if (pos_in_page >= last_page_len) {
                        K_smem[i] = half(0.0f);
                        V_smem[i] = half(0.0f);
                        continue;
                    }
                }
                
                uint32_t page_idx = kv_page_indices[page_start + page_offset];
                uint32_t global_idx = (page_idx * tokens_per_page * params.num_kv_heads * params.head_dim) +
                                     (pos_in_page * params.num_kv_heads * params.head_dim) +
                                     (head_idx * params.head_dim) + dim_idx;
                
                K_smem[i] = paged_k_cache[global_idx];
                V_smem[i] = paged_v_cache[global_idx];
            } else {
                K_smem[i] = half(0.0f);
                V_smem[i] = half(0.0f);
            }
        } else {
            K_smem[i] = half(0.0f);
            V_smem[i] = half(0.0f);
        }
    }
}

// FlashInfer-style attention score computation with block masking
inline void compute_attention_scores_flashinfer(
    threadgroup half* S_smem,
    threadgroup half* Q_smem, 
    threadgroup half* K_smem,
    BlockSchedule schedule,
    uint32_t kv_tile_start,
    uint32_t q_tile_size,
    uint32_t kv_tile_size,
    constant FlashInferParams& params,
    uint32_t tid
) {
    // Compute attention scores with efficient blocking
    for (uint32_t i = tid; i < q_tile_size * kv_tile_size; i += WARP_SIZE * 4) {
        uint32_t q_local = i / kv_tile_size;
        uint32_t kv_local = i % kv_tile_size;
        
        if (q_local < q_tile_size && kv_local < kv_tile_size) {
            uint32_t q_global_pos = schedule.q_start + q_local;
            uint32_t kv_global_pos = kv_tile_start + kv_local;
            
            // FlashInfer-style causal masking check
            if (is_causal_masked(q_global_pos, kv_global_pos, params.causal)) {
                S_smem[i] = F16_NEG_INFINITY;
                continue;
            }
            
            // Compute dot product
            half score = half(0.0f);
            for (uint32_t d = 0; d < params.head_size; d++) {
                half q_val = Q_smem[q_local * params.head_size + d];
                half k_val = K_smem[kv_local * params.head_size + d];
                score += q_val * k_val;
            }
            
            // Apply scaling and clamping for F16 stability
            score *= half(params.scale);
            score = clamp(score, -F16_SAFE_MAX, F16_SAFE_MAX);
            S_smem[i] = score;
        } else {
            S_smem[i] = F16_NEG_INFINITY;
        }
    }
}

// Online softmax update with FlashInfer optimization
inline void update_online_softmax_flashinfer(
    thread OnlineSoftmaxState& state,
    half m_j,
    half l_j,
    constant FlashInferParams& params
) {
    if (l_j <= F16_EPSILON) return; // Skip empty tiles
    
    half m_new = max(state.m_i, m_j);
    
    // Clamp differences to prevent overflow
    half m_i_diff = clamp(state.m_i - m_new, -F16_SAFE_MAX, F16_SAFE_MAX);
    half m_j_diff = clamp(m_j - m_new, -F16_SAFE_MAX, F16_SAFE_MAX);
    
    half alpha = exp(m_i_diff);
    half beta = exp(m_j_diff);
    
    // Numerical stability checks for F16
    if (alpha <= F16_EPSILON) alpha = half(0.0f);
    if (beta <= F16_EPSILON) beta = half(0.0f);
    
    half l_new = alpha * state.l_i + beta * l_j;
    
    // Prevent accumulator overflow
    if (l_new > F16_EPSILON) {
        // Update accumulator with rescaling
        for (uint32_t d = 0; d < params.head_size; d++) {
            state.acc[d] = alpha * state.acc[d];
        }
        
        state.m_i = m_new;
        state.l_i = l_new;
    }
}

// Main FlashInfer-style kernel with dynamic scheduling
kernel void flash_attention_flashinfer_f16(
    device const half* Q [[buffer(0)]],                    // Query tensor
    device const half* paged_k_cache [[buffer(1)]],        // Paged K cache
    device const half* paged_v_cache [[buffer(2)]],        // Paged V cache
    device const uint32_t* qo_indptr [[buffer(3)]],        // Query indices
    device const uint32_t* kv_indptr [[buffer(4)]],        // KV indices
    device const uint32_t* kv_page_indptr [[buffer(5)]],   // KV page indices
    device const uint32_t* kv_page_indices [[buffer(6)]],  // KV page indices
    device const uint32_t* kv_last_page_lens [[buffer(7)]], // Last page lengths
    constant FlashInferParams& params [[buffer(8)]],       // Parameters
    device half* output [[buffer(9)]],                     // Output tensor
    threadgroup half* shared_memory [[threadgroup(0)]],    // Shared memory
    uint3 gid [[thread_position_in_grid]],                // Grid position
    uint3 tgid [[threadgroup_position_in_grid]],          // Threadgroup position
    uint tid [[thread_index_in_threadgroup]],              // Thread ID
    uint3 grid_size [[threads_per_grid]]                   // Grid size
) {
    // FlashInfer-style dynamic block scheduling
    uint32_t seq_idx = tgid.x;
    uint32_t head_idx = tgid.y;
    uint32_t block_idx = tgid.z;
    
    if (seq_idx >= grid_size.x) return;
    
    uint32_t q_start = qo_indptr[seq_idx];
    uint32_t q_end = qo_indptr[seq_idx + 1]; 
    uint32_t q_len = q_end - q_start;
    uint32_t kv_start = kv_indptr[seq_idx];
    uint32_t kv_end = kv_indptr[seq_idx + 1];
    uint32_t kv_len = kv_end - kv_start;
    
    if (q_len == 0 || kv_len == 0) return;
    
    // Create dynamic block schedule
    BlockSchedule schedule = create_block_schedule(block_idx, seq_idx, q_len, kv_len, params);
    
    // Early termination for fully masked blocks
    if (is_block_masked(schedule.q_start, schedule.q_end, schedule.kv_start, schedule.kv_end, params.causal)) {
        return;
    }
    
    // Shared memory layout with FlashInfer optimization
    threadgroup half* Q_smem = shared_memory;
    threadgroup half* K_smem = Q_smem + FLASHINFER_TILE_SIZE_Q * params.head_size;
    threadgroup half* V_smem = K_smem + FLASHINFER_TILE_SIZE_KV * params.head_size;
    threadgroup half* S_smem = V_smem + FLASHINFER_TILE_SIZE_KV * params.head_size;
    
    // Initialize online softmax state
    OnlineSoftmaxState states[FLASHINFER_TILE_SIZE_Q];
    uint32_t q_tile_size = schedule.q_end - schedule.q_start;
    
    for (uint32_t q_local = 0; q_local < q_tile_size; q_local++) {
        states[q_local].m_i = F16_NEG_INFINITY;
        states[q_local].l_i = half(0.0f);
        for (uint32_t d = 0; d < params.head_size; d++) {
            states[q_local].acc[d] = half(0.0f);
        }
    }
    
    // Load Q tile once with FlashInfer optimization
    load_q_tile_flashinfer(Q_smem, Q, schedule, head_idx, params, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process KV tiles with dynamic scheduling
    uint32_t kv_tiles = (schedule.kv_end - schedule.kv_start + FLASHINFER_TILE_SIZE_KV - 1) / FLASHINFER_TILE_SIZE_KV;
    
    for (uint32_t kv_tile_idx = 0; kv_tile_idx < kv_tiles; kv_tile_idx++) {
        uint32_t kv_tile_start = schedule.kv_start + kv_tile_idx * FLASHINFER_TILE_SIZE_KV;
        uint32_t kv_tile_size = min(uint32_t(FLASHINFER_TILE_SIZE_KV), schedule.kv_end - kv_tile_start);
        
        // Load KV tile for this iteration
        load_paged_kv_tile_flashinfer(
            K_smem, V_smem, paged_k_cache, paged_v_cache,
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            schedule, kv_tile_start, kv_tile_size, head_idx, params, tid
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores with block masking
        compute_attention_scores_flashinfer(S_smem, Q_smem, K_smem, schedule, kv_tile_start, q_tile_size, kv_tile_size, params, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Online softmax and value accumulation
        for (uint32_t q_local = tid; q_local < q_tile_size; q_local += WARP_SIZE) {
            // Find max score for this query
            half m_j = F16_NEG_INFINITY;
            for (uint32_t kv_local = 0; kv_local < kv_tile_size; kv_local++) {
                half score = S_smem[q_local * FLASHINFER_TILE_SIZE_KV + kv_local];
                if (score > F16_NEG_INFINITY + half(1.0f)) {
                    m_j = max(m_j, score);
                }
            }
            
            // Compute softmax denominator for this tile
            half l_j = half(0.0f);
            for (uint32_t kv_local = 0; kv_local < kv_tile_size; kv_local++) {
                half score = S_smem[q_local * FLASHINFER_TILE_SIZE_KV + kv_local];
                if (score > F16_NEG_INFINITY + half(1.0f)) {
                    half prob = exp(score - m_j);
                    S_smem[q_local * FLASHINFER_TILE_SIZE_KV + kv_local] = prob;
                    l_j += prob;
                }
            }
            
            // Update online softmax state
            update_online_softmax_flashinfer(states[q_local], m_j, l_j, params);
            
            // Accumulate weighted values with proper scaling
            half beta = (states[q_local].l_i > F16_EPSILON) ? exp(m_j - states[q_local].m_i) : half(1.0f);
            for (uint32_t d = 0; d < params.head_size; d++) {
                half weighted_sum = half(0.0f);
                for (uint32_t kv_local = 0; kv_local < kv_tile_size; kv_local++) {
                    half prob = S_smem[q_local * FLASHINFER_TILE_SIZE_KV + kv_local];
                    if (prob > F16_EPSILON) {
                        half v_val = V_smem[kv_local * params.head_size + d];
                        weighted_sum += prob * v_val;
                    }
                }
                // Apply proper weighted accumulation
                states[q_local].acc[d] += beta * weighted_sum;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Finalize and write output with FlashInfer normalization
    for (uint32_t q_local = tid; q_local < q_tile_size; q_local += WARP_SIZE) {
        // Normalize final output with F16 safety
        half norm_factor = half(1.0f);
        if (states[q_local].l_i > F16_EPSILON) {
            norm_factor = half(1.0f) / states[q_local].l_i;
            // Clamp normalization factor to prevent overflow
            norm_factor = clamp(norm_factor, half(0.0f), half(100.0f));
        }
        
        for (uint32_t d = 0; d < params.head_size; d++) {
            uint32_t output_idx = (schedule.q_start + q_local) * params.o_stride_seq +
                                 head_idx * params.o_stride_head + d;
            
            half result = states[q_local].acc[d] * norm_factor;
            // Final clamp to prevent F16 overflow in output
            result = clamp(result, -half(50.0f), half(50.0f));
            output[output_idx] = result;
        }
    }
}