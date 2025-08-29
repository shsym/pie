#include <metal_stdlib>
using namespace metal;

// F16 numerical constants for stability
constant half F16_NEG_INFINITY = half(-65504.0f);
constant half F16_EPSILON = half(6.1e-5f);
constant half F16_SAFE_MAX = half(10.0f);

// Block processing granularity (computational)
constant int BLOCK_SIZE = 8;
constant int MAX_HEAD_DIM = 256;

struct BlockOptimizedAttentionParams {
    uint32_t head_dim;
    uint32_t head_size;
    uint32_t q_stride_seq;
    uint32_t q_stride_head;
    uint32_t o_stride_seq;
    uint32_t o_stride_head;
    float scale;
    uint32_t causal;
    uint32_t num_kv_heads;
    uint32_t group_size;
};

// Online softmax state for FlashAttention
struct OnlineSoftmaxState {
    half m_i;                          // Running maximum
    half l_i;                          // Running sum
    half acc[MAX_HEAD_DIM];           // Output accumulator
};

// Update online softmax state with new block results
inline void update_online_softmax(
    thread OnlineSoftmaxState& state,
    half m_j,
    half l_j,
    thread half* block_output,
    uint32_t head_size
) {
    if (l_j <= F16_EPSILON) return; // Skip empty blocks
    
    half m_new = max(state.m_i, m_j);
    
    // Clamp differences for F16 stability
    half m_i_diff = clamp(state.m_i - m_new, -F16_SAFE_MAX, F16_SAFE_MAX);
    half m_j_diff = clamp(m_j - m_new, -F16_SAFE_MAX, F16_SAFE_MAX);
    
    half alpha = exp(m_i_diff);
    half beta = exp(m_j_diff);
    
    // Numerical stability checks
    if (alpha <= F16_EPSILON) alpha = half(0.0f);
    if (beta <= F16_EPSILON) beta = half(0.0f);
    
    half l_new = alpha * state.l_i + beta * l_j;
    
    // Update accumulator and state
    if (l_new > F16_EPSILON) {
        for (uint32_t d = 0; d < head_size; d++) {
            state.acc[d] = alpha * state.acc[d] + beta * block_output[d];
        }
        state.m_i = m_new;
        state.l_i = l_new;
    }
}

// Block-optimized FlashAttention kernel using existing page abstraction
kernel void block_optimized_flash_attention_f16(
    device const half* Q [[buffer(0)]],                    // Query tensor
    device const half* paged_k_cache [[buffer(1)]],        // Paged K cache
    device const half* paged_v_cache [[buffer(2)]],        // Paged V cache
    device const uint32_t* qo_indptr [[buffer(3)]],        // Query indices
    device const uint32_t* kv_page_indptr [[buffer(4)]],   // KV page indices
    device const uint32_t* kv_page_indices [[buffer(5)]],  // Page indices
    device const uint32_t* kv_last_page_lens [[buffer(6)]], // Last page lengths
    constant BlockOptimizedAttentionParams& params [[buffer(7)]], // Parameters
    device half* output [[buffer(8)]],                     // Output tensor
    threadgroup half* shared_memory [[threadgroup(0)]],    // Shared memory
    uint3 gid [[thread_position_in_grid]],                // Grid position
    uint tid [[thread_index_in_threadgroup]],              // Thread ID
    uint3 tgid [[threadgroup_position_in_grid]],          // Threadgroup position
    uint3 threadgroup_size [[threads_per_threadgroup]]    // Threadgroup size
) {
    const uint32_t seq_idx = gid.x;        // Sequence index
    const uint32_t head_idx = gid.y;       // Head index
    const uint32_t q_token_idx = gid.z;    // Query token index
    
    // Get sequence boundaries using provided indices
    const uint32_t q_start = qo_indptr[seq_idx];
    const uint32_t q_end = qo_indptr[seq_idx + 1];
    const uint32_t q_len = q_end - q_start;
    
    if (q_token_idx >= q_len) return;
    
    // Get KV page boundaries using provided indices
    const uint32_t kv_page_start = kv_page_indptr[seq_idx];
    const uint32_t kv_page_end = kv_page_indptr[seq_idx + 1];
    const uint32_t num_pages = kv_page_end - kv_page_start;
    
    if (num_pages == 0) return;
    
    // Calculate KV length using provided page info
    const uint32_t tokens_per_page = 16;
    const uint32_t last_page_len = kv_last_page_lens[seq_idx];
    const uint32_t kv_len = (num_pages > 0) ? (num_pages - 1) * tokens_per_page + last_page_len : 0;
    
    if (kv_len == 0) return;
    
    // Calculate KV head for GQA
    const uint32_t kv_head_idx = head_idx / params.group_size;
    
    // Shared memory layout for block processing
    threadgroup half* Q_block = shared_memory;
    threadgroup half* K_block = Q_block + BLOCK_SIZE * params.head_size;
    threadgroup half* V_block = K_block + BLOCK_SIZE * params.head_size;
    threadgroup half* S_block = V_block + BLOCK_SIZE * params.head_size;
    
    // Load query for this token
    half q_vec[MAX_HEAD_DIM];
    const uint32_t q_global_idx = (q_start + q_token_idx) * params.q_stride_seq + head_idx * params.q_stride_head;
    for (uint32_t d = 0; d < params.head_size; d++) {
        q_vec[d] = Q[q_global_idx + d];
    }
    
    // Initialize online softmax state
    OnlineSoftmaxState state;
    state.m_i = F16_NEG_INFINITY;
    state.l_i = half(0.0f);
    for (uint32_t d = 0; d < params.head_size; d++) {
        state.acc[d] = half(0.0f);
    }
    
    // Process KV cache in blocks using provided page indices
    const uint32_t num_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const uint32_t block_start = block_idx * BLOCK_SIZE;
        const uint32_t block_size = min(uint32_t(BLOCK_SIZE), kv_len - block_start);
        
        // Load K and V block using page abstraction
        for (uint32_t i = tid; i < block_size * params.head_size; i += threadgroup_size.x) {
            uint32_t local_kv_idx = i / params.head_size;
            uint32_t dim_idx = i % params.head_size;
            uint32_t global_kv_pos = block_start + local_kv_idx;
            
            if (global_kv_pos < kv_len) {
                // Use provided page structure directly
                uint32_t page_offset = global_kv_pos / tokens_per_page;
                uint32_t pos_in_page = global_kv_pos % tokens_per_page;
                
                if (page_offset < num_pages) {
                    // Check last page constraint
                    if (page_offset == num_pages - 1 && pos_in_page >= last_page_len) {
                        K_block[i] = half(0.0f);
                        V_block[i] = half(0.0f);
                        continue;
                    }
                    
                    // Get page index from provided array
                    uint32_t page_idx = kv_page_indices[kv_page_start + page_offset];
                    uint32_t cache_idx = (page_idx * tokens_per_page * params.num_kv_heads * params.head_size) +
                                        (pos_in_page * params.num_kv_heads * params.head_size) +
                                        (kv_head_idx * params.head_size) + dim_idx;
                    
                    K_block[i] = paged_k_cache[cache_idx];
                    V_block[i] = paged_v_cache[cache_idx];
                } else {
                    K_block[i] = half(0.0f);
                    V_block[i] = half(0.0f);
                }
            } else {
                K_block[i] = half(0.0f);
                V_block[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores for this block
        if (tid == 0) {
            half m_j = F16_NEG_INFINITY;
            half block_output[MAX_HEAD_DIM];
            for (uint32_t d = 0; d < params.head_size; d++) {
                block_output[d] = half(0.0f);
            }
            
            // Compute scores and find max
            for (uint32_t kv_local = 0; kv_local < block_size; kv_local++) {
                uint32_t kv_global_pos = block_start + kv_local;
                
                // Causal masking
                if (params.causal && kv_global_pos > (q_start + q_token_idx)) {
                    S_block[kv_local] = F16_NEG_INFINITY;
                    continue;
                }
                
                // Compute dot product
                half score = half(0.0f);
                for (uint32_t d = 0; d < params.head_size; d++) {
                    half k_val = K_block[kv_local * params.head_size + d];
                    score += q_vec[d] * k_val;
                }
                
                // Apply scaling and clamp
                score *= half(params.scale);
                score = clamp(score, -F16_SAFE_MAX, F16_SAFE_MAX);
                S_block[kv_local] = score;
                
                if (score > F16_NEG_INFINITY + half(1.0f)) {
                    m_j = max(m_j, score);
                }
            }
            
            // Compute softmax and accumulate values
            half l_j = half(0.0f);
            for (uint32_t kv_local = 0; kv_local < block_size; kv_local++) {
                half score = S_block[kv_local];
                if (score > F16_NEG_INFINITY + half(1.0f)) {
                    half prob = exp(score - m_j);
                    S_block[kv_local] = prob;
                    l_j += prob;
                    
                    // Accumulate weighted values
                    for (uint32_t d = 0; d < params.head_size; d++) {
                        half v_val = V_block[kv_local * params.head_size + d];
                        block_output[d] += prob * v_val;
                    }
                } else {
                    S_block[kv_local] = half(0.0f);
                }
            }
            
            // Update online softmax state
            update_online_softmax(state, m_j, l_j, block_output, params.head_size);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final output with normalization
    if (tid == 0) {
        const uint32_t output_idx = (q_start + q_token_idx) * params.o_stride_seq + head_idx * params.o_stride_head;
        
        // Final normalization with F16 safety
        half norm_factor = half(1.0f);
        if (state.l_i > F16_EPSILON) {
            norm_factor = half(1.0f) / state.l_i;
            norm_factor = clamp(norm_factor, half(0.0f), half(100.0f));
        }
        
        for (uint32_t d = 0; d < params.head_size; d++) {
            half result = state.acc[d] * norm_factor;
            result = clamp(result, -half(50.0f), half(50.0f));
            output[output_idx + d] = result;
        }
    }
}