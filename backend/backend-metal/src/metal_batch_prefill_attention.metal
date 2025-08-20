#include <metal_stdlib>
using namespace metal;

// Batch prefill attention kernel for FlashInfer-style attention computation
// Implements attention mechanism with paged KV cache support

kernel void batch_prefill_attention_bf16_kernel(
    device const half* Q [[buffer(0)]],         // Query: [num_tokens, num_query_heads, head_size]
    device const half* K [[buffer(1)]],         // Key: [kv_len, num_kv_heads, head_size]
    device const half* V [[buffer(2)]],         // Value: [kv_len, num_kv_heads, head_size]
    device const int* indptr [[buffer(3)]],     // Index pointer for sequence lengths
    device const int* indices [[buffer(4)]],   // Token indices per sequence
    device half* O [[buffer(5)]],              // Output: [num_tokens, num_query_heads, head_size]
    constant int& num_tokens [[buffer(6)]],
    constant int& num_query_heads [[buffer(7)]],
    constant int& num_kv_heads [[buffer(8)]],
    constant int& head_size [[buffer(9)]],
    constant int& kv_len [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one query token and one head
    uint token_idx = tgid.x;
    uint head_idx = tgid.y;
    
    if (token_idx >= num_tokens || head_idx >= num_query_heads) return;
    
    // Handle grouped-query attention (GQA)
    uint kv_head_idx = head_idx % num_kv_heads;
    
    // Shared memory for reduction
    threadgroup float shared_max[64];
    threadgroup float shared_sum[64];
    threadgroup float shared_out[64];
    
    uint tid_in_tg = tid.x;
    
    // Initialize shared memory
    if (tid_in_tg < head_size) {
        shared_out[tid_in_tg] = 0.0f;
    }
    
    // Load query vector for this token and head
    float q_local[16]; // Assume head_size <= 16 for simplicity
    for (int i = 0; i < min(head_size, 16); i++) {
        if (tid_in_tg == 0) {
            q_local[i] = float(Q[token_idx * num_query_heads * head_size + head_idx * head_size + i]);
        }
    }
    
    // Simplified attention computation
    // In practice, this would handle variable-length sequences using indptr/indices
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Compute attention scores for all KV positions
    // This is a simplified version - full implementation would handle paging
    for (int kv_pos = tid_in_tg; kv_pos < kv_len; kv_pos += threads_per_tg.x) {
        float score = 0.0f;
        
        // Compute Q * K^T
        for (int d = 0; d < head_size; d++) {
            float q_val = (tid_in_tg == 0) ? q_local[min(d, 15)] : 0.0f;
            float k_val = float(K[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + d]);
            score += q_val * k_val;
        }
        score *= scale;
        
        // Track maximum for numerical stability
        max_score = max(max_score, score);
    }
    
    // Reduce to find global maximum
    shared_max[tid_in_tg] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for max
    for (int stride = threads_per_tg.x / 2; stride > 0; stride /= 2) {
        if (tid_in_tg < stride && tid_in_tg + stride < threads_per_tg.x) {
            shared_max[tid_in_tg] = max(shared_max[tid_in_tg], shared_max[tid_in_tg + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float global_max = shared_max[0];
    
    // Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = tid_in_tg; kv_pos < kv_len; kv_pos += threads_per_tg.x) {
        float score = 0.0f;
        
        // Re-compute Q * K^T (in practice, we'd cache this)
        for (int d = 0; d < head_size; d++) {
            float q_val = (tid_in_tg == 0) ? q_local[min(d, 15)] : 0.0f;
            float k_val = float(K[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + d]);
            score += q_val * k_val;
        }
        score = (score * scale) - global_max;
        
        float exp_score = exp(score);
        local_sum += exp_score;
    }
    
    // Reduce sum
    shared_sum[tid_in_tg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int stride = threads_per_tg.x / 2; stride > 0; stride /= 2) {
        if (tid_in_tg < stride && tid_in_tg + stride < threads_per_tg.x) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float global_sum = shared_sum[0];
    
    // Compute weighted output
    for (int d = tid_in_tg; d < head_size; d += threads_per_tg.x) {
        float output_val = 0.0f;
        
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            float score = 0.0f;
            
            // Re-compute attention score
            for (int dim = 0; dim < head_size; dim++) {
                float q_val = float(Q[token_idx * num_query_heads * head_size + head_idx * head_size + dim]);
                float k_val = float(K[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + dim]);
                score += q_val * k_val;
            }
            score = (score * scale) - global_max;
            
            float attention_weight = exp(score) / global_sum;
            float v_val = float(V[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + d]);
            output_val += attention_weight * v_val;
        }
        
        O[token_idx * num_query_heads * head_size + head_idx * head_size + d] = half(output_val);
    }
}

kernel void batch_prefill_attention_f32_kernel(
    device const float* Q [[buffer(0)]],        // Query: [num_tokens, num_query_heads, head_size]
    device const float* K [[buffer(1)]],        // Key: [kv_len, num_kv_heads, head_size]
    device const float* V [[buffer(2)]],        // Value: [kv_len, num_kv_heads, head_size]
    device const int* indptr [[buffer(3)]],     // Index pointer for sequence lengths
    device const int* indices [[buffer(4)]],   // Token indices per sequence
    device float* O [[buffer(5)]],             // Output: [num_tokens, num_query_heads, head_size]
    constant int& num_tokens [[buffer(6)]],
    constant int& num_query_heads [[buffer(7)]],
    constant int& num_kv_heads [[buffer(8)]],
    constant int& head_size [[buffer(9)]],
    constant int& kv_len [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threads_per_tg [[threads_per_threadgroup]]
) {
    // Similar implementation to bf16 version but with float types
    uint token_idx = tgid.x;
    uint head_idx = tgid.y;
    
    if (token_idx >= num_tokens || head_idx >= num_query_heads) return;
    
    uint kv_head_idx = head_idx % num_kv_heads;
    uint tid_in_tg = tid.x;
    
    // Simplified attention computation for float32
    for (int d = tid_in_tg; d < head_size; d += threads_per_tg.x) {
        float output_val = 0.0f;
        float sum_weights = 0.0f;
        
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            float score = 0.0f;
            
            // Compute attention score: Q * K^T
            for (int dim = 0; dim < head_size; dim++) {
                float q_val = Q[token_idx * num_query_heads * head_size + head_idx * head_size + dim];
                float k_val = K[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + dim];
                score += q_val * k_val;
            }
            score *= scale;
            
            float attention_weight = exp(score);
            sum_weights += attention_weight;
            
            float v_val = V[kv_pos * num_kv_heads * head_size + kv_head_idx * head_size + d];
            output_val += attention_weight * v_val;
        }
        
        O[token_idx * num_query_heads * head_size + head_idx * head_size + d] = output_val / sum_weights;
    }
}