#include <metal_stdlib>
using namespace metal;

// F16-specific constants for numerical stability
constant half F16_NEG_INFINITY = half(-65504.0f);  // F16 negative infinity
constant half F16_POS_INFINITY = half(65504.0f);   // F16 positive infinity
constant half F16_EPSILON = half(6.1e-5f);         // F16 minimum normal value
constant half F16_SAFE_MAX = half(10.0f);          // Safe max for exp() to prevent overflow

struct AttentionParams {
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
};

kernel void unified_batch_prefill_attention_f16_native(
    device const half* q_input [[buffer(0)]],           // F16 query input
    device const half* paged_k_cache [[buffer(1)]],     // F16 key cache
    device const half* paged_v_cache [[buffer(2)]],     // F16 value cache
    device const uint32_t* qo_indptr [[buffer(3)]],     // Query output indices
    device const uint32_t* kv_indptr [[buffer(4)]],     // KV indices
    device const uint32_t* kv_page_indptr [[buffer(5)]], // KV page indices
    device const uint32_t* kv_page_indices [[buffer(6)]], // KV page indices
    device const uint32_t* kv_last_page_lens [[buffer(7)]], // Last page lengths
    constant AttentionParams& params [[buffer(8)]],     // Parameters
    device half* output [[buffer(9)]],                  // F16 output
    uint3 gid [[thread_position_in_grid]],              // 3D grid position
    uint3 grid_size [[threads_per_grid]]                // Grid dimensions
) {
    const uint32_t seq_idx = gid.x;      // Sequence index
    const uint32_t qo_head_idx = gid.y;  // Query/Output head index
    const uint32_t q_idx = gid.z;        // Query token index
    
    // Get sequence bounds
    if (seq_idx >= grid_size.x) return;
    const uint32_t q_start = qo_indptr[seq_idx];
    const uint32_t q_end = qo_indptr[seq_idx + 1];
    const uint32_t q_len = q_end - q_start;
    
    if (q_idx >= q_len) return;
    
    // KV cache bounds
    const uint32_t kv_start = kv_indptr[seq_idx];
    const uint32_t kv_end = kv_indptr[seq_idx + 1];
    const uint32_t kv_len = kv_end - kv_start;
    
    if (kv_len == 0) return;
    
    // Calculate KV head index (for GQA support)
    const uint32_t kv_head_idx = qo_head_idx / params.group_size;
    const uint32_t actual_head_size = min(params.head_size, params.head_dim);
    
    // Load query vector (native F16, no conversion)
    half q_vec[256]; // Assume max head_size of 256
    for (uint32_t d = 0; d < actual_head_size; d++) {
        uint32_t global_q_idx = ((q_start + q_idx) * params.q_stride_seq) + 
                                (qo_head_idx * params.q_stride_head) + d;
        q_vec[d] = q_input[global_q_idx];
    }
    
    // F16-specific attention computation
    half max_score = F16_NEG_INFINITY;
    half scores[1024]; // Assume max sequence length of 1024
    uint32_t actual_kv_len = min(kv_len, uint32_t(1024));
    
    // Phase 1: Compute all attention scores and find maximum (F16 precision)
    for (uint32_t kv_pos = 0; kv_pos < actual_kv_len; kv_pos++) {
        uint32_t actual_kv_idx = kv_start + kv_pos;
        
        // Get page information
        uint32_t page_start = kv_page_indptr[seq_idx];
        uint32_t page_end = kv_page_indptr[seq_idx + 1];
        uint32_t num_pages = page_end - page_start;
        
        if (num_pages == 0) {
            scores[kv_pos] = F16_NEG_INFINITY;
            continue;
        }
        
        // Calculate which page this token belongs to
        uint32_t tokens_per_page = 16; // Standard page size
        uint32_t page_offset = kv_pos / tokens_per_page;
        uint32_t pos_in_page = kv_pos % tokens_per_page;
        
        if (page_offset >= num_pages) {
            scores[kv_pos] = F16_NEG_INFINITY;
            continue;
        }
        
        // Handle last page length restriction
        if (page_offset == num_pages - 1) {
            uint32_t last_page_len = kv_last_page_lens[seq_idx];
            if (pos_in_page >= last_page_len) {
                scores[kv_pos] = F16_NEG_INFINITY;
                continue;
            }
        }
        
        uint32_t page_idx = kv_page_indices[page_start + page_offset];
        
        // Compute attention score using native F16 arithmetic
        half score = half(0.0f);
        for (uint32_t d = 0; d < actual_head_size; d++) {
            uint32_t global_k_idx = (page_idx * tokens_per_page * params.num_kv_heads * params.head_dim) + 
                                    (pos_in_page * params.num_kv_heads * params.head_dim) + 
                                    (kv_head_idx * params.head_dim) + d;
            
            half k_val = paged_k_cache[global_k_idx];
            score += q_vec[d] * k_val; // Native F16 multiply-add
        }
        
        // Apply scaling with F16 precision
        score *= half(params.scale);
        
        // Apply causal mask if enabled
        if (params.causal && kv_pos > (q_start + q_idx)) {
            score = F16_NEG_INFINITY;
        }
        
        // Clamp to safe range to prevent F16 overflow
        score = clamp(score, -F16_SAFE_MAX, F16_SAFE_MAX);
        
        scores[kv_pos] = score;
        max_score = max(max_score, score);
    }
    
    // Phase 2: Apply softmax with F16 numerical stability
    half sum_exp = half(0.0f);
    for (uint32_t kv_pos = 0; kv_pos < actual_kv_len; kv_pos++) {
        if (scores[kv_pos] <= (F16_NEG_INFINITY + half(1.0f))) {
            // Skip masked positions
            scores[kv_pos] = half(0.0f);
        } else {
            // Compute exp with numerical stability
            half shifted_score = scores[kv_pos] - max_score;
            
            // Clamp to prevent F16 underflow in exp()
            if (shifted_score < -F16_SAFE_MAX) {
                scores[kv_pos] = half(0.0f);
            } else {
                half exp_score = exp(shifted_score);
                scores[kv_pos] = exp_score;
                sum_exp += exp_score;
            }
        }
    }
    
    // Prevent division by zero with F16 epsilon
    if (sum_exp <= F16_EPSILON) {
        sum_exp = half(1.0f);
    }
    
    // Normalize probabilities (F16 precision)
    for (uint32_t kv_pos = 0; kv_pos < actual_kv_len; kv_pos++) {
        scores[kv_pos] /= sum_exp;
    }
    
    // Phase 3: Compute weighted sum of values (F16 precision)
    half output_vec[256]; // Assume max head_size of 256
    for (uint32_t d = 0; d < actual_head_size; d++) {
        output_vec[d] = half(0.0f);
    }
    
    for (uint32_t kv_pos = 0; kv_pos < actual_kv_len; kv_pos++) {
        if (scores[kv_pos] <= F16_EPSILON) continue; // Skip negligible weights
        
        uint32_t actual_kv_idx = kv_start + kv_pos;
        
        // Get page information (same logic as for keys)
        uint32_t page_start = kv_page_indptr[seq_idx];
        uint32_t page_end = kv_page_indptr[seq_idx + 1];
        uint32_t num_pages = page_end - page_start;
        
        if (num_pages == 0) continue;
        
        uint32_t tokens_per_page = 16;
        uint32_t page_offset = kv_pos / tokens_per_page;
        uint32_t pos_in_page = kv_pos % tokens_per_page;
        
        if (page_offset >= num_pages) continue;
        
        // Handle last page length restriction
        if (page_offset == num_pages - 1) {
            uint32_t last_page_len = kv_last_page_lens[seq_idx];
            if (pos_in_page >= last_page_len) continue;
        }
        
        uint32_t page_idx = kv_page_indices[page_start + page_offset];
        
        // Accumulate weighted values with native F16 arithmetic
        for (uint32_t d = 0; d < actual_head_size; d++) {
            uint32_t global_v_idx = (page_idx * tokens_per_page * params.num_kv_heads * params.head_dim) + 
                                    (pos_in_page * params.num_kv_heads * params.head_dim) + 
                                    (kv_head_idx * params.head_dim) + d;
            
            half v_val = paged_v_cache[global_v_idx];
            output_vec[d] += scores[kv_pos] * v_val; // Native F16 multiply-add
        }
    }
    
    // Phase 4: Write output (F16, no conversion)
    for (uint32_t d = 0; d < actual_head_size; d++) {
        uint32_t global_out_idx = ((q_start + q_idx) * params.o_stride_seq) + 
                                  (qo_head_idx * params.o_stride_head) + d;
        output[global_out_idx] = output_vec[d]; // Direct F16 assignment
    }
}