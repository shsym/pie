#include <metal_stdlib>
using namespace metal;

// Unified FlashInfer-style interface (bf16). This minimal version flattens head_dim and
// treats kv pages linearly; it does not implement true paging performance yet,
// but accepts the same buffers and produces output in the expected shape.
kernel void batch_prefill_attention_unified_bf16_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant int& num_qo [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    constant int& page_size [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // Find which sequence this qo_idx belongs to by scanning qo_indptr
    int seq_id = 0;
    // Scan through qo_indptr to find the sequence containing qo_idx
    // qo_indptr[i] <= qo_idx < qo_indptr[i+1] means qo_idx belongs to sequence i
    while (seq_id < 100 && qo_indptr[seq_id + 1] <= int(qo_idx)) { // safeguard against infinite loop
        seq_id++;
    }

    // Get the KV page range for this sequence
    int kv_start = kv_page_indptr[seq_id];
    int kv_end = kv_page_indptr[seq_id + 1];

    // First pass: find maximum score for numerical stability (only thread 0)
    threadgroup float shared_max_score;

    if (tid_in_tgp == 0) {
        float max_score = -INFINITY;

        for (int page_idx_pos = kv_start; page_idx_pos < kv_end; ++page_idx_pos) {
            int page_idx = kv_page_indices[page_idx_pos];
            int valid_len = kv_last_page_lens[seq_id];
            int loop_len = page_size;
            // Last page may have shorter length
            bool is_last = (page_idx_pos == kv_end - 1);
            if (is_last && valid_len > 0) {
                loop_len = valid_len;
            }

            for (int t = 0; t < loop_len; ++t) {
                // Compute score = dot(q, k) * scale
                float score = 0.0f;
                for (int hd = 0; hd < head_dim; ++hd) {
                    float qv = float(q_input[qo_idx * head_dim + hd]);
                    float kv = float(paged_k_cache[page_idx * page_size * head_dim + t * head_dim + hd]);
                    score += qv * kv;
                }
                score *= scale;
                max_score = max(max_score, score);
            }
        }
        shared_max_score = max_score;
    }

    // Synchronize all threads to ensure max_score is computed
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: compute attention output with numerical stability
    for (int d = int(tid.x); d < head_dim; d += 32) {
        float out_val = 0.0f;
        float sum_w = 0.0f;

        for (int page_idx_pos = kv_start; page_idx_pos < kv_end; ++page_idx_pos) {
            int page_idx = kv_page_indices[page_idx_pos];
            int valid_len = kv_last_page_lens[seq_id];
            int loop_len = page_size;
            // Last page may have shorter length
            bool is_last = (page_idx_pos == kv_end - 1);
            if (is_last && valid_len > 0) {
                loop_len = valid_len;
            }

            for (int t = 0; t < loop_len; ++t) {
                // Recompute score for this specific key token
                float score = 0.0f;
                for (int hd = 0; hd < head_dim; ++hd) {
                    float qv = float(q_input[qo_idx * head_dim + hd]);
                    float kv = float(paged_k_cache[page_idx * page_size * head_dim + t * head_dim + hd]);
                    score += qv * kv;
                }
                score *= scale;

                // Apply numerical stability by subtracting max_score
                float stable_score = score - shared_max_score;
                float w = exp(stable_score);
                sum_w += w;

                float vv = float(paged_v_cache[page_idx * page_size * head_dim + t * head_dim + d]);
                out_val += w * vv;
            }
        }

        output[qo_idx * head_dim + d] = half(out_val / max(sum_w, 1e-9f));
    }
}
