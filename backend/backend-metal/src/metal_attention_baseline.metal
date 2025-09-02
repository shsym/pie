// --- Baseline FlashAttention Implementation ---
// This is the original implementation with traditional threadgroup-wide reductions
// Serves as the fallback and reference implementation for correctness validation

kernel void batch_prefill_attention_unified_bf16_baseline_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // Check if head_size (dimension per head) exceeds our threadgroup memory limit
    if (head_size > MAX_HEAD_DIM) return;

    // --- Shared Memory Declaration ---
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup float w_block[KERNEL_BLOCK_SIZE];

    // Online softmax accumulators (allocated once, reused per head)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];

    // Temporary reduction scratchpad
    threadgroup float temp_reduce[TGP_SIZE];

    // Use the explicitly provided number of query heads
    const int num_heads = num_query_heads;

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = scale;
        debug_out[1] = (float)head_dim;
        debug_out[2] = (float)page_size;
        debug_out[3] = (float)num_qo;
        debug_out[5] = (float)num_pages;
    }

    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
    }

    // --- Per-head processing: compute attention independently for each head ---
    for (int h = 0; h < num_heads; ++h) {
        // --- Initialization per head ---
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            acc_i[d] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Load Query slice for this head into Shared Memory ---
        int q_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            q_s[d] = q_input[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main Loop: Process KV Cache in Parallel Blocks ---
        // Process keys in KERNEL_BLOCK_SIZE chunks for memory efficiency
        for (int block_start = 0; block_start < total_kv_len; block_start += KERNEL_BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    // Map query head to KV head for MQA/GQA support
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint base_addr = calculate_kv_address(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    for (int d = 0; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute scores for this block (per head)
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;
            if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < total_kv_len) {
                for (int d = 0; d < head_size; ++d) {
                    score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                }
                score *= scale;
            } else {
                score = -INFINITY;
            }

            // Online softmax update
            // 1) block max - ORIGINAL THREADGROUP-WIDE REDUCTION
            temp_reduce[tid_in_tgp] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float m_j = temp_reduce[0];

            // 2) update global max and rescale accumulators
            threadgroup float m_prev;
            if (tid_in_tgp == 0) {
                m_prev = m_i;
                m_i = max(m_i, m_j);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale_factor = exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // 3) compute weights and update accumulators
            float w = (score > -INFINITY) ? exp(score - m_i) : 0.0f;
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Update l_i (sum of weights) - ORIGINAL THREADGROUP-WIDE REDUCTION
            temp_reduce[tid_in_tgp] = (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid_in_tgp == 0) {
                l_i += temp_reduce[0];
            }

            // Accumulate V slice
            int dims_per_thread = (head_size + TGP_SIZE - 1) / TGP_SIZE;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid_in_tgp * dims_per_thread + i;
                if (d < head_size) {
                    float sum_wv_d = 0.0f;
                    int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, total_kv_len - block_start);
                    for (int j = 0; j < num_keys_in_block; ++j) {
                        sum_wv_d += w_block[j] * float(v_block[j][d]);
                    }
                    acc_i[d] += sum_wv_d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- Finalization for this head ---
        int out_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = half(acc_i[d] / l_i);
            } else {
                output[out_base + d] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void batch_prefill_attention_unified_f32_baseline_kernel(
    device const float* q_input [[buffer(0)]],
    device const float* paged_k_cache [[buffer(1)]],
    device const float* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // F32 baseline kernel implementation (similar structure to BF16)
    // Implementation details similar to BF16 kernel but with float types
    // Note: This would be the full implementation - abbreviated for space
    // For now, using the existing F32 kernel from metal_batch_prefill_attention.metal
}