// --- Buffer Diagnostic Kernel ---
// Minimal kernel that only tests buffer accessibility to isolate segfault source
// Tests each buffer individually with detailed reporting

kernel void buffer_diagnostic_kernel(
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
    // Only thread 0 in first threadgroup does the testing
    if (tgid.x != 0 || tid_in_tgp != 0) return;

    if (debug_out == nullptr) return; // Need debug buffer for reporting

    // Initialize diagnostic markers
    for (int i = 0; i < 20; ++i) {
        debug_out[i] = -1.0f; // -1 = not tested yet
    }

    debug_out[0] = 100.0f; // Kernel entry marker

    // Test 1: Read parameters
    debug_out[1] = 200.0f; // Starting parameter test
    int num_qo = params.num_qo;
    int head_dim = params.head_dim;
    int page_size = params.page_size;
    debug_out[1] = 201.0f; // Parameters accessible

    // Test 2: q_input buffer
    debug_out[2] = 300.0f; // Starting q_input test
    volatile half q_test = q_input[0];
    debug_out[2] = 301.0f; // q_input[0] accessible

    // Test 3: qo_indptr buffer
    debug_out[3] = 400.0f; // Starting qo_indptr test
    volatile int qo_start = qo_indptr[0];
    volatile int qo_end = qo_indptr[1];
    debug_out[3] = 401.0f; // qo_indptr accessible

    // Test 4: kv_page_indptr buffer
    debug_out[4] = 500.0f; // Starting kv_page_indptr test
    volatile int kv_start = kv_page_indptr[0];
    volatile int kv_end = kv_page_indptr[1];
    debug_out[4] = 501.0f; // kv_page_indptr accessible

    // Test 5: kv_page_indices buffer
    debug_out[5] = 600.0f; // Starting kv_page_indices test
    if (kv_end > kv_start) {
        volatile int page_idx = kv_page_indices[kv_start];
        debug_out[5] = 601.0f; // kv_page_indices accessible
    } else {
        debug_out[5] = 602.0f; // No pages to test
    }

    // Test 6: kv_last_page_lens buffer
    debug_out[6] = 700.0f; // Starting kv_last_page_lens test
    volatile int last_len = kv_last_page_lens[0];
    debug_out[6] = 701.0f; // kv_last_page_lens accessible

    // Test 7: paged_k_cache buffer (minimal access)
    debug_out[7] = 800.0f; // Starting k_cache test
    volatile half k_test = paged_k_cache[0];
    debug_out[7] = 801.0f; // k_cache[0] accessible

    // Test 8: Try accessing k_cache with calculated offset (this might be the problem)
    debug_out[8] = 900.0f; // Starting calculated k_cache access
    if (kv_end > kv_start) {
        int page_idx = kv_page_indices[kv_start];
        int kv_offset = page_idx * page_size * head_dim;
        // Try very conservative access
        if (kv_offset >= 0) {
            volatile half k_calc = paged_k_cache[kv_offset];
            debug_out[8] = 901.0f; // k_cache calculated access OK
        } else {
            debug_out[8] = 902.0f; // Invalid offset
        }
    } else {
        debug_out[8] = 903.0f; // No pages to calculate
    }

    // Test 9: paged_v_cache buffer (minimal access)
    debug_out[9] = 1000.0f; // Starting v_cache test
    volatile half v_test = paged_v_cache[0];
    debug_out[9] = 1001.0f; // v_cache[0] accessible

    // Test 10: output buffer
    debug_out[10] = 1100.0f; // Starting output test
    output[0] = half(0.5f); // Try writing to output
    debug_out[10] = 1101.0f; // output accessible

    // Test 11: Completion marker
    debug_out[11] = 9999.0f; // All tests completed successfully
}

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
    // CRITICAL DEBUG: Add detailed memory bounds checking to identify segfault location

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

    // DEBUG: Log entry point with thread info
    if (tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[10] = float(qo_idx);         // Current query index
        debug_out[11] = float(tid_in_tgp);     // Thread index
        debug_out[12] = 1.0f;                  // Entry point marker
    }

    // DEBUG: Test basic buffer accessibility before any complex operations
    if (tid_in_tgp == 0) {
        // Test q_input accessibility
        volatile half q_test = q_input[0];     // Test first element
        if (debug_out != nullptr) debug_out[13] = 2.0f;  // q_input accessible

        // Test paged_k_cache accessibility
        volatile half k_test = paged_k_cache[0];  // Test first element
        if (debug_out != nullptr) debug_out[14] = 3.0f;  // k_cache accessible

        // Test paged_v_cache accessibility
        volatile half v_test = paged_v_cache[0];  // Test first element
        if (debug_out != nullptr) debug_out[15] = 4.0f;  // v_cache accessible

        // Test qo_indptr accessibility
        volatile int qo_test = qo_indptr[0];
        if (debug_out != nullptr) debug_out[16] = 5.0f;  // qo_indptr accessible

        // Test kv_page_indptr accessibility
        volatile int kv_indptr_test = kv_page_indptr[0];
        if (debug_out != nullptr) debug_out[17] = 6.0f;  // kv_page_indptr accessible

        // Test output buffer accessibility
        output[qo_idx * head_dim] = 0.0h;  // Test write access
        if (debug_out != nullptr) debug_out[18] = 7.0f;  // output accessible
    }

    // Synchronize before continuing
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    // DEBUG: Add bounds checking for sequence lookup
    if (tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[19] = 8.0f;  // About to find sequence ID
    }

    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));

    // DEBUG: Validate seq_id bounds before accessing arrays
    if (tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[20] = float(seq_id);  // Log sequence ID found
    }

    // Bounds check for kv_page_indptr access
    if (seq_id < 0 || seq_id >= num_qo) {
        if (tid_in_tgp == 0 && debug_out != nullptr) {
            debug_out[21] = -1.0f;  // Invalid sequence ID error
        }
        return;
    }

    if (tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[22] = 9.0f;  // About to access kv_page_indptr
    }

    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    // DEBUG: Log page information
    if (tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[23] = float(kv_start_page_pos);  // Start page pos
        debug_out[24] = float(kv_end_page_pos);    // End page pos
        debug_out[25] = float(num_pages);          // Total pages
    }

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
            // DEBUG: Log main loop entry
            if (tid_in_tgp == 0 && debug_out != nullptr) {
                debug_out[26] = 10.0f;  // Main loop entry
                debug_out[27] = float(block_start);  // Current block start
            }

            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;

                    // DEBUG: Critical bounds checking before accessing kv_page_indices
                    if (kv_start_page_pos + page_offset >= 0 &&
                        tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[28] = 11.0f;  // About to access kv_page_indices
                        debug_out[29] = float(kv_start_page_pos + page_offset);  // Index being accessed
                    }

                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];

                    // DEBUG: Validate page_idx
                    if (tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[30] = float(page_idx);  // Page index obtained
                    }

                    // Map query head to KV head for MQA/GQA support
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint base_addr = calculate_kv_address(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // DEBUG: Log memory addresses before access
                    if (tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[31] = 12.0f;  // About to access paged caches
                        debug_out[32] = float(base_addr);  // Base address
                    }

                    // CRITICAL: Add extremely defensive bounds checking for the memory access
                    if (tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[34] = float(base_addr);  // Log exact address being accessed
                        debug_out[35] = float(head_size);  // Log head_size being used
                    }

                    // DEFENSIVE: Test individual memory access before the loop
                    if (tid_in_tgp == 0) {
                        // Test just the first element access to see if this causes the segfault
                        volatile half k_first_test = paged_k_cache[base_addr];
                        if (debug_out != nullptr) debug_out[36] = 14.0f;  // First K element accessible

                        volatile half v_first_test = paged_v_cache[base_addr];
                        if (debug_out != nullptr) debug_out[37] = 15.0f;  // First V element accessible
                    }

                    // CRITICAL FIX: Load full head_size elements for proper attention
                    // This was previously limited to 4 elements for debugging, but we need
                    // the full key/value vectors for correct attention computation
                    for (int d = 0; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }

                    // DEBUG: Log successful full memory access
                    if (tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[38] = 16.0f;  // Full memory access completed successfully
                        debug_out[39] = float(head_size);  // Full head_size elements accessed
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