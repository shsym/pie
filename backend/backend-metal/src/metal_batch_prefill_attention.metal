#include <metal_stdlib>
using namespace metal;

// --- Kernel Constants ---
// TGP_SIZE: Threads per threadgroup. This should be a power of 2, e.g., 64, 128, 256.
// It determines the degree of parallelism for processing one query.
#define TGP_SIZE 128
// BLOCK_SIZE: The number of keys processed in parallel by the threadgroup in each step.
#define BLOCK_SIZE 8
// MAX_HEAD_DIM: The kernel uses fixed-size shared memory arrays for performance.
// Set to 512 to match the test configuration (8 heads × 64 head_size = 512 head_dim)
#define MAX_HEAD_DIM 512

// Small uniform parameter block passed via buffer(8)
struct Params {
    int num_qo;
    int head_dim;
    int head_size;
    int page_size;
    float scale;
};

kernel void batch_prefill_attention_unified_bf16_kernel(
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
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // This check is good practice but will cause a compile-time error if head_dim is a compile-time constant.
    if (head_dim > MAX_HEAD_DIM) return;

    // --- Shared Memory Declaration ---
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup float w_block[BLOCK_SIZE];

    // Online softmax accumulators (allocated once, reused per head)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];

    // Temporary reduction scratchpad
    threadgroup float temp_reduce[TGP_SIZE];

    // Validate head_size and determine number of heads
    const int num_heads = max(1, head_dim / max(1, head_size));

    // --- Get Sequence and KV Page Information ---
    // PERF: This linear scan can be slow for batches with many sequences. A binary search would be faster.
    int seq_id = 0;
    while (seq_id < 100 && qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; }
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
        for (int block_start = 0; block_start < total_kv_len; block_start += BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    uint base_addr = page_idx * page_size * head_dim + in_page_offset * head_dim + h * head_size;

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
            if (tid_in_tgp < BLOCK_SIZE && global_key_idx_score < total_kv_len) {
                for (int d = 0; d < head_size; ++d) {
                    score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                }
                score *= scale;
            } else {
                score = -INFINITY;
            }

            // Online softmax update
            // 1) block max
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

            // Accumulate V slice
            int dims_per_thread = (head_size + TGP_SIZE - 1) / TGP_SIZE;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid_in_tgp * dims_per_thread + i;
                if (d < head_size) {
                    float sum_wv_d = 0.0f;
                    int num_keys_in_block = min((int)BLOCK_SIZE, total_kv_len - block_start);
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

// Float32 variant of the unified prefill attention kernel
kernel void batch_prefill_attention_unified_f32_kernel(
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
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const float scale = params.scale;

    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;
    if (head_dim > MAX_HEAD_DIM) return;

    threadgroup float q_s[MAX_HEAD_DIM];
    threadgroup float k_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup float v_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup float w_block[BLOCK_SIZE];
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];
    threadgroup float temp_reduce[TGP_SIZE];

    const int num_heads = max(1, head_dim / max(1, head_size));

    int seq_id = 0;
    while (seq_id < 100 && qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; }
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
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
    }

    for (int h = 0; h < num_heads; ++h) {
        if (tid_in_tgp == 0) { m_i = -INFINITY; l_i = 0.0f; }
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) { acc_i[d] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int q_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            q_s[d] = q_input[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int block_start = 0; block_start < total_kv_len; block_start += BLOCK_SIZE) {
            if (tid_in_tgp < BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < total_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    uint base_addr = page_idx * page_size * head_dim + in_page_offset * head_dim + h * head_size;
                    for (int d = 0; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;
            if (tid_in_tgp < BLOCK_SIZE && global_key_idx_score < total_kv_len) {
                for (int d = 0; d < head_size; ++d) {
                    score += q_s[d] * k_block[tid_in_tgp][d];
                }
                score *= scale;
            } else {
                score = -INFINITY;
            }

            temp_reduce[tid_in_tgp] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float m_j = temp_reduce[0];

            threadgroup float m_prev;
            if (tid_in_tgp == 0) { m_prev = m_i; m_i = max(m_i, m_j); }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale_factor = exp(m_prev - m_i);
            if (tid_in_tgp == 0) { l_i *= scale_factor; }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) { acc_i[d] *= scale_factor; }

            float w = (score > -INFINITY) ? exp(score - m_i) : 0.0f;
            if (tid_in_tgp < BLOCK_SIZE) { w_block[tid_in_tgp] = w; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            temp_reduce[tid_in_tgp] = (tid_in_tgp < BLOCK_SIZE) ? w : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
                if (tid_in_tgp < s) temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid_in_tgp == 0) { l_i += temp_reduce[0]; }

            int dims_per_thread = (head_size + TGP_SIZE - 1) / TGP_SIZE;
            for (int i = 0; i < dims_per_thread; ++i) {
                int d = tid_in_tgp * dims_per_thread + i;
                if (d < head_size) {
                    float sum_wv_d = 0.0f;
                    int num_keys_in_block = min((int)BLOCK_SIZE, total_kv_len - block_start);
                    for (int j = 0; j < num_keys_in_block; ++j) {
                        sum_wv_d += w_block[j] * v_block[j][d];
                    }
                    acc_i[d] += sum_wv_d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        int out_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = acc_i[d] / l_i;
            } else {
                output[out_base + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}