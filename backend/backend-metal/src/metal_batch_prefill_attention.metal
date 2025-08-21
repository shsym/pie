#include <metal_stdlib>
using namespace metal;

// --- Kernel Constants ---
// TGP_SIZE: Threads per threadgroup. This should be a power of 2, e.g., 64, 128, 256.
// It determines the degree of parallelism for processing one query.
#define TGP_SIZE 128
// BLOCK_SIZE: The number of keys processed in parallel by the threadgroup in each step.
// Reduced to 8 to keep shared memory under 32KB with MAX_HEAD_DIM=512
#define BLOCK_SIZE 8
// MAX_HEAD_DIM: The kernel uses fixed-size shared memory arrays for performance.
// Set to 512 to match the test configuration (8 heads × 64 head_size = 512 head_dim)
#define MAX_HEAD_DIM 512

// Optimized Flash-Attention-style kernel for batch prefill.
// This implementation uses a single-pass online softmax algorithm to avoid recomputing scores
// and leverages threadgroup memory to cache Q, K, and V tensors.
//
// Each threadgroup computes attention for a single query token (qo_idx).
// Threads within the group collaborate:
// 1. The Q vector is cached in shared memory.
// 2. The KV cache is processed in blocks of size BLOCK_SIZE.
// 3. For each block:
//    a. K and V vectors are loaded into shared memory.
//    b. Dot-product scores are computed in parallel.
//    c. An online softmax update is performed, rescaling running totals for max_score,
//       the output accumulator, and the weight sum (l_i).
// 4. After all blocks are processed, the final normalized output is written to global memory.
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
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // Ensure the head dimension does not exceed our static shared memory allocation.
    // This will cause a compile-time error if head_dim is known, or a runtime no-op if not.
    if (head_dim > MAX_HEAD_DIM) return;

    // --- Shared Memory Declaration ---
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup float w_block[BLOCK_SIZE]; // Stores weights `w = exp(score - m_i)` for the current block

    // Online softmax accumulators, stored in shared memory
    threadgroup float m_i; // Current max score
    threadgroup float l_i; // Current sum of weights, `l_i = sum(exp(score - m_i))`
    threadgroup float acc_i[MAX_HEAD_DIM]; // Running output accumulator

    // --- Initialization ---
    if (tid_in_tgp == 0) {
        m_i = -INFINITY;
        l_i = 0.0f;
    }
    // Initialize accumulator in parallel
    for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
        acc_i[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Load Query into Shared Memory ---
    // Each thread loads a portion of the Q vector.
    for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
        q_s[d] = q_input[qo_idx * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Get Sequence and KV Page Information ---
    int seq_id = 0;
    while (seq_id < 100 && qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; } // Safeguard 100
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    // If the sequence has no KV cache pages, write zero and exit.
    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;

    // --- Main Loop: Process KV Cache in Parallel Blocks ---
    for (int block_start = 0; block_start < total_kv_len; block_start += BLOCK_SIZE) {
        int global_key_idx = block_start + tid_in_tgp;
        int local_key_idx = tid_in_tgp % BLOCK_SIZE;  // Map thread to block position

        // Load a block of K and V vectors into shared memory. Each thread loads one key/value pair.
        if (tid_in_tgp < BLOCK_SIZE && global_key_idx < total_kv_len) {
            int page_offset = global_key_idx / page_size;
            int in_page_offset = global_key_idx % page_size;
            int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
            uint base_addr = page_idx * page_size * head_dim + in_page_offset * head_dim;

            for (int d = 0; d < head_dim; ++d) {
                k_block[local_key_idx][d] = paged_k_cache[base_addr + d];
                v_block[local_key_idx][d] = paged_v_cache[base_addr + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Process the Block ---
        // Each thread computes one dot product score for elements in this block.
        float score = -INFINITY;
        if (tid_in_tgp < BLOCK_SIZE && global_key_idx < total_kv_len) {
            score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += float(q_s[d]) * float(k_block[local_key_idx][d]);
            }
            score *= scale;
        }

        // --- Online Softmax Update ---
        // 1. Find the maximum score within the block (m_j) using a parallel reduction.
        threadgroup float temp_reduce[TGP_SIZE];
        temp_reduce[tid_in_tgp] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
            if (tid_in_tgp < s) temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float m_j = temp_reduce[0]; // Block max score

        // 2. Update the global max (m_i) and rescale the running accumulators.
        threadgroup float m_prev;
        if (tid_in_tgp == 0) {
            m_prev = m_i;
            m_i = max(m_i, m_j);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale l_i and acc_i based on the new max score to maintain numerical stability.
        float scale_factor = exp(m_prev - m_i);
        if (tid_in_tgp == 0) {
            l_i *= scale_factor;
        }
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            acc_i[d] *= scale_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Compute weights `w = exp(score - m_i)` and update accumulators for the current block.
        float w = exp(score - m_i);
        if (tid_in_tgp >= BLOCK_SIZE || global_key_idx >= total_kv_len) w = 0.0f;

        // Store weight only for threads that have valid keys
        if (tid_in_tgp < BLOCK_SIZE) {
            w_block[local_key_idx] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update l_i (sum of weights) with parallel reduction over all threads.
        temp_reduce[tid_in_tgp] = w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
            if (tid_in_tgp < s) temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid_in_tgp == 0) {
            l_i += temp_reduce[0];
        }

        // Update acc_i (output accumulator). Each thread handles a few output dimensions.
        int dims_per_thread = (head_dim + TGP_SIZE - 1) / TGP_SIZE;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid_in_tgp * dims_per_thread + i;
            if (d < head_dim) {
                float sum_wv_d = 0.0f;
                // Sum the contribution of all keys in the block for dimension d.
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    sum_wv_d += w_block[j] * float(v_block[j][d]);
                }
                acc_i[d] += sum_wv_d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Finalization ---
    // Normalize the final accumulator and write the result to global memory.
    for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
        if (l_i > 1e-9f) {
            output[qo_idx * head_dim + d] = half(acc_i[d] / l_i);
        } else {
            output[qo_idx * head_dim + d] = 0.0h;
        }
    }
}
