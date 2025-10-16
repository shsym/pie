#include <metal_stdlib>
using namespace metal;

// --- Common Attention Kernel Constants and Utilities ---

// TGP_SIZE: Threads per threadgroup. This should be a power of 2, e.g., 64, 128, 256.
// It determines the degree of parallelism for processing one query.
#define TGP_SIZE 128

// BLOCK_SIZE: The number of keys processed in parallel by the threadgroup in each step.
// Should match page_size for optimal memory alignment.
// NOTE: BLOCK_SIZE is dynamically injected by Python at compilation time via #define.
// Configuration flow:
//   1. TOML config (kv_page_size) -> server.py sets PIE_METAL_PAGE_SIZE env var
//   2. metal_kernels/ops.py reads PIE_METAL_PAGE_SIZE at import time
//   3. mps_attention.py injects #define BLOCK_SIZE at shader compilation
// See: metal_kernels/_internal/mps_attention.py::_compile_attention_kernels()
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16  // Fallback for standalone compilation/testing only
#endif

// Smart kernel block size based on memory constraints (32KB limit)
// F32 uses 4x more memory than FP16, so needs smaller block size
#ifdef FP32_KERNEL
  #if BLOCK_SIZE > 16
    #define KERNEL_BLOCK_SIZE 16  // Cap at 16 for f32 (~30KB memory usage)
  #else
    #define KERNEL_BLOCK_SIZE BLOCK_SIZE
  #endif
#else  // FP16 kernel
  #define KERNEL_BLOCK_SIZE BLOCK_SIZE  // FP16 can handle up to 32
#endif

// MAX_HEAD_DIM: The kernel uses fixed-size shared memory arrays for performance.
// Memory calculation:
// FP16: 2KB + KERNEL_BLOCK_SIZE * 1028 bytes ≤ 32KB
// F32:  2.5KB + KERNEL_BLOCK_SIZE * 2052 bytes ≤ 32KB
#define MAX_HEAD_DIM 256

// Metal SIMD group size on Apple Silicon
#define SIMD_SIZE 32

// Small uniform parameter block passed via buffer(7)
struct Params {
    int num_qo;
    int head_dim;        // Query head dimension (num_query_heads * head_size)
    int kv_head_dim;     // KV head dimension (num_kv_heads * head_size)
    int head_size;
    int page_size;
    int num_query_heads; // Number of query heads
    int num_kv_heads;    // Number of KV heads (for MQA/GQA support)
    float scale;
    int total_kv_len;    // Total KV sequence length for mask indexing
};

// --- Common Utility Functions ---

// Efficient sequence ID lookup (can be optimized to binary search later)
inline int find_sequence_id(device const int* qo_indptr, int qo_idx) {
    // Walk qo_indptr until we find the sequence whose end exceeds qo_idx.
    // Assumes qo_indptr is length (num_sequences + 1) with qo_indptr[last] == total_tokens.
    // This guarantees termination without needing an arbitrary cap.
    int seq_id = 0;
    while (qo_indptr[seq_id + 1] <= qo_idx) {
        seq_id++;
    }
    return seq_id;
}

// Calculate K cache offset in unified KV cache buffer for MQA/GQA support
// Unified buffer layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
// Within each page: K at offset 0, V at offset (page_size * kv_head_dim)
// Returns: Absolute offset from unified buffer start for K data
inline uint calculate_k_offset(
    int in_page_offset,
    int page_size,
    int kv_head_dim,
    int head_size,
    int page_idx,
    int kv_head
) {
    // Each page contains both K and V: stride is 2 * page_size * kv_head_dim
    // K is at offset 0 within page
    return page_idx * (2 * page_size * kv_head_dim) +
           0 * (page_size * kv_head_dim) +  // K is at offset 0
           in_page_offset * kv_head_dim +
           kv_head * head_size;
}

// Calculate V cache offset in unified KV cache buffer for MQA/GQA support
// Returns: Absolute offset from unified buffer start for V data
inline uint calculate_v_offset(
    int in_page_offset,
    int page_size,
    int kv_head_dim,
    int head_size,
    int page_idx,
    int kv_head
) {
    // V starts at (page_size * kv_head_dim) offset within each page
    return page_idx * (2 * page_size * kv_head_dim) +
           1 * (page_size * kv_head_dim) +  // V is after K
           in_page_offset * kv_head_dim +
           kv_head * head_size;
}

// Map query head to KV head for MQA/GQA support
inline int map_query_to_kv_head(int query_head, int num_query_heads, int num_kv_heads) {
    return query_head / max(1, num_query_heads / num_kv_heads);
}

// --- Simdgroup Reduction Utilities ---

// Efficient simdgroup max reduction with inter-simdgroup combination
inline float simdgroup_max_reduction(
    float value,
    threadgroup float* scratch,
    uint tid_in_tgp,
    uint simd_lane_id,
    uint simd_group_id,
    int active_elements
) {
    // Max within simdgroup
    float simd_max_val = simd_max(value);

    // Store simdgroup result
    if (simd_lane_id == 0) {
        scratch[simd_group_id] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine simdgroup results
    float result = scratch[0];
    if (tid_in_tgp == 0) {
        int active_simd_groups = (active_elements + SIMD_SIZE - 1) / SIMD_SIZE;
        for (int i = 1; i < active_simd_groups; i++) {
            result = max(result, scratch[i]);
        }
        scratch[0] = result; // Store for broadcast
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0]; // Broadcast result
}

// Efficient simdgroup sum reduction with inter-simdgroup combination
inline float simdgroup_sum_reduction(
    float value,
    threadgroup float* scratch,
    uint tid_in_tgp,
    uint simd_lane_id,
    uint simd_group_id,
    int active_elements
) {
    // Sum within simdgroup
    float simd_sum_val = simd_sum(value);

    // Store simdgroup result
    if (simd_lane_id == 0) {
        scratch[simd_group_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine simdgroup results
    float result = 0.0f;
    if (tid_in_tgp == 0) {
        int active_simd_groups = (active_elements + SIMD_SIZE - 1) / SIMD_SIZE;
        for (int i = 0; i < active_simd_groups; i++) {
            result += scratch[i];
        }
        scratch[0] = result; // Store for broadcast
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0]; // Broadcast result
}

// Split-dimension parallel dot product computation
inline float split_d_dot_product(
    threadgroup const half* q_s,
    threadgroup const half* k_row,
    int head_size,
    uint simd_lane_id
) {
    const int elems_per_lane = (head_size + SIMD_SIZE - 1) / SIMD_SIZE;
    float partial_score = 0.0f;

    for (int i = 0; i < elems_per_lane; ++i) {
        int d = simd_lane_id * elems_per_lane + i;
        if (d < head_size) {
            partial_score += float(q_s[d]) * float(k_row[d]);
        }
    }

    return simd_sum(partial_score);
}

// Split-dimension parallel dot product computation for float arrays
inline float split_d_dot_product_f32(
    threadgroup const float* q_s,
    threadgroup const float* k_row,
    int head_size,
    uint simd_lane_id
) {
    const int elems_per_lane = (head_size + SIMD_SIZE - 1) / SIMD_SIZE;
    float partial_score = 0.0f;

    for (int i = 0; i < elems_per_lane; ++i) {
        int d = simd_lane_id * elems_per_lane + i;
        if (d < head_size) {
            partial_score += q_s[d] * k_row[d];
        }
    }

    return simd_sum(partial_score);
}