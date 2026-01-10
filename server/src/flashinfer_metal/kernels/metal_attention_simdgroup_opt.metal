// MLX Steel Attention-based Kernel
// Runtime-flexible head dimension support (64, 128)
// Both FP16 and F32 variants with paged KV cache
//
// Architecture: MLX Steel Attention with simdgroup matrix operations
// - FP16: BQ=32, 128 threads (4 simdgroups)
// - F32:  BQ=16, 64 threads (2 simdgroups) - reduced for memory constraints
//
// Grid dispatch:
// - FP16: (ceil(num_qo / 32), num_heads, 1), threadgroup (128, 1, 1)
// - F32:  (ceil(num_qo / 16), num_heads, 1), threadgroup (64, 1, 1)
//
// Paged KV cache layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
// - K at offset 0, V at offset (page_size * kv_head_dim)

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Common constants
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16  // Page size (injected by Python at compile time)
#endif

#define SIMD_SIZE 32
#define MAX_HEAD_DIM 128  // Maximum supported head dimension

// Simdgroup matrix size (Apple Silicon native)
constant constexpr int kFragSize = 8;
constant constexpr int kElemsPerFrag = 2;  // 64 elements / 32 threads = 2 per thread

// FP16 kernel constants
constant constexpr int BQ_FP16 = 32;
constant constexpr int BK = 16;
constant constexpr int kNWarps_FP16 = 4;
constant constexpr int TGP_SIZE_FP16 = kNWarps_FP16 * SIMD_SIZE;  // 128
constant constexpr int TK = BK / kFragSize;  // 2

// F32 kernel constants (reduced for memory)
constant constexpr int BQ_F32 = 16;
constant constexpr int kNWarps_F32 = 2;
constant constexpr int TGP_SIZE_F32 = kNWarps_F32 * SIMD_SIZE;  // 64

// Padding for bank conflict avoidance
constant constexpr int padQ = 8;
constant constexpr int padK = 8;
constant constexpr int padV = 8;

// Parameter struct (matches production interface)
struct Params {
    int num_qo;
    int head_dim;        // num_query_heads * head_size
    int kv_head_dim;     // num_kv_heads * head_size
    int head_size;       // dimension per head
    int page_size;
    int num_query_heads;
    int num_kv_heads;
    float scale;
    int total_kv_len;    // unused but kept for compatibility
};

// =============================================================================
// Helper functions
// =============================================================================

// Get thread coordinates within 8×8 simdgroup matrix (MLX formula)
inline short2 mlx_get_coord(ushort simd_lane_id) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);  // row 0-7
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;  // col 0,2,4,6
    return short2(fn, fm);  // (col, row)
}

// Row reduction for max (MLX pattern: reduce 2 elems, then xor(1), xor(8))
inline float row_reduce_max(float2 vals) {
    float thr = max(vals.x, vals.y);
    float qgr = simd_shuffle_xor(thr, 1);
    qgr = max(thr, qgr);
    float sgr = simd_shuffle_xor(qgr, 8);
    return max(qgr, sgr);
}

// Row reduction for sum
inline float row_reduce_sum(float2 vals) {
    float thr = vals.x + vals.y;
    float qgr = simd_shuffle_xor(thr, 1);
    qgr = thr + qgr;
    float sgr = simd_shuffle_xor(qgr, 8);
    return qgr + sgr;
}

// Paged KV cache offset calculations
inline uint calc_k_offset(int in_page_offset, int page_size, int kv_head_dim, int head_size, int page_idx, int kv_head) {
    return page_idx * (2 * page_size * kv_head_dim) + in_page_offset * kv_head_dim + kv_head * head_size;
}

inline uint calc_v_offset(int in_page_offset, int page_size, int kv_head_dim, int head_size, int page_idx, int kv_head) {
    return page_idx * (2 * page_size * kv_head_dim) + (page_size * kv_head_dim) + in_page_offset * kv_head_dim + kv_head * head_size;
}

// Find sequence ID from query index
inline int find_sequence_id(constant int* qo_indptr, int qo_idx) {
    int seq_id = 0;
    while (qo_indptr[seq_id + 1] <= qo_idx) {
        seq_id++;
    }
    return seq_id;
}

// Map query head to KV head for GQA/MQA
inline int map_query_to_kv_head(int query_head, int num_query_heads, int num_kv_heads) {
    return query_head / max(1, num_query_heads / num_kv_heads);
}

// =============================================================================
// FP16 Kernel - BQ=32, 128 threads (4 simdgroups)
// =============================================================================

kernel void batch_prefill_attention_unified_fp16_simdgroup_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    // Parse parameters
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const half scale_log2e = half(scale * M_LOG2E_F);

    // Validate head_size (must be multiple of 8, max 128)
    if (head_size > MAX_HEAD_DIM || (head_size & 7) != 0) return;

    // Compute runtime fragment count
    const int td = head_size / kFragSize;  // 8 for head_size=64, 16 for head_size=128

    // Block indices
    const int q_block_idx = tid.x;
    const int head_idx = tid.y;
    const int q_seq_start = q_block_idx * BQ_FP16;

    if (q_seq_start >= num_qo || head_idx >= uint(num_query_heads)) return;

    // Sequence info
    const int seq_id = find_sequence_id(qo_indptr, q_seq_start);
    const int kv_start_page = kv_page_indptr[seq_id];
    const int kv_end_page = kv_page_indptr[seq_id + 1];
    const int num_pages = kv_end_page - kv_start_page;

    if (num_pages <= 0) return;

    const int last_page_len = kv_last_page_lens[seq_id];
    const int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    const int seq_start = qo_indptr[seq_id];
    const int num_q_in_block = min(BQ_FP16, num_qo - q_seq_start);

    // GQA head mapping
    const int kv_head = map_query_to_kv_head(head_idx, num_query_heads, num_kv_heads);

    // KV sequence adjustment for causal attention
    const int kv_seq_start = total_kv_len - num_qo;

    // Compute strides based on actual head_size
    const int LDQ = head_size + padQ;
    const int LDK = BK + padK;
    const int LDV = head_size + padV;

    // Threadgroup memory (sized for MAX_HEAD_DIM)
    threadgroup half Qs[BQ_FP16 * (MAX_HEAD_DIM + padQ)];
    threadgroup half Ks[MAX_HEAD_DIM * (BK + padK)];
    threadgroup half Vs[BK * (MAX_HEAD_DIM + padV)];

    // Thread coordinates (MLX formula)
    const short2 simd_coord = mlx_get_coord(simd_lane_id);
    const short sm = simd_coord.y;  // row 0-7
    const short sn = simd_coord.x;  // col 0,2,4,6

    // Per-simdgroup offset in Q block
    const short tm = kFragSize * simd_group_id;  // 0, 8, 16, 24

    // Threadgroup memory offsets
    const short Qs_offset = (tm + sm) * LDQ + sn;
    const short Ks_offset = sm * LDK + sn;
    const short Vs_offset = sm * LDV + sn;

    // Tile strides
    const short Qs_tile_stride = kFragSize;
    const short Ks_tile_stride = kFragSize * LDK;

    // Thread index for cooperative loads
    const int thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

    // ==========================================================================
    // Load Q block and apply scale
    // ==========================================================================
    {
        const int total_elems = BQ_FP16 * head_size;
        for (int i = 0; i < (total_elems + TGP_SIZE_FP16 - 1) / TGP_SIZE_FP16; i++) {
            int elem_idx = thread_idx + i * TGP_SIZE_FP16;
            if (elem_idx < total_elems) {
                int row = elem_idx / head_size;
                int col = elem_idx % head_size;
                half val = 0;
                if (q_seq_start + row < num_qo) {
                    const int qo_idx = q_seq_start + row;
                    val = q_input[qo_idx * head_dim + head_idx * head_size + col] * scale_log2e;
                }
                Qs[row * LDQ + col] = val;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==========================================================================
    // Initialize output accumulators and softmax state
    // ==========================================================================

    // Output accumulator: td fragments × 2 elements each
    float O_acc[MAX_HEAD_DIM / kFragSize * kElemsPerFrag];
    for (int i = 0; i < td * kElemsPerFrag; i++) O_acc[i] = 0.0f;

    float max_score = -INFINITY;
    float sum_score = 0.0f;

    // Calculate effective KV length for this Q row
    const int my_q_row = q_seq_start + tm + sm;
    const int my_seq_pos = my_q_row - seq_start;
    const int my_effective_kv_len = min(total_kv_len, kv_seq_start + my_seq_pos + 1);

    // Max effective KV length across all Q rows in this block
    const int max_query_pos = q_seq_start + num_q_in_block - 1 - seq_start;
    const int max_effective_kv_len = min(total_kv_len, kv_seq_start + max_query_pos + 1);
    const int kb_lim = (max_effective_kv_len + BK - 1) / BK;

    // ==========================================================================
    // Main loop over K/V blocks
    // ==========================================================================
    for (int kb = 0; kb < kb_lim; kb++) {
        const int kv_start = kb * BK;
        const int kv_len = min(BK, max_effective_kv_len - kv_start);

        // ----------------------------------------------------------------------
        // Load K block (transposed) from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int k_row = thread_idx; k_row < kv_len; k_row += TGP_SIZE_FP16) {
                const int kv_idx = kv_start + k_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint k_off = calc_k_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                // Vectorized load and transpose
                for (int d = 0; d < head_size; d += 4) {
                    half4 k_vec = *reinterpret_cast<device const half4*>(&paged_kv_cache[k_off + d]);
                    Ks[(d + 0) * LDK + k_row] = k_vec.x;
                    Ks[(d + 1) * LDK + k_row] = k_vec.y;
                    Ks[(d + 2) * LDK + k_row] = k_vec.z;
                    Ks[(d + 3) * LDK + k_row] = k_vec.w;
                }
            }
            // Zero unused rows
            for (int k_row = kv_len + thread_idx; k_row < BK; k_row += TGP_SIZE_FP16) {
                for (int d = 0; d < head_size; d++) {
                    Ks[d * LDK + k_row] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ----------------------------------------------------------------------
        // Compute S = Q @ K^T using simdgroup MMA
        // ----------------------------------------------------------------------
        float S_acc[TK * kElemsPerFrag] = {0};

        for (int dd = 0; dd < td; dd++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load Q fragment
            simdgroup_matrix<float, 8, 8> Q_mat;
            {
                const threadgroup half* q_ptr = &Qs[Qs_offset + dd * Qs_tile_stride];
                float2 q_frag = float2(float(q_ptr[0]), float(q_ptr[1]));
                reinterpret_cast<thread float2&>(Q_mat.thread_elements()) = q_frag;
            }

            // Load K fragments
            simdgroup_matrix<float, 8, 8> K_mat[TK];
            for (int tk = 0; tk < TK; tk++) {
                const threadgroup half* k_ptr = &Ks[Ks_offset + dd * Ks_tile_stride + tk * kFragSize];
                float2 k_frag = float2(float(k_ptr[0]), float(k_ptr[1]));
                reinterpret_cast<thread float2&>(K_mat[tk].thread_elements()) = k_frag;
            }

            simdgroup_barrier(mem_flags::mem_none);

            // MMA: S += Q @ K^T
            for (int tk = 0; tk < TK; tk++) {
                simdgroup_matrix<float, 8, 8> S_mat;
                float2 s_frag = float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]);
                reinterpret_cast<thread float2&>(S_mat.thread_elements()) = s_frag;

                simdgroup_multiply_accumulate(S_mat, Q_mat, K_mat[tk], S_mat);

                float2 result = reinterpret_cast<thread float2&>(S_mat.thread_elements());
                S_acc[tk * kElemsPerFrag] = result.x;
                S_acc[tk * kElemsPerFrag + 1] = result.y;
            }
        }

        // ----------------------------------------------------------------------
        // Apply causal mask
        // ----------------------------------------------------------------------
        {
            for (int tk = 0; tk < TK; tk++) {
                const int col_base = kv_start + sn + tk * kFragSize;
                for (int j = 0; j < kElemsPerFrag; j++) {
                    int col_pos = col_base + j;
                    if (col_pos >= my_effective_kv_len || col_pos >= max_effective_kv_len) {
                        S_acc[tk * kElemsPerFrag + j] = -INFINITY;
                    }
                }
            }
        }

        // ----------------------------------------------------------------------
        // Load V block from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int v_row = thread_idx; v_row < kv_len; v_row += TGP_SIZE_FP16) {
                const int kv_idx = kv_start + v_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint v_off = calc_v_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup half4*>(&Vs[v_row * LDV + d]) =
                        *reinterpret_cast<device const half4*>(&paged_kv_cache[v_off + d]);
                }
            }
            // Zero unused rows
            for (int v_row = kv_len + thread_idx; v_row < BK; v_row += TGP_SIZE_FP16) {
                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup half4*>(&Vs[v_row * LDV + d]) = half4(0);
                }
            }
        }

        // ----------------------------------------------------------------------
        // Online softmax
        // ----------------------------------------------------------------------
        float new_max = max_score;
        for (int tk = 0; tk < TK; tk++) {
            float local_max = row_reduce_max(float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]));
            new_max = max(new_max, local_max);
        }

        // Compute P = exp2(S - new_max)
        float P_acc[TK * kElemsPerFrag];
        for (int tk = 0; tk < TK; tk++) {
            for (int j = 0; j < kElemsPerFrag; j++) {
                P_acc[tk * kElemsPerFrag + j] = fast::exp2(S_acc[tk * kElemsPerFrag + j] - new_max);
            }
        }

        // Correction factor for previous accumulator
        float factor = fast::exp2(max_score - new_max);
        max_score = new_max;

        // Sum of exp values
        float sum_tmp = 0.0f;
        for (int tk = 0; tk < TK; tk++) {
            float local_sum = row_reduce_sum(float2(P_acc[tk * kElemsPerFrag], P_acc[tk * kElemsPerFrag + 1]));
            sum_tmp += local_sum;
        }
        sum_score = sum_score * factor + sum_tmp;

        // Rescale previous O accumulator
        for (int i = 0; i < td * kElemsPerFrag; i++) {
            O_acc[i] *= factor;
        }

        // ----------------------------------------------------------------------
        // Compute O += P @ V using simdgroup MMA
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int id = 0; id < td; id++) {
            for (int ik = 0; ik < TK; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                // Load P fragment
                simdgroup_matrix<float, 8, 8> P_mat;
                {
                    float2 p_frag = float2(P_acc[ik * kElemsPerFrag], P_acc[ik * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(P_mat.thread_elements()) = p_frag;
                }

                // Load V fragment
                simdgroup_matrix<float, 8, 8> V_mat;
                {
                    const threadgroup half* v_ptr = &Vs[Vs_offset + ik * kFragSize * LDV + id * kFragSize];
                    float2 v_frag = float2(float(v_ptr[0]), float(v_ptr[1]));
                    reinterpret_cast<thread float2&>(V_mat.thread_elements()) = v_frag;
                }

                simdgroup_barrier(mem_flags::mem_none);

                // MMA: O += P @ V
                simdgroup_matrix<float, 8, 8> O_mat;
                {
                    float2 o_frag = float2(O_acc[id * kElemsPerFrag], O_acc[id * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(O_mat.thread_elements()) = o_frag;
                }

                simdgroup_multiply_accumulate(O_mat, P_mat, V_mat, O_mat);

                {
                    float2 o_result = reinterpret_cast<thread float2&>(O_mat.thread_elements());
                    O_acc[id * kElemsPerFrag] = o_result.x;
                    O_acc[id * kElemsPerFrag + 1] = o_result.y;
                }
            }
        }
    }

    // ==========================================================================
    // Normalize and write output
    // ==========================================================================
    float inv_sum = (sum_score > 0.0f) ? (1.0f / sum_score) : 0.0f;
    for (int i = 0; i < td * kElemsPerFrag; i++) {
        O_acc[i] *= inv_sum;
    }

    const int out_row = q_seq_start + tm + sm;
    if (out_row < num_qo) {
        device half* O_dst = output + out_row * head_dim + head_idx * head_size;

        for (int id = 0; id < td; id++) {
            int col = sn + id * kFragSize;
            O_dst[col + 0] = half(O_acc[id * kElemsPerFrag + 0]);
            O_dst[col + 1] = half(O_acc[id * kElemsPerFrag + 1]);
        }
    }
}

// =============================================================================
// F32 Kernel - BQ=16, 64 threads (2 simdgroups)
// =============================================================================

kernel void batch_prefill_attention_unified_f32_simdgroup_kernel(
    device const float* q_input [[buffer(0)]],
    device const float* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    // Parse parameters
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const float scale_log2e = scale * M_LOG2E_F;

    // Validate head_size (must be multiple of 8, max 128)
    if (head_size > MAX_HEAD_DIM || (head_size & 7) != 0) return;

    // Compute runtime fragment count
    const int td = head_size / kFragSize;

    // Block indices (BQ=16 for f32)
    const int q_block_idx = tid.x;
    const int head_idx = tid.y;
    const int q_seq_start = q_block_idx * BQ_F32;

    if (q_seq_start >= num_qo || head_idx >= uint(num_query_heads)) return;

    // Sequence info
    const int seq_id = find_sequence_id(qo_indptr, q_seq_start);
    const int kv_start_page = kv_page_indptr[seq_id];
    const int kv_end_page = kv_page_indptr[seq_id + 1];
    const int num_pages = kv_end_page - kv_start_page;

    if (num_pages <= 0) return;

    const int last_page_len = kv_last_page_lens[seq_id];
    const int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    const int seq_start = qo_indptr[seq_id];
    const int num_q_in_block = min(BQ_F32, num_qo - q_seq_start);

    // GQA head mapping
    const int kv_head = map_query_to_kv_head(head_idx, num_query_heads, num_kv_heads);

    // KV sequence adjustment for causal attention
    const int kv_seq_start = total_kv_len - num_qo;

    // Compute strides based on actual head_size
    const int LDQ = head_size + padQ;
    const int LDK = BK + padK;
    const int LDV = head_size + padV;

    // Threadgroup memory (sized for MAX_HEAD_DIM, f32 type)
    threadgroup float Qs[BQ_F32 * (MAX_HEAD_DIM + padQ)];
    threadgroup float Ks[MAX_HEAD_DIM * (BK + padK)];
    threadgroup float Vs[BK * (MAX_HEAD_DIM + padV)];

    // Thread coordinates (MLX formula)
    const short2 simd_coord = mlx_get_coord(simd_lane_id);
    const short sm = simd_coord.y;  // row 0-7
    const short sn = simd_coord.x;  // col 0,2,4,6

    // Per-simdgroup offset in Q block (only 2 simdgroups for f32)
    const short tm = kFragSize * simd_group_id;  // 0 or 8

    // Threadgroup memory offsets
    const short Qs_offset = (tm + sm) * LDQ + sn;
    const short Ks_offset = sm * LDK + sn;
    const short Vs_offset = sm * LDV + sn;

    // Tile strides
    const short Qs_tile_stride = kFragSize;
    const short Ks_tile_stride = kFragSize * LDK;

    // Thread index for cooperative loads
    const int thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

    // ==========================================================================
    // Load Q block and apply scale
    // ==========================================================================
    {
        const int total_elems = BQ_F32 * head_size;
        for (int i = 0; i < (total_elems + TGP_SIZE_F32 - 1) / TGP_SIZE_F32; i++) {
            int elem_idx = thread_idx + i * TGP_SIZE_F32;
            if (elem_idx < total_elems) {
                int row = elem_idx / head_size;
                int col = elem_idx % head_size;
                float val = 0;
                if (q_seq_start + row < num_qo) {
                    const int qo_idx = q_seq_start + row;
                    val = q_input[qo_idx * head_dim + head_idx * head_size + col] * scale_log2e;
                }
                Qs[row * LDQ + col] = val;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==========================================================================
    // Initialize output accumulators and softmax state
    // ==========================================================================
    float O_acc[MAX_HEAD_DIM / kFragSize * kElemsPerFrag];
    for (int i = 0; i < td * kElemsPerFrag; i++) O_acc[i] = 0.0f;

    float max_score = -INFINITY;
    float sum_score = 0.0f;

    // Calculate effective KV length for this Q row
    const int my_q_row = q_seq_start + tm + sm;
    const int my_seq_pos = my_q_row - seq_start;
    const int my_effective_kv_len = min(total_kv_len, kv_seq_start + my_seq_pos + 1);

    // Max effective KV length across all Q rows in this block
    const int max_query_pos = q_seq_start + num_q_in_block - 1 - seq_start;
    const int max_effective_kv_len = min(total_kv_len, kv_seq_start + max_query_pos + 1);
    const int kb_lim = (max_effective_kv_len + BK - 1) / BK;

    // ==========================================================================
    // Main loop over K/V blocks
    // ==========================================================================
    for (int kb = 0; kb < kb_lim; kb++) {
        const int kv_start = kb * BK;
        const int kv_len = min(BK, max_effective_kv_len - kv_start);

        // ----------------------------------------------------------------------
        // Load K block (transposed) from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int k_row = thread_idx; k_row < kv_len; k_row += TGP_SIZE_F32) {
                const int kv_idx = kv_start + k_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint k_off = calc_k_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                // Vectorized load and transpose
                for (int d = 0; d < head_size; d += 4) {
                    float4 k_vec = *reinterpret_cast<device const float4*>(&paged_kv_cache[k_off + d]);
                    Ks[(d + 0) * LDK + k_row] = k_vec.x;
                    Ks[(d + 1) * LDK + k_row] = k_vec.y;
                    Ks[(d + 2) * LDK + k_row] = k_vec.z;
                    Ks[(d + 3) * LDK + k_row] = k_vec.w;
                }
            }
            // Zero unused rows
            for (int k_row = kv_len + thread_idx; k_row < BK; k_row += TGP_SIZE_F32) {
                for (int d = 0; d < head_size; d++) {
                    Ks[d * LDK + k_row] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ----------------------------------------------------------------------
        // Compute S = Q @ K^T using simdgroup MMA
        // ----------------------------------------------------------------------
        float S_acc[TK * kElemsPerFrag] = {0};

        for (int dd = 0; dd < td; dd++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load Q fragment
            simdgroup_matrix<float, 8, 8> Q_mat;
            {
                const threadgroup float* q_ptr = &Qs[Qs_offset + dd * Qs_tile_stride];
                float2 q_frag = float2(q_ptr[0], q_ptr[1]);
                reinterpret_cast<thread float2&>(Q_mat.thread_elements()) = q_frag;
            }

            // Load K fragments
            simdgroup_matrix<float, 8, 8> K_mat[TK];
            for (int tk = 0; tk < TK; tk++) {
                const threadgroup float* k_ptr = &Ks[Ks_offset + dd * Ks_tile_stride + tk * kFragSize];
                float2 k_frag = float2(k_ptr[0], k_ptr[1]);
                reinterpret_cast<thread float2&>(K_mat[tk].thread_elements()) = k_frag;
            }

            simdgroup_barrier(mem_flags::mem_none);

            // MMA: S += Q @ K^T
            for (int tk = 0; tk < TK; tk++) {
                simdgroup_matrix<float, 8, 8> S_mat;
                float2 s_frag = float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]);
                reinterpret_cast<thread float2&>(S_mat.thread_elements()) = s_frag;

                simdgroup_multiply_accumulate(S_mat, Q_mat, K_mat[tk], S_mat);

                float2 result = reinterpret_cast<thread float2&>(S_mat.thread_elements());
                S_acc[tk * kElemsPerFrag] = result.x;
                S_acc[tk * kElemsPerFrag + 1] = result.y;
            }
        }

        // ----------------------------------------------------------------------
        // Apply causal mask
        // ----------------------------------------------------------------------
        {
            for (int tk = 0; tk < TK; tk++) {
                const int col_base = kv_start + sn + tk * kFragSize;
                for (int j = 0; j < kElemsPerFrag; j++) {
                    int col_pos = col_base + j;
                    if (col_pos >= my_effective_kv_len || col_pos >= max_effective_kv_len) {
                        S_acc[tk * kElemsPerFrag + j] = -INFINITY;
                    }
                }
            }
        }

        // ----------------------------------------------------------------------
        // Load V block from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int v_row = thread_idx; v_row < kv_len; v_row += TGP_SIZE_F32) {
                const int kv_idx = kv_start + v_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint v_off = calc_v_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup float4*>(&Vs[v_row * LDV + d]) =
                        *reinterpret_cast<device const float4*>(&paged_kv_cache[v_off + d]);
                }
            }
            // Zero unused rows
            for (int v_row = kv_len + thread_idx; v_row < BK; v_row += TGP_SIZE_F32) {
                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup float4*>(&Vs[v_row * LDV + d]) = float4(0);
                }
            }
        }

        // ----------------------------------------------------------------------
        // Online softmax
        // ----------------------------------------------------------------------
        float new_max = max_score;
        for (int tk = 0; tk < TK; tk++) {
            float local_max = row_reduce_max(float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]));
            new_max = max(new_max, local_max);
        }

        // Compute P = exp2(S - new_max)
        float P_acc[TK * kElemsPerFrag];
        for (int tk = 0; tk < TK; tk++) {
            for (int j = 0; j < kElemsPerFrag; j++) {
                P_acc[tk * kElemsPerFrag + j] = fast::exp2(S_acc[tk * kElemsPerFrag + j] - new_max);
            }
        }

        // Correction factor
        float factor = fast::exp2(max_score - new_max);
        max_score = new_max;

        // Sum of exp values
        float sum_tmp = 0.0f;
        for (int tk = 0; tk < TK; tk++) {
            float local_sum = row_reduce_sum(float2(P_acc[tk * kElemsPerFrag], P_acc[tk * kElemsPerFrag + 1]));
            sum_tmp += local_sum;
        }
        sum_score = sum_score * factor + sum_tmp;

        // Rescale previous O accumulator
        for (int i = 0; i < td * kElemsPerFrag; i++) {
            O_acc[i] *= factor;
        }

        // ----------------------------------------------------------------------
        // Compute O += P @ V using simdgroup MMA
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int id = 0; id < td; id++) {
            for (int ik = 0; ik < TK; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                // Load P fragment
                simdgroup_matrix<float, 8, 8> P_mat;
                {
                    float2 p_frag = float2(P_acc[ik * kElemsPerFrag], P_acc[ik * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(P_mat.thread_elements()) = p_frag;
                }

                // Load V fragment
                simdgroup_matrix<float, 8, 8> V_mat;
                {
                    const threadgroup float* v_ptr = &Vs[Vs_offset + ik * kFragSize * LDV + id * kFragSize];
                    float2 v_frag = float2(v_ptr[0], v_ptr[1]);
                    reinterpret_cast<thread float2&>(V_mat.thread_elements()) = v_frag;
                }

                simdgroup_barrier(mem_flags::mem_none);

                // MMA: O += P @ V
                simdgroup_matrix<float, 8, 8> O_mat;
                {
                    float2 o_frag = float2(O_acc[id * kElemsPerFrag], O_acc[id * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(O_mat.thread_elements()) = o_frag;
                }

                simdgroup_multiply_accumulate(O_mat, P_mat, V_mat, O_mat);

                {
                    float2 o_result = reinterpret_cast<thread float2&>(O_mat.thread_elements());
                    O_acc[id * kElemsPerFrag] = o_result.x;
                    O_acc[id * kElemsPerFrag + 1] = o_result.y;
                }
            }
        }
    }

    // ==========================================================================
    // Normalize and write output
    // ==========================================================================
    float inv_sum = (sum_score > 0.0f) ? (1.0f / sum_score) : 0.0f;
    for (int i = 0; i < td * kElemsPerFrag; i++) {
        O_acc[i] *= inv_sum;
    }

    const int out_row = q_seq_start + tm + sm;
    if (out_row < num_qo) {
        device float* O_dst = output + out_row * head_dim + head_idx * head_size;

        for (int id = 0; id < td; id++) {
            int col = sn + id * kFragSize;
            O_dst[col + 0] = O_acc[id * kElemsPerFrag + 0];
            O_dst[col + 1] = O_acc[id * kElemsPerFrag + 1];
        }
    }
}

// =============================================================================
// BF16 Kernel - BQ=32, 128 threads (4 simdgroups)
// =============================================================================

kernel void batch_prefill_attention_unified_bf16_simdgroup_kernel(
    device const bfloat* q_input [[buffer(0)]],
    device const bfloat* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    // Parse parameters
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const bfloat scale_log2e = bfloat(scale * M_LOG2E_F);

    // Validate head_size (must be multiple of 8, max 128)
    if (head_size > MAX_HEAD_DIM || (head_size & 7) != 0) return;

    // Compute runtime fragment count
    const int td = head_size / kFragSize;

    // Block indices (reuse FP16 constants as BQ=32)
    const int q_block_idx = tid.x;
    const int head_idx = tid.y;
    const int q_seq_start = q_block_idx * BQ_FP16;

    if (q_seq_start >= num_qo || head_idx >= uint(num_query_heads)) return;

    // Sequence info
    const int seq_id = find_sequence_id(qo_indptr, q_seq_start);
    const int kv_start_page = kv_page_indptr[seq_id];
    const int kv_end_page = kv_page_indptr[seq_id + 1];
    const int num_pages = kv_end_page - kv_start_page;

    if (num_pages <= 0) return;

    const int last_page_len = kv_last_page_lens[seq_id];
    const int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    const int seq_start = qo_indptr[seq_id];
    const int num_q_in_block = min(BQ_FP16, num_qo - q_seq_start);

    // GQA head mapping
    const int kv_head = map_query_to_kv_head(head_idx, num_query_heads, num_kv_heads);

    // KV sequence adjustment for causal attention
    const int kv_seq_start = total_kv_len - num_qo;

    // Compute strides based on actual head_size
    const int LDQ = head_size + padQ;
    const int LDK = BK + padK;
    const int LDV = head_size + padV;

    // Threadgroup memory (sized for MAX_HEAD_DIM)
    threadgroup bfloat Qs[BQ_FP16 * (MAX_HEAD_DIM + padQ)];
    threadgroup bfloat Ks[MAX_HEAD_DIM * (BK + padK)];
    threadgroup bfloat Vs[BK * (MAX_HEAD_DIM + padV)];

    // Thread coordinates (MLX formula)
    const short2 simd_coord = mlx_get_coord(simd_lane_id);
    const short sm = simd_coord.y;  // row 0-7
    const short sn = simd_coord.x;  // col 0,2,4,6

    // Per-simdgroup offset in Q block
    const short tm = kFragSize * simd_group_id;

    // Threadgroup memory offsets
    const short Qs_offset = (tm + sm) * LDQ + sn;
    const short Ks_offset = sm * LDK + sn;
    const short Vs_offset = sm * LDV + sn;

    // Tile strides
    const short Qs_tile_stride = kFragSize;
    const short Ks_tile_stride = kFragSize * LDK;

    // Thread index for cooperative loads
    const int thread_idx = simd_group_id * SIMD_SIZE + simd_lane_id;

    // ==========================================================================
    // Load Q block and apply scale
    // ==========================================================================
    {
        const int total_elems = BQ_FP16 * head_size;
        for (int i = 0; i < (total_elems + TGP_SIZE_FP16 - 1) / TGP_SIZE_FP16; i++) {
            int elem_idx = thread_idx + i * TGP_SIZE_FP16;
            if (elem_idx < total_elems) {
                int row = elem_idx / head_size;
                int col = elem_idx % head_size;
                bfloat val = 0;
                if (q_seq_start + row < num_qo) {
                    const int qo_idx = q_seq_start + row;
                    val = q_input[qo_idx * head_dim + head_idx * head_size + col] * scale_log2e;
                }
                Qs[row * LDQ + col] = val;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==========================================================================
    // Initialize output accumulators and softmax state
    // ==========================================================================

    // Output accumulator: td fragments × 2 elements each
    float O_acc[MAX_HEAD_DIM / kFragSize * kElemsPerFrag];
    for (int i = 0; i < td * kElemsPerFrag; i++) O_acc[i] = 0.0f;

    float max_score = -INFINITY;
    float sum_score = 0.0f;

    // Calculate effective KV length for this Q row
    const int my_q_row = q_seq_start + tm + sm;
    const int my_seq_pos = my_q_row - seq_start;
    const int my_effective_kv_len = min(total_kv_len, kv_seq_start + my_seq_pos + 1);

    // Max effective KV length across all Q rows in this block
    const int max_query_pos = q_seq_start + num_q_in_block - 1 - seq_start;
    const int max_effective_kv_len = min(total_kv_len, kv_seq_start + max_query_pos + 1);
    const int kb_lim = (max_effective_kv_len + BK - 1) / BK;

    // ==========================================================================
    // Main loop over K/V blocks
    // ==========================================================================
    for (int kb = 0; kb < kb_lim; kb++) {
        const int kv_start = kb * BK;
        const int kv_len = min(BK, max_effective_kv_len - kv_start);

        // ----------------------------------------------------------------------
        // Load K block (transposed) from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int k_row = thread_idx; k_row < kv_len; k_row += TGP_SIZE_FP16) {
                const int kv_idx = kv_start + k_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint k_off = calc_k_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                // Vectorized load and transpose
                for (int d = 0; d < head_size; d += 4) {
                    bfloat4 k_vec = *reinterpret_cast<device const bfloat4*>(&paged_kv_cache[k_off + d]);
                    Ks[(d + 0) * LDK + k_row] = k_vec.x;
                    Ks[(d + 1) * LDK + k_row] = k_vec.y;
                    Ks[(d + 2) * LDK + k_row] = k_vec.z;
                    Ks[(d + 3) * LDK + k_row] = k_vec.w;
                }
            }
            // Zero unused rows
            for (int k_row = kv_len + thread_idx; k_row < BK; k_row += TGP_SIZE_FP16) {
                for (int d = 0; d < head_size; d++) {
                    Ks[d * LDK + k_row] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ----------------------------------------------------------------------
        // Compute S = Q @ K^T using simdgroup MMA
        // ----------------------------------------------------------------------
        float S_acc[TK * kElemsPerFrag] = {0};

        for (int dd = 0; dd < td; dd++) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load Q fragment
            simdgroup_matrix<float, 8, 8> Q_mat;
            {
                const threadgroup bfloat* q_ptr = &Qs[Qs_offset + dd * Qs_tile_stride];
                float2 q_frag = float2(float(q_ptr[0]), float(q_ptr[1]));
                reinterpret_cast<thread float2&>(Q_mat.thread_elements()) = q_frag;
            }

            // Load K fragments
            simdgroup_matrix<float, 8, 8> K_mat[TK];
            for (int tk = 0; tk < TK; tk++) {
                const threadgroup bfloat* k_ptr = &Ks[Ks_offset + dd * Ks_tile_stride + tk * kFragSize];
                float2 k_frag = float2(float(k_ptr[0]), float(k_ptr[1]));
                reinterpret_cast<thread float2&>(K_mat[tk].thread_elements()) = k_frag;
            }

            simdgroup_barrier(mem_flags::mem_none);

            // MMA: S += Q @ K^T
            for (int tk = 0; tk < TK; tk++) {
                simdgroup_matrix<float, 8, 8> S_mat;
                float2 s_frag = float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]);
                reinterpret_cast<thread float2&>(S_mat.thread_elements()) = s_frag;

                simdgroup_multiply_accumulate(S_mat, Q_mat, K_mat[tk], S_mat);

                float2 result = reinterpret_cast<thread float2&>(S_mat.thread_elements());
                S_acc[tk * kElemsPerFrag] = result.x;
                S_acc[tk * kElemsPerFrag + 1] = result.y;
            }
        }

        // ----------------------------------------------------------------------
        // Apply causal mask
        // ----------------------------------------------------------------------
        {
            for (int tk = 0; tk < TK; tk++) {
                const int col_base = kv_start + sn + tk * kFragSize;
                for (int j = 0; j < kElemsPerFrag; j++) {
                    int col_pos = col_base + j;
                    if (col_pos >= my_effective_kv_len || col_pos >= max_effective_kv_len) {
                        S_acc[tk * kElemsPerFrag + j] = -INFINITY;
                    }
                }
            }
        }

        // ----------------------------------------------------------------------
        // Load V block from paged cache
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            for (int v_row = thread_idx; v_row < kv_len; v_row += TGP_SIZE_FP16) {
                const int kv_idx = kv_start + v_row;
                const int pg_off = kv_idx / page_size;
                const int in_pg = kv_idx % page_size;
                const int pg_idx = kv_page_indices[kv_start_page + pg_off];
                const uint v_off = calc_v_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);

                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup bfloat4*>(&Vs[v_row * LDV + d]) =
                        *reinterpret_cast<device const bfloat4*>(&paged_kv_cache[v_off + d]);
                }
            }
            // Zero unused rows
            for (int v_row = kv_len + thread_idx; v_row < BK; v_row += TGP_SIZE_FP16) {
                for (int d = 0; d < head_size; d += 4) {
                    *reinterpret_cast<threadgroup bfloat4*>(&Vs[v_row * LDV + d]) = bfloat4(0);
                }
            }
        }

        // ----------------------------------------------------------------------
        // Online softmax
        // ----------------------------------------------------------------------
        float new_max = max_score;
        for (int tk = 0; tk < TK; tk++) {
            float local_max = row_reduce_max(float2(S_acc[tk * kElemsPerFrag], S_acc[tk * kElemsPerFrag + 1]));
            new_max = max(new_max, local_max);
        }

        // Compute P = exp2(S - new_max)
        float P_acc[TK * kElemsPerFrag];
        for (int tk = 0; tk < TK; tk++) {
            for (int j = 0; j < kElemsPerFrag; j++) {
                P_acc[tk * kElemsPerFrag + j] = fast::exp2(S_acc[tk * kElemsPerFrag + j] - new_max);
            }
        }

        // Correction factor for previous accumulator
        float factor = fast::exp2(max_score - new_max);
        max_score = new_max;

        // Sum of exp values
        float sum_tmp = 0.0f;
        for (int tk = 0; tk < TK; tk++) {
            float local_sum = row_reduce_sum(float2(P_acc[tk * kElemsPerFrag], P_acc[tk * kElemsPerFrag + 1]));
            sum_tmp += local_sum;
        }
        sum_score = sum_score * factor + sum_tmp;

        // Rescale previous O accumulator
        for (int i = 0; i < td * kElemsPerFrag; i++) {
            O_acc[i] *= factor;
        }

        // ----------------------------------------------------------------------
        // Compute O += P @ V using simdgroup MMA
        // ----------------------------------------------------------------------
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int id = 0; id < td; id++) {
            for (int ik = 0; ik < TK; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                // Load P fragment
                simdgroup_matrix<float, 8, 8> P_mat;
                {
                    float2 p_frag = float2(P_acc[ik * kElemsPerFrag], P_acc[ik * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(P_mat.thread_elements()) = p_frag;
                }

                // Load V fragment
                simdgroup_matrix<float, 8, 8> V_mat;
                {
                    const threadgroup bfloat* v_ptr = &Vs[Vs_offset + ik * kFragSize * LDV + id * kFragSize];
                    float2 v_frag = float2(float(v_ptr[0]), float(v_ptr[1]));
                    reinterpret_cast<thread float2&>(V_mat.thread_elements()) = v_frag;
                }

                simdgroup_barrier(mem_flags::mem_none);

                // MMA: O += P @ V
                simdgroup_matrix<float, 8, 8> O_mat;
                {
                    float2 o_frag = float2(O_acc[id * kElemsPerFrag], O_acc[id * kElemsPerFrag + 1]);
                    reinterpret_cast<thread float2&>(O_mat.thread_elements()) = o_frag;
                }

                simdgroup_multiply_accumulate(O_mat, P_mat, V_mat, O_mat);

                {
                    float2 o_result = reinterpret_cast<thread float2&>(O_mat.thread_elements());
                    O_acc[id * kElemsPerFrag] = o_result.x;
                    O_acc[id * kElemsPerFrag + 1] = o_result.y;
                }
            }
        }
    }

    // ==========================================================================
    // Normalize and write output
    // ==========================================================================
    float inv_sum = (sum_score > 0.0f) ? (1.0f / sum_score) : 0.0f;
    for (int i = 0; i < td * kElemsPerFrag; i++) {
        O_acc[i] *= inv_sum;
    }

    const int out_row = q_seq_start + tm + sm;
    if (out_row < num_qo) {
        device bfloat* O_dst = output + out_row * head_dim + head_idx * head_size;

        for (int id = 0; id < td; id++) {
            int col = sn + id * kFragSize;
            O_dst[col + 0] = bfloat(O_acc[id * kElemsPerFrag + 0]);
            O_dst[col + 1] = bfloat(O_acc[id * kElemsPerFrag + 1]);
        }
    }
}

// =============================================================================
// Decode Attention Kernel V2 - MLX sdpa_vector Architecture
// =============================================================================
//
// Optimized for single-query decode with:
// - 1024 threads (32 simdgroups × 32 lanes)
// - simd_sum for hardware reduction
// - Direct device -> register K/V loading (no shared memory)
// - 32 KV positions processed in parallel per iteration
//
// Grid: (num_heads, 1, 1) - one threadgroup per head
// Threadgroup: 1024 threads

// Decode constants
#define DECODE_BN 32           // Number of simdgroups = KV positions processed in parallel
#define DECODE_BD 32           // Threads per simdgroup (SIMD width)
#define DECODE_TGP_SIZE (DECODE_BN * DECODE_BD)  // 1024 threads total

template <typename T, int HEAD_DIM>
[[kernel]] void attention_decode_v2(
    device const T* q_input [[buffer(0)]],
    device const T* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device T* output [[buffer(6)]],
    device const float* params_raw [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Parse parameters
    const int kv_head_dim = (int)params_raw[2];
    const int head_size = (int)params_raw[3];
    const int page_size = (int)params_raw[4];
    const int num_query_heads = (int)params_raw[5];
    const int num_kv_heads = (int)params_raw[6];
    const float scale = params_raw[7];

    const int head_idx = tgid.x;
    if (head_idx >= num_query_heads) return;

    // Elements per thread for Q, K, V (head_dim / 32 threads)
    constexpr int qk_per_thread = HEAD_DIM / DECODE_BD;
    constexpr int v_per_thread = HEAD_DIM / DECODE_BD;

    // Thread-local storage
    float q[qk_per_thread];
    float k[qk_per_thread];
    float o[v_per_thread];

    // Shared memory for final reduction only
    threadgroup float outputs_smem[DECODE_BN * DECODE_BD];
    threadgroup float max_scores_smem[DECODE_BN];
    threadgroup float sum_exp_scores_smem[DECODE_BN];

    // Sequence info (single sequence for decode)
    const int seq_id = 0;
    const int kv_start_page = kv_page_indptr[seq_id];
    const int kv_end_page = kv_page_indptr[seq_id + 1];
    const int num_pages = kv_end_page - kv_start_page;

    if (num_pages <= 0) {
        // Write zeros for empty sequence
        if (simd_gid == 0) {
            for (int i = 0; i < v_per_thread; i++) {
                output[head_idx * head_size + simd_lid * v_per_thread + i] = T(0);
            }
        }
        return;
    }

    const int last_page_len = kv_last_page_lens[seq_id];
    const int total_kv = (num_pages - 1) * page_size + last_page_len;
    const int kv_head = head_idx / max(1, num_query_heads / num_kv_heads);

    // Load Q into registers (scaled)
    device const T* q_ptr = q_input + head_idx * head_size + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = float(q_ptr[i]) * scale;
    }

    // Initialize output accumulator
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0.0f;
    }

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    // Main loop: each simdgroup processes one KV position at a time
    // 32 simdgroups process 32 KV positions in parallel
    for (int kv_idx = simd_gid; kv_idx < total_kv; kv_idx += DECODE_BN) {
        // Calculate paged KV offset
        const int pg_off = kv_idx / page_size;
        const int in_pg = kv_idx % page_size;
        const int pg_idx = kv_page_indices[kv_start_page + pg_off];

        // Load K directly to registers
        const uint k_off = calc_k_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);
        device const T* k_ptr = paged_kv_cache + k_off + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            k[i] = float(k_ptr[i]);
        }

        // Compute Q·K dot product using simd_sum (hardware reduction)
        float score = 0.0f;
        for (int i = 0; i < qk_per_thread; i++) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);  // Hardware reduction across 32 threads

        // Online softmax update
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Load V and accumulate weighted output
        const uint v_off = calc_v_offset(in_pg, page_size, kv_head_dim, head_size, pg_idx, kv_head);
        device const T* v_ptr = paged_kv_cache + v_off + simd_lid * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) {
            o[i] = o[i] * factor + exp_score * float(v_ptr[i]);
        }
    }

    // Final reduction across simdgroups (MLX sdpa_vector pattern)
    // Step 1: Communicate max and sum_exp across simdgroups
    if (simd_lid == 0) {
        max_scores_smem[simd_gid] = max_score;
        sum_exp_scores_smem[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread loads a DIFFERENT simdgroup's max (transpose access pattern)
    float loaded_max = max_scores_smem[simd_lid];
    float global_max = simd_max(loaded_max);

    // Correction factor for simdgroup simd_lid
    float factor = fast::exp(loaded_max - global_max);

    // Compute global sum with correction
    float loaded_sum = sum_exp_scores_smem[simd_lid];
    float global_sum = simd_sum(loaded_sum * factor);

    // Step 2: Aggregate outputs using shared memory transpose
    for (int i = 0; i < v_per_thread; i++) {
        // Write UNCORRECTED output to shared memory (transpose layout: row=lid, col=gid)
        outputs_smem[simd_lid * DECODE_BN + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Read from transposed position and apply correction for that simdgroup
        o[i] = simd_sum(outputs_smem[simd_gid * DECODE_BD + simd_lid] * factor);

        // Normalize by global sum
        o[i] = (global_sum > 0.0f) ? (o[i] / global_sum) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (only lane 0 of each simdgroup writes its portion)
    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            output[head_idx * head_size + simd_gid * v_per_thread + i] = T(o[i]);
        }
    }
}

// Explicit instantiations for fp16
// HEAD_DIM=64: 64/32 = 2 elements per thread
template [[host_name("attention_decode_v2_fp16_64")]] [[kernel]]
void attention_decode_v2<half, 64>(
    device const half*, device const half*,
    constant int*, constant int*, constant int*, constant int*,
    device half*, device const float*, device float*,
    uint3, uint, uint);

// HEAD_DIM=128: 128/32 = 4 elements per thread
template [[host_name("attention_decode_v2_fp16_128")]] [[kernel]]
void attention_decode_v2<half, 128>(
    device const half*, device const half*,
    constant int*, constant int*, constant int*, constant int*,
    device half*, device const float*, device float*,
    uint3, uint, uint);

// Explicit instantiations for f32
template [[host_name("attention_decode_v2_f32_64")]] [[kernel]]
void attention_decode_v2<float, 64>(
    device const float*, device const float*,
    constant int*, constant int*, constant int*, constant int*,
    device float*, device const float*, device float*,
    uint3, uint, uint);

template [[host_name("attention_decode_v2_f32_128")]] [[kernel]]
void attention_decode_v2<float, 128>(
    device const float*, device const float*,
    constant int*, constant int*, constant int*, constant int*,
    device float*, device const float*, device float*,
    uint3, uint, uint);

// Explicit instantiations for bf16
template [[host_name("attention_decode_v2_bf16_64")]] [[kernel]]
void attention_decode_v2<bfloat, 64>(
    device const bfloat*, device const bfloat*,
    constant int*, constant int*, constant int*, constant int*,
    device bfloat*, device const float*, device float*,
    uint3, uint, uint);

template [[host_name("attention_decode_v2_bf16_128")]] [[kernel]]
void attention_decode_v2<bfloat, 128>(
    device const bfloat*, device const bfloat*,
    constant int*, constant int*, constant int*, constant int*,
    device bfloat*, device const float*, device float*,
    uint3, uint, uint);
