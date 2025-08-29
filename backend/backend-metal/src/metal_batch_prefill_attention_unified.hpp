#pragma once

#include "metal_common.hpp"
#include <Metal/Metal.h>
#include <cstdint>
#include <vector>

/**
 * @brief Unified FlashAttention-like Metal Implementation
 * 
 * This provides a scalable FlashAttention implementation that handles:
 * - Arbitrary sequence lengths through 2D tiling
 * - Parallel head processing via 3D dispatch grid  
 * - Dynamic tile size configuration based on sequence characteristics
 * - Efficient memory access patterns for paged KV cache
 */

namespace metal::unified_attention {

// === Configuration Constants ===
constexpr int MAX_TILE_SIZE = 128;
constexpr int MIN_TILE_SIZE = 32;
constexpr int DEFAULT_THREADGROUP_SIZE = 256;

// === Parameter Structures ===

/**
 * @brief Unified kernel parameters passed to Metal compute shader
 */
struct UnifiedParams {
    int32_t num_qo;           // Total query tokens across all sequences
    int32_t num_sequences;    // Number of sequences in batch
    int32_t head_dim;         // Total head dimension (num_heads * head_size)
    int32_t head_size;        // Size per attention head
    int32_t num_heads;        // Number of attention heads
    int32_t page_size;        // Tokens per KV cache page
    int32_t max_seq_len;      // Maximum sequence length for tile sizing
    float scale;              // Attention scaling factor (1/sqrt(head_size))
};

/**
 * @brief Tile configuration for dynamic sizing
 */
struct TileConfig {
    int q_tile_size;          // Query tile size
    int kv_tile_size;         // KV tile size  
    int num_q_tiles;          // Number of query tiles per sequence
    int num_kv_tiles;         // Number of KV tiles per sequence
    int max_q_tiles_global;   // Maximum Q tiles across all sequences
    int max_kv_tiles_global;  // Maximum KV tiles across all sequences
};

/**
 * @brief Dispatch configuration for 3D grid
 */
struct DispatchConfig {
    MTLSize threadgroups_per_grid;  // 3D grid dimensions
    MTLSize threads_per_threadgroup; // Threadgroup size
    std::vector<int> function_constants; // Dynamic constants for kernel
};

// === Core Functions ===

/**
 * @brief Initialize the unified attention subsystem
 * @return true if successful, false otherwise
 */
bool initialize();

/**
 * @brief Cleanup unified attention resources
 */
void cleanup();

/**
 * @brief Check if unified attention is initialized
 */
bool is_initialized();

/**
 * @brief Calculate optimal tile configuration for given sequence characteristics
 * 
 * @param sequences_info Array of sequence lengths
 * @param num_sequences Number of sequences
 * @param head_dim Head dimension
 * @param page_size KV cache page size
 * @return TileConfig Optimal tiling configuration
 */
TileConfig calculate_tile_config(
    const std::vector<int>& sequence_lengths,
    int head_dim,
    int page_size
);

/**
 * @brief Generate 3D dispatch configuration for unified kernel
 * 
 * @param tile_config Tile configuration from calculate_tile_config
 * @param num_sequences Number of sequences in batch
 * @param num_heads Number of attention heads
 * @return DispatchConfig 3D dispatch grid configuration
 */
DispatchConfig generate_dispatch_config(
    const TileConfig& tile_config,
    int num_sequences, 
    int num_heads
);

/**
 * @brief Unified batch prefill attention with 2D tiling and 3D dispatch
 * 
 * This function implements a FlashAttention-like algorithm that scales to arbitrary
 * sequence lengths by using:
 * - 2D tiling of query and KV sequences for memory efficiency
 * - 3D dispatch grid: [q_tiles, kv_tiles, batch_heads]
 * - Dynamic tile size configuration based on sequence characteristics
 * - Parallel head processing instead of sequential loops
 * 
 * @param q_input Query tensor [num_qo, head_dim] in bfloat16
 * @param paged_k_cache Paged key cache [num_pages_total, page_size, head_dim]
 * @param paged_v_cache Paged value cache [num_pages_total, page_size, head_dim]
 * @param qo_indptr Query output indices [num_sequences+1] - sequence boundaries
 * @param kv_page_indptr KV page indices [num_sequences+1] - page boundaries per sequence
 * @param kv_page_indices Page indices array [total_pages_across_sequences]
 * @param kv_last_page_lens Last page lengths [num_sequences] - tokens in final page
 * @param output Output tensor [num_qo, head_dim] in bfloat16 (pre-allocated)
 * @param params Unified kernel parameters
 * @param debug_out Optional debug output buffer
 */
void unified_batch_prefill_attention_bf16(
    const bfloat16_t* q_input,
    const bfloat16_t* paged_k_cache,
    const bfloat16_t* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    bfloat16_t* output,
    const UnifiedParams& params,
    float* debug_out = nullptr
);

/**
 * @brief F32 version of unified batch prefill attention with 2D tiling and 3D dispatch
 * 
 * Higher precision version for accuracy testing and validation.
 * Uses the same FlashAttention algorithm but with full float32 precision.
 */
void unified_batch_prefill_attention_f32(
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    const UnifiedParams& params,
    float* debug_out = nullptr
);

/**
 * @brief High-level interface with automatic configuration
 * 
 * Automatically calculates optimal tile sizes and dispatch configuration based on
 * input characteristics. This is the recommended interface for most use cases.
 * 
 * @param q_input Query tensor [num_qo, head_dim]  
 * @param paged_k_cache Paged key cache [num_pages_total, page_size, head_dim]
 * @param paged_v_cache Paged value cache [num_pages_total, page_size, head_dim] 
 * @param qo_indptr Query output indices [num_sequences+1]
 * @param kv_page_indptr KV page indices [num_sequences+1]
 * @param kv_page_indices Page indices array [total_pages_across_sequences]
 * @param kv_last_page_lens Last page lengths [num_sequences]
 * @param output Output tensor [num_qo, head_dim] (pre-allocated)
 * @param num_qo Total number of query tokens
 * @param num_sequences Number of sequences in batch
 * @param head_dim Total head dimension (num_heads * head_size)
 * @param head_size Size of each attention head  
 * @param page_size Number of tokens per KV cache page
 * @param scale Attention scaling factor (typically 1/sqrt(head_size))
 * @param debug_out Optional debug output buffer
 */
void unified_batch_prefill_attention_auto(
    const bfloat16_t* q_input,
    const bfloat16_t* paged_k_cache,
    const bfloat16_t* paged_v_cache, 
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    bfloat16_t* output,
    int num_qo,
    int num_sequences,
    int head_dim,
    int head_size,
    int page_size,
    float scale,
    float* debug_out = nullptr
);

// === Utility Functions ===

/**
 * @brief Calculate memory requirements for unified attention
 * 
 * @param max_seq_len Maximum sequence length
 * @param num_sequences Number of sequences
 * @param head_dim Head dimension
 * @param num_heads Number of heads
 * @return Required workspace size in bytes
 */
size_t calculate_workspace_size(
    int max_seq_len,
    int num_sequences, 
    int head_dim,
    int num_heads
);

/**
 * @brief Validate input parameters for unified attention
 * 
 * @param params Unified kernel parameters
 * @return true if parameters are valid, false otherwise
 */
bool validate_parameters(const UnifiedParams& params);

/**
 * @brief Get performance characteristics for given configuration
 * 
 * @param tile_config Tile configuration
 * @param num_sequences Number of sequences
 * @param num_heads Number of heads
 * @return Estimated FLOPS and memory bandwidth utilization
 */
struct PerformanceEstimate {
    double estimated_flops;      // Estimated FLOPS for the operation
    double memory_bandwidth_gb;  // Estimated memory bandwidth usage
    double estimated_time_ms;    // Estimated execution time
};

PerformanceEstimate estimate_performance(
    const TileConfig& tile_config,
    int num_sequences,
    int num_heads,
    const std::vector<int>& sequence_lengths
);

} // namespace metal::unified_attention