// GPU ops using the exact CUDA kernels/utilities from backend/backend-cuda
#pragma once

#include <cstdint>
#include <string>

namespace ops {

struct EmbeddingConfig {
    int num_tokens;   // number of indices to lookup (rows in output)
    int hidden_size;  // embedding width (columns)
    int vocab_size;   // number of embedding rows
};

// Runs the embedding lookup via backend/common.cu::embed<T,I>() and records artifacts
void run_embedding_lookup(const std::string& case_id,
                          const EmbeddingConfig& cfg,
                          uint64_t seed);

struct ExtractKConfig {
    int M;  // rows
    int N;  // cols
    int k;  // values to extract per row
};

void run_extract_k_values(const std::string& case_id,
                          const ExtractKConfig& cfg,
                          uint64_t seed);

// RMS Normalization test
struct RMSNormConfig {
    int num_tokens;
    int hidden_size;
    float eps;
};

void run_rms_norm(const std::string& case_id,
                  const RMSNormConfig& cfg,
                  uint64_t seed);

// SiLU activation and multiplication test
struct SiLUAndMulConfig {
    int num_tokens;
    int intermediate_size;
};

void run_silu_and_mul(const std::string& case_id,
                      const SiLUAndMulConfig& cfg,
                      uint64_t seed);

// Residual addition test
struct AddResidualConfig {
    int num_tokens;
    int hidden_size;
};

void run_add_residual(const std::string& case_id,
                      const AddResidualConfig& cfg,
                      uint64_t seed);

// GEMM operation test
struct GemmConfig {
    int m;          // output rows
    int n;          // output cols
    int k;          // reduction dimension
    bool transa;    // transpose A
    bool transb;    // transpose B
    bool use_bias;  // whether to include bias
};

void run_gemm(const std::string& case_id,
              const GemmConfig& cfg,
              uint64_t seed);

// Type casting test
struct CastTypeConfig {
    int num_elements;
    std::string input_dtype;   // "fp32", "fp16", "bf16"
    std::string output_dtype;  // "fp32", "fp16", "bf16"
};

void run_cast_type(const std::string& case_id,
                   const CastTypeConfig& cfg,
                   uint64_t seed);

// RoPE (Rotary Position Embedding) test
struct RoPEConfig {
    int num_tokens;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    float rope_theta;
    float rope_factor;
    float rope_low_frequency_factor;
    float rope_high_frequency_factor;
    int max_position_embeddings;
};

void run_rope(const std::string& case_id,
              const RoPEConfig& cfg,
              uint64_t seed);

// Top-K mask logits test (softmax + masking)
struct TopKMaskConfig {
    int num_tokens;
    int vocab_size;
    int k;
};

void run_topk_mask_logits(const std::string& case_id,
                          const TopKMaskConfig& cfg,
                          uint64_t seed);

// Softmax test
struct SoftmaxConfig {
    int batch_size;
    int vocab_size;
    float temperature;
};

void run_softmax(const std::string& case_id,
                 const SoftmaxConfig& cfg,
                 uint64_t seed);

// Batch prefill attention test
struct BatchPrefillAttentionConfig {
    int num_tokens;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    int kv_len;
    int page_size;
};

void run_batch_prefill_attention(const std::string& case_id,
                                 const BatchPrefillAttentionConfig& cfg,
                                 uint64_t seed);

// Grouped GEMM operation test (multiple GEMMs in batch)
struct GroupedGemmConfig {
    int num_groups;     // Number of separate GEMM operations
    int m;             // Rows per GEMM
    int n;             // Columns per GEMM
    int k;             // Reduction dimension per GEMM
    bool transa;       // Transpose A matrices
    bool transb;       // Transpose B matrices
    bool use_bias;     // Whether to include bias per GEMM
};

void run_grouped_gemm(const std::string& case_id,
                      const GroupedGemmConfig& cfg,
                      uint64_t seed);

// Append paged KV cache operation test
struct AppendPagedKVCacheConfig {
    int num_tokens;
    int num_kv_heads;
    int head_size;
    int page_size;
    int max_num_pages;
    int batch_size;
};

void run_append_paged_kv_cache(const std::string& case_id,
                               const AppendPagedKVCacheConfig& cfg,
                               uint64_t seed);

// Template versions of operations for multi-dtype testing
template<typename T>
void run_gemm_typed(const std::string& case_id, const GemmConfig& cfg, uint64_t seed);

// Typed embedding lookup (T: data type, I: index type)
template<typename T, typename I>
void run_embedding_lookup_typed(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed);

// Typed extract_k_values
template<typename T>
void run_extract_k_values_typed(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed);

template<typename T>
void run_rms_norm_typed(const std::string& case_id, const RMSNormConfig& cfg, uint64_t seed);

template<typename T>
void run_silu_and_mul_typed(const std::string& case_id, const SiLUAndMulConfig& cfg, uint64_t seed);

template<typename T>
void run_add_residual_typed(const std::string& case_id, const AddResidualConfig& cfg, uint64_t seed);

template<typename T>
void run_rope_typed(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed);

template<typename T>
void run_grouped_gemm_typed(const std::string& case_id, const GroupedGemmConfig& cfg, uint64_t seed);

// Automatic multi-dtype testing function
void run_all_dtypes_for_operation(const std::string& op_name, const std::string& base_case_id,
                                 const void* config_ptr, uint64_t seed);

// Metal backend implementations
void run_gemm_metal(const std::string& case_id, const GemmConfig& cfg, uint64_t seed);
void run_embedding_lookup_metal(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed);
void run_silu_and_mul_metal(const std::string& case_id, const SiLUAndMulConfig& cfg, uint64_t seed);
void run_extract_k_values_metal(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed);
void run_softmax_metal(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed);
void run_rms_norm_metal(const std::string& case_id, const RMSNormConfig& cfg, uint64_t seed);

} // namespace ops
