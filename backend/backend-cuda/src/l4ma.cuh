#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_vector.h>
#include <cuda_bf16.h>

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

// Forward Declarations
struct L4maConfig;
template <typename T>
class L4maMlp;
template <typename T>
class L4maAttention;
template <typename T>
class L4maDecoderLayer;
template <typename T>
class L4maModel;

// Constants
constexpr int PAGE_SIZE = 16;

// Model Configuration
struct L4maConfig
{
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    bool use_qkv_bias;
    float rms_norm_eps;
    int vocab_size;
    int pad_token_id;
    float rope_base = 10000.0f;

    int head_dim() const { return hidden_size / num_attention_heads; }
    void print() const;
};

L4maConfig load_l4ma_config_from_yaml(const std::string &yaml_path);

/**
 * @class L4maMlp
 * @brief Implements the SwiGLU MLP module.
 */
template <typename T>
class L4maMlp
{
public:
    L4maMlp(const L4maConfig &config,
            const thrust::device_vector<T> &gate_proj_weights,
            const thrust::device_vector<T> &up_proj_weights,
            const thrust::device_vector<T> &down_proj_weights);
    ~L4maMlp();

    void forward(thrust::device_vector<T> &output,
                 const thrust::device_vector<T> &x,
                 int num_tokens,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

private:
    const L4maConfig &config_;
    thrust::device_vector<T> gate_proj_weights_;
    thrust::device_vector<T> up_proj_weights_;
    thrust::device_vector<T> down_proj_weights_;
};

/**
 * @class L4maAttention
 * @brief Implements paged multi-head attention with RoPE.
 */
template <typename T>
class L4maAttention
{
public:
    L4maAttention(const L4maConfig &config,
                  const thrust::device_vector<T> &q_proj_weights,
                  const thrust::device_vector<T> &k_proj_weights,
                  const thrust::device_vector<T> &v_proj_weights,
                  const thrust::device_vector<T> &o_proj_weights);
    ~L4maAttention();

    void forward(thrust::device_vector<T> &attn_output,
                 const thrust::device_vector<T> &hidden_states,
                 const int32_t *position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int nnz,
                 int batch_size,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

private:
    const L4maConfig &config_;
    thrust::device_vector<T> q_proj_weights_, k_proj_weights_, v_proj_weights_, o_proj_weights_;
};

/**
 * @class L4maDecoderLayer
 * @brief A single transformer decoder layer.
 */
template <typename T>
class L4maDecoderLayer
{
public:
    L4maDecoderLayer(const L4maConfig &config, const std::unordered_map<std::string, thrust::device_vector<T>> &weights);
    ~L4maDecoderLayer();

    void forward(thrust::device_vector<T> &hidden_states, // In-place
                 const int32_t *position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int nnz,
                 int batch_size,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

private:
    const L4maConfig &config_;
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;

    thrust::device_vector<T> input_layernorm_weight_;
    thrust::device_vector<T> post_attention_layernorm_weight_;

    // Persistent buffers to avoid reallocation
    thrust::device_vector<T> residual_;
    thrust::device_vector<T> normed_hidden_states_;
};

/**
 * @class L4maModel
 * @brief The main L4MA transformer model.
 */
template <typename T>
class L4maModel
{
public:
    static L4maModel<T> from_files(const std::string &yaml_path, const std::string &ztensor_path);

    L4maModel(const L4maConfig &config, const std::unordered_map<std::string, thrust::device_vector<T>> &all_weights);
    ~L4maModel();

    // The forward pass now takes token IDs and produces logits
    void forward(thrust::device_vector<float> &logits,
                 const thrust::device_vector<int32_t> &input_ids,
                 const thrust::device_vector<int32_t> &position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int batch_size,
                 cudaStream_t stream);

private:
    L4maConfig config_;
    cublasLtHandle_t cublaslt_handle_;

    // Model weights
    thrust::device_vector<T> embedding_weights_;
    thrust::device_vector<T> lm_head_weights_;
    thrust::device_vector<T> final_norm_weight_;

    std::vector<L4maDecoderLayer<T>> layers_;

    // Persistent buffers
    thrust::device_vector<T> hidden_states_;
    thrust::device_vector<T> temp_bwd_buffer_;
};