#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "config.hpp"
#include "tensor.hpp"
#include "profiler.hpp"
#include "flashinfer_ops.cuh"

// Forward declarations
class StackAllocator;
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct CUstream_st* cudaStream_t;


/**
 * @brief Base class for all model components (modules).
 */
template <typename T>
class Module {
public:
    virtual ~Module() = default;
    virtual std::map<std::string, Tensor<T>*> get_parameters() = 0;
};

// ---

/**
 * @brief RMS Normalization layer.
 */
template <typename T>
class RMSNorm : public Module<T> {
public:
    explicit RMSNorm(const L4maConfig& config);
    
    // Unchanged: This layer does not use a workspace buffer.
    void forward(PerformanceLogger& logger,
                 T* output,
                 const T* input,
                 int num_tokens,
                 cudaStream_t stream);

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> weight_;
};

// ---

/**
 * @brief The MLP block of the L4MA model.
 */
template <typename T>
class L4maMlp : public Module<T> {
public:
    explicit L4maMlp(const L4maConfig& config);

    // REFACTORED: Now accepts a StackAllocator.
    void forward(PerformanceLogger& logger,
                 StackAllocator& allocator,
                 T* output,
                 const T* x,
                 int num_tokens,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> gate_proj_weights_;
    Tensor<T> up_proj_weights_;
    Tensor<T> down_proj_weights_;
};

// ---

/**
 * @brief The attention block of the L4MA model.
 */
template <typename T>
class L4maAttention : public Module<T> {
public:
    explicit L4maAttention(const L4maConfig& config);

    // REFACTORED: Now accepts a StackAllocator.
    void forward(PerformanceLogger& logger,
                 StackAllocator& allocator,
                 T* attn_output,
                 const T* hidden_states,
                 thrust::device_vector<int32_t>& position_ids,
                 T* kv_cache_k,
                 T* kv_cache_v,
                 thrust::device_vector<int32_t>& kv_page_indices,
                 thrust::device_vector<int32_t>& kv_page_indptr,
                 thrust::device_vector<int32_t>& kv_last_page_lens,
                 thrust::device_vector<int32_t>& qo_indptr,
                 thrust::device_vector<uint8_t>& custom_mask,
                 thrust::device_vector<int32_t>& mask_indptr,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream,
                 flashinfer::BatchPrefillHandler& prefill_handler,
                 const int32_t page_size,
                 thrust::device_vector<int32_t>& kv_batch_indices,
                 thrust::device_vector<int32_t>& kv_positions
                );

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> q_proj_weights_;
    Tensor<T> k_proj_weights_;
    Tensor<T> v_proj_weights_;
    Tensor<T> o_proj_weights_;
    // Tensor<T> q_proj_bias_;
    // Tensor<T> k_proj_bias_;
    // Tensor<T> v_proj_bias_;
};

// ---

/**
 * @brief A single decoder layer of the L4MA model.
 */
template <typename T>
class L4maDecoderLayer : public Module<T> {
public:
    explicit L4maDecoderLayer(const L4maConfig& config);

    // REFACTORED: Now accepts a StackAllocator.
    void forward(PerformanceLogger& logger,
                 StackAllocator& allocator,
                 T* hidden_states, // In-place
                 thrust::device_vector<int32_t>& position_ids,
                 T* kv_cache_k,
                 T* kv_cache_v,
                 thrust::device_vector<int32_t>& kv_page_indices,
                 thrust::device_vector<int32_t>& kv_page_indptr,
                 thrust::device_vector<int32_t>& kv_last_page_lens,
                 thrust::device_vector<int32_t>& qo_indptr,
                 thrust::device_vector<uint8_t>& custom_mask,
                 thrust::device_vector<int32_t>& mask_indptr,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream,
                 flashinfer::BatchPrefillHandler& prefill_handler,
                 const int32_t page_size,
                 thrust::device_vector<int32_t>& kv_batch_indices,
                 thrust::device_vector<int32_t>& kv_positions
                );

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;
    RMSNorm<T> input_layernorm_;
    RMSNorm<T> post_attention_layernorm_;
};

// ---

/**
 * @brief The main body of the L4MA model.
 */
template <typename T>
class L4maModel : public Module<T> {
public:
    explicit L4maModel(const L4maConfig& config);

    // REFACTORED: Now accepts a StackAllocator.
    void forward(PerformanceLogger& logger,
                 StackAllocator& allocator,
                 T* final_norm_output,
                 const thrust::device_vector<uint32_t>& input_ids,
                 thrust::device_vector<int32_t>& position_ids,
                 thrust::device_vector<T>& kv_cache_k,
                 thrust::device_vector<T>& kv_cache_v,
                 thrust::device_vector<int32_t>& kv_page_indices,
                 thrust::device_vector<int32_t>& kv_page_indptr,
                 thrust::device_vector<int32_t>& kv_last_page_lens,
                 thrust::device_vector<int32_t>& qo_indptr,
                 thrust::device_vector<uint8_t>& custom_mask,
                 thrust::device_vector<int32_t>& mask_indptr,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream,
                 flashinfer::BatchPrefillHandler& prefill_handler,
                 const int32_t page_size,
                 thrust::device_vector<int32_t>& kv_batch_indices,
                 thrust::device_vector<int32_t>& kv_positions
                );

    std::map<std::string, Tensor<T>*> get_parameters() override;

    Tensor<T>& get_embed_tokens_weight() { return embed_tokens_weight_; }

private:
    L4maConfig config_;
    Tensor<T> embed_tokens_weight_;
    std::vector<L4maDecoderLayer<T>> layers_;
    RMSNorm<T> norm_;
};

// ---

/**
 * @brief The L4MA model with a causal language model head.
 */
template <typename T>
class L4maForCausalLM : public Module<T> {
public:
    explicit L4maForCausalLM(const L4maConfig& config);


    void forward(PerformanceLogger& logger,
                 StackAllocator& allocator,
                 Tensor<T>& output,
                 const thrust::device_vector<uint32_t>& input_ids,
                 thrust::device_vector<int32_t>& position_ids,
                 thrust::device_vector<int32_t>& kv_page_indices,
                 thrust::device_vector<int32_t>& kv_page_indptr,
                 std::vector<int32_t>& kv_page_indptr_host,
                 thrust::device_vector<int32_t>& kv_last_page_lens,
                 thrust::device_vector<int32_t>& qo_indptr,
                 std::vector<int32_t>& qo_indptr_host,
                 thrust::device_vector<uint8_t>& custom_mask,
                 thrust::device_vector<int32_t>& mask_indptr,
                 cudaStream_t stream,
                 const int32_t page_size,
                 thrust::device_vector<int32_t>& kv_batch_indices,
                 thrust::device_vector<int32_t>& kv_positions
                );

    std::map<std::string, Tensor<T>*> get_parameters() override;
    void create_kv_device_vectors(int max_kv_num);
    size_t get_workspace_size(int max_num_tokens) const;

    L4maConfig& get_config() { return config_; }

private:
    L4maConfig config_;
    cublasLtHandle_t cublaslt_handle_;

    L4maModel<T> model_;

    thrust::device_vector<T> kv_cache_k_;
    thrust::device_vector<T> kv_cache_v_;
};