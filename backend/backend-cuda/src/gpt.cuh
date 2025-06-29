#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "config.hpp"

// Forward declarations for CUDA types
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct CUstream_st* cudaStream_t;


/**
 * @brief Base class for all model components (modules).
 */
template <typename T>
class Module {
public:
    virtual ~Module() = default;

    /**
     * @brief Retrieves pointers to the internally managed parameters.
     */
    virtual std::map<std::string, thrust::device_vector<T>*> get_parameters() = 0;
};

/**
 * @brief RMS Normalization layer.
 */
template <typename T>
class RMSNorm : public Module<T> {
public:
    explicit RMSNorm(const L4maConfig& config);
    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;

private:
    thrust::device_vector<T> weight_;
};

/**
 * @brief The MLP block of the L4MA model.
 */
template <typename T>
class L4maMlp : public Module<T> {
public:
    explicit L4maMlp(const L4maConfig& config);

    void forward(thrust::device_vector<T>& output,
                 const thrust::device_vector<T>& x,
                 int num_tokens,
                 thrust::device_vector<T>& temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;

private:
    L4maConfig config_;
    thrust::device_vector<T> gate_proj_weights_;
    thrust::device_vector<T> up_proj_weights_;
    thrust::device_vector<T> down_proj_weights_;
};

/**
 * @brief The attention block of the L4MA model.
 */
template <typename T>
class L4maAttention : public Module<T> {
public:
    explicit L4maAttention(const L4maConfig& config);
    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;

private:
    L4maConfig config_;
    thrust::device_vector<T> q_proj_weights_;
    thrust::device_vector<T> k_proj_weights_;
    thrust::device_vector<T> v_proj_weights_;
    thrust::device_vector<T> o_proj_weights_;
    thrust::device_vector<T> q_proj_bias_;
    thrust::device_vector<T> k_proj_bias_;
    thrust::device_vector<T> v_proj_bias_;
};

/**
 * @brief A single decoder layer of the L4MA model.
 */
template <typename T>
class L4maDecoderLayer : public Module<T> {
public:
    explicit L4maDecoderLayer(const L4maConfig& config);
    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;

private:
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;
    RMSNorm<T> input_layernorm_;
    RMSNorm<T> post_attention_layernorm_;
};

/**
 * @brief The main body of the L4MA model.
 */
template <typename T>
class L4maModel : public Module<T> {
public:
    explicit L4maModel(const L4maConfig& config);
    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;
    
    // Getter to allow weight tying for lm_head
    const thrust::device_vector<T>& get_embed_tokens_weight() const;

private:
    L4maConfig config_;
    thrust::device_vector<T> embed_tokens_weight_;
    std::vector<L4maDecoderLayer<T>> layers_;
    RMSNorm<T> norm_;
};

/**
 * @brief The L4MA model with a causal language model head.
 */
template <typename T>
class L4maForCausalLM : public Module<T> {
public:
    explicit L4maForCausalLM(const L4maConfig& config);
    std::map<std::string, thrust::device_vector<T>*> get_parameters() override;

private:
    L4maConfig config_;
    L4maModel<T> model_;
    thrust::device_vector<T> lm_head_weight_;
};
