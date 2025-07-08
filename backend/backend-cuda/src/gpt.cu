#include "gpt.cuh"
#include "config.hpp"
#include "common.cuh"   // Your helper functions header

#include <stdexcept>
#include <iostream>
#include <utility>
#include <algorithm> // for std::max

#include "flashinfer/norm.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"

// --- Helper CUDA Kernels ---
// These are still needed for operations not covered by your common.cuh or FlashInfer.

template <typename T>
__global__ void add_residual_kernel(T* x, const T* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform addition in float32 for precision, then cast back.
        x[idx] = static_cast<T>(static_cast<float>(x[idx]) + static_cast<float>(residual[idx]));
    }
}

template <typename T>
__global__ void swiglu_kernel(T* gate_proj, const T* up_proj, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate_val = static_cast<float>(gate_proj[idx]);
        float up_val = static_cast<float>(up_proj[idx]);
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float silu_val = gate_val * (1.0f / (1.0f + expf(-gate_val)));
        gate_proj[idx] = static_cast<T>(silu_val * up_val);
    }
}
// --- Constructor Implementations ---

template <typename T>
RMSNorm<T>::RMSNorm(const L4maConfig& config)
    : weight_(config.hidden_size) {}

template <typename T>
L4maMlp<T>::L4maMlp(const L4maConfig& config)
    : config_(config),
      gate_proj_weights_(config.hidden_size * config.intermediate_size),
      up_proj_weights_(config.hidden_size * config.intermediate_size),
      down_proj_weights_(config.intermediate_size * config.hidden_size) {}

template <typename T>
L4maAttention<T>::L4maAttention(const L4maConfig& config)
    : config_(config),
      q_proj_weights_(config.hidden_size * (config.num_query_heads * config.head_size)),
      k_proj_weights_(config.hidden_size * (config.num_key_value_heads * config.head_size)),
      v_proj_weights_(config.hidden_size * (config.num_key_value_heads * config.head_size)),
      o_proj_weights_((config.num_query_heads * config.head_size) * config.hidden_size) {
    if (config_.use_qkv_bias) {
        q_proj_bias_.resize(config.num_query_heads * config.head_size);
        k_proj_bias_.resize(config.num_key_value_heads * config.head_size);
        v_proj_bias_.resize(config.num_key_value_heads * config.head_size);
    }
}

template <typename T>
L4maDecoderLayer<T>::L4maDecoderLayer(const L4maConfig& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_(config),
      post_attention_layernorm_(config) {}

template <typename T>
L4maModel<T>::L4maModel(const L4maConfig& config)
    : config_(config),
      embed_tokens_weight_(config.vocab_size * config.hidden_size),
      norm_(config) {

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config);
    }
    CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle_));
}

template <typename T>
L4maForCausalLM<T>::L4maForCausalLM(const L4maConfig& config)
    : config_(config), model_(config) {
    lm_head_weight_ = model_.get_embed_tokens_weight(); // Weight tying
}

// --- get_parameters() Implementations ---

template <typename T>
std::map<std::string, thrust::device_vector<T>*> RMSNorm<T>::get_parameters() {
    return {{"weight", &weight_}};
}

template <typename T>
std::map<std::string, thrust::device_vector<T>*> L4maMlp<T>::get_parameters() {
    return {{"gate_proj.weight", &gate_proj_weights_},
            {"up_proj.weight", &up_proj_weights_},
            {"down_proj.weight", &down_proj_weights_}};
}

template <typename T>
std::map<std::string, thrust::device_vector<T>*> L4maAttention<T>::get_parameters() {
    auto params = std::map<std::string, thrust::device_vector<T>*>{
        {"q_proj.weight", &q_proj_weights_},
        {"k_proj.weight", &k_proj_weights_},
        {"v_proj.weight", &v_proj_weights_},
        {"o_proj.weight", &o_proj_weights_}};
    if (config_.use_qkv_bias) {
        params["q_proj.bias"] = &q_proj_bias_;
        params["k_proj.bias"] = &k_proj_bias_;
        params["v_proj.bias"] = &v_proj_bias_;
    }
    return params;
}

template <typename T>
std::map<std::string, thrust::device_vector<T>*> L4maDecoderLayer<T>::get_parameters() {
    std::map<std::string, thrust::device_vector<T>*> params;
    for (auto const& [key, val] : self_attn_.get_parameters()) { params["self_attn." + key] = val; }
    for (auto const& [key, val] : mlp_.get_parameters()) { params["mlp." + key] = val; }
    for (auto const& [key, val] : input_layernorm_.get_parameters()) { params["input_layernorm." + key] = val; }
    for (auto const& [key, val] : post_attention_layernorm_.get_parameters()) { params["post_attention_layernorm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, thrust::device_vector<T>*> L4maModel<T>::get_parameters() {
    std::map<std::string, thrust::device_vector<T>*> params;
    params["embed_tokens.weight"] = &embed_tokens_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        for (auto const& [key, val] : layers_[i].get_parameters()) {
            params["layers." + std::to_string(i) + "." + key] = val;
        }
    }
    for (auto const& [key, val] : norm_.get_parameters()) { params["norm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, thrust::device_vector<T>*> L4maForCausalLM<T>::get_parameters() {
    std::map<std::string, thrust::device_vector<T>*> params;
    for (auto const& [key, val] : model_.get_parameters()) {
        params["model." + key] = val;
    }
    params["lm_head.weight"] = &lm_head_weight_;
    return params;
}

template <typename T>
const thrust::device_vector<T>& L4maModel<T>::get_embed_tokens_weight() const {
    return embed_tokens_weight_;
}

// --- Forward Pass Stub ---


template <typename T>
void RMSNorm<T>::forward(thrust::device_vector<T>& output, const thrust::device_vector<T>& input, int num_tokens, cudaStream_t stream) {
    
    uint32_t batch_size = input.size() / config_.hidden_size;
    uint32_t stride = config_.hidden_size;
    uint32_t d = config_.hidden_size;

    flashinfer::norm::RMSNorm(
        const_cast<T *>(thrust::raw_pointer_cast(input.data())),
        const_cast<T *>(thrust::raw_pointer_cast(weight_.data())),
        thrust::raw_pointer_cast(output.data()),
        batch_size, d, stride, stride, config_.rms_norm_eps
    );
}

template <typename T>
void L4maMlp<T>::forward(thrust::device_vector<T>& output, const thrust::device_vector<T>& x, int num_tokens, thrust::device_vector<T>& temp_buffer, cublasLtHandle_t ltHandle, cudaStream_t stream) {
    std::cerr << "Warning: L4maMlp<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maAttention<T>::forward(thrust::device_vector<T>& attn_output, const thrust::device_vector<T>& hidden_states, const int32_t* position_ids, thrust::device_vector<T>& kv_cache_k, thrust::device_vector<T>& kv_cache_v, const int32_t* kv_page_indices, const int32_t* kv_page_indptr, const int32_t* kv_last_page_lens, const int32_t* qo_indptr, int nnz, int batch_size, thrust::device_vector<T>& temp_buffer, cublasLtHandle_t ltHandle, cudaStream_t stream) {
    std::cerr << "Warning: L4maAttention<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maDecoderLayer<T>::forward(thrust::device_vector<T>& hidden_states, const int32_t* position_ids, thrust::device_vector<T>& kv_cache_k, thrust::device_vector<T>& kv_cache_v, const int32_t* kv_page_indices, const int32_t* kv_page_indptr, const int32_t* kv_last_page_lens, const int32_t* qo_indptr, int nnz, int batch_size, thrust::device_vector<T>& temp_buffer, cublasLtHandle_t ltHandle, cudaStream_t stream) {
    std::cerr << "Warning: L4maDecoderLayer<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maModel<T>::forward(thrust::device_vector<T>& hidden_states, const thrust::device_vector<int32_t>& input_ids, const thrust::device_vector<int32_t>& position_ids, thrust::device_vector<T>& kv_cache_k, thrust::device_vector<T>& kv_cache_v, const int32_t* kv_page_indices, const int32_t* kv_page_indptr, const int32_t* kv_last_page_lens, const int32_t* qo_indptr, int batch_size, cudaStream_t stream) {
    std::cerr << "Warning: L4maModel<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maForCausalLM<T>::forward(thrust::device_vector<float>& logits, const thrust::device_vector<int32_t>& input_ids, const thrust::device_vector<int32_t>& position_ids, thrust::device_vector<T>& kv_cache_k, thrust::device_vector<T>& kv_cache_v, const int32_t* kv_page_indices, const int32_t* kv_page_indptr, const int32_t* kv_last_page_lens, const int32_t* qo_indptr, int batch_size, cudaStream_t stream) {
    std::cerr << "Warning: L4maForCausalLM<T>::forward is not implemented." << std::endl;
}

// --- Explicit Template Instantiations ---
template class RMSNorm<float>;
template class L4maMlp<float>;
template class L4maAttention<float>;
template class L4maDecoderLayer<float>;
template class L4maModel<float>;
template class L4maForCausalLM<float>;

template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;