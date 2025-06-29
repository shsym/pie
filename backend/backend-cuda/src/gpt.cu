#include "gpt.cuh"
#include "config.hpp"

#include <stdexcept>
#include <iostream>
#include <utility>

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
void L4maMlp<T>::forward(thrust::device_vector<T>&, const thrust::device_vector<T>&, int, thrust::device_vector<T>&, cublasLtHandle_t, cudaStream_t) {
    std::cerr << "Warning: L4maMlp::forward is not implemented." << std::endl;
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