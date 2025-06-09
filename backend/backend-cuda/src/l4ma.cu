#include "l4ma.cuh"
#include <cassert>
#include <thrust/copy.h>
#include <iostream>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdexcept>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <flashinfer/activation.cuh>
#include "ztensor.hpp"

#include <fstream>
#include <yaml-cpp/yaml.h>

L4maConfig load_l4ma_config_from_yaml(const std::string& yaml_path) {
    L4maConfig config;
    YAML::Node root = YAML::LoadFile(yaml_path);
    // Navigate YAML structure
    auto arch = root["architecture"];
    if (!arch) throw std::runtime_error("Missing architecture section in YAML");
    config.hidden_size = arch["hidden_size"].as<int>();
    config.intermediate_size = arch["intermediate_size"].as<int>();
    config.num_attention_heads = arch["num_heads"].as<int>();
    config.num_key_value_heads = arch["num_heads_kv"].as<int>();
    config.num_hidden_layers = arch["num_layers"].as<int>();
    config.use_qkv_bias = arch["use_qkv_bias"] ? arch["use_qkv_bias"].as<bool>() : false;
    config.rms_norm_eps = arch["rms_norm_eps"] ? arch["rms_norm_eps"].as<float>() : 1e-5f;
    // Tokenizer section
    auto tokenizer = root["tokenizer"];
    if (!tokenizer) throw std::runtime_error("Missing tokenizer section in YAML");
    config.vocab_size = tokenizer["vocab_size"].as<int>();
    // Pad token id (optional)
    config.pad_token_id = root["pad_token_id"] ? root["pad_token_id"].as<int>() : 0;
    return config;
}


// Macro for cuBLAS error checking
#define CUBLAS_CHECK(status)                                                    \
    do                                                                          \
    {                                                                           \
        cublasStatus_t _status = (status);                                      \
        if (_status != CUBLAS_STATUS_SUCCESS)                                   \
        {                                                                       \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, _status); \
            throw std::runtime_error("cuBLAS error");                           \
        }                                                                       \
    } while (0)

// Helper for SiLU activation (device function)
__device__ __forceinline__ float silu(const float &val) { return val / (1.0f + __expf(-val)); }

// SiLU and elementwise multiply kernel launcher
template <typename T>
void silu_and_mul(
    thrust::device_vector<T> &out,
    const thrust::device_vector<T> &input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl)
{
    T *out_ptr = thrust::raw_pointer_cast(out.data());
    const T *input_ptr = thrust::raw_pointer_cast(input.data());
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    cudaLaunchKernelEx(&config, kernel, out_ptr, input_ptr, d);
}

// Explicit template instantiation for float

template void silu_and_mul<float>(
    thrust::device_vector<float> &out,
    const thrust::device_vector<float> &input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl);

// --- L4maAttention (stub: identity) ---
template <typename T>
L4maAttention<T>::L4maAttention(const L4maConfig &config, int layer_idx,
                  const thrust::device_vector<T> &q_proj_weights,
                  const thrust::device_vector<T> &k_proj_weights,
                  const thrust::device_vector<T> &v_proj_weights,
                  const thrust::device_vector<T> &o_proj_weights,
                  const thrust::device_vector<T> &q_proj_bias,
                  const thrust::device_vector<T> &k_proj_bias,
                  const thrust::device_vector<T> &v_proj_bias)
    : config_(config), layer_idx_(layer_idx) {}

template <typename T>
L4maAttention<T>::~L4maAttention() {}

template <typename T>
void L4maAttention<T>::forward(
    thrust::device_vector<T> &attn_output,
    void *handler,
    const thrust::device_vector<T> &hidden_states,
    const int32_t *position_ids,
    thrust::device_vector<T> &kv_cache_for_layer_k,
    thrust::device_vector<T> &kv_cache_for_layer_v,
    const int32_t *kv_page_indices,
    const int32_t *kv_page_indptr,
    const int32_t *kv_last_page_lens,
    const int32_t *qo_indptr,
    int nnz,
    int batch_size,
    thrust::device_vector<T> &temp_buffer_attn,
    cudaStream_t stream)
{
    // For now, just copy hidden_states to attn_output
    thrust::copy(hidden_states.begin(), hidden_states.end(), attn_output.begin());
}

// --- L4maDecoderLayer ---
template <typename T>
L4maDecoderLayer<T>::L4maDecoderLayer(const L4maConfig &config, int layer_idx)
    : config_(config),
      self_attn_(config, layer_idx, {}, {}, {}, {}, {}, {}, {}),
      mlp_(config, {}, {}, {}) {}

// You should load real weights in a real implementation

template <typename T>
L4maDecoderLayer<T>::~L4maDecoderLayer() {}

template <typename T>
void L4maDecoderLayer<T>::forward(
    thrust::device_vector<T> &output_hidden_states,
    void *handler,
    const thrust::device_vector<T> &input_hidden_states,
    const int32_t *position_ids,
    thrust::device_vector<T> &kv_cache_for_layer_k,
    thrust::device_vector<T> &kv_cache_for_layer_v,
    const int32_t *kv_page_indices,
    const int32_t *kv_page_indptr,
    const int32_t *kv_last_page_lens,
    const int32_t *qo_indptr,
    int nnz,
    int batch_size,
    thrust::device_vector<T> &temp_buffer_layer,
    cudaStream_t stream)
{
    // RMSNorm stub: copy input to normed_hidden_states_
    if (normed_hidden_states_.size() != input_hidden_states.size())
        normed_hidden_states_.resize(input_hidden_states.size());
    thrust::copy(input_hidden_states.begin(), input_hidden_states.end(), normed_hidden_states_.begin());

    // Attention
    if (attn_output_.size() != input_hidden_states.size())
        attn_output_.resize(input_hidden_states.size());
    self_attn_.forward(attn_output_, handler, normed_hidden_states_, position_ids,
                      kv_cache_for_layer_k, kv_cache_for_layer_v,
                      kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr,
                      nnz, batch_size, temp_buffer_layer, stream);

    // MLP
    mlp_.forward(output_hidden_states, attn_output_, nnz, temp_buffer_layer, stream);
}

// --- L4maModel ---
template <typename T>
L4maModel<T>::L4maModel(const L4maConfig &config)
    : config_(config)
{
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }
}

template <typename T>
L4maModel<T>::~L4maModel() {}

template <typename T>
void L4maModel<T>::forward(
    thrust::device_vector<T> &output_hidden_states,
    const thrust::device_vector<T> &input_embeds,
    const int32_t *position_ids,
    thrust::device_vector<T> &kv_cache_ptr_k,
    thrust::device_vector<T> &kv_cache_ptr_v,
    const int32_t *kv_page_indices,
    const int32_t *kv_page_indptr,
    const int32_t *kv_last_page_lens,
    const int32_t *qo_indptr,
    const float *custom_mask,
    bool single_token_inference_mode,
    int nnz,
    int batch_size,
    int max_qo_len,
    int max_kv_len,
    cudaStream_t stream)
{
    // For demo: just run all layers sequentially
    current_hidden_states_ = input_embeds;
    thrust::device_vector<T> temp_buffer(current_hidden_states_.size() * 2);
    for (auto &layer : layers_) {
        layer.forward(current_hidden_states_, nullptr, current_hidden_states_, position_ids,
                      kv_cache_ptr_k, kv_cache_ptr_v, kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr,
                      nnz, batch_size, temp_buffer, stream);
    }
    output_hidden_states = current_hidden_states_;
}

// Static factory method for L4maModel
// Loads config from YAML and weights from ztensor file
// Only supports __nv_bfloat16 for now

template <>
L4maModel<__nv_bfloat16> L4maModel<__nv_bfloat16>::from_files(const std::string& yaml_path, const std::string& ztensor_path) {
    // 1. Load config
    L4maConfig config = load_l4ma_config_from_yaml(yaml_path);

    // 2. Load weights from ztensor
    ztensor::zTensorReader reader(ztensor_path);
    std::unordered_map<std::string, thrust::device_vector<__nv_bfloat16>> device_tensors;
    auto tensor_names = reader.list_tensors();
    for (const auto& name : tensor_names) {
        const auto& info = reader.get_tensor_info(name);
        size_t numel = info.num_elements();
        const void* raw_ptr = reader.get_raw_tensor_pointer(name);
        thrust::device_vector<__nv_bfloat16> dev_bf16(numel);
        cudaMemcpy(thrust::raw_pointer_cast(dev_bf16.data()), raw_ptr, numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        device_tensors[name] = std::move(dev_bf16);
    }

    // 3. Organize weights per layer
    std::vector<std::unordered_map<std::string, thrust::device_vector<__nv_bfloat16>>> all_layer_weights(config.num_hidden_layers);
    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer) + ".";
        std::vector<std::string> param_names = {
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
            // Add bias and norm weights if present
        };
        for (const auto& pname : param_names) {
            std::string full_name = prefix + pname;
            if (device_tensors.count(full_name)) {
                all_layer_weights[layer][pname] = device_tensors.at(full_name);
            }
        }
    }

    // 4. Construct model (for now, just pass config, you can extend to pass weights)
    L4maModel<__nv_bfloat16> model(config);
    // TODO: Actually wire weights into layers/MLP/Attention as needed
    return model;
}

// Explicit instantiations

template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;