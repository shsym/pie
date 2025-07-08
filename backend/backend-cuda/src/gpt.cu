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
#include "flashinfer_ops.cuh"

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
void L4maMlp<T>::forward(thrust::device_vector<T>& output, const thrust::device_vector<T>& x, int num_tokens, thrust::device_vector<T>& temp_buffer, cublasLtHandle_t ltHandle, cudaStream_t stream, thrust::device_vector<char>& workspace) {
    const int hidden_size = config_.hidden_size;
    const int intermediate_size = config_.intermediate_size;
    
    // Partition the temp buffer
    thrust::device_vector<T> gate_proj_out(thrust::device_pointer_cast(temp_buffer.data()), thrust::device_pointer_cast(temp_buffer.data() + (size_t)num_tokens * intermediate_size));
    thrust::device_vector<T> up_proj_out(thrust::device_pointer_cast(gate_proj_out.data().get() + (size_t)num_tokens * intermediate_size), thrust::device_pointer_cast(gate_proj_out.data().get() + 2 * (size_t)num_tokens * intermediate_size));

    // 1. Gate projection: gate_proj_out = x @ W_gate^T
    gemm_cublasLt<T>(ltHandle, stream, x, gate_proj_weights_, nullptr, gate_proj_out, num_tokens, intermediate_size, hidden_size, workspace, false, true);
    
    // 2. Up projection: up_proj_out = x @ W_up^T
    gemm_cublasLt<T>(ltHandle, stream, x, up_proj_weights_, nullptr, up_proj_out, num_tokens, intermediate_size, hidden_size, workspace, false, true);

    // 3. SwiGLU activation: silu(gate_proj) * up_proj (in-place into gate_proj_out)
    int swiglu_elements = num_tokens * intermediate_size;
    swiglu_kernel<<<(swiglu_elements + 255) / 256, 256, 0, stream>>>(thrust::raw_pointer_cast(gate_proj_out.data()), thrust::raw_pointer_cast(up_proj_out.data()), swiglu_elements);

    // 4. Down projection: output = (activated_output) @ W_down^T
    gemm_cublasLt<T>(ltHandle, stream, gate_proj_out, down_proj_weights_, nullptr, output, num_tokens, hidden_size, intermediate_size, workspace, false, true);
}

template <typename T>
void L4maAttention<T>::forward(
    thrust::device_vector<T>& attn_output,
    const thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<uint32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    const int32_t* kv_page_indices,
    const int32_t* kv_page_indptr,
    const int32_t* kv_last_page_lens,
    const int32_t* qo_indptr,
    thrust::device_vector<T>& temp_buffer,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace,
    flashinfer::BatchPrefillHandler& prefill_handler
) {

    const int batch = hidden_states.size() / config_.hidden_size;
    const int hidden_size = config_.hidden_size;
    const int head_dim = config_.head_size;
    const int num_q_heads = config_.num_query_heads;
    const int num_kv_heads = config_.num_key_value_heads;
    
    size_t q_size = (size_t)batch * num_q_heads * head_dim;
    size_t k_size = (size_t)batch * num_kv_heads * head_dim;
    size_t v_size = (size_t)batch * num_kv_heads * head_dim;

    if(temp_buffer.size() < q_size + k_size + v_size) {
        // panic if the temp buffer is too small
        throw std::runtime_error("Temporary buffer size is too small for Q, K, V projections.");
    }

    // Partition buffer
    thrust::device_vector<T> q_proj(thrust::device_pointer_cast(temp_buffer.data()), thrust::device_pointer_cast(temp_buffer.data() + q_size));
    thrust::device_vector<T> k_proj(thrust::device_pointer_cast(q_proj.data().get() + q_size), thrust::device_pointer_cast(q_proj.data().get() + q_size + k_size));
    thrust::device_vector<T> v_proj(thrust::device_pointer_cast(k_proj.data().get() + k_size), thrust::device_pointer_cast(k_proj.data().get() + k_size + v_size));
    
    // 1. Q, K, V projections
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, q_proj_weights_, config_.use_qkv_bias ? &q_proj_bias_ : nullptr, q_proj, batch, num_q_heads * head_dim, hidden_size, workspace, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, k_proj_weights_, config_.use_qkv_bias ? &k_proj_bias_ : nullptr, k_proj, batch, num_kv_heads * head_dim, hidden_size, workspace, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, v_proj_weights_, config_.use_qkv_bias ? &v_proj_bias_ : nullptr, v_proj, batch, num_kv_heads * head_dim, hidden_size, workspace, false, true);

    // 2. Apply RoPE (in-place)
    flashinfer::BatchQKApplyLlama31RotaryPosIds(
        const_cast<T *>(thrust::raw_pointer_cast(q_proj.data())), // q
        const_cast<T *>(thrust::raw_pointer_cast(k_proj.data())), // k
        thrust::raw_pointer_cast(q_proj.data()),                  // q_rope (not available)
        thrust::raw_pointer_cast(k_proj.data()),                  // k_rope (not available)
        thrust::raw_pointer_cast(position_ids.data()),                 // pos_ids (uint32_t*)
        batch,                                                    // nnz (assuming batch size for now)
        num_q_heads,                                                       // num_qo_heads
        num_kv_heads,                                                      // num_kv_heads
        head_dim,                                       // rotary_dim
        head_dim,                                       // head_dim
        num_q_heads * head_dim,                                                  // q_stride_n
        head_dim,                                       // q_stride_h
        num_kv_heads * head_dim,                                                 // k_stride_n
        head_dim,
        ///----                                                      // k_stride_h
        // q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h (not available)
        num_q_heads * head_dim,
        head_dim,
        num_kv_heads * head_dim,
        head_dim,
        ///----                                                      
        false, // interleave
        8.0f,  // rope_scale
        5e5f,  // rope_theta
        1.0f,  // low_freq_factor
        4.0f,  // high_freq_factor
        8192,  // old_context_length
        stream // cudaStream_t
    );

//     const int page_size = 32;


//     // 3. Create paged KV-cache object
//     flashinfer::paged_kv_t<T, int32_t> paged_kv(
//         num_kv_heads, page_size, head_dim, batch,
//         flashinfer::QKVLayout::kNHD,
//         thrust::raw_pointer_cast(kv_cache_k.data()),
//         thrust::raw_pointer_cast(kv_cache_v.data()),
//         thrust::raw_pointer_cast(kv_page_indices.data()), 
//         thrust::raw_pointer_cast(kv_page_indptr.data()), 
//         thrust::raw_pointer_cast(kv_last_page_lens.data()));


//     std::vector<int32_t> batch_indices_host{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     std::vector<int32_t> positions_host{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
//                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
//     thrust::device_vector<int32_t> batch_indices(batch_indices_host);
//     thrust::device_vector<int32_t> positions(positions_host);

//     // populate the kv cache.
//     flashinfer::AppendPagedKVCache<T, int32_t>(
//         paged_kv,
//         thrust::raw_pointer_cast(k_proj.data()), // append_key
//         thrust::raw_pointer_cast(v_proj.data()), // append_value
//         thrust::raw_pointer_cast(batch_indices.data()),
//         thrust::raw_pointer_cast(positions.data()),
//         32,
//         num_kv_heads * head_dim, head_dim,
//         num_kv_heads * head_dim, head_dim);


//     // 4. Compute attention using FlashInfer
//     thrust::device_vector<T> attn_context = q_proj; // Reuse buffer
//     flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
//         thrust::raw_pointer_cast(q_proj.data()),
//         thrust::raw_pointer_cast(qo_indptr.data()),
//         paged_kv,
//         thrust::raw_pointer_cast(attn_context.data()),
//         (uint32_t)num_q_heads, /*causal=*/true, stream);

//     // 5. Output projection
//     attn_output.resize((size_t)batch * hidden_size);
//     gemm_cublasLt<T>(ltHandle, stream, attn_context, o_proj_weights_, nullptr, attn_output, nnz, hidden_size, num_q_heads * head_dim, workspace, false, true);
// }
}

template <typename T>
void L4maDecoderLayer<T>::forward(
    thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<uint32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    const int32_t* kv_page_indices,
    const int32_t* kv_page_indptr,
    const int32_t* kv_last_page_lens,
    const int32_t* qo_indptr,
    thrust::device_vector<T>& temp_buffer,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace,
    flashinfer::BatchPrefillHandler& prefill_handler
) {
    std::cerr << "Warning: L4maDecoderLayer<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maModel<T>::forward(
    thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<uint32_t>& input_ids,
    const thrust::device_vector<uint32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    const int32_t* kv_page_indices,
    const int32_t* kv_page_indptr,
    const int32_t* kv_last_page_lens,
    const int32_t* qo_indptr,
    int batch_size,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace,
    flashinfer::BatchPrefillHandler& prefill_handler
) {
    std::cerr << "Warning: L4maModel<T>::forward is not implemented." << std::endl;
}

template <typename T>
void L4maForCausalLM<T>::forward(thrust::device_vector<float>& logits, const thrust::device_vector<uint32_t>& input_ids, const thrust::device_vector<uint32_t>& position_ids, thrust::device_vector<T>& kv_cache_k, thrust::device_vector<T>& kv_cache_v, const int32_t* kv_page_indices, const int32_t* kv_page_indptr, const int32_t* kv_last_page_lens, const int32_t* qo_indptr, int batch_size, cudaStream_t stream, thrust::device_vector<char>& workspace) {
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