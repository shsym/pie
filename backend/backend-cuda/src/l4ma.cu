#include "l4ma.cuh"
#include "config.hpp"
#include "common.cuh"   // Your helper functions header

#include <stdexcept>
#include <iostream>
#include <utility>
#include <algorithm> // for std::max

#include "flashinfer/norm.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/activation.cuh"
#include "flashinfer_ops.cuh"

// --- Helper CUDA Kernels ---
// These are still needed for operations not covered by your common.cuh or FlashInfer.

template <typename T>
__global__ void add_residual_kernel(T* x, const T* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + residual[idx];
    }
}

__device__ __forceinline__ float silu(const float &val) { return val / (1.0f + __expf(-val)); }

template <typename T>
void silu_and_mul(
    T *out_ptr,
    const T *in_ptr,
    int num_tokens,
    int d_half,
    cudaStream_t stream)
{
    // This function wraps the flashinfer fused kernel for SwiGLU: out = silu(gate) * up
    // The kernel expects a single concatenated input tensor and internally splits it.
    // The 'd_half' parameter is the dimension of the gate/up projection (intermediate_size).
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d_half / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    //attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    // Pass a single input pointer to the underlying flashinfer kernel
    cudaLaunchKernelEx(&config, kernel, out_ptr, in_ptr, d_half);
}

template <typename T>
RMSNorm<T>::RMSNorm(const L4maConfig& config)
    :  weight_(std::make_shared<thrust::device_vector<T>>(config.hidden_size)) {}

template <typename T>
L4maMlp<T>::L4maMlp(const L4maConfig& config)
    : config_(config),
      gate_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * config.intermediate_size)),
      up_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * config.intermediate_size)),
      down_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.intermediate_size * config.hidden_size)) {}

template <typename T>
L4maAttention<T>::L4maAttention(const L4maConfig& config)
    : config_(config),
      q_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_query_heads * config.head_size))),
      k_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      v_proj_weights_(std::make_shared<thrust::device_vector<T>>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      o_proj_weights_(std::make_shared<thrust::device_vector<T>>((config.num_query_heads * config.head_size) * config.hidden_size)) {
    if (config_.use_qkv_bias) {
        q_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_query_heads * config.head_size);
        k_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_key_value_heads * config.head_size);
        v_proj_bias_ = std::make_shared<thrust::device_vector<T>>(config.num_key_value_heads * config.head_size);
    }
}

template <typename T>
L4maDecoderLayer<T>::L4maDecoderLayer(const L4maConfig& config)
    : config_(config),
      self_attn_(config),
      mlp_(config),
      input_layernorm_(config),
      post_attention_layernorm_(config) {}

template <typename T>
L4maModel<T>::L4maModel(const L4maConfig& config)
    : config_(config),
      embed_tokens_weight_(std::make_shared<thrust::device_vector<T>>(config.vocab_size * config.hidden_size)),
      norm_(config) {

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config);
    }
}

template <typename T>
L4maForCausalLM<T>::L4maForCausalLM(const L4maConfig& config)
    : config_(config), model_(config) {
    CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle_));
}


template <typename T>
void L4maForCausalLM<T>::create_kv_device_vectors(int max_kv_num) {
    // Preallocate KV-cache device vectors with the correct size
    size_t kv_cache_size = static_cast<size_t>(max_kv_num) * config_.num_key_value_heads * config_.head_size;

    // Only resize if needed to avoid unnecessary allocations
    if (kv_cache_k_.size() != kv_cache_size) {
        kv_cache_k_.resize(kv_cache_size);
    }
    if (kv_cache_v_.size() != kv_cache_size) {
        kv_cache_v_.resize(kv_cache_size);
    }
    
}

// --- get_parameters() Implementations ---

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> RMSNorm<T>::get_parameters() {
    return {{"weight", weight_}};
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maMlp<T>::get_parameters() {
    return {{"gate_proj.weight", gate_proj_weights_},
            {"up_proj.weight", up_proj_weights_},
            {"down_proj.weight", down_proj_weights_}};
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maAttention<T>::get_parameters() {
    auto params = std::map<std::string, std::shared_ptr<thrust::device_vector<T>>>{
        {"q_proj.weight", q_proj_weights_},
        {"k_proj.weight", k_proj_weights_},
        {"v_proj.weight", v_proj_weights_},
        {"o_proj.weight", o_proj_weights_}};
    if (config_.use_qkv_bias) {
        params["q_proj.bias"] = q_proj_bias_;
        params["k_proj.bias"] = k_proj_bias_;
        params["v_proj.bias"] = v_proj_bias_;
    }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maDecoderLayer<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    for (auto const& [key, val] : self_attn_.get_parameters()) { params["self_attn." + key] = val; }
    for (auto const& [key, val] : mlp_.get_parameters()) { params["mlp." + key] = val; }
    for (auto const& [key, val] : input_layernorm_.get_parameters()) { params["input_layernorm." + key] = val; }
    for (auto const& [key, val] : post_attention_layernorm_.get_parameters()) { params["post_attention_layernorm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>>  L4maModel<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    params["embed_tokens.weight"] = embed_tokens_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        for (auto const& [key, val] : layers_[i].get_parameters()) {
            params["layers." + std::to_string(i) + "." + key] = val;
        }
    }
    for (auto const& [key, val] : norm_.get_parameters()) { params["norm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> L4maForCausalLM<T>::get_parameters() {
    std::map<std::string, std::shared_ptr<thrust::device_vector<T>>> params;
    for (auto const& [key, val] : model_.get_parameters()) {
        params["model." + key] = val;
    }
    params["lm_head.weight"] = lm_head_weight_;
    return params;
}

// --- Forward Pass Stub ---


template <typename T>
void RMSNorm<T>::forward(
    thrust::device_vector<T>& output,
    const thrust::device_vector<T>& input,
    int num_tokens,
    cudaStream_t stream) {
    
    uint32_t batch_size = input.size() / config_.hidden_size;
    uint32_t stride = config_.hidden_size;
    uint32_t d = config_.hidden_size;

    flashinfer::norm::RMSNorm(
        const_cast<T *>(thrust::raw_pointer_cast(input.data())),
        const_cast<T *>(thrust::raw_pointer_cast(weight_->data())),
        thrust::raw_pointer_cast(output.data()),
        batch_size, d, stride, stride, config_.rms_norm_eps, false, stream
    );
}

template <typename T>
void L4maMlp<T>::forward(
    thrust::device_vector<T>& output, 
    const thrust::device_vector<T>& x, 
    int num_tokens, 
    thrust::device_vector<T>& temp_buffer, 
    cublasLtHandle_t ltHandle, 
    cudaStream_t stream, 
    thrust::device_vector<char>& workspace_buffer_float
) {
    const int hidden_size = config_.hidden_size;
    const int intermediate_size = config_.intermediate_size;

    // Partition the temp buffer
    thrust::device_vector<T> gate_proj_out(thrust::device_pointer_cast(temp_buffer.data()), thrust::device_pointer_cast(temp_buffer.data() + (size_t)num_tokens * intermediate_size));
    thrust::device_vector<T> up_proj_out(thrust::device_pointer_cast(gate_proj_out.data().get() + (size_t)num_tokens * intermediate_size), thrust::device_pointer_cast(gate_proj_out.data().get() + 2 * (size_t)num_tokens * intermediate_size));

    // 1. Gate projection: gate_proj_out = x @ W_gate^T
    gemm_cublasLt<T>(ltHandle, stream, x, *gate_proj_weights_, nullptr, gate_proj_out, num_tokens, intermediate_size, hidden_size, workspace_buffer_float, false, true);
    
    // 2. Up projection: up_proj_out = x @ W_up^T
    gemm_cublasLt<T>(ltHandle, stream, x, *up_proj_weights_, nullptr, up_proj_out, num_tokens, intermediate_size, hidden_size, workspace_buffer_float, false, true);

    // 3. SwiGLU activation: silu(gate_proj) * up_proj (in-place into gate_proj_out)
    silu_and_mul<T>(
        thrust::raw_pointer_cast(temp_buffer.data()),
        const_cast<T *>(thrust::raw_pointer_cast(temp_buffer.data())),
        num_tokens,
        intermediate_size,
        stream
    );

    // 4. Down projection: output = (activated_output) @ W_down^T
    gemm_cublasLt<T>(ltHandle, stream, gate_proj_out, *down_proj_weights_, nullptr, output, num_tokens, hidden_size, intermediate_size, workspace_buffer_float, false, true);
}

template <typename T>
void L4maAttention<T>::forward(
    thrust::device_vector<T>& attn_output,
    const thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    thrust::device_vector<T>& temp_buffer,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {

    const int batch_size = hidden_states.size() / config_.hidden_size;
    const int hidden_size = config_.hidden_size;
    const int head_size = config_.head_size;
    const int num_query_heads = config_.num_query_heads;
    const int num_key_value_heads = config_.num_key_value_heads;
    
    size_t q_size = (size_t)batch_size * num_query_heads * head_size;
    size_t k_size = (size_t)batch_size * num_key_value_heads * head_size;
    size_t v_size = (size_t)batch_size * num_key_value_heads * head_size;

    if(temp_buffer.size() < q_size + k_size + v_size) {
        throw std::runtime_error("Temporary buffer size is too small for Q, K, V projections.");
    }

    // Partition buffer
    thrust::device_vector<T> q_proj(thrust::device_pointer_cast(temp_buffer.data()), thrust::device_pointer_cast(temp_buffer.data() + q_size));
    thrust::device_vector<T> k_proj(thrust::device_pointer_cast(q_proj.data().get() + q_size), thrust::device_pointer_cast(q_proj.data().get() + q_size + k_size));
    thrust::device_vector<T> v_proj(thrust::device_pointer_cast(k_proj.data().get() + k_size), thrust::device_pointer_cast(k_proj.data().get() + k_size + v_size));
    
    // 1. Q, K, V projections
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, *q_proj_weights_, config_.use_qkv_bias ? q_proj_bias_.get() : nullptr, q_proj, batch_size, num_query_heads * head_size, hidden_size, workspace_buffer_float, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, *k_proj_weights_, config_.use_qkv_bias ? k_proj_bias_.get() : nullptr, k_proj, batch_size, num_key_value_heads * head_size, hidden_size, workspace_buffer_float, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, *v_proj_weights_, config_.use_qkv_bias ? v_proj_bias_.get() : nullptr, v_proj, batch_size, num_key_value_heads * head_size, hidden_size, workspace_buffer_float, false, true);

    // 2. Apply RoPE (in-place)
    flashinfer::BatchQKApplyLlama31RotaryPosIds(
        const_cast<T *>(thrust::raw_pointer_cast(q_proj.data())), // q
        const_cast<T *>(thrust::raw_pointer_cast(k_proj.data())), // k
        thrust::raw_pointer_cast(q_proj.data()),                  // q_rope (not available)
        thrust::raw_pointer_cast(k_proj.data()),                  // k_rope (not available)
        thrust::raw_pointer_cast(position_ids.data()),                 // pos_ids (uint32_t*)
        batch_size,                                                    // nnz (assuming batch size for now)
        num_query_heads,                                                       // num_qo_heads
        num_key_value_heads,                                                      // num_kv_heads
        head_size,                                       // rotary_dim
        head_size,                                       // head_dim
        num_query_heads * head_size,                                                  // q_stride_n
        head_size,                                       // q_stride_h
        num_key_value_heads * head_size,                                                 // k_stride_n
        head_size,
        ///----                                                      // k_stride_h
        // q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h (not available)
        num_query_heads * head_size,
        head_size,
        num_key_value_heads * head_size,
        head_size,
        ///----                                                      
        false, // interleave
        config_.rope_factor,  // rope_scale
        config_.rope_theta,  // rope_theta
        config_.rope_low_frequency_factor,  // low_freq_factor
        config_.rope_high_frequency_factor,  // high_freq_factor
        8192,  // old_context_length
        stream // cudaStream_t
    );


    // 3. Create paged KV-cache object
    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_key_value_heads, page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        thrust::raw_pointer_cast(kv_cache_k.data()),
        thrust::raw_pointer_cast(kv_cache_v.data()),
        thrust::raw_pointer_cast(kv_page_indices.data()), 
        thrust::raw_pointer_cast(kv_page_indptr.data()), 
        thrust::raw_pointer_cast(kv_last_page_lens.data())
    );


    flashinfer::AppendPagedKVCache<T, int32_t>(
        paged_kv,
        thrust::raw_pointer_cast(k_proj.data()), // append_key
        thrust::raw_pointer_cast(v_proj.data()), // append_value
        thrust::raw_pointer_cast(kv_batch_indices.data()),
        thrust::raw_pointer_cast(kv_positions.data()),
        kv_batch_indices.size(),
        num_key_value_heads * head_size, head_size,
        num_key_value_heads * head_size, head_size);


    thrust::device_vector<T> o_proj = q_proj; // Reuse buffer
    cudaError_t status = flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
        &prefill_handler,
        thrust::raw_pointer_cast(q_proj.data()),
        thrust::raw_pointer_cast(qo_indptr.data()),
        /*q_rope_offset=*/nullptr,
        paged_kv,
        thrust::raw_pointer_cast(o_proj.data()),
        /*lse=*/nullptr, 
        num_query_heads,
        flashinfer::MaskMode::kCustom,
        thrust::raw_pointer_cast(custom_mask.data()),
        thrust::raw_pointer_cast(mask_indptr.data()),
        flashinfer::PosEncodingMode::kNone);

    gemm_cublasLt<T>(ltHandle, stream, o_proj, *o_proj_weights_, nullptr, attn_output, batch_size, hidden_size, num_query_heads * head_size, workspace_buffer_float, false, true);

}

template <typename T>
void L4maDecoderLayer<T>::forward(
    thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    thrust::device_vector<T>& temp_buffer,
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = hidden_states.size() / config_.hidden_size;

    // --- 1. Self-Attention Block ---
    
    // Store residual for the first connection
    thrust::device_vector<T> residual = hidden_states;

    // Apply input layer normalization
    thrust::device_vector<T> normed_input(hidden_states.size());
    input_layernorm_.forward(normed_input, hidden_states, num_tokens, stream);

    // Perform attention
    thrust::device_vector<T> attn_output(hidden_states.size());
    self_attn_.forward(attn_output, normed_input, position_ids, kv_cache_k, kv_cache_v, 
                       kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr, custom_mask, mask_indptr, temp_buffer, 
                       ltHandle, stream, workspace_buffer_float, prefill_handler, page_size,
                       kv_batch_indices, kv_positions);
    
    // First residual connection: hidden_states = residual + attn_output
    add_residual_kernel<<<(hidden_states.size() + 255) / 256, 256, 0, stream>>>(
        thrust::raw_pointer_cast(hidden_states.data()), 
        thrust::raw_pointer_cast(attn_output.data()), 
        hidden_states.size());

    // --- 2. MLP Block ---

    // Store residual for the second connection
    residual = hidden_states;

    // Apply post-attention layer normalization
    post_attention_layernorm_.forward(normed_input, hidden_states, num_tokens, stream); // Re-use normed_input buffer

    // Perform MLP
    thrust::device_vector<T> mlp_output(hidden_states.size());
    mlp_.forward(mlp_output, normed_input, num_tokens, temp_buffer, ltHandle, stream, workspace_buffer_float);

    // Second residual connection: hidden_states = residual + mlp_output
    add_residual_kernel<<<(hidden_states.size() + 255) / 256, 256, 0, stream>>>(
        thrust::raw_pointer_cast(hidden_states.data()), 
        thrust::raw_pointer_cast(mlp_output.data()), 
        hidden_states.size());
}

template <typename T>
void L4maModel<T>::forward(
    thrust::device_vector<T>& hidden_states,
    const thrust::device_vector<uint32_t>& input_ids,
    const thrust::device_vector<int32_t>& position_ids,
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
    thrust::device_vector<char>& workspace_buffer_float,
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = input_ids.size();
    
    // 1. Token Embeddings
    embed<T>(*embed_tokens_weight_, input_ids, &hidden_states, config_.hidden_size, stream);

    size_t temp_buffer_size = 2 * (size_t)num_tokens * config_.intermediate_size;
    thrust::device_vector<T> temp_buffer(temp_buffer_size);

    for (auto& layer : layers_) {
        layer.forward(hidden_states, position_ids, kv_cache_k, kv_cache_v,
                      kv_page_indices, kv_page_indptr, kv_last_page_lens,
                      qo_indptr, custom_mask, mask_indptr, temp_buffer, ltHandle, stream, workspace_buffer_float,
                      prefill_handler, page_size, kv_batch_indices, kv_positions);
    }

    thrust::device_vector<T> final_norm_input = hidden_states;
    norm_.forward(hidden_states, final_norm_input, num_tokens, stream);
}

template <typename T>
void L4maForCausalLM<T>::forward(
    thrust::device_vector<T>& logits, 
    const thrust::device_vector<uint32_t>& input_ids,
    const thrust::device_vector<int32_t>& position_ids,
    thrust::device_vector<int32_t>& kv_page_indices,
    thrust::device_vector<int32_t>& kv_page_indptr,
    std::vector<int32_t>& kv_page_indptr_host,
    thrust::device_vector<int32_t>& kv_last_page_lens,
    thrust::device_vector<int32_t>& qo_indptr,
    std::vector<int32_t>& qo_indptr_host,
    thrust::device_vector<uint8_t>& custom_mask,
    thrust::device_vector<int32_t>& mask_indptr,
    cudaStream_t stream,
    thrust::device_vector<char>& workspace_buffer_float,
    thrust::device_vector<char>& workspace_buffer_int,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
    ) {
    
    const int head_size = config_.head_size;
    const int num_query_heads = config_.num_query_heads;
    const int num_key_value_heads = config_.num_key_value_heads;
    const int num_tokens = input_ids.size();
    flashinfer::BatchPrefillHandler handler;

    handler.Plan<T, int32_t>(
        (void *)thrust::raw_pointer_cast(workspace_buffer_float.data()), workspace_buffer_float.size(),
        (void *)thrust::raw_pointer_cast(workspace_buffer_int.data()), workspace_buffer_int.size(),
        qo_indptr_host.data(), 
        kv_page_indptr_host.data(),
         /*total_num_rows=*/num_tokens,
         /*batch=*/qo_indptr_host.size() - 1,
        num_query_heads,
        num_key_value_heads,
        head_size,
        page_size);


   // --- Start of Forward Pass Implementation ---

    // 1. Run the core model forward pass
    // This vector will hold the final hidden states from the last decoder layer.
    thrust::device_vector<T> hidden_states(num_tokens * config_.hidden_size);

    // Execute the forward pass of the base L4maModel
    model_.forward(
        hidden_states,
        input_ids,
        position_ids,
        kv_cache_k_, // Pass the pre-allocated KV cache
        kv_cache_v_, // Pass the pre-allocated KV cache
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        custom_mask,
        mask_indptr,
        cublaslt_handle_,
        stream,
        workspace_buffer_float,
        handler, // The planned prefill handler
        page_size,
        kv_batch_indices,
        kv_positions
    );

    // 2. Compute logits using the LM head
    // The output logits are computed by multiplying the final hidden states
    // with the language modeling head weights (which are tied to the token embeddings).
    // logits = hidden_states @ lm_head_weight^T
    gemm_cublasLt<T>(
        cublaslt_handle_, // Use the cuBLAS handle from the model
        stream,
        hidden_states,           // Input matrix A (m x k)
        *lm_head_weight_,         // Input matrix B (n x k)
        nullptr,                 // Bias (not used)
        logits,                  // Output matrix C (m x n)
        num_tokens,              // m: number of tokens
        config_.vocab_size,      // n: vocabulary size
        config_.hidden_size,     // k: hidden dimension
        workspace_buffer_float,  // Workspace for cuBLAS
        false,                   // transa: Don't transpose hidden_states
        true                     // transb: Transpose the lm_head_weight matrix
    );


}

// --- Explicit Template Instantiations ---
// template class RMSNorm<float>;
// template class L4maMlp<float>;
// template class L4maAttention<float>;
// template class L4maDecoderLayer<float>;
// template class L4maModel<float>;
// template class L4maForCausalLM<float>;

template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;