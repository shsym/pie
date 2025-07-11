#include "l4ma.cuh"
#include "config.hpp"
#include "common.cuh"   // Your helper functions header
#include "stack_allocator.cuh" // Import the new stack allocator

#include <stdexcept>
#include <iostream>
#include <utility>
#include <algorithm> // for std::max

#include "flashinfer/norm.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/activation.cuh"
#include "flashinfer_ops.cuh"

// --- Helper CUDA Kernels (Unchanged) ---
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
    uint32_t vec_size = 16 / sizeof(T);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d_half / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<T, silu>;
    cudaLaunchKernelEx(&config, kernel, out_ptr, in_ptr, d_half);
}

// --- Constructor Implementations (Unchanged) ---
template <typename T>
RMSNorm<T>::RMSNorm(const L4maConfig& config)
    : config_(config), weight_(std::make_shared<thrust::device_vector<T>>(config.hidden_size)) {}

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

// --- KV Cache and Workspace Management (REFACTORED) ---

template <typename T>
void L4maForCausalLM<T>::create_kv_device_vectors(int max_kv_num) {
    size_t kv_cache_size = static_cast<size_t>(max_kv_num) * config_.num_key_value_heads * config_.head_size;
    if (kv_cache_k_.size() != kv_cache_size) {
        kv_cache_k_.resize(kv_cache_size);
    }
    if (kv_cache_v_.size() != kv_cache_size) {
        kv_cache_v_.resize(kv_cache_size);
    }
}

template <typename T>
size_t L4maForCausalLM<T>::get_workspace_size(int max_num_tokens) const {
    const size_t alignment = 256;
    const size_t hidden_size = config_.hidden_size;
    const size_t intermediate_size = config_.intermediate_size;
    const size_t num_q_heads = config_.num_query_heads;
    const size_t num_kv_heads = config_.num_key_value_heads;
    const size_t head_size = config_.head_size;

    size_t peak_bytes = 0;

    // --- Trace allocations in L4maForCausalLM::forward ---
    size_t offset0 = 0;
    size_t hidden_states_offset = align_up(offset0, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);
    size_t flash_float_offset = align_up(hidden_states_offset, alignment) + 256 * 1024 * 1024;
    size_t flash_int_offset = align_up(flash_float_offset, alignment) + 8 * 1024 * 1024;

    // --- Trace into L4maModel::forward ---
    size_t model_working_hidden_offset = align_up(flash_int_offset, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);

    // --- Trace into L4maDecoderLayer::forward ---
    // The decoder layer reuses space, so we find the peak within one layer invocation.
    
    // Path 1: Attention block
    size_t attn_norm_offset = align_up(model_working_hidden_offset, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);
    size_t attn_out_offset = align_up(attn_norm_offset, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);
    // After attention, these two buffers are deallocated, rewinding the offset to model_working_hidden_offset.
    // Now trace the allocations *inside* the attention block forward call.
    size_t q_proj_offset = align_up(attn_out_offset, alignment) + (size_t)max_num_tokens * num_q_heads * head_size * sizeof(T);
    size_t k_proj_offset = align_up(q_proj_offset, alignment) + (size_t)max_num_tokens * num_kv_heads * head_size * sizeof(T);
    size_t v_proj_offset = align_up(k_proj_offset, alignment) + (size_t)max_num_tokens * num_kv_heads * head_size * sizeof(T);
    size_t attn_peak = v_proj_offset; // Does not include cublas workspace, as allocate_rest() is used.

    // Path 2: MLP block (starts from the same point)
    size_t mlp_norm_offset = align_up(model_working_hidden_offset, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);
    size_t mlp_out_offset = align_up(mlp_norm_offset, alignment) + (size_t)max_num_tokens * hidden_size * sizeof(T);
    // Trace inside MLP forward call
    size_t gate_proj_offset = align_up(mlp_out_offset, alignment) + (size_t)max_num_tokens * intermediate_size * sizeof(T);
    size_t up_proj_offset = align_up(gate_proj_offset, alignment) + (size_t)max_num_tokens * intermediate_size * sizeof(T);
    size_t mlp_peak = up_proj_offset;

    // The peak usage is the highest offset reached during the entire forward pass.
    peak_bytes = std::max(attn_peak, mlp_peak);

    // Finally, account for the lm_head GEMM, which uses allocate_rest(). To be safe,
    // we add a fixed large buffer for any `allocate_rest` calls.
    const size_t rest_buffer = 32 * 1024 * 1024; // 32MB buffer for all cublas workspaces (https://docs.nvidia.com/cuda/cublas/#cublassetworkspace)
    
    return peak_bytes + rest_buffer;
}


// --- get_parameters() Implementations (Unchanged) ---
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


template <typename T>
void RMSNorm<T>::forward(
    T* output,
    const T* input,
    int num_tokens,
    cudaStream_t stream) {
    
    uint32_t d = config_.hidden_size;

    flashinfer::norm::RMSNorm(
        const_cast<T *>(input),
        const_cast<T *>(thrust::raw_pointer_cast(weight_->data())),
        output,
        num_tokens, d, d, d, config_.rms_norm_eps, false, stream
    );
}

template <typename T>
void L4maMlp<T>::forward(
    StackAllocator& allocator,
    T* output, 
    const T* x, 
    int num_tokens, 
    cublasLtHandle_t ltHandle, 
    cudaStream_t stream
) {
    const int hidden_size = config_.hidden_size;
    const int intermediate_size = config_.intermediate_size;
    const size_t proj_count = (size_t)num_tokens * intermediate_size;

    // 1. Allocate buffers from the stack allocator
    T* gate_proj_out_ptr = allocator.template allocate<T>(proj_count);
    T* up_proj_out_ptr = allocator.template allocate<T>(proj_count);
    
    // Use allocate_rest for the cublas workspace as requested
    size_t cublas_workspace_size = 32 * 1024 * 1024; // 32MB workspace size for cublasLt
    void* cublas_workspace_ptr = allocator.allocate_bytes(cublas_workspace_size); // Allocate aligned workspace

    // 2. Gate and Up projections
    gemm_cublasLt<T>(ltHandle, stream, x, thrust::raw_pointer_cast(gate_proj_weights_->data()), nullptr, gate_proj_out_ptr, num_tokens, intermediate_size, hidden_size, cublas_workspace_ptr, cublas_workspace_size, false, true);
    gemm_cublasLt<T>(ltHandle, stream, x, thrust::raw_pointer_cast(up_proj_weights_->data()), nullptr, up_proj_out_ptr, num_tokens, intermediate_size, hidden_size, cublas_workspace_ptr, cublas_workspace_size, false, true);

    // 3. SwiGLU activation (gate * silu(up))
    // We can reuse the gate_proj_out_ptr buffer for the output of SwiGLU
    silu_and_mul<T>(gate_proj_out_ptr, up_proj_out_ptr, num_tokens, intermediate_size, stream);

    // 4. Down projection
    gemm_cublasLt<T>(ltHandle, stream, gate_proj_out_ptr, thrust::raw_pointer_cast(down_proj_weights_->data()), nullptr, output, num_tokens, hidden_size, intermediate_size, cublas_workspace_ptr, cublas_workspace_size, false, true);

    // 5. Deallocate buffers in reverse order of allocation (LIFO)
    allocator.deallocate_bytes(cublas_workspace_ptr, cublas_workspace_size);
    allocator.template deallocate<T>(up_proj_out_ptr, proj_count);
    allocator.template deallocate<T>(gate_proj_out_ptr, proj_count);
}

template <typename T>
void L4maAttention<T>::forward(
    StackAllocator& allocator,
    T* attn_output,
    const T* hidden_states,
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
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = position_ids.size();
    const int hidden_size = config_.hidden_size;
    const int head_size = config_.head_size;
    const int num_query_heads = config_.num_query_heads;
    const int num_key_value_heads = config_.num_key_value_heads;
    
    const size_t q_proj_count = (size_t)num_tokens * num_query_heads * head_size;
    const size_t kv_proj_count = (size_t)num_tokens * num_key_value_heads * head_size;
    
    // 1. Allocate buffers from the stack allocator
    T* q_proj_ptr = allocator.template allocate<T>(q_proj_count);
    T* k_proj_ptr = allocator.template allocate<T>(kv_proj_count);
    T* v_proj_ptr = allocator.template allocate<T>(kv_proj_count);
    size_t cublas_workspace_size = 32 * 1024 * 1024; // 32MB workspace size for cublasLt
    void* cublas_workspace_ptr = allocator.allocate_bytes(cublas_workspace_size); // Allocate aligned workspace

    // 2. Q, K, V projections
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(q_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(q_proj_bias_->data()) : nullptr, q_proj_ptr, num_tokens, num_query_heads * head_size, hidden_size, cublas_workspace_ptr, cublas_workspace_size, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(k_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(k_proj_bias_->data()) : nullptr, k_proj_ptr, num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace_ptr, cublas_workspace_size, false, true);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, thrust::raw_pointer_cast(v_proj_weights_->data()), config_.use_qkv_bias ? thrust::raw_pointer_cast(v_proj_bias_->data()) : nullptr, v_proj_ptr, num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace_ptr, cublas_workspace_size, false, true);

    // 3. Apply RoPE (in-place)
    flashinfer::BatchQKApplyLlama31RotaryPosIds(
        q_proj_ptr, k_proj_ptr, q_proj_ptr, k_proj_ptr,
        thrust::raw_pointer_cast(position_ids.data()),
        num_tokens, num_query_heads, num_key_value_heads, head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        false, config_.rope_factor, config_.rope_theta, config_.rope_low_frequency_factor,
        config_.rope_high_frequency_factor, 8192, stream
    );

    // 4. Paged KV-cache operations
    const int batch_size = (position_ids.size() > 0) ? (qo_indptr.size() - 1) : 0;
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
        paged_kv, k_proj_ptr, v_proj_ptr,
        thrust::raw_pointer_cast(kv_batch_indices.data()),
        thrust::raw_pointer_cast(kv_positions.data()),
        kv_batch_indices.size(),
        num_key_value_heads * head_size, head_size,
        num_key_value_heads * head_size, head_size);

    // Reuse a buffer for the attention output before the final projection
    T* o_proj_input_ptr = k_proj_ptr; 
    flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
        &prefill_handler, q_proj_ptr, thrust::raw_pointer_cast(qo_indptr.data()),
        nullptr, paged_kv, o_proj_input_ptr, nullptr, num_query_heads,
        flashinfer::MaskMode::kCustom,
        thrust::raw_pointer_cast(custom_mask.data()),
        thrust::raw_pointer_cast(mask_indptr.data()),
        flashinfer::PosEncodingMode::kNone);
    
    // 5. Final output projection
    gemm_cublasLt<T>(ltHandle, stream, o_proj_input_ptr, thrust::raw_pointer_cast(o_proj_weights_->data()), nullptr, attn_output, num_tokens, hidden_size, num_query_heads * head_size, cublas_workspace_ptr, cublas_workspace_size, false, true);

    // 6. Deallocate buffers in reverse order
    allocator.deallocate_bytes(cublas_workspace_ptr, cublas_workspace_size);
    allocator.template deallocate<T>(v_proj_ptr, kv_proj_count);
    allocator.template deallocate<T>(k_proj_ptr, kv_proj_count);
    allocator.template deallocate<T>(q_proj_ptr, q_proj_count);
}

template <typename T>
void L4maDecoderLayer<T>::forward(
    StackAllocator& allocator,
    T* hidden_states,
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
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = position_ids.size();
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;

    // --- 1. Self-Attention Block ---
    // The input `hidden_states` serves as the first residual.
    T* normed_input_ptr = allocator.template allocate<T>(hidden_size_elements);
    input_layernorm_.forward(normed_input_ptr, hidden_states, num_tokens, stream);

    T* attn_output_ptr = allocator.template allocate<T>(hidden_size_elements);
    self_attn_.forward(allocator, attn_output_ptr, 
                       normed_input_ptr, position_ids, kv_cache_k, kv_cache_v, 
                       kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr, custom_mask, mask_indptr, 
                       ltHandle, stream, prefill_handler, page_size,
                       kv_batch_indices, kv_positions);

    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, attn_output_ptr, hidden_size_elements);
    
    // Deallocate attn_output and then normed_input to free up space for the MLP block
    allocator.template deallocate<T>(attn_output_ptr, hidden_size_elements);
    allocator.template deallocate<T>(normed_input_ptr, hidden_size_elements);


    // --- 2. MLP Block ---
    // The result of the attention block, `hidden_states`, is the residual for the MLP block.
    T* normed_mlp_input_ptr = allocator.template allocate<T>(hidden_size_elements);
    post_attention_layernorm_.forward(normed_mlp_input_ptr, hidden_states, num_tokens, stream);

    T* mlp_output_ptr = allocator.template allocate<T>(hidden_size_elements);
    mlp_.forward(allocator, mlp_output_ptr, normed_mlp_input_ptr, num_tokens, ltHandle, stream);
    
    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, mlp_output_ptr, hidden_size_elements);
        
    // Deallocate MLP buffers
    allocator.template deallocate<T>(mlp_output_ptr, hidden_size_elements);
    allocator.template deallocate<T>(normed_mlp_input_ptr, hidden_size_elements);
}

template <typename T>
void L4maModel<T>::forward(
    StackAllocator& allocator,
    T* final_norm_output,
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
    flashinfer::BatchPrefillHandler& prefill_handler,
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
) {
    const int num_tokens = input_ids.size();
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;
    
    // Allocate a working buffer for the layers. The layers will operate in-place on this buffer.
    T* working_hidden_buffer = allocator.template allocate<T>(hidden_size_elements);

    embed<T>(
        thrust::raw_pointer_cast(embed_tokens_weight_->data()),
        embed_tokens_weight_->size() / config_.hidden_size,
        thrust::raw_pointer_cast(input_ids.data()),
        input_ids.size(),
        working_hidden_buffer, // Embeddings are written to the allocated working buffer
        config_.hidden_size, 
        stream
    );

    for (auto& layer : layers_) {
        // Pass the allocator down to the layer. The layer will use it for its own scratch space.
        layer.forward(allocator, working_hidden_buffer,
                      position_ids, kv_cache_k, kv_cache_v,
                      kv_page_indices, kv_page_indptr, kv_last_page_lens,
                      qo_indptr, custom_mask, mask_indptr, ltHandle, stream,
                      prefill_handler, page_size, kv_batch_indices, kv_positions);
    }

    // Final norm reads from the working buffer and writes to the final output buffer.
    norm_.forward(final_norm_output, working_hidden_buffer, num_tokens, stream);
    
    // Deallocate the working buffer.
    allocator.template deallocate<T>(working_hidden_buffer, hidden_size_elements);
}

template <typename T>
void L4maForCausalLM<T>::forward(
    StackAllocator& allocator,
    T* output,
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
    const int32_t page_size,
    thrust::device_vector<int32_t>& kv_batch_indices,
    thrust::device_vector<int32_t>& kv_positions
    ) {
    
    const int num_tokens = input_ids.size();
    
    // Allocate the buffer for the final hidden states (output of the model, input to lm_head)
    T* hidden_states_ptr = allocator.template allocate<T>((size_t)num_tokens * config_.hidden_size);
    
    // Allocate fixed-size buffers for FlashInfer
    // Check the appendix D2 of https://arxiv.org/pdf/2501.01005
    void* flashinfer_float_buffer = allocator.allocate_bytes(256 * 1024 * 1024);
    void* flashinfer_int_buffer = allocator.allocate_bytes(8 * 1024 * 1024);

    flashinfer::BatchPrefillHandler handler;
    handler.Plan<T, int32_t>(
        flashinfer_float_buffer, 256 * 1024 * 1024,
        flashinfer_int_buffer, 8 * 1024 * 1024,
        qo_indptr_host.data(), 
        kv_page_indptr_host.data(),
        num_tokens,
        qo_indptr_host.size() - 1,
        config_.num_query_heads,
        config_.num_key_value_heads,
        config_.head_size,
        page_size);

    model_.forward(
        allocator, // Pass the allocator down to the model.
        hidden_states_ptr,
        input_ids, position_ids,
        kv_cache_k_, kv_cache_v_,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        qo_indptr, custom_mask, mask_indptr,
        cublaslt_handle_, stream, handler, page_size,
        kv_batch_indices, kv_positions
    );

    size_t lm_head_workspace_size = 32 * 1024 * 1024; // 32MB workspace size for cublasLt
    void* lm_head_workspace_ptr = allocator.allocate_bytes(lm_head_workspace_size); // Allocate aligned workspace
    
    gemm_cublasLt<T>(
        cublaslt_handle_, stream,
        hidden_states_ptr,
        thrust::raw_pointer_cast(lm_head_weight_->data()),
        nullptr,
        output,
        num_tokens, config_.vocab_size, config_.hidden_size,
        lm_head_workspace_ptr, lm_head_workspace_size, false, true
    );
    
    // Deallocations will happen automatically as the allocator goes out of scope.
    // However, to be explicit and use the implemented checks:
    allocator.deallocate_bytes(lm_head_workspace_ptr, lm_head_workspace_size);
    allocator.deallocate_bytes(flashinfer_int_buffer, 8 * 1024 * 1024);
    allocator.deallocate_bytes(flashinfer_float_buffer, 256 * 1024 * 1024);
    allocator.template deallocate<T>(hidden_states_ptr, (size_t)num_tokens * config_.hidden_size);
}

// --- Explicit Template Instantiations (Unchanged) ---
template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;