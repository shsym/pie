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
#include "flashinfer/vec_dtypes.cuh"

#include "flashinfer_ops.cuh"




// --- Helper CUDA Kernels (Unchanged) ---
template <typename T>
__global__ void add_residual_kernel(T* x, const T* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + residual[idx];
    }
}


__device__ __forceinline__ float silu(const float &val) {
    return val / (1.0f + __expf(-val));
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const int d
) {
    // Each thread processes one element at a time, with vectorization
    constexpr uint32_t vec_size = 16 / sizeof(T);

    // The block index corresponds to the token index in the batch
    const int64_t token_idx = blockIdx.x;
    // The thread index within the block
    const int64_t thread_idx = threadIdx.x;
    // The total number of threads in the block
    const int64_t stride = blockDim.x;

    // Calculate the base offset for the current token for each input
    const int64_t token_offset = token_idx * d;

    // Main loop for vectorized processing
    // Each thread processes multiple elements in a strided pattern
    #pragma unroll 1
    for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
        flashinfer::vec_t<float, vec_size> x_vec, y_vec, out_vec;

        // Load data from the two separate input tensors
        x_vec.cast_load(input1 + token_offset + idx * vec_size);
        y_vec.cast_load(input2 + token_offset + idx * vec_size);

        // Apply activation to the first vector and multiply element-wise
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            out_vec[i] = Activation(x_vec[i]) * y_vec[i];
        }

        // Store the result back to the output tensor
        out_vec.cast_store(out + token_offset + idx * vec_size);
    }

    // Handle remaining elements that don't fit into a full vector
    const int64_t remaining_offset = (d / vec_size) * vec_size;
    #pragma unroll 1
    for (int64_t idx = thread_idx + remaining_offset; idx < d; idx += stride) {
        // Load single elements from each input tensor
        float x = static_cast<float>(__ldg(input1 + token_offset + idx));
        float y = static_cast<float>(__ldg(input2 + token_offset + idx));
        // Compute and store the result
        out[token_offset + idx] = static_cast<T>(Activation(x) * y);
    }
}



// Host-side function to launch the silu_and_mul kernel
template <typename T>
void silu_and_mul(
    T* out_ptr,
    const T* in1_ptr, // Pointer to the first input tensor
    const T* in2_ptr, // Pointer to the second input tensor
    int num_tokens,   // Batch size or number of tokens
    int d,            // The hidden dimension size
    cudaStream_t stream
) {
    // Vector size depends on the data type (e.g., 8 for half, 4 for float)
    constexpr uint32_t vec_size = 16 / sizeof(T);
    
    // Each block processes one token
    dim3 grid_dim(num_tokens);
    // Number of threads per block, capped at 256 for efficiency
    // We choose a block size that is a power of two and related to the problem size
    uint32_t block_dim = std::min(static_cast<uint32_t>(d / vec_size), 256U);
    if (block_dim == 0) { // Ensure at least one thread block if d is small
        block_dim = std::min(static_cast<uint32_t>(d), 256U);
    }

    // Launch the kernel with a standard execution configuration
    act_and_mul_kernel<T, silu><<<grid_dim, block_dim, 0, stream>>>(
        out_ptr, in1_ptr, in2_ptr, d
    );
}


// --- Constructor Implementations (Unchanged) ---
template <typename T>
RMSNorm<T>::RMSNorm(const L4maConfig& config)
    : config_(config), weight_(Tensor<T>(config.hidden_size)) {}

template <typename T>
L4maMlp<T>::L4maMlp(const L4maConfig& config)
    : config_(config),
      gate_proj_weights_(Tensor<T>(config.hidden_size * config.intermediate_size)),
      up_proj_weights_(Tensor<T>(config.hidden_size * config.intermediate_size)),
      down_proj_weights_(Tensor<T>(config.intermediate_size * config.hidden_size)) {}

template <typename T>
L4maAttention<T>::L4maAttention(const L4maConfig& config)
    : config_(config),
      q_proj_weights_(Tensor<T>(config.hidden_size * (config.num_query_heads * config.head_size))),
      k_proj_weights_(Tensor<T>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      v_proj_weights_(Tensor<T>(config.hidden_size * (config.num_key_value_heads * config.head_size))),
      o_proj_weights_(Tensor<T>((config.num_query_heads * config.head_size) * config.hidden_size)) {
    // if (config_.use_qkv_bias) {
    //     q_proj_bias_ = Tensor<T>(config.num_query_heads * config.head_size);
    //     k_proj_bias_ = Tensor<T>(config.num_key_value_heads * config.head_size);
    //     v_proj_bias_ = Tensor<T>(config.num_key_value_heads * config.head_size);
    // }
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
      embed_tokens_weight_(Tensor<T>(config.vocab_size * config.hidden_size)),
      norm_(config) {

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config);
    }
}

template <typename T>
L4maForCausalLM<T>::L4maForCausalLM(const L4maConfig& config)
    : config_(config), 
      model_(config) {
    CUBLAS_CHECK(cublasLtCreate(&cublaslt_handle_));
}

// --- KV Cache and Workspace Management (REFACTORED) ---

template <typename T>
void L4maForCausalLM<T>::create_kv_device_vectors(int max_kv_num) {
    size_t kv_cache_size = static_cast<size_t>(max_kv_num) * config_.num_key_value_heads * config_.head_size * config_.num_layers;
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


// --- get_parameters() Implementations (Corrected) ---
template <typename T>
std::map<std::string, Tensor<T>*> RMSNorm<T>::get_parameters() {
    // Return a pointer to the weight tensor
    return {{"weight", &weight_}};
}

template <typename T>
std::map<std::string, Tensor<T>*> L4maMlp<T>::get_parameters() {
    // Return pointers to the weight tensors
    return {{"gate_proj.weight", &gate_proj_weights_},
            {"up_proj.weight", &up_proj_weights_},
            {"down_proj.weight", &down_proj_weights_}};
}

template <typename T>
std::map<std::string, Tensor<T>*> L4maAttention<T>::get_parameters() {
    // Initialize the map with pointers
    auto params = std::map<std::string, Tensor<T>*>{
        {"q_proj.weight", &q_proj_weights_},
        {"k_proj.weight", &k_proj_weights_},
        {"v_proj.weight", &v_proj_weights_},
        {"o_proj.weight", &o_proj_weights_}};
    // Bias handling (if you re-enable it)
    // if (config_.use_qkv_bias) {
    //     params["q_proj.bias"] = &q_proj_bias_;
    //     params["k_proj.bias"] = &k_proj_bias_;
    //     params["v_proj.bias"] = &v_proj_bias_;
    // }
    return params;
}

template <typename T>
std::map<std::string, Tensor<T>*> L4maDecoderLayer<T>::get_parameters() {
    // The map now correctly stores pointers
    std::map<std::string, Tensor<T>*> params;
    // The 'val' from the sub-calls is now a Tensor<T>*, which can be assigned directly.
    for (auto const& [key, val] : self_attn_.get_parameters()) { params["self_attn." + key] = val; }
    for (auto const& [key, val] : mlp_.get_parameters()) { params["mlp." + key] = val; }
    for (auto const& [key, val] : input_layernorm_.get_parameters()) { params["input_layernorm." + key] = val; }
    for (auto const& [key, val] : post_attention_layernorm_.get_parameters()) { params["post_attention_layernorm." + key] = val; }
    return params;
}

template <typename T>
std::map<std::string, Tensor<T>*> L4maModel<T>::get_parameters() {
    std::map<std::string, Tensor<T>*> params;
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
std::map<std::string, Tensor<T>*> L4maForCausalLM<T>::get_parameters() {
    std::map<std::string, Tensor<T>*> params;
    for (auto const& [key, val] : model_.get_parameters()) {
        params["model." + key] = val;
    }
    
    return params;
}

template <typename T>
void RMSNorm<T>::forward(
    PerformanceLogger& logger,
    T* output,
    const T* input,
    int num_tokens,
    cudaStream_t stream) {
    
    uint32_t d = config_.hidden_size;

    flashinfer::norm::RMSNorm<T>(
        const_cast<T *>(input),
        weight_.data(),
        output,
        num_tokens, d, d, d, config_.rms_norm_eps, false, stream
    );
}

template <typename T>
void L4maMlp<T>::forward(
    PerformanceLogger& logger,
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

    Tensor<T> up_proj_out = allocator.allocate<T>(proj_count);
    Tensor<T> gate_proj_out = allocator.allocate<T>(proj_count);

    // Use a Tensor<uint8_t> for the raw byte buffer
    size_t cublas_workspace_size = 32 * 1024 * 1024;
    Tensor<uint8_t> cublas_workspace = allocator.allocate<uint8_t>(cublas_workspace_size);

    // 2. Gate and Up projections. TODO: Fuse them into a single GEMM if possible
    gemm_cublasLt<T>(ltHandle, stream, x, up_proj_weights_.data(), nullptr, up_proj_out.data(), num_tokens, intermediate_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.up_projection", stream);
    gemm_cublasLt<T>(ltHandle, stream, x, gate_proj_weights_.data(), nullptr, gate_proj_out.data(), num_tokens, intermediate_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.gate_projection", stream);
    // print the mean of gate_proj_out_ptr and up_proj_out_ptr for debugging
    //std::cout << "Gate mean: " << gate_proj_out.mean() << ", Up mean: " << up_proj_out.mean() << std::endl;

    // 3. SwiGLU activation (gate * silu(up))
    // We can reuse the gate_proj_out_ptr buffer for the output of SwiGLU
    //silu_and_mul<T>(up_proj_out.data(), up_proj_out.data(), num_tokens, intermediate_size, stream);

    silu_and_mul<T>(
        up_proj_out.data(), 
        gate_proj_out.data(), 
        up_proj_out.data(), 
        num_tokens, 
        intermediate_size, 
        stream
    );
    logger.record("mlp.silu_and_mul", stream);
    //std::cout << "SwiGLU output mean: " << up_proj_out.mean() << std::endl;

    // 4. Down projection
    gemm_cublasLt<T>(ltHandle, stream, up_proj_out.data(), down_proj_weights_.data(), nullptr, output, num_tokens, hidden_size, intermediate_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.down_projection", stream);
    // 5. Deallocate buffers in reverse order of allocation (LIFO)
    allocator.deallocate(cublas_workspace);
    allocator.deallocate(gate_proj_out);
    allocator.deallocate(up_proj_out);

}

template <typename T>
void L4maAttention<T>::forward(
    PerformanceLogger& logger,
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
) {



    // sanity check. Compute the mean of qkv weights - no problem
    // float q_mean = compute_mean(thrust::raw_pointer_cast(q_proj_weights_->data()), q_proj_weights_->size());
    // float k_mean = compute_mean(thrust::raw_pointer_cast(k_proj_weights_->data()), k_proj_weights_->size());
    // float v_mean = compute_mean(thrust::raw_pointer_cast(v_proj_weights_->data()), v_proj_weights_->size());
    // std::cout << "Q mean: " << q_mean << ", K mean: " << k_mean << ", V mean: " << v_mean << std::endl;

    const size_t num_tokens = position_ids.size();
    const size_t hidden_size = config_.hidden_size;
    const size_t head_size = config_.head_size;
    const size_t num_query_heads = config_.num_query_heads;
    const size_t num_key_value_heads = config_.num_key_value_heads;
    const size_t batch_size = kv_page_indptr.size() - 1;

    const size_t q_proj_count = (size_t)num_tokens * num_query_heads * head_size;
    const size_t kv_proj_count = (size_t)num_tokens * num_key_value_heads * head_size;
    
    // 1. Allocate buffers from the stack allocator
    Tensor<T> q_proj = allocator.allocate<T>(q_proj_count);
    Tensor<T> k_proj = allocator.allocate<T>(kv_proj_count);
    Tensor<T> v_proj = allocator.allocate<T>(kv_proj_count);
    size_t cublas_workspace_size = 32 * 1024 * 1024;
    Tensor<uint8_t> cublas_workspace = allocator.allocate<uint8_t>(cublas_workspace_size);

    // 2. Q, K, V projections. TODO: Fuse them into a single GEMM if possible
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, q_proj_weights_.data(), nullptr, q_proj.data(), num_tokens, num_query_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.q_projection", stream);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, k_proj_weights_.data(), nullptr, k_proj.data(), num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.k_projection", stream);
    gemm_cublasLt<T>(ltHandle, stream, hidden_states, v_proj_weights_.data(), nullptr, v_proj.data(), num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.v_projection", stream);

    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_key_value_heads, page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        kv_cache_k, kv_cache_v,
        thrust::raw_pointer_cast(kv_page_indices.data()), 
        thrust::raw_pointer_cast(kv_page_indptr.data()), 
        thrust::raw_pointer_cast(kv_last_page_lens.data())
    );
    logger.record("attn.kv_page_create", stream);

    // 3. Apply RoPE (in-place)
    cudaError_t status = flashinfer::BatchQKApplyLlama31RotaryPosIds(
        q_proj.data(), k_proj.data(), q_proj.data(),  k_proj.data(),
        thrust::raw_pointer_cast(position_ids.data()),
        (uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_key_value_heads, (uint32_t)head_size, (uint32_t)head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        false, config_.rope_factor, config_.rope_theta, config_.rope_low_frequency_factor,
        config_.rope_high_frequency_factor, 8192, stream
    );

    logger.record("attn.apply_rope", stream);

    flashinfer::AppendPagedKVCache<T, int32_t>(
        paged_kv, k_proj.data(), v_proj.data(),
        thrust::raw_pointer_cast(kv_batch_indices.data()),
        thrust::raw_pointer_cast(kv_positions.data()),
        num_tokens,
        num_key_value_heads * head_size, head_size,
        num_key_value_heads * head_size, head_size,
        stream
    );
    logger.record("attn.append_kv_cache", stream);
    // Reuse a buffer for the attention output before the final projection
    T* o_proj_input_ptr = q_proj.data(); 
    flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
        &prefill_handler, q_proj.data(), thrust::raw_pointer_cast(qo_indptr.data()),
        nullptr, paged_kv, o_proj_input_ptr, nullptr, num_query_heads,
        flashinfer::MaskMode::kCustom,
        thrust::raw_pointer_cast(custom_mask.data()),
        thrust::raw_pointer_cast(mask_indptr.data()),
        flashinfer::PosEncodingMode::kNone,
        false, // use_fp16_qk_reduction
        std::nullopt, // maybe_sm_scale
        1.f, // rope_scale
        1e4, // rope_theta
        stream
    );
    logger.record("attn.attention", stream);

    // 5. Final output projection
    gemm_cublasLt<T>(ltHandle, stream, o_proj_input_ptr, o_proj_weights_.data(), nullptr, attn_output, num_tokens, hidden_size, num_query_heads * head_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.o_projection", stream);
    
    // 6. Deallocate buffers in reverse order
    allocator.deallocate(cublas_workspace);
    allocator.deallocate(v_proj);
    allocator.deallocate(k_proj);
    allocator.deallocate(q_proj);
}

template <typename T>
void L4maDecoderLayer<T>::forward(
    PerformanceLogger& logger,
    StackAllocator& allocator,
    T* hidden_states,
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
) {
    const int num_tokens = position_ids.size();
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;

    // --- 1. Self-Attention Block ---
    // The input `hidden_states` serves as the first residual.
    Tensor<T> normed_input = allocator.allocate<T>(hidden_size_elements);
    input_layernorm_.forward(logger, normed_input.data(), hidden_states, num_tokens, stream);
    logger.record("decoder.norm_1", stream);

    Tensor<T> attn_output = allocator.allocate<T>(hidden_size_elements);

    auto attn_logger = logger.scope("decoder.self_attn", stream);
    self_attn_.forward(attn_logger, allocator, attn_output.data(), 
                       normed_input.data(), position_ids, kv_cache_k, kv_cache_v, 
                       kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr, custom_mask, mask_indptr, 
                       ltHandle, stream, prefill_handler, page_size,
                       kv_batch_indices, kv_positions);
    logger.record("decoder.self_attn", stream);


    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, attn_output.data(), hidden_size_elements);
    logger.record("decoder.attn_residual_add", stream);

    // Deallocate attn_output and then normed_input to free up space for the MLP block
    allocator.deallocate(attn_output);
    allocator.deallocate(normed_input);


    // --- 2. MLP Block ---
    // The result of the attention block, `hidden_states`, is the residual for the MLP block.
    Tensor<T> normed_mlp_input = allocator.allocate<T>(hidden_size_elements);
    post_attention_layernorm_.forward(logger, normed_mlp_input.data(), hidden_states, num_tokens, stream);
    logger.record("decoder.norm_2", stream);

    Tensor<T> mlp_output = allocator.allocate<T>(hidden_size_elements);
    auto mlp_logger = logger.scope("decoder.mlp", stream);
    mlp_.forward(mlp_logger, allocator, mlp_output.data(), normed_mlp_input.data(), num_tokens, ltHandle, stream);
    logger.record("decoder.mlp", stream);
    // print the attn_output_ptr mean for debugging
    // float attn_output_mean = compute_mean(mlp_output_ptr, hidden_size_elements);
    // std::cout << "mlp_output_ptr mean: " << attn_output_mean << std::endl;

    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, stream>>>(
        hidden_states, mlp_output.data(), hidden_size_elements);
    logger.record("decoder.mlp_residual_add", stream);
    
    // Deallocate MLP buffers
    allocator.deallocate(mlp_output);
    allocator.deallocate(normed_mlp_input);
}

template <typename T>
void L4maModel<T>::forward(
    PerformanceLogger& logger,
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
) {
    const int num_tokens = input_ids.size();
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;
    
    // Allocate a working buffer for the layers. The layers will operate in-place on this buffer.
    Tensor<T> working_hidden_buffer = allocator.allocate<T>(hidden_size_elements);

    embed<T>(
        embed_tokens_weight_.data(),
        embed_tokens_weight_.size() / config_.hidden_size,
        thrust::raw_pointer_cast(input_ids.data()),
        input_ids.size(),
        working_hidden_buffer.data(), // Embeddings are written to the allocated working buffer
        config_.hidden_size, 
        stream
    );
    logger.record("model.embedding_lookup", stream);

    // print out the mean of the embeddings
    // float embed_mean = compute_mean(working_hidden_buffer, hidden_size_elements);
    // std::cout << "Embed mean: " << embed_mean << std::endl;

    size_t kv_cache_size = kv_cache_k.size() / config_.num_layers;

    T* k_cache_ptr = thrust::raw_pointer_cast(kv_cache_k.data());
    T* v_cache_ptr = thrust::raw_pointer_cast(kv_cache_v.data());

    const size_t max_num_pages = kv_cache_k.size() / (config_.num_key_value_heads * config_.head_size);
    for (size_t i = 0; i < layers_.size(); ++i) {

        auto& layer = layers_[i];

        // compute offset
        T* layer_k_cache_ptr = k_cache_ptr + i * kv_cache_size;
        T* layer_v_cache_ptr = v_cache_ptr + i * kv_cache_size;

        auto layer_logger = logger.scope("model.decoder_layer", stream);
        // auto start_time = std::chrono::high_resolution_clock::now();

        // Pass the allocator down to the layer. The layer will use it for its own scratch space.
        layer.forward(layer_logger, allocator, working_hidden_buffer.data(),
                      position_ids, layer_k_cache_ptr, layer_v_cache_ptr,
                      kv_page_indices, kv_page_indptr, kv_last_page_lens,
                      qo_indptr, custom_mask, mask_indptr, ltHandle, stream,
                      prefill_handler, page_size, kv_batch_indices, kv_positions);

// cudaStreamSynchronize(stream);
//             auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
//     std::cout << "one layer took " << (elapsed.count()) << " ms." << std::endl;

        //std::cout << "layer [" << i << "] :" << working_hidden_buffer.mean() << std::endl;
        //working_hidden_buffer.print(0, 10);
        logger.record("model.decoder_layer", stream);

    }

    // Final norm reads from the working buffer and writes to the final output buffer.
    norm_.forward(logger, final_norm_output, working_hidden_buffer.data(), num_tokens, stream);
    logger.record("model.norm_", stream);

    // Deallocate the working buffer.
    allocator.deallocate(working_hidden_buffer);
}

template <typename T>
void L4maForCausalLM<T>::forward(
    PerformanceLogger& logger,
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
    ) {
    
    const int num_tokens = input_ids.size();
    
    const size_t hidden_elements = (size_t)num_tokens * config_.hidden_size;
    const size_t flash_float_bytes = 256 * 1024 * 1024;
    const size_t flash_int_bytes = 8 * 1024 * 1024;
    const size_t lm_head_workspace_bytes = 32 * 1024 * 1024;

    Tensor<T> hidden_states = allocator.allocate<T>(hidden_elements);
    Tensor<uint8_t> flashinfer_float_buffer = allocator.allocate<uint8_t>(flash_float_bytes);
    Tensor<uint8_t> flashinfer_int_buffer = allocator.allocate<uint8_t>(flash_int_bytes);

    flashinfer::BatchPrefillHandler handler;
    handler.Plan<T, int32_t>(
        flashinfer_float_buffer.data(), flash_float_bytes,
        flashinfer_int_buffer.data(), flash_int_bytes,
        qo_indptr_host.data(), 
        kv_page_indptr_host.data(),
        num_tokens,
        qo_indptr_host.size() - 1,
        config_.num_query_heads,
        config_.num_key_value_heads,
        config_.head_size,
        page_size);
    logger.record("model.flashinfer_plan", stream);

    model_.forward(
        logger,
        allocator, // Pass the allocator down to the model.
        hidden_states.data(),
        input_ids, position_ids,
        kv_cache_k_, kv_cache_v_,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        qo_indptr, custom_mask, mask_indptr,
        cublaslt_handle_, stream, handler, page_size,
        kv_batch_indices, kv_positions
    );



    Tensor<uint8_t> lm_head_workspace = allocator.allocate<uint8_t>(lm_head_workspace_bytes);
    
    gemm_cublasLt<T>(
        cublaslt_handle_, stream,
        hidden_states.data(),
        model_.get_embed_tokens_weight().data(),
        nullptr,
        output.data(),
        num_tokens, config_.vocab_size, config_.hidden_size,
        lm_head_workspace.data(), lm_head_workspace_bytes, false, true
    );
    logger.record("model.lm_head", stream);
    
    // Deallocations will happen automatically as the allocator goes out of scope.
    // However, to be explicit and use the implemented checks:
    allocator.deallocate(lm_head_workspace);
    allocator.deallocate(flashinfer_int_buffer);
    allocator.deallocate(flashinfer_float_buffer);
    allocator.deallocate(hidden_states);
}

// --- Explicit Template Instantiations (Unchanged) ---
template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;