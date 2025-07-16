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
#include "flashinfer/sampling.cuh"
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

std::vector<uint8_t> packbits_little(const std::vector<bool>& data) {
    // Calculate the number of bytes needed, padding with zeros for the last byte if necessary.
    const size_t num_bytes = (data.size() + 7) / 8;
    std::vector<uint8_t> packed(num_bytes, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        // The first element in each chunk of 8 corresponds to the LSB.
        // The '& 7' is equivalent to 'i % 8' but can be faster.
        if (data[i]) {
            packed[i / 8] |= (1 << (i & 7));
        }
    }

    return packed;
}


template <typename T>
L4maBuffer<T>::L4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size)
    : config(cfg),
      page_size(page_size),
      dist_size(dist_size),
      num_tokens(0),
      batch_size(0),
      stream(nullptr),
      buffer_size_(workspace_size) {
    allocator_ = std::make_unique<StackAllocator>(buffer_size_);
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));
}

// destructor
template <typename T>
L4maBuffer<T>::~L4maBuffer() {
    // Clean up the CUBLAS handle
    CUBLAS_CHECK(cublasLtDestroy(ltHandle));
    // The StackAllocator will automatically free its buffer when it goes out of scope
}

template <typename T>
size_t L4maBuffer<T>::get_workspace_size(
    const L4maConfig& config,
    size_t max_num_tokens,
    size_t max_batch_size,
    size_t max_kv_seqlens, // TODO: max_dist_size is needed for sampling buffers
    size_t dist_size
) {
    const size_t alignment = 256;
    const size_t hidden_size = config.hidden_size;
    const size_t intermediate_size = config.intermediate_size;
    const size_t num_q_heads = config.num_query_heads;
    const size_t num_kv_heads = config.num_key_value_heads;
    const size_t head_size = config.head_size;

    // --- Peak memory within a decoder layer ---
    size_t decoder_layer_peak = 0;
    {
        // Buffers allocated in L4maDecoderLayer::forward
        size_t decoder_wrapper_bytes = 2 * align_up((size_t)max_num_tokens * hidden_size * sizeof(T), alignment);

        // Path 1: Attention block peak
        size_t attn_path_peak = 0;
        attn_path_peak += align_up((size_t)max_num_tokens * num_q_heads * head_size * sizeof(T), alignment); // q_proj
        attn_path_peak += align_up((size_t)max_num_tokens * num_kv_heads * head_size * sizeof(T), alignment); // k_proj
        attn_path_peak += align_up((size_t)max_num_tokens * num_kv_heads * head_size * sizeof(T), alignment); // v_proj
        attn_path_peak += align_up(32 * 1024 * 1024, alignment); // cublas_workspace

        // Path 2: MLP block peak
        size_t mlp_path_peak = 0;
        mlp_path_peak += align_up((size_t)max_num_tokens * intermediate_size * sizeof(T), alignment); // up_proj
        mlp_path_peak += align_up((size_t)max_num_tokens * intermediate_size * sizeof(T), alignment); // gate_proj
        mlp_path_peak += align_up(32 * 1024 * 1024, alignment); // cublas_workspace

        decoder_layer_peak = decoder_wrapper_bytes + std::max(attn_path_peak, mlp_path_peak);
    }

    // --- Peak memory for the final LM head and sampling ---
    size_t final_stage_peak = 0;
    {
        final_stage_peak += align_up((size_t)max_num_tokens * hidden_size * sizeof(T), alignment);                      // hidden_states
        final_stage_peak += align_up((size_t)max_num_tokens * config.vocab_size * sizeof(T), alignment);                // output_logits
        final_stage_peak += align_up((size_t)max_num_tokens * config.vocab_size * sizeof(float), alignment);             // output_logits_fp32
        final_stage_peak += align_up((size_t)max_num_tokens * config.vocab_size * sizeof(float), alignment);             // output_logits_masked
        final_stage_peak += align_up((size_t)max_num_tokens * dist_size * sizeof(float), alignment);                     // final_logits_val
        final_stage_peak += align_up((size_t)max_num_tokens * dist_size * sizeof(int32_t), alignment);                   // final_logits_indices
        final_stage_peak += align_up(32 * 1024 * 1024, alignment);                                                      // lm_head_workspace
        final_stage_peak += align_up((size_t)max_num_tokens * hidden_size * sizeof(T), alignment);                      // gathered_states (worst case)
    }

    // --- Other persistent buffers ---
    size_t persistent_buffers = 0;
    // Memory for FlashInfer handlers
    persistent_buffers += align_up(256 * 1024 * 1024, alignment);
    persistent_buffers += align_up(8 * 1024 * 1024, alignment);
    // Working buffer in L4maModel::forward
    persistent_buffers += align_up((size_t)max_num_tokens * hidden_size * sizeof(T), alignment);

    // Total size is the max of the two main stages, plus persistent metadata/handler buffers.
    size_t total_bytes = persistent_buffers + std::max(decoder_layer_peak, final_stage_peak);

    // Part 2: Index and Metadata Vectors
    total_bytes += align_up(max_num_tokens * sizeof(uint32_t), alignment);
    total_bytes += align_up(max_num_tokens * sizeof(int32_t), alignment);
    total_bytes += align_up(max_num_tokens * sizeof(int32_t), alignment);
    total_bytes += align_up((max_batch_size + 1) * sizeof(int32_t), alignment);
    total_bytes += align_up(max_batch_size * sizeof(int32_t), alignment);
    total_bytes += align_up((max_batch_size + 1) * sizeof(int32_t), alignment);
    size_t max_mask_elements = max_num_tokens * max_kv_seqlens;
    total_bytes += align_up((max_mask_elements + 7) / 8, alignment);
    total_bytes += align_up((max_batch_size + 1) * sizeof(int32_t), alignment);
    total_bytes += align_up(max_num_tokens * sizeof(int32_t), alignment);
    total_bytes += align_up(max_num_tokens * sizeof(int32_t), alignment);

    return total_bytes;
}

template <typename T>
void L4maBuffer<T>::plan(
    cudaStream_t strm,
     std::vector<int32_t>& input_ids_host,
     std::vector<int32_t>& position_ids_host,
     std::vector<int32_t>& kv_page_indices_host,
     std::vector<int32_t>& kv_page_indptr_host,
     std::vector<int32_t>& kv_last_page_lens_host,
     std::vector<int32_t>& qo_indptr_host,
     std::vector<bool>& custom_masks_host,
     std::vector<int32_t>& mask_indptr_host,
     std::vector<int32_t>& kv_batch_indices_host,
     std::vector<int32_t>& kv_positions_host,
     std::vector<int32_t>& output_indices_src_host
) {
    this->stream = strm;
    this->num_tokens = input_ids_host.size();
    this->batch_size = kv_page_indptr_host.empty() ? 0 : kv_page_indptr_host.size() - 1;
    
    std::vector<uint8_t> packed_custom_mask_host = packbits_little(custom_masks_host);

    allocator_->reset();

    input_ids          = allocator_->allocate_and_copy_async<int32_t>(input_ids_host, stream);
    position_ids       = allocator_->allocate_and_copy_async<int32_t>(position_ids_host, stream);
    kv_page_indices    = allocator_->allocate_and_copy_async<int32_t>(kv_page_indices_host, stream);
    kv_page_indptr     = allocator_->allocate_and_copy_async<int32_t>(kv_page_indptr_host, stream);
    kv_last_page_lens  = allocator_->allocate_and_copy_async<int32_t>(kv_last_page_lens_host, stream);
    qo_indptr          = allocator_->allocate_and_copy_async<int32_t>(qo_indptr_host, stream);
    custom_mask        = allocator_->allocate_and_copy_async<uint8_t>(packed_custom_mask_host, stream);
    mask_indptr        = allocator_->allocate_and_copy_async<int32_t>(mask_indptr_host, stream);
    kv_batch_indices   = allocator_->allocate_and_copy_async<int32_t>(kv_batch_indices_host, stream);
    kv_positions       = allocator_->allocate_and_copy_async<int32_t>(kv_positions_host, stream);

    if (!output_indices_src_host.empty()) {
        output_indices_src = allocator_->allocate_and_copy_async<int32_t>(output_indices_src_host, stream);
    }

    Tensor<uint8_t> flashinfer_float_buffer = this->allocate<uint8_t>(256 * 1024 * 1024);
    Tensor<uint8_t> flashinfer_int_buffer = this->allocate<uint8_t>(8 * 1024 * 1024);

    prefill_handler.Plan<T, int32_t>(
        flashinfer_float_buffer.data(), flashinfer_float_buffer.size(),
        flashinfer_int_buffer.data(), flashinfer_int_buffer.size(),
        qo_indptr_host.data(),
        kv_page_indptr_host.data(),
        num_tokens,
        batch_size,
        config.num_query_heads,
        config.num_key_value_heads,
        config.head_size,
        page_size
    );

}

template <typename T> template <typename U>
Tensor<U> L4maBuffer<T>::allocate(size_t count) {
    return allocator_->template allocate<U>(count);
}

template <typename T>
Tensor<uint8_t> L4maBuffer<T>::allocate_rest() {
    return allocator_->allocate_rest();
}

template <typename T> template <typename U>
void L4maBuffer<T>::deallocate(Tensor<U>& tensor) {
    allocator_->deallocate(tensor);
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
    L4maBuffer<T>& buffer,
    T* output, 
    const T* x
) {
    const int hidden_size = config_.hidden_size;
    const int intermediate_size = config_.intermediate_size;
    const size_t proj_count = (size_t)buffer.num_tokens * intermediate_size;

    Tensor<T> up_proj_out = buffer.template allocate<T>(proj_count);
    Tensor<T> gate_proj_out = buffer.template allocate<T>(proj_count);

    // Use a Tensor<uint8_t> for the raw byte buffer
    size_t cublas_workspace_size = 32 * 1024 * 1024;
    Tensor<uint8_t> cublas_workspace = buffer.template allocate<uint8_t>(cublas_workspace_size);

    // 2. Gate and Up projections. TODO: Fuse them into a single GEMM if possible
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, x, up_proj_weights_.data(), nullptr, up_proj_out.data(), buffer.num_tokens, intermediate_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.up_projection", buffer.stream);
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, x, gate_proj_weights_.data(), nullptr, gate_proj_out.data(), buffer.num_tokens, intermediate_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.gate_projection", buffer.stream);

    // 3. SwiGLU activation (gate * silu(up))
    // We can reuse the gate_proj_out_ptr buffer for the output of SwiGLU
    silu_and_mul<T>(
        up_proj_out.data(), 
        gate_proj_out.data(), 
        up_proj_out.data(), 
        buffer.num_tokens, 
        intermediate_size, 
        buffer.stream
    );
    logger.record("mlp.silu_and_mul", buffer.stream);
    //std::cout << "SwiGLU output mean: " << up_proj_out.mean() << std::endl;

    // 4. Down projection
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, up_proj_out.data(), down_proj_weights_.data(), nullptr, output, buffer.num_tokens, hidden_size, intermediate_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("mlp.down_projection", buffer.stream);

    // 5. Deallocate buffers in reverse order of allocation (LIFO)
    buffer.deallocate(cublas_workspace);
    buffer.deallocate(gate_proj_out);
    buffer.deallocate(up_proj_out);

}

template <typename T>
void L4maAttention<T>::forward(
    PerformanceLogger& logger,
    L4maBuffer<T>& buffer,
    T* attn_output,
    const T* hidden_states,
    //thrust::device_vector<int32_t>& position_ids,
    T* kv_cache_k,
    T* kv_cache_v
    // thrust::device_vector<int32_t>& kv_page_indices,
    // thrust::device_vector<int32_t>& kv_page_indptr,
    // thrust::device_vector<int32_t>& kv_last_page_lens,
    // thrust::device_vector<int32_t>& qo_indptr,
    // thrust::device_vector<uint8_t>& custom_mask,
    // thrust::device_vector<int32_t>& mask_indptr,
    // cublasLtHandle_t ltHandle,
    // cudaStream_t stream,
    // flashinfer::BatchPrefillHandler& prefill_handler,
    // const int32_t page_size,
    // thrust::device_vector<int32_t>& kv_batch_indices,
    // thrust::device_vector<int32_t>& kv_positions
) {



    // sanity check. Compute the mean of qkv weights - no problem
    // float q_mean = compute_mean(thrust::raw_pointer_cast(q_proj_weights_->data()), q_proj_weights_->size());
    // float k_mean = compute_mean(thrust::raw_pointer_cast(k_proj_weights_->data()), k_proj_weights_->size());
    // float v_mean = compute_mean(thrust::raw_pointer_cast(v_proj_weights_->data()), v_proj_weights_->size());
    // std::cout << "Q mean: " << q_mean << ", K mean: " << k_mean << ", V mean: " << v_mean << std::endl;

    const size_t num_tokens = buffer.num_tokens;
    const size_t hidden_size = config_.hidden_size;
    const size_t head_size = config_.head_size;
    const size_t num_query_heads = config_.num_query_heads;
    const size_t num_key_value_heads = config_.num_key_value_heads;
    const size_t batch_size = buffer.batch_size;

    const size_t q_proj_count = (size_t)num_tokens * num_query_heads * head_size;
    const size_t kv_proj_count = (size_t)num_tokens * num_key_value_heads * head_size;
    
    // 1. Allocate buffers from the stack allocator
    Tensor<T> q_proj = buffer.template allocate<T>(q_proj_count);
    Tensor<T> k_proj = buffer.template allocate<T>(kv_proj_count);
    Tensor<T> v_proj = buffer.template allocate<T>(kv_proj_count);
    size_t cublas_workspace_size = 32 * 1024 * 1024;
    Tensor<uint8_t> cublas_workspace = buffer.template allocate<uint8_t>(cublas_workspace_size);

    // 2. Q, K, V projections. TODO: Fuse them into a single GEMM if possible
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, hidden_states, q_proj_weights_.data(), nullptr, q_proj.data(), num_tokens, num_query_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.q_projection", buffer.stream);
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, hidden_states, k_proj_weights_.data(), nullptr, k_proj.data(), num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.k_projection", buffer.stream);
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, hidden_states, v_proj_weights_.data(), nullptr, v_proj.data(), num_tokens, num_key_value_heads * head_size, hidden_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.v_projection", buffer.stream);

    flashinfer::paged_kv_t<T, int32_t> paged_kv(
        num_key_value_heads, buffer.page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        kv_cache_k, kv_cache_v,
        buffer.kv_page_indices.data(), 
        buffer.kv_page_indptr.data(), 
        buffer.kv_last_page_lens.data()
    );
    logger.record("attn.kv_page_create", buffer.stream);

    // 3. Apply RoPE (in-place)
    cudaError_t status = flashinfer::BatchQKApplyLlama31RotaryPosIds(
        q_proj.data(), k_proj.data(), q_proj.data(),  k_proj.data(),
        buffer.position_ids.data(),
        (uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_key_value_heads, (uint32_t)head_size, (uint32_t)head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        num_query_heads * head_size, head_size, num_key_value_heads * head_size, head_size,
        false, config_.rope_factor, config_.rope_theta, config_.rope_low_frequency_factor,
        config_.rope_high_frequency_factor, 8192, buffer.stream
    );

    logger.record("attn.apply_rope", buffer.stream);

    flashinfer::AppendPagedKVCache<T, int32_t>(
        paged_kv, k_proj.data(), v_proj.data(),
        buffer.kv_batch_indices.data(),
        buffer.kv_positions.data(),
        num_tokens,
        num_key_value_heads * head_size, head_size,
        num_key_value_heads * head_size, head_size,
        buffer.stream
    );
    logger.record("attn.append_kv_cache", buffer.stream);

    // Reuse a buffer for the attention output before the final projection
    T* o_proj_input_ptr = q_proj.data(); 
    flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
        &buffer.prefill_handler, q_proj.data(), buffer.qo_indptr.data(),
        nullptr, paged_kv, o_proj_input_ptr, nullptr, num_query_heads,
        flashinfer::MaskMode::kCustom,
        buffer.custom_mask.data(),
        buffer.mask_indptr.data(),
        flashinfer::PosEncodingMode::kNone,
        false, // use_fp16_qk_reduction -> unused
        std::nullopt, // maybe_sm_scale -> unused
        1.f, // rope_scale -> unused
        1e4, // rope_theta -> unused
        buffer.stream
    );
    logger.record("attn.attention", buffer.stream);

    // 5. Final output projection
    gemm_cublasLt<T>(buffer.ltHandle, buffer.stream, o_proj_input_ptr, o_proj_weights_.data(), nullptr, attn_output, num_tokens, hidden_size, num_query_heads * head_size, cublas_workspace.data(), cublas_workspace_size, false, true);
    logger.record("attn.o_projection", buffer.stream);
    
    // 6. Deallocate buffers in reverse order
    buffer.deallocate(cublas_workspace);
    buffer.deallocate(v_proj);
    buffer.deallocate(k_proj);
    buffer.deallocate(q_proj);
}

template <typename T>
void L4maDecoderLayer<T>::forward(
    PerformanceLogger& logger,
    L4maBuffer<T>& buffer,
    T* hidden_states,
    T* kv_cache_k,
    T* kv_cache_v
) {
    const int num_tokens = buffer.num_tokens;
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;

    // --- 1. Self-Attention Block ---
    // The input `hidden_states` serves as the first residual.
    Tensor<T> normed_input = buffer.template allocate<T>(hidden_size_elements);
    input_layernorm_.forward(logger, normed_input.data(), hidden_states, num_tokens, buffer.stream);
    logger.record("decoder.norm_1", buffer.stream);

    Tensor<T> attn_output = buffer.template allocate<T>(hidden_size_elements);

    auto attn_logger = logger.scope("decoder.self_attn", buffer.stream);
    self_attn_.forward(attn_logger, buffer, attn_output.data(), 
                       normed_input.data() , kv_cache_k, kv_cache_v);

    logger.record("decoder.self_attn", buffer.stream);


    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, buffer.stream>>>(
        hidden_states, attn_output.data(), hidden_size_elements);
    logger.record("decoder.attn_residual_add", buffer.stream);

    // Deallocate attn_output and then normed_input to free up space for the MLP block
    buffer.deallocate(attn_output);
    buffer.deallocate(normed_input);


    // --- 2. MLP Block ---
    // The result of the attention block, `hidden_states`, is the residual for the MLP block.
    Tensor<T> normed_mlp_input = buffer.template allocate<T>(hidden_size_elements);
    post_attention_layernorm_.forward(logger, normed_mlp_input.data(), hidden_states, num_tokens, buffer.stream);
    logger.record("decoder.norm_2", buffer.stream);

    Tensor<T> mlp_output = buffer.template allocate<T>(hidden_size_elements);
    auto mlp_logger = logger.scope("decoder.mlp", buffer.stream);
    mlp_.forward(mlp_logger, buffer, mlp_output.data(), normed_mlp_input.data());
    logger.record("decoder.mlp", buffer.stream);
    // print the attn_output_ptr mean for debugging
    // float attn_output_mean = compute_mean(mlp_output_ptr, hidden_size_elements);
    // std::cout << "mlp_output_ptr mean: " << attn_output_mean << std::endl;

    add_residual_kernel<<<(hidden_size_elements + 255) / 256, 256, 0, buffer.stream>>>(
        hidden_states, mlp_output.data(), hidden_size_elements);
    logger.record("decoder.mlp_residual_add", buffer.stream);
    
    // Deallocate MLP buffers
    buffer.deallocate(mlp_output);
    buffer.deallocate(normed_mlp_input);
}

template <typename T>
void L4maModel<T>::forward(
    PerformanceLogger& logger,
    L4maBuffer<T>& buffer,
    T* final_norm_output,
    thrust::device_vector<T>& kv_cache_k,
    thrust::device_vector<T>& kv_cache_v
) {
    const int num_tokens = buffer.num_tokens;
    const size_t hidden_size_elements = (size_t)num_tokens * config_.hidden_size;
    
    // Allocate a working buffer for the layers. The layers will operate in-place on this buffer.
    Tensor<T> working_hidden_buffer = buffer.template allocate<T>(hidden_size_elements);

    embed<T, int32_t>(
        embed_tokens_weight_.data(),
        embed_tokens_weight_.size() / config_.hidden_size,
        buffer.input_ids.data(),
        buffer.num_tokens,
        working_hidden_buffer.data(), // Embeddings are written to the allocated working buffer
        config_.hidden_size, 
        buffer.stream
    );
    logger.record("model.embedding_lookup", buffer.stream);

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

        auto layer_logger = logger.scope("model.decoder_layer", buffer.stream);
        // auto start_time = std::chrono::high_resolution_clock::now();

        // Pass the allocator down to the layer. The layer will use it for its own scratch space.
        layer.forward(layer_logger, buffer, working_hidden_buffer.data(),
                      layer_k_cache_ptr, layer_v_cache_ptr);

// cudaStreamSynchronize(stream);
//             auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
//     std::cout << "one layer took " << (elapsed.count()) << " ms." << std::endl;

        //std::cout << "layer [" << i << "] :" << working_hidden_buffer.mean() << std::endl;
        //working_hidden_buffer.print(0, 10);
        logger.record("model.decoder_layer", buffer.stream);

    }

    // Final norm reads from the working buffer and writes to the final output buffer.
    norm_.forward(logger, final_norm_output, working_hidden_buffer.data(), num_tokens, buffer.stream);
    logger.record("model.norm_", buffer.stream);

    // Deallocate the working buffer.
    buffer.deallocate(working_hidden_buffer);
}

template <typename T>
std::pair<std::vector<float>, std::vector<int32_t>> L4maForCausalLM<T>::forward(
    PerformanceLogger& logger,
    L4maBuffer<T>& buffer) {

    const int num_tokens = buffer.num_tokens;
    const int num_output_tokens = buffer.output_indices_src.size();
    const int dist_size = buffer.dist_size;
    const size_t hidden_elements = (size_t)num_tokens * config_.hidden_size;
    const size_t output_elements = (size_t)num_output_tokens * config_.hidden_size;
    const size_t lm_head_workspace_bytes = 32 * 1024 * 1024;

    // 1. Allocate all necessary temporary buffers from the stack allocator.
    Tensor<T> hidden_states = buffer.template allocate<T>(hidden_elements);

    // 2. Run the main model layers to produce hidden states.
    model_.forward(
        logger,
        buffer,
        hidden_states.data(),
        kv_cache_k_,
        kv_cache_v_
    );
    logger.record("model.forward", buffer.stream);


    if (num_output_tokens == 0) {
        // If there are no output tokens, we can return empty vectors.
        return std::make_pair(std::vector<float>(), std::vector<int32_t>());
    }


    Tensor<T> output_logits = buffer.template allocate<T>(num_output_tokens * config_.vocab_size);
    Tensor<float> output_logits_fp32 = buffer.template allocate<float>(num_output_tokens * config_.vocab_size);
    Tensor<float> output_logits_masked = buffer.template allocate<float>(num_output_tokens * config_.vocab_size);
    Tensor<float> final_logits_val = buffer.template allocate<float>(num_output_tokens * dist_size);
    Tensor<int32_t> final_logits_indices = buffer.template allocate<int32_t>(num_output_tokens * dist_size);
    Tensor<uint8_t> lm_head_workspace = buffer.template allocate<uint8_t>(lm_head_workspace_bytes);



    // 3. Handle the hidden states for the final projection.
    Tensor<T>* final_hidden_states_ptr = &hidden_states;
    Tensor<T> gathered_states;
    bool needs_gather = (hidden_elements != output_elements);

    if (needs_gather) {
        // NOTE: gathered_states is allocated last, so it must be deallocated first.
        gathered_states = buffer.template allocate<T>(output_elements);
        embed<T, int32_t>(
            hidden_states.data(),
            num_tokens,
            buffer.output_indices_src.data(),
            num_output_tokens,
            gathered_states.data(),
            config_.hidden_size,
            buffer.stream
        );
        final_hidden_states_ptr = &gathered_states;
        logger.record("model.gather_hidden_states", buffer.stream);

    }

    // 4. Compute logits
    gemm_cublasLt<T>(
        buffer.ltHandle, buffer.stream,
        final_hidden_states_ptr->data(),
        model_.get_embed_tokens_weight().data(),
        nullptr,
        output_logits.data(),
        num_output_tokens,
        config_.vocab_size, config_.hidden_size,
        lm_head_workspace.data(), lm_head_workspace_bytes, false, true
    );
    logger.record("model.lm_head", buffer.stream);

    cast_type<T, float>(
        output_logits.data(),
        output_logits_fp32.data(),
        num_output_tokens * config_.vocab_size,
        buffer.stream
    );
    logger.record("model.casting", buffer.stream);

    // 5. Perform sampling
    flashinfer::sampling::TopKMaskLogits<float, int32_t>(
        output_logits_fp32.data(),
        output_logits_masked.data(),
        nullptr,
        num_output_tokens,
        dist_size,
        config_.vocab_size,
        buffer.stream
    );
    logger.record("model.topkmask", buffer.stream);

    extract_k_values<float>(
        output_logits_masked.data(),
        final_logits_val.data(),
        final_logits_indices.data(),
        num_output_tokens,
        config_.vocab_size,
        dist_size,
        buffer.stream
    );
    logger.record("model.extract", buffer.stream);

    // 6. Copy final results back to the host.
    std::vector<float> final_logits_val_host = final_logits_val.to_vector();
    std::vector<int32_t> final_logits_indices_host = final_logits_indices.to_vector();
    
    // 7. DEALLOCATE ALL BUFFERS IN REVERSE ORDER (LIFO)


    if (needs_gather) {
        buffer.deallocate(gathered_states);
    }

    buffer.deallocate(lm_head_workspace);
    buffer.deallocate(final_logits_indices);
    buffer.deallocate(final_logits_val);
    buffer.deallocate(output_logits_masked);
    buffer.deallocate(output_logits_fp32); // Deallocate the fp32 version
    buffer.deallocate(output_logits);
    buffer.deallocate(hidden_states);

    // 8. Return the results.
    return std::make_pair(final_logits_val_host, final_logits_indices_host);
}

// --- Explicit Template Instantiations (Unchanged) ---
template class L4maBuffer<__nv_bfloat16>;
template class RMSNorm<__nv_bfloat16>;
template class L4maMlp<__nv_bfloat16>;
template class L4maAttention<__nv_bfloat16>;
template class L4maDecoderLayer<__nv_bfloat16>;
template class L4maModel<__nv_bfloat16>;
template class L4maForCausalLM<__nv_bfloat16>;