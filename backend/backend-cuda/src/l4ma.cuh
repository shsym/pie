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
#include "stack_allocator.cuh"
// Forward declarations

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



/**
 * @brief Consolidates all workspace memory, intermediate computations, and index
 * pointers for a single forward pass. This object manages the lifecycle of the
 * temporary workspace buffer required for the model execution.
 */
template <typename T>
class L4maBuffer {
public:
    // --- Device-side pointers, valid after `plan()` is called ---
    Tensor<uint32_t> input_ids;
    Tensor<int32_t> position_ids;
    Tensor<int32_t> kv_page_indices;
    Tensor<int32_t> kv_page_indptr;
    Tensor<int32_t> kv_last_page_lens;
    Tensor<int32_t> qo_indptr;
    Tensor<uint8_t> custom_mask;
    Tensor<int32_t> mask_indptr;
    Tensor<int32_t> kv_batch_indices;
    Tensor<int32_t> kv_positions;

    // --- Shape, Config, and Execution Context ---
    const L4maConfig& config;
    const int32_t page_size;
    size_t num_tokens;
    size_t batch_size;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;

    // --- Handlers ---
    flashinfer::BatchPrefillHandler prefill_handler;

    // --- Constructor / Destructor ---
    L4maBuffer(const L4maConfig& cfg, int32_t p_size, size_t workspace_size);
    ~L4maBuffer();

    // --- Deleted Functions ---
    L4maBuffer(const L4maBuffer&) = delete;
    L4maBuffer& operator=(const L4maBuffer&) = delete;
    L4maBuffer(L4maBuffer&&) = delete;
    L4maBuffer& operator=(L4maBuffer&&) = delete;

    // --- Static Method for Workspace Calculation ---
    static size_t get_workspace_size(
        const L4maConfig& config,
        size_t max_num_tokens,
        size_t max_batch_size,
        size_t max_kv_seqlens
    );

    // --- Public Methods ---
    void plan(
        cudaStream_t strm,
         std::vector<uint32_t>& input_ids_host,
         std::vector<int32_t>& position_ids_host,
         std::vector<int32_t>& kv_page_indices_host,
         std::vector<int32_t>& kv_page_indptr_host,
         std::vector<int32_t>& kv_last_page_lens_host,
         std::vector<int32_t>& qo_indptr_host,
         std::vector<bool>& packed_custom_mask_host,
         std::vector<int32_t>& mask_indptr_host,
         std::vector<int32_t>& kv_batch_indices_host,
         std::vector<int32_t>& kv_positions_host
    );

    // --- Allocator Wrappers ---
    template <typename U> Tensor<U> allocate(size_t count);
    Tensor<uint8_t> allocate_rest();
    template <typename U> void deallocate(Tensor<U>& tensor);

private:
    size_t buffer_size_;
    std::unique_ptr<StackAllocator> allocator_;
};


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
                 L4maBuffer<T>& buffer,
                 T* output,
                 const T* x);

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

    void forward(PerformanceLogger& logger,
                 L4maBuffer<T>& buffer,
                 T* attn_output,
                 const T* hidden_states,
                 T* kv_cache_k,
                 T* kv_cache_v
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

    void forward(PerformanceLogger& logger,
                 L4maBuffer<T>& buffer,
                 T* hidden_states, 
                 T* kv_cache_k,
                 T* kv_cache_v);

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

    void forward(PerformanceLogger& logger,
                 L4maBuffer<T>& buffer,
                 T* final_norm_output,
                 thrust::device_vector<T>& kv_cache_k,
                 thrust::device_vector<T>& kv_cache_v);

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
                 L4maBuffer<T>& buffer,
                 Tensor<T>& output);

    std::map<std::string, Tensor<T>*> get_parameters() override;
    void create_kv_device_vectors(int max_kv_num);

    L4maConfig& get_config() { return config_; }

private:
    L4maConfig config_;

    L4maModel<T> model_;

    thrust::device_vector<T> kv_cache_k_;
    thrust::device_vector<T> kv_cache_v_;
};