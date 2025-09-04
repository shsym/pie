#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_buffer.hpp"
#include "metal_kv_cache.hpp"
#include "metal_gemm_wrapper.hpp"
#include "metal_rmsnorm_wrapper.hpp"
#include "metal_rope_wrapper.hpp"
#include "metal_silu_and_mul_wrapper.hpp"
#include "metal_batch_prefill_handle.hpp"
#include <Metal/Metal.h>
#include <map>
#include <memory>
#include <vector>
#include <string>

// Forward declarations
struct L4maConfig;
struct AppConfig;
struct ModelMetadata;

/**
 * @brief Base class for all Metal model components (modules)
 * Metal equivalent of the CUDA Module<T> base class
 */
template <typename T>
class MetalModule {
public:
    virtual ~MetalModule() = default;
    virtual std::map<std::string, MetalTensor<T>*> get_parameters() = 0;
};

/**
 * @brief Metal RMS Normalization layer
 * Equivalent to RMSNorm<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maRMSNorm : public MetalModule<T> {
public:
    explicit MetalL4maRMSNorm(const L4maConfig& config);

    // Move constructor and assignment
    MetalL4maRMSNorm(MetalL4maRMSNorm&& other) noexcept = default;
    MetalL4maRMSNorm& operator=(MetalL4maRMSNorm&& other) noexcept = default;

    // Delete copy constructor and assignment
    MetalL4maRMSNorm(const MetalL4maRMSNorm&) = delete;
    MetalL4maRMSNorm& operator=(const MetalL4maRMSNorm&) = delete;

    // Forward pass using existing metal_rmsnorm implementation
    void forward(T* output,
                 const T* input,
                 int num_tokens,
                 id<MTLCommandBuffer> commandBuffer);

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    MetalTensor<T> weight_;
};

/**
 * @brief Metal MLP block of the L4MA model
 * Equivalent to L4maMlp<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maMlp : public MetalModule<T> {
public:
    explicit MetalL4maMlp(const L4maConfig& config);

    // Move constructor and assignment
    MetalL4maMlp(MetalL4maMlp&& other) noexcept = default;
    MetalL4maMlp& operator=(MetalL4maMlp&& other) noexcept = default;

    // Delete copy constructor and assignment
    MetalL4maMlp(const MetalL4maMlp&) = delete;
    MetalL4maMlp& operator=(const MetalL4maMlp&) = delete;

    // Forward pass with Metal buffer management
    void forward(MetalProfileScope profiler,
                 MetalL4maBuffer<T>& buffer,
                 T* output,
                 const T* x);

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    MetalTensor<T> gate_proj_weights_;
    MetalTensor<T> up_proj_weights_;
    MetalTensor<T> down_proj_weights_;
};

/**
 * @brief Metal attention block of the L4MA model
 * Equivalent to L4maAttention<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maAttention : public MetalModule<T> {
public:
    explicit MetalL4maAttention(const L4maConfig& config);
    ~MetalL4maAttention();

    // Move constructor and assignment
    MetalL4maAttention(MetalL4maAttention&& other) noexcept;
    MetalL4maAttention& operator=(MetalL4maAttention&& other) noexcept;

    // Delete copy constructor and assignment
    MetalL4maAttention(const MetalL4maAttention&) = delete;
    MetalL4maAttention& operator=(const MetalL4maAttention&) = delete;

    void forward(MetalProfileScope profiler,
                 MetalL4maBuffer<T>& buffer,
                 T* attn_output,
                 const T* hidden_states,
                 T* kv_cache_k,
                 T* kv_cache_v);

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    MetalTensor<T> q_proj_weights_;
    MetalTensor<T> k_proj_weights_;
    MetalTensor<T> v_proj_weights_;
    MetalTensor<T> o_proj_weights_;
    metal::batch_prefill_attention::MetalBatchPrefillHandle* attention_handle_;
};

/**
 * @brief Metal decoder layer of the L4MA model
 * Equivalent to L4maDecoderLayer<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maDecoderLayer : public MetalModule<T> {
public:
    explicit MetalL4maDecoderLayer(const L4maConfig& config);

    // Move constructor and assignment
    MetalL4maDecoderLayer(MetalL4maDecoderLayer&& other) noexcept = default;
    MetalL4maDecoderLayer& operator=(MetalL4maDecoderLayer&& other) noexcept = default;

    // Delete copy constructor and assignment
    MetalL4maDecoderLayer(const MetalL4maDecoderLayer&) = delete;
    MetalL4maDecoderLayer& operator=(const MetalL4maDecoderLayer&) = delete;

    void forward(MetalProfileScope profiler,
                 MetalL4maBuffer<T>& buffer,
                 T* hidden_states,
                 T* kv_cache_k,
                 T* kv_cache_v);

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    MetalL4maAttention<T> self_attn_;
    MetalL4maMlp<T> mlp_;
    MetalL4maRMSNorm<T> input_layernorm_;
    MetalL4maRMSNorm<T> post_attention_layernorm_;
};

/**
 * @brief Metal main body of the L4MA model
 * Equivalent to L4maModel<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maModel : public MetalModule<T> {
public:
    explicit MetalL4maModel(const L4maConfig& config);

    // Move constructor and assignment
    MetalL4maModel(MetalL4maModel&& other) noexcept = default;
    MetalL4maModel& operator=(MetalL4maModel&& other) noexcept = default;

    // Delete copy constructor and assignment
    MetalL4maModel(const MetalL4maModel&) = delete;
    MetalL4maModel& operator=(const MetalL4maModel&) = delete;

    void forward(MetalProfileScope profiler,
                 MetalL4maBuffer<T>& buffer,
                 MetalL4maKVCache<T>& kv_cache,
                 T* final_norm_output);

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

    MetalTensor<T>& get_embed_tokens_weight() { return embed_tokens_weight_; }

private:
    L4maConfig config_;
    MetalTensor<T> embed_tokens_weight_;
    std::vector<MetalL4maDecoderLayer<T>> layers_;
    MetalL4maRMSNorm<T> norm_;
};

/**
 * @brief Metal L4MA model with causal language model head
 * Equivalent to L4maForCausalLM<T> from l4ma.cuh
 */
template <typename T>
class MetalL4maForCausalLM : public MetalModule<T> {
public:
    explicit MetalL4maForCausalLM(const L4maConfig& config);

    // Main forward pass - returns top-k values and indices
    std::pair<std::vector<float>, std::vector<int32_t>> forward(
        MetalProfileScope profiler,
        MetalL4maBuffer<T>& buffer,
        MetalL4maKVCache<T>& kv_cache
    );

    std::map<std::string, MetalTensor<T>*> get_parameters() override;

    L4maConfig& get_config() { return config_; }

private:
    L4maConfig config_;
    MetalL4maModel<T> model_;
};

/**
 * @brief Metal model parameter loading utilities
 */
namespace MetalModelUtils {

    /**
     * @brief Load model parameters from checkpoint files
     * Similar to load_model_internal from model.cu
     */
    template<typename T>
    std::unique_ptr<MetalL4maForCausalLM<T>> load_model_internal(
        const AppConfig& config,
        const ModelMetadata& metadata
    );

    /**
     * @brief Copy parameters from host memory to Metal tensors
     */
    template<typename T>
    bool load_parameter_tensor(
        MetalTensor<T>& target_tensor,
        const T* host_data,
        size_t num_elements,
        const std::string& tensor_name
    );

    /**
     * @brief Validate model configuration for Metal backend
     */
    bool validate_model_config(const L4maConfig& config);
}

// Template implementations

template <typename T>
MetalL4maRMSNorm<T>::MetalL4maRMSNorm(const L4maConfig& config) : config_(config) {
    // Initialize weight tensor
    weight_ = MetalTensor<T>({static_cast<size_t>(config_.hidden_size)});
}

template <typename T>
void MetalL4maRMSNorm<T>::forward(T* output, const T* input, int num_tokens, id<MTLCommandBuffer> commandBuffer) {
    // Use existing metal_rmsnorm implementation via namespace
    MetalRMSNorm::rmsnorm(input, weight_.data(), output,
                         num_tokens, config_.hidden_size,
                         config_.rms_norm_eps, commandBuffer);
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maRMSNorm<T>::get_parameters() {
    std::map<std::string, MetalTensor<T>*> params;
    params["weight"] = &weight_;
    return params;
}

template <typename T>
MetalL4maMlp<T>::MetalL4maMlp(const L4maConfig& config) : config_(config) {
    size_t hidden_size = config_.hidden_size;
    size_t intermediate_size = config_.intermediate_size;

    // Initialize weight tensors
    gate_proj_weights_ = MetalTensor<T>({intermediate_size, hidden_size});
    up_proj_weights_ = MetalTensor<T>({intermediate_size, hidden_size});
    down_proj_weights_ = MetalTensor<T>({hidden_size, intermediate_size});
}

template <typename T>
void MetalL4maMlp<T>::forward(MetalProfileScope profiler, MetalL4maBuffer<T>& buffer, T* output, const T* x) {
    size_t hidden_size = config_.hidden_size;
    size_t intermediate_size = config_.intermediate_size;

    // Allocate intermediate tensors from buffer
    auto gate_proj_out = buffer.template allocate<T>(buffer.num_tokens * intermediate_size);
    auto up_proj_out = buffer.template allocate<T>(buffer.num_tokens * intermediate_size);

    auto& context = MetalContext::getInstance();

    // 1. Gate and Up projections using GEMM namespace
    MetalGEMM::gemm(buffer.commandBuffer, x, up_proj_weights_.data(), static_cast<const T*>(nullptr), up_proj_out.data(),
                    buffer.num_tokens, intermediate_size, hidden_size,
                    nullptr, 0, false, true);
    profiler.record("up_projection");

    MetalGEMM::gemm(buffer.commandBuffer, x, gate_proj_weights_.data(), static_cast<const T*>(nullptr), gate_proj_out.data(),
                    buffer.num_tokens, intermediate_size, hidden_size,
                    nullptr, 0, false, true);
    profiler.record("gate_projection");

    // 2. SiLU activation and multiplication
    MetalSiLUMul::silu_and_mul(gate_proj_out.data(), up_proj_out.data(),
                               up_proj_out.data(), // reuse for result
                               buffer.num_tokens, intermediate_size, buffer.commandBuffer);
    profiler.record("silu_and_mul");

    // 3. Down projection
    MetalGEMM::gemm(buffer.commandBuffer, up_proj_out.data(), down_proj_weights_.data(), static_cast<const T*>(nullptr), output,
                    buffer.num_tokens, hidden_size, intermediate_size,
                    nullptr, 0, false, true);
    profiler.record("down_projection");

    // Deallocate intermediate tensors
    buffer.deallocate(gate_proj_out);
    buffer.deallocate(up_proj_out);
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maMlp<T>::get_parameters() {
    std::map<std::string, MetalTensor<T>*> params;
    params["gate_proj.weight"] = &gate_proj_weights_;
    params["up_proj.weight"] = &up_proj_weights_;
    params["down_proj.weight"] = &down_proj_weights_;
    return params;
}

template <typename T>
MetalL4maAttention<T>::MetalL4maAttention(const L4maConfig& config) : config_(config) {
    size_t hidden_size = config_.hidden_size;
    size_t num_query_heads = config_.num_query_heads;
    size_t num_key_value_heads = config_.num_key_value_heads;
    size_t head_size = config_.head_size;

    // Initialize weight tensors
    q_proj_weights_ = MetalTensor<T>({num_query_heads * head_size, hidden_size});
    k_proj_weights_ = MetalTensor<T>({num_key_value_heads * head_size, hidden_size});
    v_proj_weights_ = MetalTensor<T>({num_key_value_heads * head_size, hidden_size});
    o_proj_weights_ = MetalTensor<T>({hidden_size, num_query_heads * head_size});

    // Create batch prefill attention handle
    attention_handle_ = metal::batch_prefill_attention::metal_batch_prefill_create_handle(
        1024,  // max_batch_size
        8192,  // max_seq_length
        static_cast<int>(num_query_heads),
        static_cast<int>(num_query_heads * head_size)  // max_head_dim
    );
}

template <typename T>
MetalL4maAttention<T>::~MetalL4maAttention() {
    if (attention_handle_) {
        metal::batch_prefill_attention::metal_batch_prefill_destroy_handle(attention_handle_);
        attention_handle_ = nullptr;
    }
}

template <typename T>
MetalL4maAttention<T>::MetalL4maAttention(MetalL4maAttention&& other) noexcept
    : config_(std::move(other.config_))
    , q_proj_weights_(std::move(other.q_proj_weights_))
    , k_proj_weights_(std::move(other.k_proj_weights_))
    , v_proj_weights_(std::move(other.v_proj_weights_))
    , o_proj_weights_(std::move(other.o_proj_weights_))
    , attention_handle_(other.attention_handle_) {
    other.attention_handle_ = nullptr;
}

template <typename T>
MetalL4maAttention<T>& MetalL4maAttention<T>::operator=(MetalL4maAttention&& other) noexcept {
    if (this != &other) {
        // Clean up current attention handle
        if (attention_handle_) {
            metal::batch_prefill_attention::metal_batch_prefill_destroy_handle(attention_handle_);
        }

        // Move data
        config_ = std::move(other.config_);
        q_proj_weights_ = std::move(other.q_proj_weights_);
        k_proj_weights_ = std::move(other.k_proj_weights_);
        v_proj_weights_ = std::move(other.v_proj_weights_);
        o_proj_weights_ = std::move(other.o_proj_weights_);
        attention_handle_ = other.attention_handle_;

        // Clear other's handle
        other.attention_handle_ = nullptr;
    }
    return *this;
}

template <typename T>
void MetalL4maAttention<T>::forward(MetalProfileScope profiler, MetalL4maBuffer<T>& buffer, T* attn_output,
                                   const T* hidden_states, T* kv_cache_k, T* kv_cache_v) {
    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;
    size_t num_query_heads = config_.num_query_heads;
    size_t num_key_value_heads = config_.num_key_value_heads;
    size_t head_size = config_.head_size;

    // Allocate Q, K, V projection outputs
    auto q_proj = buffer.template allocate<T>(num_tokens * num_query_heads * head_size);
    auto k_proj = buffer.template allocate<T>(num_tokens * num_key_value_heads * head_size);
    auto v_proj = buffer.template allocate<T>(num_tokens * num_key_value_heads * head_size);

    auto& context = MetalContext::getInstance();

    // 1. Q, K, V projections using GEMM namespace
    MetalGEMM::gemm(buffer.commandBuffer, hidden_states, q_proj_weights_.data(), static_cast<const T*>(nullptr), q_proj.data(),
                    num_tokens, num_query_heads * head_size, hidden_size,
                    nullptr, 0, false, true);
    profiler.record("q_projection");

    MetalGEMM::gemm(buffer.commandBuffer, hidden_states, k_proj_weights_.data(), static_cast<const T*>(nullptr), k_proj.data(),
                    num_tokens, num_key_value_heads * head_size, hidden_size,
                    nullptr, 0, false, true);
    profiler.record("k_projection");

    MetalGEMM::gemm(buffer.commandBuffer, hidden_states, v_proj_weights_.data(), static_cast<const T*>(nullptr), v_proj.data(),
                    num_tokens, num_key_value_heads * head_size, hidden_size,
                    nullptr, 0, false, true);
    profiler.record("v_projection");

    // 2. Apply RoPE to Q and K using RoPE namespace
    MetalRoPE::rope_inplace(q_proj.data(),
                           reinterpret_cast<const int32_t*>(buffer.position_ids.data()),
                           num_tokens, num_query_heads, head_size,
                           config_.rope_theta, config_.rope_factor);

    MetalRoPE::rope_inplace(k_proj.data(),
                           reinterpret_cast<const int32_t*>(buffer.position_ids.data()),
                           num_tokens, num_key_value_heads, head_size,
                           config_.rope_theta, config_.rope_factor);
    profiler.record("apply_rope");

    // 3. Batch prefill attention with handle-based API
    auto context_out = buffer.template allocate<T>(num_tokens * num_query_heads * head_size);

    // Get workspace requirements
    auto workspace = metal::batch_prefill_attention::metal_batch_prefill_get_workspace(
        attention_handle_,
        static_cast<int>(num_tokens),
        static_cast<int>(config_.num_query_heads * head_size),    // head_dim
        static_cast<int>(config_.num_key_value_heads * head_size), // kv_head_dim
        static_cast<int>(buffer.page_size),
        static_cast<int>(buffer.kv_page_indices.size())
    );

    // Allocate workspace buffer
    auto& metalContext = MetalContext::getInstance();
    id<MTLBuffer> workspace_buffer = [metalContext.getDevice() newBufferWithLength:workspace.total_size
                                                                            options:MTLResourceStorageModeShared];

    if constexpr (std::is_same_v<T, float>) {
        metal::batch_prefill_attention::batch_prefill_attention_unified_f32(
            attention_handle_,
            [workspace_buffer contents],
            [workspace_buffer length],
            reinterpret_cast<const float*>(q_proj.data()),
            reinterpret_cast<const float*>(kv_cache_k),
            reinterpret_cast<const float*>(kv_cache_v),
            reinterpret_cast<const int32_t*>(buffer.qo_indptr.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_page_indptr.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_page_indices.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_last_page_lens.data()),
            reinterpret_cast<float*>(context_out.data()),
            static_cast<int>(num_tokens),
            static_cast<int>(config_.num_query_heads * head_size),    // head_dim
            static_cast<int>(config_.num_key_value_heads * head_size), // kv_head_dim
            static_cast<int>(head_size),
            static_cast<int>(buffer.page_size),
            static_cast<int>(config_.num_query_heads),
            static_cast<int>(config_.num_key_value_heads),
            1.0f / std::sqrt(float(head_size)), // scale
            static_cast<int>(buffer.kv_page_indices.size())
        );
    } else {
        // For non-float types, use bf16 version
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            attention_handle_,
            [workspace_buffer contents],
            [workspace_buffer length],
            q_proj.data(), kv_cache_k, kv_cache_v,
            reinterpret_cast<const int32_t*>(buffer.qo_indptr.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_page_indptr.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_page_indices.data()),
            reinterpret_cast<const int32_t*>(buffer.kv_last_page_lens.data()),
            context_out.data(),
            static_cast<int>(num_tokens),
            static_cast<int>(config_.num_query_heads * head_size),    // head_dim
            static_cast<int>(config_.num_key_value_heads * head_size), // kv_head_dim
            static_cast<int>(head_size),
            static_cast<int>(buffer.page_size),
            static_cast<int>(config_.num_query_heads),
            static_cast<int>(config_.num_key_value_heads),
            1.0f / std::sqrt(float(head_size)), // scale
            static_cast<int>(buffer.kv_page_indices.size())
        );
    }
    profiler.record("batch_prefill_attention");

    // 4. Output projection
    MetalGEMM::gemm(buffer.commandBuffer, context_out.data(), o_proj_weights_.data(), static_cast<const T*>(nullptr), attn_output,
                    num_tokens, hidden_size, num_query_heads * head_size,
                    nullptr, 0, false, true);
    profiler.record("output_projection");

    // Deallocate intermediate tensors
    buffer.deallocate(q_proj);
    buffer.deallocate(k_proj);
    buffer.deallocate(v_proj);
    buffer.deallocate(context_out);
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maAttention<T>::get_parameters() {
    std::map<std::string, MetalTensor<T>*> params;
    params["q_proj.weight"] = &q_proj_weights_;
    params["k_proj.weight"] = &k_proj_weights_;
    params["v_proj.weight"] = &v_proj_weights_;
    params["o_proj.weight"] = &o_proj_weights_;
    return params;
}

template <typename T>
MetalL4maDecoderLayer<T>::MetalL4maDecoderLayer(const L4maConfig& config)
    : config_(config)
    , self_attn_(config)
    , mlp_(config)
    , input_layernorm_(config)
    , post_attention_layernorm_(config) {
}

template <typename T>
void MetalL4maDecoderLayer<T>::forward(MetalProfileScope profiler, MetalL4maBuffer<T>& buffer,
                                      T* hidden_states, T* kv_cache_k, T* kv_cache_v) {
    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;

    // Allocate intermediate tensors
    auto normed_hidden_states = buffer.template allocate<T>(num_tokens * hidden_size);
    auto attn_output = buffer.template allocate<T>(num_tokens * hidden_size);
    auto normed_attn_output = buffer.template allocate<T>(num_tokens * hidden_size);
    auto mlp_output = buffer.template allocate<T>(num_tokens * hidden_size);

    // 1. Input layer normalization
    input_layernorm_.forward(normed_hidden_states.data(), hidden_states, num_tokens, buffer.commandBuffer);
    profiler.record("input_layernorm");

    // 2. Self attention
    self_attn_.forward(profiler.scope("self_attention"), buffer, attn_output.data(),
                      normed_hidden_states.data(), kv_cache_k, kv_cache_v);

    // 3. Residual connection (add attention output to original hidden states)
    // TODO: Use metal_add_residual when implemented
    for (size_t i = 0; i < num_tokens * hidden_size; ++i) {
        hidden_states[i] += attn_output.data()[i];
    }
    profiler.record("attention_residual");

    // 4. Post attention layer normalization
    post_attention_layernorm_.forward(normed_attn_output.data(), hidden_states, num_tokens, buffer.commandBuffer);
    profiler.record("post_attention_layernorm");

    // 5. MLP
    mlp_.forward(profiler.scope("mlp"), buffer, mlp_output.data(), normed_attn_output.data());

    // 6. Residual connection (add MLP output to hidden states)
    // TODO: Use metal_add_residual when implemented
    for (size_t i = 0; i < num_tokens * hidden_size; ++i) {
        hidden_states[i] += mlp_output.data()[i];
    }
    profiler.record("mlp_residual");

    // Deallocate intermediate tensors
    buffer.deallocate(normed_hidden_states);
    buffer.deallocate(attn_output);
    buffer.deallocate(normed_attn_output);
    buffer.deallocate(mlp_output);
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maDecoderLayer<T>::get_parameters() {
    std::map<std::string, MetalTensor<T>*> params;

    // Get parameters from sub-modules
    auto attn_params = self_attn_.get_parameters();
    auto mlp_params = mlp_.get_parameters();
    auto input_norm_params = input_layernorm_.get_parameters();
    auto post_norm_params = post_attention_layernorm_.get_parameters();

    // Add with prefixes
    for (auto& [name, tensor] : attn_params) {
        params["self_attn." + name] = tensor;
    }
    for (auto& [name, tensor] : mlp_params) {
        params["mlp." + name] = tensor;
    }
    for (auto& [name, tensor] : input_norm_params) {
        params["input_layernorm." + name] = tensor;
    }
    for (auto& [name, tensor] : post_norm_params) {
        params["post_attention_layernorm." + name] = tensor;
    }

    return params;
}

// Additional template implementations would continue here...
// For brevity, I'll provide the key structural elements