#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h> // <--- Changed header
#include <thrust/functional.h>
#include <cuda_bf16.h>
#include "common.cuh"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include "flashinfer/norm.cuh"
// Forward Declarations
struct L4maConfig;
template <typename T>
class L4maMlp;
template <typename T>
class L4maAttention;
template <typename T>
class L4maDecoderLayer;
template <typename T>
class L4maModel;

// Helper structure to convert a float to a __nv_bfloat16 for initialization.
// This uses the CUDA library's conversion functions.
// Helper structure to convert a float to a __nv_bfloat16 for initialization.
// This uses the CUDA library's conversion functions.
struct float_to_bfloat16
{
    __host__ __device__
        __nv_bfloat16
        operator()(const float &f) const
    {
        return __float2bfloat16(f);
    }
};

/**
 * @brief Computes the mean of a thrust::device_vector of __nv_bfloat16.
 *
 * @param d_vec The input vector of __nv_bfloat16 elements stored on the GPU.
 */
inline void compute_bfloat16_mean(const thrust::device_vector<__nv_bfloat16> &d_vec)
{
    if (d_vec.empty())
    {
        std::cout << "Vector is empty. Mean is 0." << std::endl;
        return;
    }

    // --- Step 1: Compute the sum of all elements using transform_reduce ---

    // Define a lambda function for the Unary Transform Operation.
    // This lambda will be applied to each element of the input vector.
    auto bfloat16_to_float = [] __host__ __device__(const __nv_bfloat16 &x) -> float {
        return __bfloat162float(x);
    };

    // Use thrust::transform_reduce.
    // 1. d_vec.begin(), d_vec.end(): Input iterator range.
    // 2. bfloat16_to_float: The Unary Transform operation.
    // 3. 0.0f: The initial value for the sum (a float).
    // 4. thrust::plus<float>(): The Binary Reduction operation (adds two floats).
    float sum = thrust::transform_reduce(
        d_vec.begin(),
        d_vec.end(),
        bfloat16_to_float,
        0.0f,
        thrust::plus<float>());

    // --- Step 2: Compute the mean ---
    float mean = sum / d_vec.size();

    // --- Step 3: Print the final result ---
    std::cout << "Computed Mean: " << mean << std::endl;
}
// Constants
constexpr int PAGE_SIZE = 16;

// Model Configuration
struct L4maConfig
{
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    bool use_qkv_bias;
    float rms_norm_eps;
    int vocab_size;
    int pad_token_id;
    float rope_base = 10000.0f;

    int head_dim() const { return hidden_size / num_attention_heads; }
    void print() const;
};

L4maConfig load_l4ma_config_from_yaml(const std::string &yaml_path);

/**
 * @class L4maMlp
 * @brief Implements the SwiGLU MLP module.
 */
template <typename T>
class L4maMlp
{
public:
    L4maMlp(const L4maConfig &config,
            const thrust::device_vector<T> &gate_proj_weights,
            const thrust::device_vector<T> &up_proj_weights,
            const thrust::device_vector<T> &down_proj_weights);
    ~L4maMlp();

    void forward(thrust::device_vector<T> &output,
                 const thrust::device_vector<T> &x,
                 int num_tokens,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

private:
    L4maConfig config_;
    thrust::device_vector<T> gate_proj_weights_;
    thrust::device_vector<T> up_proj_weights_;
    thrust::device_vector<T> down_proj_weights_;
};

/**
 * @class L4maAttention
 * @brief Implements paged multi-head attention with RoPE.
 */
template <typename T>
class L4maAttention
{
public:
    L4maAttention(const L4maConfig &config,
                  const thrust::device_vector<T> &q_proj_weights,
                  const thrust::device_vector<T> &k_proj_weights,
                  const thrust::device_vector<T> &v_proj_weights,
                  const thrust::device_vector<T> &o_proj_weights);
    ~L4maAttention();

    void forward(thrust::device_vector<T> &attn_output,
                 const thrust::device_vector<T> &hidden_states,
                 const int32_t *position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int nnz,
                 int batch_size,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

    void simple_forward(const thrust::device_vector<T> &input)
    {
        // Only project Q, K, V using gemm_cublasLt
        int batch = input.size() / config_.hidden_size;
        int hs = config_.hidden_size;
        int nq = config_.num_attention_heads;
        int nkv = config_.num_key_value_heads;
        int hd = config_.head_dim();

        // Print batch, hs, nq, ...
        std::cout << "Batch: " << batch << ", HS: " << hs
                  << ", NQ: " << nq << ", NKV: " << nkv
                  << ", HD: " << hd << std::endl;

        // Allocate output buffers for Q, K, V
        thrust::device_vector<T> q_proj(batch * nq * hd);
        thrust::device_vector<T> k_proj(batch * nkv * hd);
        thrust::device_vector<T> v_proj(batch * nkv * hd);
        thrust::device_vector<char> workspace(1024 * 1024 * 128);

        // No bias for projections
        const thrust::device_vector<T> *no_bias = nullptr;
        cublasLtHandle_t ltHandle;
        cublasLtCreate(&ltHandle);
        cudaStream_t stream = 0; // Default stream

        compute_bfloat16_mean(input);

        // Q projection: [batch, nq*hd] = [batch, hs] x [hs, nq*hd]^T
        // gemm_cublasLt2<T>(ltHandle, stream,
        //                   thrust::raw_pointer_cast(input.data()),
        //                   thrust::raw_pointer_cast(q_proj_weights_.data()),
        //                   thrust::raw_pointer_cast(q_proj.data()),
        //                   batch, nq * hd, hs, false, true, false);

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        multiply_bf16_cublas(handle,
                             thrust::raw_pointer_cast(input.data()),
                             thrust::raw_pointer_cast(q_proj_weights_.data()),
                             thrust::raw_pointer_cast(q_proj.data()),
                             batch, nq * hd, hs, false, true);

        compute_bfloat16_mean(q_proj);
        
        gemm_cublasLt2<__nv_bfloat16>(
                    ltHandle, stream,
                    thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(q_proj_weights_.data()),
                    thrust::raw_pointer_cast(q_proj.data()),
                    batch, nq * hd, hs, false, true);


        // gemm_cublasLt<__nv_bfloat16>(
        //     ltHandle, stream,
        //     input,
        //     q_proj_weights_,
        //     no_bias,
        //     q_proj,
        //     batch, nq * hd, hs, workspace,
        //     false, false);

        compute_bfloat16_mean(q_proj);

        cublasLtDestroy(ltHandle);
    }

private:
    L4maConfig config_;
    thrust::device_vector<T> q_proj_weights_, k_proj_weights_, v_proj_weights_, o_proj_weights_;
};

/**
 * @class L4maDecoderLayer
 * @brief A single transformer decoder layer.
 */
template <typename T>
class L4maDecoderLayer
{
public:
    L4maDecoderLayer(const L4maConfig &config, const std::unordered_map<std::string, thrust::device_vector<T>> &weights);
    ~L4maDecoderLayer();

    void forward(thrust::device_vector<T> &hidden_states, // In-place
                 const int32_t *position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int nnz,
                 int batch_size,
                 thrust::device_vector<T> &temp_buffer,
                 cublasLtHandle_t ltHandle,
                 cudaStream_t stream);

    void simple_forward(const thrust::device_vector<T> &input)
    {
        thrust::device_vector<T> out1(input.size());

        uint32_t batch_size = input.size() / config_.hidden_size;
        uint32_t stride = config_.hidden_size;
        uint32_t d = config_.hidden_size;

        flashinfer::norm::RMSNorm<T>(
            const_cast<T *>(thrust::raw_pointer_cast(input.data())),
            const_cast<T *>(thrust::raw_pointer_cast(input_layernorm_weight_.data())),
            thrust::raw_pointer_cast(out1.data()),
            batch_size, d, stride, stride, config_.rms_norm_eps);

        // compute_bfloat16_mean(out1);

        self_attn_.simple_forward(out1);
    }

private:
    L4maConfig config_;
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;

    thrust::device_vector<T> input_layernorm_weight_;
    thrust::device_vector<T> post_attention_layernorm_weight_;

    // Persistent buffers to avoid reallocation
    thrust::device_vector<T> residual_;
    thrust::device_vector<T> normed_hidden_states_;
};

/**
 * @class L4maModel
 * @brief The main L4MA transformer model.
 */
template <typename T>
class L4maModel
{
public:
    static L4maModel<T> from_files(const std::string &yaml_path, const std::string &ztensor_path);

    L4maModel(const L4maConfig &config, const std::unordered_map<std::string, thrust::device_vector<T>> &all_weights);
    // New constructor for per-layer weights
    L4maModel(const L4maConfig &config,
              const std::unordered_map<std::string, thrust::device_vector<T>> &global_weights,
              const std::vector<std::unordered_map<std::string, thrust::device_vector<T>>> &layer_weights_vec);
    ~L4maModel();

    // The forward pass now takes token IDs and produces logits
    void forward(thrust::device_vector<float> &logits,
                 const thrust::device_vector<int32_t> &input_ids,
                 const thrust::device_vector<int32_t> &position_ids,
                 thrust::device_vector<T> &kv_cache_k,
                 thrust::device_vector<T> &kv_cache_v,
                 const int32_t *kv_page_indices,
                 const int32_t *kv_page_indptr,
                 const int32_t *kv_last_page_lens,
                 const int32_t *qo_indptr,
                 int batch_size,
                 cudaStream_t stream);

    L4maConfig get_config() const { return config_; }

    void embed_input_ids(const thrust::device_vector<uint32_t> &input_ids,
                         thrust::device_vector<T> &output,
                         cudaStream_t stream = 0)
    {
        int embedding_dim = config_.hidden_size;
        // Resize output to hold the embeddings
        // Call the embed function (from common.cuh)
        embed(
            embedding_weights_,
            input_ids,
            &output,
            embedding_dim,
            stream);

        layers_.front().simple_forward(output);
        // // create a copy of output
        // thrust::device_vector<T> out1(output.size());

        // // first item of layer
        // layers_.front().

        // flashinfer::norm::RMSNorm<T> norm(config_.hidden_size, config_.rms_norm_eps);
    }

private:
    L4maConfig config_;
    cublasLtHandle_t cublaslt_handle_;

    // Model weights
    thrust::device_vector<T> embedding_weights_;
    thrust::device_vector<T> lm_head_weights_;
    thrust::device_vector<T> final_norm_weight_;

    std::vector<L4maDecoderLayer<T>> layers_;

    // Persistent buffers
    thrust::device_vector<T> hidden_states_;
    thrust::device_vector<T> temp_bwd_buffer_;
};