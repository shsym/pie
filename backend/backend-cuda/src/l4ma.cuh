#pragma once

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

#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <optional>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <vector>
#include <memory>

constexpr int PAGE_SIZE = 16; // Or get from config

struct L4maConfig
{
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    bool use_qkv_bias;
    float rms_norm_eps;
    int vocab_size;
    int pad_token_id;
    int num_hidden_layers;
    // Add any other config parameters used

    // Helper for head_dim
    int head_dim() const { return hidden_size / num_attention_heads; }
};

template <typename T = float>
class L4maMlp
{
public:
    L4maMlp(const L4maConfig &config,
            const thrust::device_vector<T> &gate_proj_weights,
            const thrust::device_vector<T> &up_proj_weights,
            const thrust::device_vector<T> &down_proj_weights,
            std::optional<thrust::device_vector<T>> gate_proj_bias = std::nullopt,
            std::optional<thrust::device_vector<T>> up_proj_bias = std::nullopt,
            std::optional<thrust::device_vector<T>> down_proj_bias = std::nullopt);
    ~L4maMlp();

    // x: [num_tokens, hidden_size]
    // output: [num_tokens, hidden_size]
    // temp_buffer_mlp: caller-provided temporary buffer for intermediate results
    void forward(thrust::device_vector<T> &output,
                 const thrust::device_vector<T> &x,
                 int num_tokens,
                 thrust::device_vector<T> &temp_buffer_mlp,
                 cudaStream_t stream);

private:
    const L4maConfig &config_;
    thrust::device_vector<float> gate_proj_weights_;
    thrust::device_vector<float> up_proj_weights_;
    thrust::device_vector<float> down_proj_weights_;
    std::optional<thrust::device_vector<float>> gate_proj_bias_;
    std::optional<thrust::device_vector<float>> up_proj_bias_;
    std::optional<thrust::device_vector<float>> down_proj_bias_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
};

template <typename T = float>
class L4maAttention
{
public:
    L4maAttention(const L4maConfig &config, int layer_idx,
                  const thrust::device_vector<T> &q_proj_weights,
                  const thrust::device_vector<T> &k_proj_weights,
                  const thrust::device_vector<T> &v_proj_weights,
                  const thrust::device_vector<T> &o_proj_weights,
                  const thrust::device_vector<T> &q_proj_bias = thrust::device_vector<T>(),
                  const thrust::device_vector<T> &k_proj_bias = thrust::device_vector<T>(),
                  const thrust::device_vector<T> &v_proj_bias = thrust::device_vector<T>());
    ~L4maAttention();

    void forward(
        thrust::device_vector<T> &attn_output, // [nnz, hidden_size]
        void *handler,                         // placeholder for attention handler
        const thrust::device_vector<T> &hidden_states,
        const int32_t *position_ids,
        thrust::device_vector<T> &kv_cache_for_layer_k, // K cache pages for this layer
        thrust::device_vector<T> &kv_cache_for_layer_v, // V cache pages for this layer
        const int32_t *kv_page_indices,
        const int32_t *kv_page_indptr,
        const int32_t *kv_last_page_lens,
        const int32_t *qo_indptr,
        int nnz,                                    // total number of tokens
        int batch_size,                             // from qo_indptr or kv_page_indptr
        thrust::device_vector<T> &temp_buffer_attn, // For intermediate Q, K, V, rope outputs etc.
        cudaStream_t stream);

private:
    const L4maConfig &config_;
    int layer_idx_;
    thrust::device_vector<T> q_proj_weights_, k_proj_weights_, v_proj_weights_, o_proj_weights_;
    thrust::device_vector<T> q_proj_bias_, k_proj_bias_, v_proj_bias_;
};

template <typename T = float>
class L4maDecoderLayer
{
public:
    L4maDecoderLayer(const L4maConfig &config, int layer_idx /*, weight pointers */);
    ~L4maDecoderLayer();

    void forward(
        thrust::device_vector<T> &output_hidden_states,      // [nnz, hidden_size]
        void *handler,                                       // placeholder for attention handler
        const thrust::device_vector<T> &input_hidden_states, // [nnz, hidden_size]
        const int32_t *position_ids,
        thrust::device_vector<T> &kv_cache_for_layer_k,
        thrust::device_vector<T> &kv_cache_for_layer_v,
        const int32_t *kv_page_indices,
        const int32_t *kv_page_indptr,
        const int32_t *kv_last_page_lens,
        const int32_t *qo_indptr,
        int nnz,
        int batch_size,
        thrust::device_vector<T> &temp_buffer_layer, // Sufficiently large temporary buffer for this layer
        cudaStream_t stream);

private:
    const L4maConfig &config_;
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;

    // RMSNorm parameters (eps is in config)
    // No explicit weights for RMSNorm in the Python means they are implicitly 1.0 or part of fused ops.
    // If RMSNorm has learnable weights, they need to be added.

    // Temporary storage for intermediate results within the layer
    thrust::device_vector<T> residual_;
    thrust::device_vector<T> normed_hidden_states_;
    thrust::device_vector<T> attn_output_;
};

template <typename T = float>
class L4maModel
{
public:
    L4maModel(const L4maConfig &config /* paths to weights or pre-loaded weight pointers */);
    ~L4maModel();

    // input_embeds: [nnz, hidden_size]
    // position_ids: [nnz]
    // kv_cache_ptr: Pointer to the entire KV cache buffer (all layers)
    // kv_page_indices, kv_page_indptr, kv_last_page_lens, qo_indptr, custom_mask: As in Python
    // single_token_inference_mode: bool
    // output_hidden_states: [nnz, hidden_size] (output buffer)
    void forward(
        thrust::device_vector<T> &output_hidden_states,
        const thrust::device_vector<T> &input_embeds,
        const int32_t *position_ids,
        thrust::device_vector<T> &kv_cache_ptr_k, // Full K cache for all layers
        thrust::device_vector<T> &kv_cache_ptr_v, // Full V cache for all layers
        const int32_t *kv_page_indices,           // [total_num_pages_for_batch_across_layers] or per layer? Python suggests one set for all layers
        const int32_t *kv_page_indptr,            // [num_layers, batch_size + 1] or [batch_size+1]? Python seems to pass one.
        const int32_t *kv_last_page_lens,         // [num_layers, batch_size] or [batch_size]?
        const int32_t *qo_indptr,                 // [batch_size + 1]
        const float *custom_mask,                 // [num_qo_heads, max_qo_len, max_kv_len] - or format FlashInfer expects
        bool single_token_inference_mode,
        int nnz,        // total number of query tokens
        int batch_size, // can be derived from qo_indptr
        int max_qo_len, // for custom_mask if prefill
        int max_kv_len, // for custom_mask if prefill
        cudaStream_t stream);

private:
    const L4maConfig &config_;
    std::vector<L4maDecoderLayer<T>> layers_;

    uint8_t *d_workspace_buffer_;
    size_t workspace_buffer_size_ = 128 * 1024 * 1024; // 128 MiB

    thrust::device_vector<T> current_hidden_states_;
};

// --- L4maMlp<T> template method definitions ---

// Helper for SiLU activation (device function)
__device__ __forceinline__ float silu(const float &val);

template <typename T>
void silu_and_mul(
    thrust::device_vector<T> &out,
    const thrust::device_vector<T> &input,
    int num_tokens,
    int d,
    cudaStream_t stream,
    bool enable_pdl = false);

// Helper for GEMM with bias using cublasLt
inline void gemm_bias_cublasLt(
    cublasLtHandle_t ltHandle,
    cudaStream_t stream,
    const void *A,    // [m, k]
    const void *B,    // [n, k] (will be transposed)
    const void *bias, // [n] or nullptr
    void *C,          // [m, n]
    int m, int n, int k,
    cudaDataType_t dtype,           // CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF
    cublasComputeType_t computeType // CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_16BF
)
{
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    float alpha = 1.0f, beta = 0.0f;
    cublasLtEpilogue_t epilogue = bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, dtype));
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (bias)
    {
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, dtype, m, k, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, dtype, n, k, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, dtype, m, n, m));

    CUBLAS_CHECK(cublasLtMatmul(
        ltHandle,
        matmulDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        C, Cdesc,
        C, Cdesc,
        nullptr, nullptr, 0, stream));

    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}

template <typename T>
L4maMlp<T>::L4maMlp(const L4maConfig &config,
                    const thrust::device_vector<T> &gate_proj_weights,
                    const thrust::device_vector<T> &up_proj_weights,
                    const thrust::device_vector<T> &down_proj_weights,
                    std::optional<thrust::device_vector<T>> gate_proj_bias,
                    std::optional<thrust::device_vector<T>> up_proj_bias,
                    std::optional<thrust::device_vector<T>> down_proj_bias)
    : config_(config),
      gate_proj_weights_(gate_proj_weights),
      up_proj_weights_(up_proj_weights),
      down_proj_weights_(down_proj_weights),
      gate_proj_bias_(std::move(gate_proj_bias)),
      up_proj_bias_(std::move(up_proj_bias)),
      down_proj_bias_(std::move(down_proj_bias))
{
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
}

template <typename T>
L4maMlp<T>::~L4maMlp()
{
    cublasDestroy(cublas_handle_);
    cublasLtDestroy(cublaslt_handle_);
}

template <typename T>
void L4maMlp<T>::forward(
    thrust::device_vector<T> &output,
    const thrust::device_vector<T> &x,
    int num_tokens,
    thrust::device_vector<T> &temp_buffer_mlp,
    cudaStream_t stream)
{
    // Shapes:
    // x: [num_tokens, hidden_size]
    // gate_proj_weights_: [intermediate_size, hidden_size]
    // up_proj_weights_: [intermediate_size, hidden_size]
    // down_proj_weights_: [hidden_size, intermediate_size]
    // output: [num_tokens, hidden_size]
    // temp_buffer_mlp: must be at least 2 * num_tokens * intermediate_size + num_tokens * hidden_size

    assert(x.size() == num_tokens * config_.hidden_size);
    assert(output.size() == num_tokens * config_.hidden_size);
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;

    const T *x_ptr = thrust::raw_pointer_cast(x.data());
    T *gate_proj_ptr = thrust::raw_pointer_cast(temp_buffer_mlp.data()); // [num_tokens, intermediate_size]
    T *up_proj_ptr = gate_proj_ptr + num_tokens * is;                    // [num_tokens, intermediate_size]
    T *silu_ptr = up_proj_ptr + num_tokens * is;                         // [num_tokens, intermediate_size]
    T *mul_ptr = silu_ptr;                                               // reuse silu_ptr for mul result
    T *out_ptr = thrust::raw_pointer_cast(output.data());

    // 1. gate_proj = x * W_g^T (+ b_g)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        x_ptr, thrust::raw_pointer_cast(gate_proj_weights_.data()),
        gate_proj_bias_ ? thrust::raw_pointer_cast(gate_proj_bias_->data()) : nullptr,
        gate_proj_ptr,
        num_tokens, is, hs,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);

    // 2. up_proj = x * W_u^T (+ b_u)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        x_ptr, thrust::raw_pointer_cast(up_proj_weights_.data()),
        up_proj_bias_ ? thrust::raw_pointer_cast(up_proj_bias_->data()) : nullptr,
        up_proj_ptr,
        num_tokens, is, hs,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);

    // 3+4. Fused SiLU activation and elementwise multiply using flashinfer kernel
    silu_and_mul<T>(
        temp_buffer_mlp, // out: silu_ptr (same as mul_ptr)
        temp_buffer_mlp, // input: concat(gate_proj_ptr, up_proj_ptr)
        num_tokens,
        is,
        stream);

    // 5. Down projection: output = mul_ptr * W_d^T (+ b_d)
    gemm_bias_cublasLt(
        cublaslt_handle_, stream,
        mul_ptr, thrust::raw_pointer_cast(down_proj_weights_.data()),
        down_proj_bias_ ? thrust::raw_pointer_cast(down_proj_bias_->data()) : nullptr,
        out_ptr,
        num_tokens, hs, is,
        CUDA_R_32F, CUBLAS_COMPUTE_32F);
}