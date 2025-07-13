#include "common.cuh"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include <limits> // Required for std::numeric_limits

/**
 * @brief High-performance CUDA kernel for embedding lookup.
 *
 * This version is specialized for uint32_t indices and uses 128-bit
 * vectorized memory operations for maximum bandwidth.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
template <typename T>
__global__ void embedding_lookup_kernel_128bit(T *output,
                                               const T *embedding_matrix,
                                               const uint32_t *indices,
                                               int n,
                                               int hidden_dim_div_16)
{
    // Each block processes one lookup index.
    int idx_n = blockIdx.x;
    if (idx_n >= n)
    {
        return;
    }

    // Use shared memory to broadcast the source row index for the block.
    __shared__ uint32_t source_row_idx;
    if (threadIdx.x == 0)
    {
        source_row_idx = indices[idx_n];
    }
    __syncthreads();

    // Cast pointers to a 128-bit type (float4) to perform 16-byte memory transfers.
    // This is the core optimization, significantly increasing memory throughput.
    const float4 *source_row_ptr = reinterpret_cast<const float4 *>(embedding_matrix) + (long long)source_row_idx * hidden_dim_div_16;
    float4 *dest_row_ptr = reinterpret_cast<float4 *>(output) + (long long)idx_n * hidden_dim_div_16;

    // Use a grid-stride loop for threads to collectively copy the entire row.
    // This ensures that all data is copied regardless of the number of threads per block.
    for (int i = threadIdx.x; i < hidden_dim_div_16; i += blockDim.x)
    {
        dest_row_ptr[i] = source_row_ptr[i];
    }
}

/**
 * @brief Host-side launch function with a Thrust-based API.
 *
 * This function provides a clean, library-style interface using thrust::device_vector
 * and an explicit CUDA stream for asynchronous execution. It assumes uint32_t indices.
 *
 * @tparam T The base data type (float, __half, etc.).
 */
/**
 * @brief Host-side launch function for embedding lookup with a raw pointer API.
 */
template <typename T>
void embed(
    const T* embedding,
    size_t embedding_num_rows,
    const uint32_t* indices,
    size_t num_indices,
    T* result,
    int embed_width,
    cudaStream_t stream)
{
    // --- Input Validation ---
    if (embedding_num_rows == 0 || num_indices == 0) return;
    if ((embedding_num_rows * embed_width) == 0) return;
    if ((embedding_num_rows * embed_width) % embed_width != 0) {
        throw std::invalid_argument("Embedding vector size is not divisible by the embed_width.");
    }
    if ((embed_width * sizeof(T)) % 16 != 0) {
        throw std::invalid_argument("Total byte size of a slice (embed_width * sizeof(T)) must be a multiple of 16.");
    }

    // --- Prepare Parameters ---
    // The result buffer is assumed to be pre-allocated by the caller.
    const int threads_per_block = 256;
    const int hidden_dim_div_16 = (embed_width * sizeof(T)) / 16;

    dim3 blocks(num_indices);
    dim3 threads(threads_per_block);

    // --- Kernel Launch ---
    embedding_lookup_kernel_128bit<T><<<blocks, threads, 0, stream>>>(
        result,
        embedding,
        indices,
        num_indices,
        hidden_dim_div_16);
}


// template <typename T>
// void embed(
//     const thrust::device_vector<T> &embedding,
//     const thrust::device_vector<uint32_t> &indices,
//     thrust::device_vector<T> *result,
//     int embed_width,
//     cudaStream_t stream)
// {
//     // --- Input Validation ---
//     if (embedding.size() == 0 || indices.size() == 0)
//         return;
//     if (embedding.size() % embed_width != 0)
//     {
//         throw std::invalid_argument("Embedding vector size is not divisible by the embed_width.");
//     }
//     if ((embed_width * sizeof(T)) % 16 != 0)
//     {
//         throw std::invalid_argument("Total byte size of a slice (embed_width * sizeof(T)) must be a multiple of 16.");
//     }

//     // --- Prepare Parameters ---
//     const int num_indices = indices.size();
//     result->resize((long long)num_indices * embed_width);

//     const int threads_per_block = 256;
//     const int hidden_dim_div_16 = (embed_width * sizeof(T)) / 16;

//     dim3 blocks(num_indices);
//     dim3 threads(threads_per_block);

//     // --- Kernel Launch ---
//     embedding_lookup_kernel_128bit<T><<<blocks, threads, 0, stream>>>(
//         thrust::raw_pointer_cast(result->data()),
//         thrust::raw_pointer_cast(embedding.data()),
//         thrust::raw_pointer_cast(indices.data()),
//         num_indices,
//         hidden_dim_div_16);
// }



// Define a reasonable maximum K that can be handled by the shared memory implementation.
// If you need k > 256, this value should be increased, and you may need more shared memory.
#define MAX_K 256

/**
 * @brief CUDA kernel to find top-k values/indices and scatter them.
 *
 * Each thread block processes one row from the input `all_logits` tensor. It finds the top 'k'
 * values and their original indices. It then writes these results to a potentially non-contiguous
 * location in the destination storage buffers, as specified by `dest_embed_ids`.
 *
 * This implementation uses a spin-lock to manage a critical section for updating
 * the list of top-k candidates held in shared memory, ensuring correctness when multiple
 * threads find a candidate simultaneously.
 */
template <typename T>
__global__ void topk_scatter_kernel(
    const T* __restrict__ all_logits,
    const size_t* __restrict__ logit_indices,
    const uint32_t* __restrict__ dest_embed_ids,
    size_t vocab_size,
    size_t k,
    T* __restrict__ topk_probs_storage,
    int32_t* __restrict__ topk_tokens_storage)
{
    // --- Shared Memory Allocation (volatile removed) ---
    __shared__ T sm_top_vals[MAX_K];
    __shared__ int32_t sm_top_idxs[MAX_K];
    __shared__ int lock;

    // --- Block-level variable setup ---
    const int block_row_idx = blockIdx.x;
    const size_t source_row_idx = logit_indices[block_row_idx];
    const uint32_t dest_embed_id = dest_embed_ids[block_row_idx];
    const T* logit_row_ptr = all_logits + source_row_idx * vocab_size;

    // --- Phase 1: Initialize shared memory ---
    if (threadIdx.x == 0) {
        lock = 0;
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        // FIX: Explicitly cast the initial value to the correct type T
        sm_top_vals[i] = static_cast<T>(-1000000.0);
        sm_top_idxs[i] = -1;
    }
    __syncthreads();


    // --- Phase 2: Iterate over the input row and find top candidates ---
    for (size_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        T val = __ldg(logit_row_ptr + i);
        int32_t idx = static_cast<int32_t>(i);

        // FIX: Comparisons now work because sm_top_vals is no longer volatile
        if (val > sm_top_vals[k-1]) {
            while(atomicCAS(&lock, 0, 1) != 0); // Acquire lock

            if (val > sm_top_vals[k-1]) {
                sm_top_vals[k-1] = val;
                sm_top_idxs[k-1] = idx;

                for (int j = k - 2; j >= 0; --j) {
                    // FIX: This comparison also works now
                    if (sm_top_vals[j+1] > sm_top_vals[j]) {
                        // Swap
                        T temp_val = sm_top_vals[j];
                        int32_t temp_idx = sm_top_idxs[j];
                        sm_top_vals[j] = sm_top_vals[j+1];
                        sm_top_idxs[j] = sm_top_idxs[j+1];
                        sm_top_vals[j+1] = temp_val;
                        sm_top_idxs[j+1] = temp_idx;
                    } else {
                        break;
                    }
                }
            }
            atomicExch(&lock, 0); // Release lock
        }
    }

    __syncthreads();

    // --- Phase 3: Write results from shared memory to global memory ---
    T* dest_probs_ptr = topk_probs_storage + dest_embed_id * k;
    int32_t* dest_tokens_ptr = topk_tokens_storage + dest_embed_id * k;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dest_probs_ptr[i] = sm_top_vals[i];
        dest_tokens_ptr[i] = sm_top_idxs[i];
    }
}

/**
 * @brief Performs top-k selection on specified rows of a logit tensor and scatters the results.
 * @tparam T The data type of the logits (e.g., __nv_bfloat16 or float).
 * @param logits The flattened device vector containing all batched logits.
 * @param logit_indices_dev A device vector of indices specifying which rows of `logits` to process.
 * @param dest_embed_ids_dev A device vector specifying the destination slot in storage for each row's result.
 * @param vocab_size The number of columns in the logit tensor (the vocabulary size).
 * @param k The number of top elements to select. Must be <= MAX_K.
 * @param topk_probs_storage Output device vector for storing top-k probabilities/scores.
 * @param topk_tokens_storage Output device vector for storing top-k token IDs.
 * @param stream The CUDA stream for the operation.
 */
template<typename T>
void topk_scatter(
    T* logits,
    const thrust::device_vector<size_t>& logit_indices_dev,
    const thrust::device_vector<uint32_t>& dest_embed_ids_dev,
    size_t vocab_size,
    size_t k,
    thrust::device_vector<T>& topk_probs_storage,
    thrust::device_vector<int32_t>& topk_tokens_storage,
    cudaStream_t stream)
{
    if (k > MAX_K) {
        // In a real application, you should throw an exception or handle this error.
        printf("Error: k=%zu is greater than MAX_K=%d\n", k, MAX_K);
        return;
    }
    if (logit_indices_dev.empty()) {
        return; // Nothing to do
    }

    // --- Kernel Launch Configuration ---
    dim3 blockDim(256); // Threads per block (a common choice)
    dim3 gridDim(logit_indices_dev.size()); // One block per row to process

    // --- Launch Kernel ---
    topk_scatter_kernel<T><<<gridDim, blockDim, 0, stream>>>(
        logits,
        thrust::raw_pointer_cast(logit_indices_dev.data()),
        thrust::raw_pointer_cast(dest_embed_ids_dev.data()),
        vocab_size,
        k,
        thrust::raw_pointer_cast(topk_probs_storage.data()),
        thrust::raw_pointer_cast(topk_tokens_storage.data())
    );
}



// --- Explicit Template Instantiations ---
// We explicitly instantiate the templates for the supported types. This forces
// the compiler to generate the code for each of these types, which will then
// be linked against when another file includes embedding.h.

// template void embed<float>(
//     const thrust::device_vector<float> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<float> *,
//     int, cudaStream_t);

// template void embed<__half>(
//     const thrust::device_vector<__half> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<__half> *,
//     int, cudaStream_t);

// template void embed<__nv_bfloat16>(
//     const thrust::device_vector<__nv_bfloat16> &,
//     const thrust::device_vector<uint32_t> &,
//     thrust::device_vector<__nv_bfloat16> *,
//     int, cudaStream_t);

template void embed<float>(
    const float*,
    size_t,
    const uint32_t*,
    size_t,
    float* result,
    int,
    cudaStream_t);

template void embed<__nv_bfloat16>(
    const __nv_bfloat16*,
    size_t,
    const uint32_t*,
    size_t,
    __nv_bfloat16* result,
    int,
    cudaStream_t);


template void topk_scatter<float>(
    float* ,
    const thrust::device_vector<size_t>& ,
    const thrust::device_vector<uint32_t>& ,
    size_t ,
    size_t ,
    thrust::device_vector<float>& ,
    thrust::device_vector<int32_t>& ,
    cudaStream_t);



template void topk_scatter<__nv_bfloat16>(
    __nv_bfloat16* ,
    const thrust::device_vector<size_t>& ,
    const thrust::device_vector<uint32_t>& ,
    size_t ,
    size_t ,
    thrust::device_vector<__nv_bfloat16>& ,
    thrust::device_vector<int32_t>& ,
    cudaStream_t);


template <typename T>
constexpr cudaDataType_t get_cuda_data_type()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return CUDA_R_32F;
    }
    else if constexpr (std::is_same_v<T, __half>)
    {
        return CUDA_R_16F;
    }
#if __CUDACC_VER_MAJOR__ >= 11
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return CUDA_R_16BF;
    }
#endif
    else if constexpr (std::is_same_v<T, double>)
    {
        return CUDA_R_64F;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return CUDA_R_8I;
    }
    // Add other types here as needed...
    else
    {
        // This will produce a compile-time error if an unsupported type is used.
        static_assert(std::is_same_v<T, void>, "Unsupported data type for gemm_cublasLt_improved");
        return CUDA_R_32F; // Dummy return to satisfy compiler
    }
}

/**
 * @brief Performs General Matrix Multiplication (GEMM) using cuBLASLt with raw device pointers.
 * * This function computes C = alpha * op(A) * op(B) + beta * C.
 * It uses a strategy of swapping A and B to handle column-major layout requirements of cuBLAS
 * while allowing the caller to think in terms of row-major layouts.
 * * @tparam T The data type of the matrices (e.g., float, __nv_bfloat16).
 * @param ltHandle The cuBLASLt library handle.
 * @param stream The CUDA stream for the operation.
 * @param d_A Pointer to matrix A on the device.
 * @param d_B Pointer to matrix B on the device.
 * @param d_bias Pointer to the bias vector on the device (can be nullptr).
 * @param d_C Pointer to the output matrix C on the device.
 * @param m The number of rows of matrix op(A) and C.
 * @param n The number of columns of matrix op(B) and C.
 * @param k The number of columns of op(A) and rows of op(B).
 * @param d_workspace Pointer to the workspace buffer on the device.
 * @param workspaceSize The size of the workspace buffer in bytes.
 * @param transa Specifies if matrix A should be transposed.
 * @param transb Specifies if matrix B should be transposed.
 */
template <typename T>
void gemm_cublasLt(cublasLtHandle_t ltHandle,
                   cudaStream_t stream,
                   const T *d_A,
                   const T *d_B,
                   const T *d_bias,
                   T *d_C,
                   int m, int n, int k,
                   void *d_workspace,
                   size_t workspaceSize,
                   bool transa,
                   bool transb)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }

    // --- Scaling Factors ---
    float alpha = 1.0f;
    float beta = (d_bias != nullptr) ? 1.0f : 0.0f;

    // --- Descriptors for cuBLASLt ---
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    // --- Data and Compute Type Configuration ---
    cudaDataType_t cuda_dtype = get_cuda_data_type<T>();
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type = CUDA_R_32F; // Use FP32 for alpha/beta for precision

    if (std::is_same<T, float>::value)
    {
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32; // Use TF32 for float GEMM
    }
    else
    {
        compute_type = CUBLAS_COMPUTE_32F; // Accumulate in FP32 for mixed-precision
    }

    // --- Core Correction using (A*B)^T = op(B)^T * op(A)^T strategy ---
    // We ask cuBLAS to compute C_col(n,m) = op(B)^T_col(n,k) * op(A)^T_col(k,m).
    cublasOperation_t opA_swapped = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB_swapped = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Create the Matmul Descriptor with the swapped & transformed operations.
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, compute_type, scale_type));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB_swapped, sizeof(opB_swapped)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA_swapped, sizeof(opA_swapped)));

    // Create matrix layouts. Note that the dimensions are for the *swapped* matrices.
    // A (now B) has dimensions (n, k)
    if (transb)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, k, n, k));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, n, k, n));
    }

    // B (now A) has dimensions (k, m)
    if (transa)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, m, k, m));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, k, m, k));
    }
    
    // C has dimensions (n, m)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype, n, m, n));

    // Configure Epilogue (Bias Addition)
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (d_bias != nullptr)
    {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
        void *d_bias_nonconst = const_cast<void *>(static_cast<const void *>(d_bias));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_nonconst, sizeof(d_bias_nonconst)));
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // --- Algorithm Heuristics ---
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // Note the order of descriptors: Adesc (for B), Bdesc (for A), Cdesc
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        fprintf(stderr, "Error: No suitable cuBLASLt algorithm found!\n");
    }
    else
    {
        // Execute the Matmul. Note the order of pointers: d_B, d_A, d_C
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                    d_B, Adesc, // First matrix is B
                                    d_A, Bdesc, // Second matrix is A
                                    &beta,
                                    d_C, Cdesc,
                                    d_C, Cdesc, // D is the same as C for this operation
                                    &heuristicResult.algo, d_workspace, workspaceSize, stream));
    }

    // --- Cleanup ---
    if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
}

// --- Explicit Template Instantiations for raw pointer version ---

template void gemm_cublasLt<__nv_bfloat16>(cublasLtHandle_t, cudaStream_t,
                                           const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *,
                                           int, int, int,
                                           void *, size_t,
                                           bool, bool);

template void gemm_cublasLt<float>(cublasLtHandle_t, cudaStream_t,
                                   const float *, const float *, const float *, float *,
                                   int, int, int,
                                   void *, size_t,
                                   bool, bool);

void multiply_bf16_cublas(cublasHandle_t handle,
                          const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
                          int m, int n, int k, bool transa, bool transb)
{

    // Use FP32 for accumulation to preserve precision.
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transa ? m : k; // Leading dimension
    int ldb = transb ? k : n; // Leading dimension
    int ldc = n;              // Leading dimension for C

    CUBLAS_CHECK(cublasGemmEx(handle,
                              opB,
                              opA,
                              n,
                              m,
                              k,
                              &alpha,
                              B, CUDA_R_16BF, ldb,
                              A, CUDA_R_16BF, lda,
                              &beta,
                              C, CUDA_R_16BF, ldc,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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