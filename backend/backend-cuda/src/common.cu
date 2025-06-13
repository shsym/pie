#include "common.cuh"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>

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
template <typename T>
void embed(
    const thrust::device_vector<T> &embedding,
    const thrust::device_vector<uint32_t> &indices,
    thrust::device_vector<T> *result,
    int embed_width,
    cudaStream_t stream)
{
    // --- Input Validation ---
    if (embedding.size() == 0 || indices.size() == 0)
        return;
    if (embedding.size() % embed_width != 0)
    {
        throw std::invalid_argument("Embedding vector size is not divisible by the embed_width.");
    }
    if ((embed_width * sizeof(T)) % 16 != 0)
    {
        throw std::invalid_argument("Total byte size of a slice (embed_width * sizeof(T)) must be a multiple of 16.");
    }

    // --- Prepare Parameters ---
    const int num_indices = indices.size();
    result->resize((long long)num_indices * embed_width);

    const int threads_per_block = 256;
    const int hidden_dim_div_16 = (embed_width * sizeof(T)) / 16;

    dim3 blocks(num_indices);
    dim3 threads(threads_per_block);

    // --- Kernel Launch ---
    embedding_lookup_kernel_128bit<T><<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(result->data()),
        thrust::raw_pointer_cast(embedding.data()),
        thrust::raw_pointer_cast(indices.data()),
        num_indices,
        hidden_dim_div_16);
}

// --- Explicit Template Instantiations ---
// We explicitly instantiate the templates for the supported types. This forces
// the compiler to generate the code for each of these types, which will then
// be linked against when another file includes embedding.h.

template void embed<float>(
    const thrust::device_vector<float> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<float> *,
    int, cudaStream_t);

template void embed<__half>(
    const thrust::device_vector<__half> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<__half> *,
    int, cudaStream_t);

template void embed<__nv_bfloat16>(
    const thrust::device_vector<__nv_bfloat16> &,
    const thrust::device_vector<uint32_t> &,
    thrust::device_vector<__nv_bfloat16> *,
    int, cudaStream_t);

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

// Helper to calculate memory alignment in bytes from a raw pointer
static uint32_t getAlignment(const void *ptr)
{
    uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    if (address == 0)
        return 256;
    if (address % 256 == 0)
        return 256;
    if (address % 128 == 0)
        return 128;
    if (address % 64 == 0)
        return 64;
    if (address % 32 == 0)
        return 32;
    if (address % 16 == 0)
        return 16;
    if (address % 8 == 0)
        return 8;
    if (address % 4 == 0)
        return 4;
    if (address % 2 == 0)
        return 2;
    return 1;
}

template <typename T>
void gemm_cublasLt2(cublasLtHandle_t ltHandle, cudaStream_t stream, const T *A, const T *B, T *C,
                    int m, int n, int k, bool transa, bool transb)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    cudaDataType_t cuda_dtype = get_cuda_data_type<T>();
    cublasComputeType_t compute_type;

    // Determine compute type based on data type T
    if (cuda_dtype == CUDA_R_16F || cuda_dtype == CUDA_R_16BF)
    {
        compute_type = CUBLAS_COMPUTE_32F; // A common and safe choice for FP16/BF16
    }
    else if (cuda_dtype == CUDA_R_32F)
    {
        compute_type = CUBLAS_COMPUTE_32F;
    }
    else
    {
        compute_type = CUBLAS_COMPUTE_32F;
    }

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, compute_type, CUDA_R_32F));

    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, n, k, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype, n, m, n));

    CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha, B, Adesc, A, Bdesc, &beta, C, Cdesc, C, Cdesc, nullptr, nullptr, 0, stream));

    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}

template void gemm_cublasLt2(cublasLtHandle_t, cudaStream_t, const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *,
                             int, int, int, bool, bool);

template <typename T>
void gemm_cublasLt(cublasLtHandle_t ltHandle,
                   cudaStream_t stream,
                   const thrust::device_vector<T> &A,
                   const thrust::device_vector<T> &B,
                   const thrust::device_vector<T> *bias,
                   thrust::device_vector<T> &C,
                   int m, int n, int k,
                   thrust::device_vector<char> &workspace,
                   bool transa,
                   bool transb)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return;
    }

    // --- Pointers and Workspace Setup ---
    const T *d_A = thrust::raw_pointer_cast(A.data());
    const T *d_B = thrust::raw_pointer_cast(B.data());
    T *d_C = thrust::raw_pointer_cast(C.data());
    const T *d_bias = (bias != nullptr && !bias->empty()) ? thrust::raw_pointer_cast(bias->data()) : nullptr;
    void *d_workspace = thrust::raw_pointer_cast(workspace.data());
    size_t workspaceSize = workspace.size();

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
    // This is achieved by swapping the inputs (B becomes the first matrix, A the second)
    // and providing the correctly transformed operations and layouts.

    // 1. Determine the operations for the swapped multiplication.
    // To get op(M)^T: if the original op was N, the new op is T. If the original was T, the new op is N.
    cublasOperation_t opA_swapped = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB_swapped = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    // 2. Create the Matmul Descriptor with the swapped & transformed operations.
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, compute_type, scale_type));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB_swapped, sizeof(opB_swapped)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA_swapped, sizeof(opA_swapped)));

    if (transb)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, k, n, k));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, cuda_dtype, n, k, n));
    }

    if (transa)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, m, k, m));
    }
    else
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, cuda_dtype, k, m, k));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, cuda_dtype, n, m, n));

    // 4. Configure Epilogue (Bias Addition)
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

    // Note the order of descriptors: Bdesc, Adesc, Cdesc
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        fprintf(stderr, "Error: No suitable cuBLASLt algorithm found!\n");
    }
    else
    {
        // 5. Execute the Matmul
        // Note the order of pointers: d_B, d_A, d_C
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                    d_B, Adesc, // First matrix is B
                                    d_A, Bdesc, // Second matrix is A
                                    &beta,
                                    d_C, Cdesc,
                                    d_C, Cdesc, // D is the same as C for this operation
                                    &heuristicResult.algo, d_workspace, workspaceSize, stream));
    }

    // --- Cleanup ---
    if (preference)
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc)
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
}

template void gemm_cublasLt(cublasLtHandle_t,
                            cudaStream_t,
                            const thrust::device_vector<__nv_bfloat16> &,
                            const thrust::device_vector<__nv_bfloat16> &,
                            const thrust::device_vector<__nv_bfloat16> *,
                            thrust::device_vector<__nv_bfloat16> &,
                            int, int, int,
                            thrust::device_vector<char> &,
                            bool,
                            bool);

template void gemm_cublasLt(cublasLtHandle_t,
                            cudaStream_t,
                            const thrust::device_vector<float> &,
                            const thrust::device_vector<float> &,
                            const thrust::device_vector<float> *,
                            thrust::device_vector<float> &,
                            int, int, int,
                            thrust::device_vector<char> &,
                            bool,
                            bool);

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