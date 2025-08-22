#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include "common.cuh"

// Test-local implementation of GEMM for native fp16 support
// This uses cuBLASLt with CUBLAS_COMPUTE_16F for native fp16 computation

template<>
void gemm_cublasLt<__half>(cublasLtHandle_t ltHandle, cudaStream_t stream,
                          const __half* d_A, const __half* d_B, const __half* d_bias, __half* d_C,
                          int m, int n, int k, void* d_workspace, size_t workspaceSize,
                          bool transa, bool transb) {
    
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, biasDesc;
    
    // Create matrix multiplication descriptor with native fp16 compute
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    
    // Set transpose operations
    cublasOperation_t op_A = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(op_A));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(op_B));
    
    // Set bias pointer if provided
    if (d_bias) {
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
    }
    
    // Create matrix layout descriptors
    int lda = transa ? m : k;
    int ldb = transb ? k : n;
    int ldc = n;
    
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa ? k : m, transa ? m : k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb ? n : k, transb ? k : n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc);
    
    if (d_bias) {
        cublasLtMatrixLayoutCreate(&biasDesc, CUDA_R_16F, n, 1, n);
    }
    
    // Set scaling factors (alpha=1, beta=0)
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // Find the best algorithm
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &workspaceSize, sizeof(workspaceSize));
    
    // Get heuristic
    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                  preference, 1, &heuristic, &returnedResults);
    
    // Perform matrix multiplication
    if (returnedResults > 0) {
        cublasLtMatmul(ltHandle, matmulDesc,
                      &alpha, d_A, Adesc, d_B, Bdesc,
                      &beta, d_C, Cdesc, d_C, Cdesc,
                      &heuristic.algo, d_workspace, workspaceSize, stream);
    }
    
    // Clean up
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    if (d_bias) cublasLtMatrixLayoutDestroy(biasDesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmulDesc);
}