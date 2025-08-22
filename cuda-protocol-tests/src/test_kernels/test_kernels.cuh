#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>

// Test-local kernel declarations for dtype variants not supported by backend

// Embedding lookup for __half
template<typename T, typename I>
void embed_test_local(const T* embedding_table, size_t vocab_size,
                     const I* indices, size_t num_tokens,
                     T* output, int hidden_size, cudaStream_t stream);

// Extract k values for __half
template<typename T>
void extract_k_values_test_local(const T* input, T* values, int32_t* indices,
                                int M, int N, int k, cudaStream_t stream);

// Forward declaration for template specialization (defined in common.cuh)
template<typename T>
void gemm_cublasLt(cublasLtHandle_t ltHandle, cudaStream_t stream,
                   const T* A, const T* B, const T* bias, T* C,
                   int m, int n, int k, void* workspace, size_t workspace_size,
                   bool transa, bool transb);

// Template specialization for __half is implemented in gemm_fp16.cu