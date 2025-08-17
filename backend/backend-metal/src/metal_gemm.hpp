#pragma once

#include <Metal/Metal.h>
#include <cstdint>
#include <string>

// Metal bfloat16 type mapping
using bfloat16_t = uint16_t;  // Metal bfloat maps to uint16_t in host code

/**
 * @brief Metal implementation of GEMM operation matching cuBLAS behavior
 * 
 * Implements C = A × B^T + bias (optional) using Metal Performance Shaders
 * Matches the API and behavior of gemm_cublasLt<__nv_bfloat16> from common.cu
 * 
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param A Input matrix A [m, k] or [k, m] if transposed (bfloat16)
 * @param B Input matrix B [k, n] or [n, k] if transposed (bfloat16)  
 * @param bias Optional bias vector [n] (bfloat16), can be nullptr
 * @param C Output matrix C [m, n] (bfloat16)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A / rows in B
 * @param workspace Metal buffer for intermediate computations (unused for compatibility)
 * @param workspace_size Size of workspace buffer (unused for compatibility)
 * @param transa Whether matrix A is transposed
 * @param transb Whether matrix B is transposed
 */
void metal_gemm_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* A,
    const bfloat16_t* B, 
    const bfloat16_t* bias,
    bfloat16_t* C,
    int m, int n, int k,
    void* workspace,        // Unused, for API compatibility with cuBLAS
    size_t workspace_size,  // Unused, for API compatibility with cuBLAS
    bool transa,
    bool transb
);

/**
 * @brief Initialize Metal compute environment
 * 
 * Sets up Metal device, command queue, and loads the gemm compute shader
 * Must be called before using metal_gemm_bfloat16
 * 
 * @return true if initialization successful, false otherwise
 */
bool initialize_metal_gemm();

/**
 * @brief Cleanup Metal compute environment
 * 
 * Releases Metal resources allocated by initialize_metal_gemm
 */
void cleanup_metal_gemm();