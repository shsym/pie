#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Grouped GEMM using Metal
// Corresponds to FlashInfer Grouped GEMM operations
// Performs multiple GEMM operations in parallel, each with potentially different dimensions
// - A_ptrs: Array of pointers to A matrices [num_groups]
// - B_ptrs: Array of pointers to B matrices [num_groups]  
// - C_ptrs: Array of pointers to C matrices [num_groups]
// - bias_ptrs: Array of pointers to bias vectors [num_groups] (optional)
// - m_array: Array of m dimensions for each GEMM [num_groups]
// - n_array: Array of n dimensions for each GEMM [num_groups]
// - k_array: Array of k dimensions for each GEMM [num_groups]
// - num_groups: Number of GEMM operations to perform
// - transa: Whether to transpose A matrices
// - transb: Whether to transpose B matrices
int metal_grouped_gemm_bfloat16(
    void** A_ptrs,                   // Array of pointers to A matrices
    void** B_ptrs,                   // Array of pointers to B matrices
    void** C_ptrs,                   // Array of pointers to output C matrices
    void** bias_ptrs,                // Array of pointers to bias vectors (can be null)
    const int* m_array,              // Array of m dimensions [num_groups]
    const int* n_array,              // Array of n dimensions [num_groups]
    const int* k_array,              // Array of k dimensions [num_groups]
    unsigned int num_groups,         // Number of GEMM operations
    bool transa,                     // Transpose A matrices
    bool transb                      // Transpose B matrices
);

int metal_grouped_gemm_float32(
    float** A_ptrs,                  // Array of pointers to A matrices
    float** B_ptrs,                  // Array of pointers to B matrices
    float** C_ptrs,                  // Array of pointers to output C matrices
    float** bias_ptrs,               // Array of pointers to bias vectors (can be null)
    const int* m_array,              // Array of m dimensions [num_groups]
    const int* n_array,              // Array of n dimensions [num_groups]
    const int* k_array,              // Array of k dimensions [num_groups]
    unsigned int num_groups,         // Number of GEMM operations
    bool transa,                     // Transpose A matrices
    bool transb                      // Transpose B matrices
);

#ifdef __cplusplus
}
#endif