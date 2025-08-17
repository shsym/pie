#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Extract k non-infinity values per row from a sparse matrix using Metal
// Extracts the first k valid (non -inf) values/indices per row from a sparse matrix
// - A: [M, N] input sparse matrix where empty values are -infinity
// - V: [M, k] output values matrix  
// - I: [M, k] output indices matrix (column positions where values were found)
int metal_extract_k_values_bfloat16(
    const void* A,           // Input matrix [M, N] 
    void* V,                 // Output values [M, k]
    int32_t* I,              // Output indices [M, k]
    unsigned int M,          // Number of rows
    unsigned int N,          // Number of columns
    unsigned int k           // Number of values to extract per row
);

int metal_extract_k_values_float32(
    const float* A,          // Input matrix [M, N]
    float* V,                // Output values [M, k]
    int32_t* I,              // Output indices [M, k]
    unsigned int M,          // Number of rows
    unsigned int N,          // Number of columns
    unsigned int k           // Number of values to extract per row
);

#ifdef __cplusplus
}
#endif