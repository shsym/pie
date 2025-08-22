#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

// Test-local implementation of extract_k_values function for __half support
// This follows the same algorithm as backend/backend-cuda but with __half

template<typename T>
__global__ void extract_k_values_kernel(const T* input, T* values, int32_t* indices, 
                                       int M, int N, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M) return;
    
    const T* row_data = input + row * N;
    T* row_values = values + row * k;
    int32_t* row_indices = indices + row * k;
    
    // Simple selection sort to find top k values
    // For test purposes, this doesn't need to be the most efficient
    for (int i = 0; i < k; i++) {
        int max_idx = -1;
        T max_val = -INFINITY;
        
        // Find the next maximum that hasn't been selected
        for (int j = 0; j < N; j++) {
            bool already_selected = false;
            for (int prev = 0; prev < i; prev++) {
                if (row_indices[prev] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && (__half2float(row_data[j]) > __half2float(max_val) || max_idx == -1)) {
                max_val = row_data[j];
                max_idx = j;
            }
        }
        
        if (max_idx != -1) {
            row_values[i] = max_val;
            row_indices[i] = max_idx;
        }
    }
}

template<typename T>
void extract_k_values_test_local(const T* input, T* values, int32_t* indices,
                                int M, int N, int k, cudaStream_t stream) {
    
    int threads_per_block = 256;
    int blocks = (M + threads_per_block - 1) / threads_per_block;
    
    extract_k_values_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
        input, values, indices, M, N, k);
}

// Explicit instantiation for __half
template void extract_k_values_test_local<__half>(
    const __half* input, __half* values, int32_t* indices,
    int M, int N, int k, cudaStream_t stream);