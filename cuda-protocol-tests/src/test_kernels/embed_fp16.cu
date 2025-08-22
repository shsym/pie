#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Test-local implementation of embed function for __half support
// This follows the same algorithm as backend/backend-cuda/src/common.cu but with __half

template<typename T, typename I>
__global__ void embed_kernel(const T* embedding_table, size_t vocab_size,
                           const I* indices, size_t num_tokens, 
                           T* output, int hidden_size) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= num_tokens || dim_idx >= hidden_size) return;
    
    I vocab_idx = indices[token_idx];
    if (vocab_idx >= 0 && vocab_idx < vocab_size) {
        output[token_idx * hidden_size + dim_idx] = 
            embedding_table[vocab_idx * hidden_size + dim_idx];
    }
}

template<typename T, typename I>
void embed_test_local(const T* embedding_table, size_t vocab_size,
                     const I* indices, size_t num_tokens,
                     T* output, int hidden_size, cudaStream_t stream) {
    
    // Grid configuration: one block per token, distributed over dimensions
    int threads_per_block = min(hidden_size, 256);
    int blocks_y = (hidden_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_size(num_tokens, blocks_y);
    dim3 block_size(threads_per_block);
    
    embed_kernel<T, I><<<grid_size, block_size, 0, stream>>>(
        embedding_table, vocab_size, indices, num_tokens, output, hidden_size);
}

// Explicit instantiation for __half
template void embed_test_local<__half, int32_t>(
    const __half* embedding_table, size_t vocab_size,
    const int32_t* indices, size_t num_tokens,
    __half* output, int hidden_size, cudaStream_t stream);