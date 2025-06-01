// src/main.cu

#include <iostream>
#include <vector>
#include <cuda_runtime.h> // For CUDA API calls (cudaMalloc, cudaMemcpy, cudaFree)
#include <algorithm>      // For std::fill
#include <cmath>          // For std::isnan

// Include your common utilities if you have them (e.g., CUDA_CHECK macro)

// FlashInfer headers
// Adjust path based on your lib/flashinfer/include structure

#include "flashinfer_ops.cuh"

// --- Helper for checking CUDA errors (from common.h or define here) ---
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)
#endif
// --- End Helper ---


int main() {
    std::cout << "Starting FlashInfer decode example..." << std::endl;

    // // --- Define fixed parameters for the decode operation ---
    // size_t num_qo_heads = 4;   // Number of query/output heads
    // size_t num_kv_heads = 4;   // Number of key/value heads
    // size_t seq_len = 128;      // Sequence length (number of past tokens in KV cache)
    // size_t head_dim = 128;     // Dimension of each head
    // QKVLayout kv_layout = QKVLayout::kHND; // HND (Heads, Sequence, Dim) or BNH (Batch, Num Heads, Seq Len)
    // PosEncodingMode pos_encoding_mode = PosEncodingMode::kNONE; // No positional encoding

    // // Data types (float for simplicity)
    // using DTypeQO = float; // Query/Output data type
    // using DTypeKV = float; // Key/Value data type

    // // --- Allocate Host Memory (CPU) ---
    // // Q: Query for current token (num_qo_heads * head_dim)
    // // K/V: KV Cache for past tokens (seq_len * num_kv_heads * head_dim)
    // // O: Output (num_qo_heads * head_dim)

    // std::vector<DTypeQO> Q_host(num_qo_heads * head_dim);
    // std::vector<DTypeKV> K_host(seq_len * num_kv_heads * head_dim);
    // std::vector<DTypeKV> V_host(seq_len * num_kv_heads * head_dim);
    // std::vector<DTypeQO> O_host(num_qo_heads * head_dim); // Output buffer on host

    // // Initialize host data (e.g., with some dummy values)
    // // In a real application, these would come from your model's forward pass
    // for (size_t i = 0; i < Q_host.size(); ++i) Q_host[i] = static_cast<DTypeQO>(i % 100) * 0.01f + 0.1f;
    // for (size_t i = 0; i < K_host.size(); ++i) K_host[i] = static_cast<DTypeKV>(i % 100) * 0.005f + 0.2f;
    // for (size_t i = 0; i < V_host.size(); ++i) V_host[i] = static_cast<DTypeKV>(i % 100) * 0.008f + 0.3f;
    // std::fill(O_host.begin(), O_host.end(), 0.0f); // Initialize output to zero

    // // --- Allocate Device Memory (GPU) ---
    // DTypeQO *d_Q, *d_O;
    // DTypeKV *d_K, *d_V;
    // void *d_workspace; // FlashInfer might need a temporary workspace

    // CUDA_CHECK(cudaMalloc(&d_Q, Q_host.size() * sizeof(DTypeQO)));
    // CUDA_CHECK(cudaMalloc(&d_K, K_host.size() * sizeof(DTypeKV)));
    // CUDA_CHECK(cudaMalloc(&d_V, V_host.size() * sizeof(DTypeKV)));
    // CUDA_CHECK(cudaMalloc(&d_O, O_host.size() * sizeof(DTypeQO)));

    // // FlashInfer decode_with_kv_cache typically needs a workspace.
    // // The size can be obtained from an estimation function or chosen conservatively.
    // // The original test used 32MB.
    // size_t workspace_size = 32 * 1024 * 1024; // 32MB
    // CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));


    // // --- Copy data from Host to Device ---
    // CUDA_CHECK(cudaMemcpy(d_Q, Q_host.data(), Q_host.size() * sizeof(DTypeQO), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_K, K_host.data(), K_host.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_V, V_host.data(), V_host.size() * sizeof(DTypeKV), cudaMemcpyHostToDevice));
    // // No need to copy O_host to device, as it's an output buffer


    // // --- Call FlashInfer Kernel ---
    // // The flashinfer::SingleDecodeWithKVCache function expects device pointers.
    // // It's part of the flashinfer:: namespace.
    // cudaError_t status = SingleDecodeWithKVCache<DTypeQO, DTypeKV, DTypeQO>(
    //     d_Q,       // Query on device
    //     d_K,       // Key cache on device
    //     d_V,       // Value cache on device
    //     d_O,       // Output on device
    //     d_workspace, // Workspace on device
    //     num_qo_heads,
    //     num_kv_heads,
    //     seq_len,
    //     head_dim,
    //     kv_layout,
    //     pos_encoding_mode
    // );

    // CUDA_CHECK(status); // Use our macro to check for kernel launch errors

    // // --- Copy result back from Device to Host ---
    // CUDA_CHECK(cudaMemcpy(O_host.data(), d_O, O_host.size() * sizeof(DTypeQO), cudaMemcpyDeviceToHost));

    // // --- Verify/Print a sample of the result (optional) ---
    // std::cout << "FlashInfer decode operation completed successfully." << std::endl;
    // std::cout << "Sample of output (O_host[0]): " << O_host[0] << std::endl;
    // // You can print more elements or perform a simple check here.
    // // Since we removed the CPU reference, we can't do a full correctness check.
    // // But if O_host[0] is not NaN and looks reasonable, it's a good sign.
    // if (std::isnan(O_host[0])) {
    //     std::cerr << "Warning: NaN detected in output!" << std::endl;
    // }


    // // --- Free Device Memory ---
    // CUDA_CHECK(cudaFree(d_Q));
    // CUDA_CHECK(cudaFree(d_K));
    // CUDA_CHECK(cudaFree(d_V));
    // CUDA_CHECK(cudaFree(d_O));
    // CUDA_CHECK(cudaFree(d_workspace));

    std::cout << "Memory freed. Example finished." << std::endl;

    return 0;
}