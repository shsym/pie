#include <metal_stdlib>
using namespace metal;

// Metal implementation of GEMM operation matching cuBLAS behavior
// C = A × B^T (when transb=true) for bfloat16 precision
// Corresponds to gemm_cublasLt<__nv_bfloat16> in common.cu

struct GemmParams {
    uint32_t m;           // Number of rows in A and C
    uint32_t n;           // Number of columns in B and C  
    uint32_t k;           // Number of columns in A / rows in B
    uint32_t lda;         // Leading dimension of A
    uint32_t ldb;         // Leading dimension of B
    uint32_t ldc;         // Leading dimension of C
    bool transa;          // Whether A is transposed
    bool transb;          // Whether B is transposed
    bool use_bias;        // Whether to add bias vector
};

// Tile size for efficient matrix multiplication using threadgroup memory
constant uint TILE_SIZE = 16;

kernel void metal_gemm_bfloat16(
    device const bfloat* A             [[buffer(0)]],  // Input matrix A [m, k] or [k, m] if transposed
    device const bfloat* B             [[buffer(1)]],  // Input matrix B [k, n] or [n, k] if transposed  
    device const bfloat* bias          [[buffer(2)]],  // Optional bias vector [n] (can be null)
    device bfloat* C                   [[buffer(3)]],  // Output matrix C [m, n]
    constant GemmParams& params        [[buffer(4)]],
    threadgroup bfloat* tile_A         [[threadgroup(0)]],  // Tile memory for A
    threadgroup bfloat* tile_B         [[threadgroup(1)]],  // Tile memory for B
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]],
    uint3 tid                          [[threadgroup_position_in_grid]]
) {
    const uint row = gid.y;
    const uint col = gid.x;
    const uint local_row = lid.y;
    const uint local_col = lid.x;
    
    // Return if thread is out of bounds
    if (row >= params.m || col >= params.n) {
        return;
    }
    
    float sum = 0.0f;  // Use float for accumulation to match cuBLAS precision
    
    // Tile-based matrix multiplication
    for (uint tile_k = 0; tile_k < params.k; tile_k += TILE_SIZE) {
        // Load tile from A into threadgroup memory
        uint a_row = row;
        uint a_col = tile_k + local_col;
        
        if (a_row < params.m && a_col < params.k) {
            if (params.transa) {
                // A is transposed: access A[a_col][a_row] stored as A[a_col * params.lda + a_row]
                tile_A[local_row * TILE_SIZE + local_col] = A[a_col * params.lda + a_row];
            } else {
                // A is not transposed: access A[a_row][a_col] stored as A[a_row * params.lda + a_col]
                tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.lda + a_col];
            }
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0h;
        }
        
        // Load tile from B into threadgroup memory
        uint b_row = tile_k + local_row;
        uint b_col = col;
        
        if (b_row < params.k && b_col < params.n) {
            if (params.transb) {
                // B is transposed: access B[b_col][b_row] stored as B[b_col * params.ldb + b_row]
                tile_B[local_row * TILE_SIZE + local_col] = B[b_col * params.ldb + b_row];
            } else {
                // B is not transposed: access B[b_row][b_col] stored as B[b_row * params.ldb + b_col]
                tile_B[local_row * TILE_SIZE + local_col] = B[b_row * params.ldb + b_col];
            }
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0h;
        }
        
        // Synchronize threads in threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            bfloat a_val = tile_A[local_row * TILE_SIZE + k];
            bfloat b_val = tile_B[k * TILE_SIZE + local_col];
            sum += float(a_val) * float(b_val);
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Add bias if provided (matches cuBLAS behavior: beta = 1.0f when bias is present)
    if (params.use_bias && bias != nullptr) {
        sum += float(bias[col]);
    }
    
    // Write result to output matrix C[row][col]
    C[row * params.ldc + col] = bfloat(sum);
}