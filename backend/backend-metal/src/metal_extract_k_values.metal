#include <metal_stdlib>
using namespace metal;

// Extract k non-infinity values per row from a sparse matrix
// Input: A [M, N] - sparse matrix where empty values are -infinity
// Output: V [M, k] - extracted values, I [M, k] - column indices

kernel void extract_k_values_bfloat16_kernel(
    device const bfloat* A [[buffer(0)]],           // Input matrix [M, N]
    device bfloat* V [[buffer(1)]],                 // Output values [M, k]
    device int32_t* I [[buffer(2)]],                // Output indices [M, k]
    constant uint& M [[buffer(3)]],                 // Number of rows
    constant uint& N [[buffer(4)]],                 // Number of columns
    constant uint& k [[buffer(5)]],                 // Number of values to extract per row
    threadgroup atomic_int* output_count [[threadgroup(0)]], // Shared counter
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // One threadgroup per row
    uint row_idx = bid;
    
    if (row_idx >= M) {
        return;
    }
    
    // Initialize shared counter for this threadgroup
    if (tid == 0) {
        atomic_store_explicit(output_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Pointers for current row
    device const bfloat* input_row = A + row_idx * N;
    device bfloat* value_output_row = V + row_idx * k;
    device int32_t* index_output_row = I + row_idx * k;
    
    // Negative infinity representation for bfloat16
    const bfloat neg_inf = bfloat(-INFINITY);
    
    // Parallel scan through the row
    uint threads_per_group = 256; // Should match the dispatch
    for (uint col_base = 0; col_base < N; col_base += threads_per_group) {
        // Early exit if k elements already found
        int current_count = atomic_load_explicit(output_count, memory_order_relaxed);
        if (current_count >= int(k)) {
            break;
        }
        
        uint col_idx = col_base + tid;
        
        // Boundary check
        if (col_idx < N) {
            bfloat val = input_row[col_idx];
            
            // Check if value is not negative infinity
            if (val != neg_inf) {
                // Atomically increment counter to get write position
                int write_idx = atomic_fetch_add_explicit(output_count, 1, memory_order_relaxed);
                
                // Write result if within top k
                if (write_idx < int(k)) {
                    value_output_row[write_idx] = val;
                    index_output_row[write_idx] = int32_t(col_idx);
                }
            }
        }
    }
}

kernel void extract_k_values_float32_kernel(
    device const float* A [[buffer(0)]],            // Input matrix [M, N]
    device float* V [[buffer(1)]],                  // Output values [M, k]
    device int32_t* I [[buffer(2)]],                // Output indices [M, k]
    constant uint& M [[buffer(3)]],                 // Number of rows
    constant uint& N [[buffer(4)]],                 // Number of columns
    constant uint& k [[buffer(5)]],                 // Number of values to extract per row
    threadgroup atomic_int* output_count [[threadgroup(0)]], // Shared counter
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // One threadgroup per row
    uint row_idx = bid;
    
    if (row_idx >= M) {
        return;
    }
    
    // Initialize shared counter for this threadgroup
    if (tid == 0) {
        atomic_store_explicit(output_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Pointers for current row
    device const float* input_row = A + row_idx * N;
    device float* value_output_row = V + row_idx * k;
    device int32_t* index_output_row = I + row_idx * k;
    
    // Negative infinity
    const float neg_inf = -INFINITY;
    
    // Parallel scan through the row
    uint threads_per_group = 256; // Should match the dispatch
    for (uint col_base = 0; col_base < N; col_base += threads_per_group) {
        // Early exit if k elements already found
        int current_count = atomic_load_explicit(output_count, memory_order_relaxed);
        if (current_count >= int(k)) {
            break;
        }
        
        uint col_idx = col_base + tid;
        
        // Boundary check
        if (col_idx < N) {
            float val = input_row[col_idx];
            
            // Check if value is not negative infinity
            if (val != neg_inf) {
                // Atomically increment counter to get write position
                int write_idx = atomic_fetch_add_explicit(output_count, 1, memory_order_relaxed);
                
                // Write result if within top k
                if (write_idx < int(k)) {
                    value_output_row[write_idx] = val;
                    index_output_row[write_idx] = int32_t(col_idx);
                }
            }
        }
    }
}