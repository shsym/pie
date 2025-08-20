#include <metal_stdlib>
using namespace metal;

// Top-K Mask Logits Metal implementation
// Corresponds to flashinfer::sampling::TopKMaskLogits from FlashInfer
// Applies top-k masking by setting non-top-k values to -infinity

struct TopKMaskParams {
    uint32_t num_tokens;     // Number of tokens (batch dimension)
    uint32_t vocab_size;     // Vocabulary size
    uint32_t k;              // Number of top-k values to keep per token
};

// Helper function for finding k-th largest element using partial sort
template<typename T>
inline T find_kth_largest(threadgroup T* shared_logits, uint32_t vocab_size, uint32_t k, uint32_t thread_id, uint32_t threads_per_group) {
    // Use parallel partial sort to find k-th largest element
    // This is a simplified version - for production use, consider more efficient algorithms
    
    // Parallel bubble sort for k iterations to get k largest elements
    for (uint32_t iter = 0; iter < k && iter < vocab_size; ++iter) {
        // Find maximum in current range
        for (uint32_t stride = 1; stride < vocab_size - iter; stride *= 2) {
            if (thread_id < vocab_size - iter && thread_id + stride < vocab_size - iter) {
                if (shared_logits[thread_id] < shared_logits[thread_id + stride]) {
                    T temp = shared_logits[thread_id];
                    shared_logits[thread_id] = shared_logits[thread_id + stride];
                    shared_logits[thread_id + stride] = temp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Move maximum to the end
        if (thread_id == 0 && vocab_size - 1 - iter > 0) {
            T temp = shared_logits[0];
            shared_logits[0] = shared_logits[vocab_size - 1 - iter];
            shared_logits[vocab_size - 1 - iter] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Return k-th largest (now at position vocab_size - k)
    return shared_logits[vocab_size - k];
}

// Top-K mask kernel using threadgroup memory for sorting
// Each threadgroup processes one token, threads cooperate to find top-k
kernel void metal_topk_mask_logits_float32(
    device float* logits              [[buffer(0)]],  // [num_tokens, vocab_size] input/output logits
    constant TopKMaskParams& params   [[buffer(1)]],
    threadgroup float* shared_logits  [[threadgroup(0)]], // Shared memory for sorting
    uint3 gid                         [[thread_position_in_grid]],
    uint3 lid                         [[thread_position_in_threadgroup]],
    uint3 tid                         [[threadgroup_position_in_grid]]
) {
    const uint32_t token_idx = tid.x;
    const uint32_t thread_id = lid.x;
    const uint32_t threads_per_group = 256; // Match typical Metal threadgroup size

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Calculate base pointer for this token
    device float* token_logits = logits + token_idx * params.vocab_size;

    // Load logits into shared memory
    for (uint32_t i = thread_id; i < params.vocab_size; i += threads_per_group) {
        shared_logits[i] = token_logits[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find k-th largest value using parallel sorting
    float kth_largest = find_kth_largest(shared_logits, params.vocab_size, params.k, thread_id, threads_per_group);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply mask: set values below k-th largest to -infinity
    for (uint32_t i = thread_id; i < params.vocab_size; i += threads_per_group) {
        if (token_logits[i] < kth_largest) {
            token_logits[i] = -INFINITY;
        }
    }
}

// bfloat16 version
kernel void metal_topk_mask_logits_bfloat16(
    device bfloat* logits             [[buffer(0)]],  // [num_tokens, vocab_size] input/output logits
    constant TopKMaskParams& params   [[buffer(1)]],
    threadgroup float* shared_logits  [[threadgroup(0)]], // Use float for sorting precision
    uint3 gid                         [[thread_position_in_grid]],
    uint3 lid                         [[thread_position_in_threadgroup]],
    uint3 tid                         [[threadgroup_position_in_grid]]
) {
    const uint32_t token_idx = tid.x;
    const uint32_t thread_id = lid.x;
    const uint32_t threads_per_group = 256;

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Calculate base pointer for this token
    device bfloat* token_logits = logits + token_idx * params.vocab_size;

    // Load logits into shared memory (convert to float for precision)
    for (uint32_t i = thread_id; i < params.vocab_size; i += threads_per_group) {
        shared_logits[i] = float(token_logits[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find k-th largest value using parallel sorting
    float kth_largest = find_kth_largest(shared_logits, params.vocab_size, params.k, thread_id, threads_per_group);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply mask: set values below k-th largest to -infinity
    for (uint32_t i = thread_id; i < params.vocab_size; i += threads_per_group) {
        if (float(token_logits[i]) < kth_largest) {
            token_logits[i] = bfloat(-INFINITY);
        }
    }
}