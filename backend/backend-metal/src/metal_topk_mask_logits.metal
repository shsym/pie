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

// Helper: in-place insert into descending-sorted top-k buffer
inline void insert_topk(thread float* topk, thread uint& count, float val, uint K) {
    if (count < K) {
        // Insert at end then bubble up to keep descending order
        uint i = count;
        topk[i] = val;
        while (i > 0 && topk[i] > topk[i - 1]) {
            float tmp = topk[i - 1];
            topk[i - 1] = topk[i];
            topk[i] = tmp;
            --i;
        }
        count++;
    } else if (K > 0 && val > topk[K - 1]) {
        // Replace smallest (tail) and bubble up
        topk[K - 1] = val;
        uint i = K - 1;
        while (i > 0 && topk[i] > topk[i - 1]) {
            float tmp = topk[i - 1];
            topk[i - 1] = topk[i];
            topk[i] = tmp;
            --i;
        }
    }
}

// Top-K mask kernel using threadgroup memory for sorting
// Each threadgroup processes one token, threads cooperate to find top-k
kernel void metal_topk_mask_logits_float32(
    device float* logits              [[buffer(0)]],  // [num_tokens, vocab_size] input/output logits
    constant TopKMaskParams& params   [[buffer(1)]],
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

    // Compute threshold (k-th largest) using a single thread to ensure correctness
    threadgroup float threshold_shared[1];
    if (thread_id == 0) {
        const uint MAX_K = 128u; // safety cap
        const uint K = min(params.k, MAX_K);
        float topk[128];
        uint count = 0u;
        for (uint i = 0u; i < params.vocab_size; ++i) {
            insert_topk(topk, count, token_logits[i], K);
        }
        float threshold = (K > 0u) ? topk[K - 1u] : -INFINITY;
        threshold_shared[0] = threshold;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kth_largest = threshold_shared[0];

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
    // Compute threshold using single-thread top-k selection
    threadgroup float threshold_shared[1];
    if (thread_id == 0) {
        const uint MAX_K = 128u;
        const uint K = min(params.k, MAX_K);
        float topk[128];
        uint count = 0u;
        for (uint i = 0u; i < params.vocab_size; ++i) {
            insert_topk(topk, count, float(token_logits[i]), K);
        }
        float threshold = (K > 0u) ? topk[K - 1u] : -INFINITY;
        threshold_shared[0] = threshold;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kth_largest = threshold_shared[0];

    // Apply mask: set values below k-th largest to -infinity
    for (uint32_t i = thread_id; i < params.vocab_size; i += threads_per_group) {
        if (float(token_logits[i]) < kth_largest) {
            token_logits[i] = bfloat(-INFINITY);
        }
    }
}