#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Top-K Mask Logits using Metal
// Corresponds to flashinfer::sampling::TopKMaskLogits from FlashInfer
// Applies top-k masking to logits by setting non-top-k values to -infinity before softmax
// - logits: [num_tokens, vocab_size] input logits (modified in-place)
// - k: number of top values to keep per token
// - num_tokens: number of tokens (batch dimension)
// - vocab_size: vocabulary size
int metal_topk_mask_logits_float32(
    float* logits,                   // Input/output logits [num_tokens, vocab_size] (modified in-place)
    unsigned int num_tokens,         // Number of tokens (batch dimension)
    unsigned int vocab_size,         // Vocabulary size
    unsigned int k                   // Number of top-k values to keep per token
);

int metal_topk_mask_logits_bfloat16(
    void* logits,                    // Input/output logits [num_tokens, vocab_size] (modified in-place)
    unsigned int num_tokens,         // Number of tokens (batch dimension)
    unsigned int vocab_size,         // Vocabulary size
    unsigned int k                   // Number of top-k values to keep per token
);

#ifdef __cplusplus
}
#endif