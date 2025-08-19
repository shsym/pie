#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Metal implementation of softmax operation with numerical stability
 * 
 * @param input: Input logits [batch_size, vocab_size] 
 * @param output: Output probabilities [batch_size, vocab_size]
 * @param batch_size: Number of sequences in batch
 * @param vocab_size: Size of vocabulary dimension
 * @param temperature: Temperature scaling factor (default: 1.0)
 * @return: 0 on success, non-zero on error
 */
int metal_softmax_float(
    const float* input,
    float* output,
    int batch_size,
    int vocab_size,
    float temperature
);

#ifdef __cplusplus
}
#endif