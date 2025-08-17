#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// RMS Normalization (Root Mean Square Normalization) using Metal
// Corresponds to flashinfer::norm::RMSNorm<T> from FlashInfer
// Formula: output = input * rsqrt(mean(input^2) + eps) * weight
// - input: [num_tokens, hidden_size] input tensor
// - weight: [hidden_size] scale weights  
// - output: [num_tokens, hidden_size] output tensor
int metal_rmsnorm_bfloat16(
    const void* input,           // Input tensor [num_tokens, hidden_size]
    const void* weight,          // Weight tensor [hidden_size]
    void* output,                // Output tensor [num_tokens, hidden_size]
    unsigned int num_tokens,     // Number of tokens (sequence length)
    unsigned int hidden_size,    // Hidden dimension size
    float eps                    // Epsilon for numerical stability (e.g., 1e-5)
);

int metal_rmsnorm_float32(
    const float* input,          // Input tensor [num_tokens, hidden_size]
    const float* weight,         // Weight tensor [hidden_size]
    float* output,               // Output tensor [num_tokens, hidden_size]
    unsigned int num_tokens,     // Number of tokens (sequence length)  
    unsigned int hidden_size,    // Hidden dimension size
    float eps                    // Epsilon for numerical stability (e.g., 1e-5)
);

#ifdef __cplusplus
}
#endif