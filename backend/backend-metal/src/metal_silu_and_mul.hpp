#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// SiLU and multiply operation using Metal
// Performs: output = silu(gate) * up where silu(x) = x / (1 + exp(-x))
// - gate: [num_tokens, intermediate_size] gate projection output
// - up: [num_tokens, intermediate_size] up projection output  
// - output: [num_tokens, intermediate_size] result buffer
int metal_silu_and_mul_bfloat16(
    const void* gate,
    const void* up,
    void* output,
    unsigned int num_tokens,
    unsigned int intermediate_size
);

int metal_silu_and_mul_float32(
    const float* gate,
    const float* up,
    float* output,
    unsigned int num_tokens,
    unsigned int intermediate_size
);

#ifdef __cplusplus
}
#endif