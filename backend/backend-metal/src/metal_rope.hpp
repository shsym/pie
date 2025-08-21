#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// RoPE (Rotary Position Embedding) using Metal
// Corresponds to flashinfer::pos_enc::apply_llama31_rope_pos_ids_inplace from FlashInfer
// Applies rotary position embedding to query and key tensors in-place
// - input_qk: [num_tokens, num_heads, head_size] query/key tensor (modified in-place)
// - position_ids: [num_tokens] position indices for each token
// - num_tokens: Number of tokens in the sequence
// - num_heads: Number of attention heads
// - head_size: Size of each attention head
// - rope_theta: Base for the rotary frequency computation (default 10000.0)
// - rope_factor: Scaling factor for RoPE (default 1.0)
int metal_rope_bfloat16(
    void* input_qk,                  // Input/output tensor [num_tokens, num_heads, head_size]
    const int32_t* position_ids,     // Position IDs [num_tokens]
    unsigned int num_tokens,         // Number of tokens (sequence length)
    unsigned int num_heads,          // Number of attention heads
    unsigned int head_size,          // Size of each attention head
    float rope_theta,                // Base for rotary frequency (e.g., 10000.0)
    float rope_factor                // Scaling factor for RoPE (e.g., 1.0)
);

int metal_rope_float32(
    float* input_qk,                 // Input/output tensor [num_tokens, num_heads, head_size]
    const int32_t* position_ids,     // Position IDs [num_tokens]
    unsigned int num_tokens,         // Number of tokens (sequence length)
    unsigned int num_heads,          // Number of attention heads
    unsigned int head_size,          // Size of each attention head
    float rope_theta,                // Base for rotary frequency (e.g., 10000.0)
    float rope_factor                // Scaling factor for RoPE (e.g., 1.0)
);

// Float16 API: input/output buffer is half (IEEE fp16). Internally compute in float for accuracy.
int metal_rope_float16(
    uint16_t* input_qk,              // Input/output tensor [num_tokens, num_heads, head_size] in IEEE half (fp16)
    const int32_t* position_ids,     // Position IDs [num_tokens]
    unsigned int num_tokens,         // Number of tokens (sequence length)
    unsigned int num_heads,          // Number of attention heads
    unsigned int head_size,          // Size of each attention head
    float rope_theta,                // Base for rotary frequency (e.g., 10000.0)
    float rope_factor                // Scaling factor for RoPE (e.g., 1.0)
);

#ifdef __cplusplus
}
#endif