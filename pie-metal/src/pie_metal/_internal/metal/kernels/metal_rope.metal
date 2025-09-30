#include <metal_stdlib>
using namespace metal;

// RoPE (Rotary Position Embedding) Metal implementation
// Corresponds to flashinfer::pos_enc::apply_llama31_rope_pos_ids_inplace from FlashInfer
// Applies rotary position embedding to query and key tensors in-place
//
// Supports both layout modes:
// - Non-interleaved (default): Rotates first half [0, head_size/2) with second half [head_size/2, head_size)
// - Interleaved: Rotates even indices [0,2,4...] with odd indices [1,3,5...]

// RoPE parameters are passed as a float array for torch.mps.compile_shader compatibility
// params_raw[0] = num_tokens
// params_raw[1] = num_heads
// params_raw[2] = head_size
// params_raw[3] = rope_theta
// params_raw[4] = rope_factor
// params_raw[5] = interleaved (0 or 1)

// RoPE kernel that processes pairs of elements in each head
// Each thread processes one pair (x, y) -> (x', y') where:
// x' = x * cos(θ) - y * sin(θ)
// y' = x * sin(θ) + y * cos(θ)
// θ = position_id / (rope_theta^(2*i/head_size)) * rope_factor
kernel void metal_rope_bfloat16(
    device bfloat* input_qk           [[buffer(0)]],  // [num_tokens, num_heads, head_size] input/output tensor
    device const int* position_ids    [[buffer(1)]],  // [num_tokens] position indices
    device const float* params_raw    [[buffer(2)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;  // Index of the (x, y) pair within the head

    // Bounds check
    if (token_idx >= ((uint32_t)params_raw[0]) ||
        head_idx >= ((uint32_t)params_raw[1]) ||
        pair_idx >= ((uint32_t)params_raw[2]) / 2) {
        return;
    }

    // Calculate tensor indices based on layout
    const uint32_t base_idx = token_idx * ((uint32_t)params_raw[1]) * ((uint32_t)params_raw[2]) +
                              head_idx * ((uint32_t)params_raw[2]);
    
    uint32_t x_idx, y_idx;
    if ((((int)params_raw[5]) != 0)) {
        // Interleaved layout: even/odd indices [0,1,2,3...] -> pairs (0,1), (2,3), etc.
        x_idx = base_idx + pair_idx * 2;
        y_idx = base_idx + pair_idx * 2 + 1;
    } else {
        // Non-interleaved layout: split halves [0,1,2,3...] -> pairs (0,2), (1,3), etc.
        x_idx = base_idx + pair_idx;
        y_idx = base_idx + pair_idx + ((uint32_t)params_raw[2]) / 2;
    }

    // Get position for this token
    const float position = float(position_ids[token_idx]);

    // Compute rotary frequency for this pair
    // freq = 1.0 / (rope_theta^(2*pair_idx/head_size)) * rope_factor
    const float exponent = (2.0f * float(pair_idx)) / float(((uint32_t)params_raw[2]));
    const float freq_base = powr(params_raw[3], exponent);
    const float freq = params_raw[4] / freq_base;

    // Compute angle
    const float theta = position * freq;
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);

    // Load current values
    const float x = float(input_qk[x_idx]);
    const float y = float(input_qk[y_idx]);

    // Apply rotation
    const float x_rot = x * cos_theta - y * sin_theta;
    const float y_rot = x * sin_theta + y * cos_theta;

    // Store rotated values
    input_qk[x_idx] = bfloat(x_rot);
    input_qk[y_idx] = bfloat(y_rot);
}

// Float32 version for higher precision
kernel void metal_rope_float32(
    device float* input_qk            [[buffer(0)]],  // [num_tokens, num_heads, head_size] input/output tensor
    device const int* position_ids    [[buffer(1)]],  // [num_tokens] position indices
    device const float* params_raw    [[buffer(2)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;  // Index of the (x, y) pair within the head

    // Bounds check
    if (token_idx >= ((uint32_t)params_raw[0]) ||
        head_idx >= ((uint32_t)params_raw[1]) ||
        pair_idx >= ((uint32_t)params_raw[2]) / 2) {
        return;
    }
    
    // Calculate tensor indices based on layout
    const uint32_t base_idx = token_idx * ((uint32_t)params_raw[1]) * ((uint32_t)params_raw[2]) +
                              head_idx * ((uint32_t)params_raw[2]);
    
    uint32_t x_idx, y_idx;
    if ((((int)params_raw[5]) != 0)) {
        // Interleaved layout: even/odd indices [0,1,2,3...] -> pairs (0,1), (2,3), etc.
        x_idx = base_idx + pair_idx * 2;
        y_idx = base_idx + pair_idx * 2 + 1;
    } else {
        // Non-interleaved layout: split halves [0,1,2,3...] -> pairs (0,2), (1,3), etc.
        x_idx = base_idx + pair_idx;
        y_idx = base_idx + pair_idx + ((uint32_t)params_raw[2]) / 2;
    }
    

    // Get position for this token
    const float position = float(position_ids[token_idx]);

    // Compute rotary frequency for this pair
    // freq = 1.0 / (rope_theta^(2*pair_idx/head_size)) * rope_factor
    const float exponent = (2.0f * float(pair_idx)) / float(((uint32_t)params_raw[2]));
    const float freq_base = powr(params_raw[3], exponent);
    const float freq = params_raw[4] / freq_base;

    // Compute angle
    const float theta = position * freq;
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);

    // Load current values
    const float x = input_qk[x_idx];
    const float y = input_qk[y_idx];

    // Apply rotation
    const float x_rot = x * cos_theta - y * sin_theta;
    const float y_rot = x * sin_theta + y * cos_theta;

    // Store rotated values
    input_qk[x_idx] = x_rot;
    input_qk[y_idx] = y_rot;
}

// Float16 (half) I/O kernel: compute in float for accuracy, store as half
kernel void metal_rope_float16(
    device half* input_qk             [[buffer(0)]],  // [num_tokens, num_heads, head_size] input/output tensor (half)
    device const int* position_ids    [[buffer(1)]],  // [num_tokens] position indices
    device const float* params_raw    [[buffer(2)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;  // Index of the (x, y) pair within the head

    if (token_idx >= ((uint32_t)params_raw[0]) ||
        head_idx >= ((uint32_t)params_raw[1]) ||
        pair_idx >= ((uint32_t)params_raw[2]) / 2) {
        return;
    }

    const uint32_t base_idx = token_idx * ((uint32_t)params_raw[1]) * ((uint32_t)params_raw[2]) +
                              head_idx * ((uint32_t)params_raw[2]);
    const uint32_t x_idx = base_idx + pair_idx;
    const uint32_t y_idx = base_idx + pair_idx + ((uint32_t)params_raw[2]) / 2;

    const float position = float(position_ids[token_idx]);
    const float exponent = (2.0f * float(pair_idx)) / float(((uint32_t)params_raw[2]));
    const float freq_base = powr(params_raw[3], exponent);
    const float freq = params_raw[4] / freq_base;
    const float theta = position * freq;
    const float cos_theta = cos(theta);
    const float sin_theta = sin(theta);

    // Load as half then promote to float
    const float x = float(input_qk[x_idx]);
    const float y = float(input_qk[y_idx]);
    const float x_rot = x * cos_theta - y * sin_theta;
    const float y_rot = x * sin_theta + y * cos_theta;

    // Store back as half
    input_qk[x_idx] = half(x_rot);
    input_qk[y_idx] = half(y_rot);
}