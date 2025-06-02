#pragma once

#include <cstddef>
#include <cstdint>

// Forward declaration for FlashInfer data types if needed
// For example, if __nv_bfloat16 is used extensively
#include <cuda_bf16.h>

// Corresponds to NUM_TOKENS_IN_BLOCK in Python
constexpr int NUM_TOKENS_IN_BLOCK_CPP = 16; // Or get from config

struct L4maConfig {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    bool use_qkv_bias;
    float rms_norm_eps;
    int vocab_size;
    int pad_token_id;
    int num_hidden_layers;
    // Add any other config parameters used

    // Helper for head_dim
    int head_dim() const { return hidden_size / num_attention_heads; }
};