#pragma once

#include <Metal/Metal.h>
#include <cstdint>
#include <string>

// Metal bfloat16 type mapping
using bfloat16_t = uint16_t;

/**
 * @brief Metal implementation of embedding lookup matching CUDA backend behavior
 *
 * Implements embedding table lookup: output[i] = embedding_matrix[indices[i]]
 * Matches the API and behavior of embed<T,I> from common.cu
 *
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param embedding_matrix Embedding lookup table [vocab_size, hidden_size] (bfloat16)
 * @param vocab_size Number of vocabulary entries
 * @param indices Token indices to lookup [num_tokens] (int32)
 * @param num_tokens Number of tokens to process
 * @param output Output embeddings [num_tokens, hidden_size] (bfloat16)
 * @param hidden_size Embedding dimension size
 */
void metal_embedding_lookup_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* embedding_matrix,
    size_t vocab_size,
    const int32_t* indices,
    size_t num_tokens,
    bfloat16_t* output,
    int hidden_size
);

/**
 * @brief Native float32 implementation of embedding lookup
 *
 * Runs the float32 Metal kernel without host-side dtype conversions.
 */
void metal_embedding_lookup_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* embedding_matrix,
    size_t vocab_size,
    const int32_t* indices,
    size_t num_tokens,
    float* output,
    int hidden_size
);

/**
 * @brief Initialize Metal embedding compute environment
 *
 * Sets up Metal device, command queue, and loads the embedding compute shader
 * Must be called before using metal_embedding_lookup_bfloat16
 *
 * @return true if initialization successful, false otherwise
 */
bool initialize_metal_embedding();

/**
 * @brief Cleanup Metal embedding compute environment
 *
 * Releases Metal resources allocated by initialize_metal_embedding
 */
void cleanup_metal_embedding();