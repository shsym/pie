#pragma once

#include <Metal/Metal.h>
#include <cstdint>

// Metal bfloat16 type mapping
using bfloat16_t = uint16_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Metal implementation of residual addition operation
 * 
 * Performs element-wise addition: output = input + residual
 * This matches the add_residual function from kernels.cuh in the CUDA backend.
 * 
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param input Input tensor data (bfloat16)
 * @param residual Residual tensor data to add (bfloat16)
 * @param output Output tensor data (bfloat16), can be same as input for in-place operation
 * @param num_elements Total number of elements to process
 */
void metal_add_residual_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* input,
    const bfloat16_t* residual,
    bfloat16_t* output,
    size_t num_elements
);

/**
 * @brief Metal implementation of residual addition for float32
 * 
 * Performs element-wise addition: output = input + residual
 * 
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param input Input tensor data (float32)
 * @param residual Residual tensor data to add (float32)
 * @param output Output tensor data (float32), can be same as input for in-place operation
 * @param num_elements Total number of elements to process
 */
void metal_add_residual_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* input,
    const float* residual,
    float* output,
    size_t num_elements
);

/**
 * @brief Metal implementation of in-place residual addition for bfloat16
 * 
 * Performs in-place element-wise addition: input += residual
 * Matches the CUDA add_residual(x, residual, n, stream) interface
 * 
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param input_output Input/output tensor data (bfloat16), modified in-place
 * @param residual Residual tensor data to add (bfloat16)
 * @param num_elements Total number of elements to process
 */
void metal_add_residual_inplace_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    bfloat16_t* input_output,
    const bfloat16_t* residual,
    size_t num_elements
);

/**
 * @brief Metal implementation of in-place residual addition for float32
 * 
 * Performs in-place element-wise addition: input += residual
 * 
 * @param device Metal device to run computation on
 * @param commandQueue Metal command queue for GPU commands
 * @param input_output Input/output tensor data (float32), modified in-place
 * @param residual Residual tensor data to add (float32)
 * @param num_elements Total number of elements to process
 */
void metal_add_residual_inplace_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    float* input_output,
    const float* residual,
    size_t num_elements
);

/**
 * @brief Initialize Metal compute environment for residual operations
 * 
 * Sets up Metal device, command queue, and loads the residual compute shaders
 * Must be called before using metal_add_residual functions
 * 
 * @return true if initialization successful, false otherwise
 */
bool initialize_metal_add_residual();

/**
 * @brief Cleanup Metal compute environment for residual operations
 * 
 * Releases Metal resources allocated by initialize_metal_add_residual
 */
void cleanup_metal_add_residual();

#ifdef __cplusplus
}
#endif

/**
 * @brief Template wrapper for type-safe residual addition
 * 
 * Automatically selects the appropriate Metal function based on template parameter
 */
template<typename T>
void metal_add_residual(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const T* input,
    const T* residual,
    T* output,
    size_t num_elements
);

/**
 * @brief Template wrapper for type-safe in-place residual addition
 */
template<typename T>
void metal_add_residual_inplace(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    T* input_output,
    const T* residual,
    size_t num_elements
);

// Template specializations
template<>
inline void metal_add_residual<bfloat16_t>(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* input,
    const bfloat16_t* residual,
    bfloat16_t* output,
    size_t num_elements
) {
    metal_add_residual_bfloat16(device, commandQueue, input, residual, output, num_elements);
}

template<>
inline void metal_add_residual<float>(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* input,
    const float* residual,
    float* output,
    size_t num_elements
) {
    metal_add_residual_float32(device, commandQueue, input, residual, output, num_elements);
}

template<>
inline void metal_add_residual_inplace<bfloat16_t>(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    bfloat16_t* input_output,
    const bfloat16_t* residual,
    size_t num_elements
) {
    metal_add_residual_inplace_bfloat16(device, commandQueue, input_output, residual, num_elements);
}

template<>
inline void metal_add_residual_inplace<float>(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    float* input_output,
    const float* residual,
    size_t num_elements
) {
    metal_add_residual_inplace_float32(device, commandQueue, input_output, residual, num_elements);
}