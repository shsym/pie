#import "metal_add_residual.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

// Global state for Metal add_residual operations
static id<MTLComputePipelineState> g_add_residual_bfloat16_pipeline = nil;
static id<MTLComputePipelineState> g_add_residual_float32_pipeline = nil;
static id<MTLComputePipelineState> g_add_residual_inplace_bfloat16_pipeline = nil;
static id<MTLComputePipelineState> g_add_residual_inplace_float32_pipeline = nil;
static id<MTLComputePipelineState> g_add_residual_bfloat16_vectorized_pipeline = nil;
static id<MTLComputePipelineState> g_add_residual_float32_vectorized_pipeline = nil;

bool initialize_metal_add_residual() {
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        std::cerr << "Metal context not initialized" << std::endl;
        return false;
    }
    
    id<MTLDevice> device = context.getDevice();
    
    // Get the kernel library - try to get from pipeline manager first
    auto& pipelineManager = MetalComputePipelineManager::getInstance();
    
    // Create compute pipeline states for each kernel
    g_add_residual_bfloat16_pipeline = pipelineManager.getComputePipeline("add_residual_bfloat16_kernel");
    g_add_residual_float32_pipeline = pipelineManager.getComputePipeline("add_residual_float32_kernel");
    g_add_residual_inplace_bfloat16_pipeline = pipelineManager.getComputePipeline("add_residual_inplace_bfloat16_kernel");
    g_add_residual_inplace_float32_pipeline = pipelineManager.getComputePipeline("add_residual_inplace_float32_kernel");
    g_add_residual_bfloat16_vectorized_pipeline = pipelineManager.getComputePipeline("add_residual_bfloat16_vectorized_kernel");
    g_add_residual_float32_vectorized_pipeline = pipelineManager.getComputePipeline("add_residual_float32_vectorized_kernel");
    
    // Check if all pipelines were created successfully
    bool success = (g_add_residual_bfloat16_pipeline != nil) &&
                   (g_add_residual_float32_pipeline != nil) &&
                   (g_add_residual_inplace_bfloat16_pipeline != nil) &&
                   (g_add_residual_inplace_float32_pipeline != nil);
    
    if (!success) {
        std::cerr << "Failed to create Metal compute pipelines for add_residual operations" << std::endl;
        cleanup_metal_add_residual();
        return false;
    }
    
    std::cout << "Metal add_residual operations initialized successfully" << std::endl;
    return true;
}

void cleanup_metal_add_residual() {
    g_add_residual_bfloat16_pipeline = nil;
    g_add_residual_float32_pipeline = nil;
    g_add_residual_inplace_bfloat16_pipeline = nil;
    g_add_residual_inplace_float32_pipeline = nil;
    g_add_residual_bfloat16_vectorized_pipeline = nil;
    g_add_residual_float32_vectorized_pipeline = nil;
    
    std::cout << "Metal add_residual operations cleaned up" << std::endl;
}

void metal_add_residual_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* input,
    const bfloat16_t* residual,
    bfloat16_t* output,
    size_t num_elements
) {
    if (num_elements == 0) return;
    
    // Check if pipeline is available
    if (g_add_residual_bfloat16_pipeline == nil) {
        std::cerr << "Metal add_residual bfloat16 pipeline not initialized" << std::endl;
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g_add_residual_bfloat16_pipeline];
    
    // Create buffers for input, residual, and output
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input
                                                    length:num_elements * sizeof(bfloat16_t)
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual
                                                       length:num_elements * sizeof(bfloat16_t)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:num_elements * sizeof(bfloat16_t)
                                                     options:MTLResourceStorageModeShared];
    
    uint32_t numElementsUint = static_cast<uint32_t>(num_elements);
    id<MTLBuffer> numElementsBuffer = [device newBufferWithBytes:&numElementsUint
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    
    // Set buffers
    [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:residualBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:numElementsBuffer offset:0 atIndex:3];
    
    // Calculate thread group sizes
    auto [threadsPerThreadgroup, threadgroupsPerGrid] = MetalDispatch::calculateThreadGroups1D(num_elements);
    
    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];
    
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    std::memcpy((void*)output, [outputBuffer contents], num_elements * sizeof(bfloat16_t));
}

void metal_add_residual_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* input,
    const float* residual,
    float* output,
    size_t num_elements
) {
    if (num_elements == 0) return;
    
    // Check if pipeline is available
    if (g_add_residual_float32_pipeline == nil) {
        std::cerr << "Metal add_residual float32 pipeline not initialized" << std::endl;
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g_add_residual_float32_pipeline];
    
    // Create buffers
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input
                                                    length:num_elements * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual
                                                       length:num_elements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:num_elements * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    
    uint32_t numElementsUint = static_cast<uint32_t>(num_elements);
    id<MTLBuffer> numElementsBuffer = [device newBufferWithBytes:&numElementsUint
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    
    // Set buffers
    [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:residualBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:2];
    [computeEncoder setBuffer:numElementsBuffer offset:0 atIndex:3];
    
    // Calculate thread group sizes
    auto [threadsPerThreadgroup, threadgroupsPerGrid] = MetalDispatch::calculateThreadGroups1D(num_elements);
    
    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];
    
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    std::memcpy(output, [outputBuffer contents], num_elements * sizeof(float));
}

void metal_add_residual_inplace_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    bfloat16_t* input_output,
    const bfloat16_t* residual,
    size_t num_elements
) {
    if (num_elements == 0) return;
    
    // Check if pipeline is available
    if (g_add_residual_inplace_bfloat16_pipeline == nil) {
        std::cerr << "Metal add_residual inplace bfloat16 pipeline not initialized" << std::endl;
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g_add_residual_inplace_bfloat16_pipeline];
    
    // Create buffers - input_output buffer needs to be writable
    id<MTLBuffer> inputOutputBuffer = [device newBufferWithBytes:input_output
                                                          length:num_elements * sizeof(bfloat16_t)
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual
                                                       length:num_elements * sizeof(bfloat16_t)
                                                      options:MTLResourceStorageModeShared];
    
    uint32_t numElementsUint = static_cast<uint32_t>(num_elements);
    id<MTLBuffer> numElementsBuffer = [device newBufferWithBytes:&numElementsUint
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    
    // Set buffers
    [computeEncoder setBuffer:inputOutputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:residualBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:numElementsBuffer offset:0 atIndex:2];
    
    // Calculate thread group sizes
    auto [threadsPerThreadgroup, threadgroupsPerGrid] = MetalDispatch::calculateThreadGroups1D(num_elements);
    
    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];
    
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back to input_output
    std::memcpy(input_output, [inputOutputBuffer contents], num_elements * sizeof(bfloat16_t));
}

void metal_add_residual_inplace_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    float* input_output,
    const float* residual,
    size_t num_elements
) {
    if (num_elements == 0) return;
    
    // Check if pipeline is available
    if (g_add_residual_inplace_float32_pipeline == nil) {
        std::cerr << "Metal add_residual inplace float32 pipeline not initialized" << std::endl;
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g_add_residual_inplace_float32_pipeline];
    
    // Create buffers
    id<MTLBuffer> inputOutputBuffer = [device newBufferWithBytes:input_output
                                                          length:num_elements * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> residualBuffer = [device newBufferWithBytes:residual
                                                       length:num_elements * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    
    uint32_t numElementsUint = static_cast<uint32_t>(num_elements);
    id<MTLBuffer> numElementsBuffer = [device newBufferWithBytes:&numElementsUint
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    
    // Set buffers
    [computeEncoder setBuffer:inputOutputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:residualBuffer offset:0 atIndex:1];  
    [computeEncoder setBuffer:numElementsBuffer offset:0 atIndex:2];
    
    // Calculate thread group sizes
    auto [threadsPerThreadgroup, threadgroupsPerGrid] = MetalDispatch::calculateThreadGroups1D(num_elements);
    
    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];
    
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back to input_output
    std::memcpy(input_output, [inputOutputBuffer contents], num_elements * sizeof(float));
}