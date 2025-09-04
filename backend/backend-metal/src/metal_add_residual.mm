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
    @autoreleasepool {
        // Get default Metal device (similar to GEMM approach)
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to create Metal device" << std::endl;
            return false;
        }

        // Load Metal shader library from source file
        NSError* error = nil;
        NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
        NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
        NSString* metalFilePath = [dirPath stringByAppendingPathComponent:@"metal_add_residual.metal"];
        NSString* metalSource = [NSString stringWithContentsOfFile:metalFilePath
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];

        if (error || !metalSource) {
            std::cerr << "Failed to load Metal add_residual shader source: " <<
                         (error ? error.localizedDescription.UTF8String : "unknown error") << std::endl;
            return false;
        }

        // Compile Metal library
        id<MTLLibrary> library = [device newLibraryWithSource:metalSource options:nil error:&error];
        if (error || !library) {
            std::cerr << "Failed to compile Metal add_residual library: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        // Create kernel functions from the library
        id<MTLFunction> bfloat16Kernel = [library newFunctionWithName:@"add_residual_bfloat16_kernel"];
        id<MTLFunction> float32Kernel = [library newFunctionWithName:@"add_residual_float32_kernel"];
        id<MTLFunction> inplaceBfloat16Kernel = [library newFunctionWithName:@"add_residual_inplace_bfloat16_kernel"];
        id<MTLFunction> inplaceFloat32Kernel = [library newFunctionWithName:@"add_residual_inplace_float32_kernel"];

        if (!bfloat16Kernel || !float32Kernel || !inplaceBfloat16Kernel || !inplaceFloat32Kernel) {
            std::cerr << "Failed to find required Metal kernel functions in add_residual library" << std::endl;
            return false;
        }

        // Create compute pipeline states for each kernel
        g_add_residual_bfloat16_pipeline = [device newComputePipelineStateWithFunction:bfloat16Kernel error:&error];
        if (error) {
            std::cerr << "Failed to create add_residual_bfloat16_pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_add_residual_float32_pipeline = [device newComputePipelineStateWithFunction:float32Kernel error:&error];
        if (error) {
            std::cerr << "Failed to create add_residual_float32_pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_add_residual_inplace_bfloat16_pipeline = [device newComputePipelineStateWithFunction:inplaceBfloat16Kernel error:&error];
        if (error) {
            std::cerr << "Failed to create add_residual_inplace_bfloat16_pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_add_residual_inplace_float32_pipeline = [device newComputePipelineStateWithFunction:inplaceFloat32Kernel error:&error];
        if (error) {
            std::cerr << "Failed to create add_residual_inplace_float32_pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        // Vectorized kernels are optional - set to nil for now
        g_add_residual_bfloat16_vectorized_pipeline = nil;
        g_add_residual_float32_vectorized_pipeline = nil;

        // Check if main pipelines were created successfully
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

    // Calculate thread group sizes - function returns {threadgroupsPerGrid, threadsPerThreadgroup}
    auto [threadgroupsPerGrid, threadsPerThreadgroup] = MetalDispatch::calculateThreadGroups1D(num_elements);

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

    // Calculate thread group sizes - function returns {threadgroupsPerGrid, threadsPerThreadgroup}
    auto [threadgroupsPerGrid, threadsPerThreadgroup] = MetalDispatch::calculateThreadGroups1D(num_elements);

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

    // Calculate thread group sizes - function returns {threadgroupsPerGrid, threadsPerThreadgroup}
    auto [threadgroupsPerGrid, threadsPerThreadgroup] = MetalDispatch::calculateThreadGroups1D(num_elements);

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

    // Calculate thread group sizes - function returns {threadgroupsPerGrid, threadsPerThreadgroup}
    auto [threadgroupsPerGrid, threadsPerThreadgroup] = MetalDispatch::calculateThreadGroups1D(num_elements);

    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadsPerThreadgroup];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy results back to input_output
    std::memcpy(input_output, [inputOutputBuffer contents], num_elements * sizeof(float));
}