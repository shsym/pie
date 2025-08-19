#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_silu_and_mul.hpp"
#include <iostream>

// Global Metal context
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> pipelineStateBfloat16 = nil;
static id<MTLComputePipelineState> pipelineStateFloat32 = nil;
static bool initialized = false;

static bool initialize_metal_silu_and_mul() {
    if (initialized) return true;

    @autoreleasepool {
        // Get Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to create Metal device" << std::endl;
            return false;
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return false;
        }

        // Load the Metal file from the same directory as this .mm file
        NSError* error = nil;
        NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
        NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
        NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_silu_and_mul.metal"];

        NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (error) {
            std::cerr << "Failed to read Metal source: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        library = [device newLibraryWithSource:source options:nil error:&error];

        if (!library) {
            std::cerr << "Failed to load Metal library: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        // Create compute pipeline states
        id<MTLFunction> functionBfloat16 = [library newFunctionWithName:@"silu_and_mul_bfloat16_kernel"];
        id<MTLFunction> functionFloat32 = [library newFunctionWithName:@"silu_and_mul_float32_kernel"];

        if (!functionBfloat16 || !functionFloat32) {
            std::cerr << "Failed to load Metal functions" << std::endl;
            return false;
        }

        pipelineStateBfloat16 = [device newComputePipelineStateWithFunction:functionBfloat16 error:&error];
        pipelineStateFloat32 = [device newComputePipelineStateWithFunction:functionFloat32 error:&error];

        if (!pipelineStateBfloat16 || !pipelineStateFloat32) {
            std::cerr << "Failed to create Metal compute pipeline states: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        initialized = true;
        return true;
    }
}

int metal_silu_and_mul_bfloat16(
    const void* gate,
    const void* up,
    void* output,
    unsigned int num_tokens,
    unsigned int intermediate_size
) {
    if (!initialize_metal_silu_and_mul()) {
        return -1;
    }

    @autoreleasepool {
        // Calculate buffer sizes
        size_t buffer_size = num_tokens * intermediate_size * sizeof(uint16_t); // bfloat16 = 2 bytes

        // Create Metal buffers
        id<MTLBuffer> gateBuffer = [device newBufferWithBytes:gate
                                                       length:buffer_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> upBuffer = [device newBufferWithBytes:up
                                                     length:buffer_size
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:buffer_size
                                                         options:MTLResourceStorageModeShared];

        if (!gateBuffer || !upBuffer || !outputBuffer) {
            std::cerr << "Failed to create Metal buffers" << std::endl;
            return -1;
        }

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set pipeline state and buffers
        [encoder setComputePipelineState:pipelineStateBfloat16];
        [encoder setBuffer:gateBuffer offset:0 atIndex:0];
        [encoder setBuffer:upBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&num_tokens length:sizeof(unsigned int) atIndex:3];
        [encoder setBytes:&intermediate_size length:sizeof(unsigned int) atIndex:4];

        // Calculate thread groups: use 1D threadgroups in X to respect max threads per group (<=1024)
        const uint32_t threadsX = std::min(intermediate_size, 256u);
        MTLSize threadsPerGroup = MTLSizeMake(threadsX, 1, 1);
        MTLSize groupsPerGrid = MTLSizeMake(
            (intermediate_size + threadsX - 1) / threadsX,
            num_tokens,
            1
        );

        [encoder dispatchThreadgroups:groupsPerGrid threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(output, outputBuffer.contents, buffer_size);

        return 0;
    }
}

int metal_silu_and_mul_float32(
    const float* gate,
    const float* up,
    float* output,
    unsigned int num_tokens,
    unsigned int intermediate_size
) {
    if (!initialize_metal_silu_and_mul()) {
        return -1;
    }

    @autoreleasepool {
        // Calculate buffer sizes
        size_t buffer_size = num_tokens * intermediate_size * sizeof(float);

        // Create Metal buffers
        id<MTLBuffer> gateBuffer = [device newBufferWithBytes:gate
                                                       length:buffer_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> upBuffer = [device newBufferWithBytes:up
                                                     length:buffer_size
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:buffer_size
                                                         options:MTLResourceStorageModeShared];

        if (!gateBuffer || !upBuffer || !outputBuffer) {
            std::cerr << "Failed to create Metal buffers" << std::endl;
            return -1;
        }

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set pipeline state and buffers
        [encoder setComputePipelineState:pipelineStateFloat32];
        [encoder setBuffer:gateBuffer offset:0 atIndex:0];
        [encoder setBuffer:upBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&num_tokens length:sizeof(unsigned int) atIndex:3];
        [encoder setBytes:&intermediate_size length:sizeof(unsigned int) atIndex:4];

        // Calculate thread groups
        MTLSize threadsPerGroup = MTLSizeMake(std::min(intermediate_size, 256u), std::min(num_tokens, 256u), 1);
        MTLSize groupsPerGrid = MTLSizeMake(
            (intermediate_size + threadsPerGroup.width - 1) / threadsPerGroup.width,
            (num_tokens + threadsPerGroup.height - 1) / threadsPerGroup.height,
            1
        );

        [encoder dispatchThreadgroups:groupsPerGrid threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(output, outputBuffer.contents, buffer_size);

        return 0;
    }
}