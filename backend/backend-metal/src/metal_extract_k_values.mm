#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_extract_k_values.hpp"
#include <iostream>

// Global Metal context
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> pipelineStateBfloat16 = nil;
static id<MTLComputePipelineState> pipelineStateFloat32 = nil;
static bool initialized = false;

static bool initialize_metal_extract_k_values() {
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
        NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_extract_k_values.metal"];
        
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
        id<MTLFunction> functionBfloat16 = [library newFunctionWithName:@"extract_k_values_bfloat16_kernel"];
        id<MTLFunction> functionFloat32 = [library newFunctionWithName:@"extract_k_values_float32_kernel"];
        
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

int metal_extract_k_values_bfloat16(
    const void* A,
    void* V,
    int32_t* I,
    unsigned int M,
    unsigned int N,
    unsigned int k
) {
    if (!initialize_metal_extract_k_values()) {
        return -1;
    }
    
    @autoreleasepool {
        // Calculate buffer sizes
        size_t input_size = M * N * sizeof(uint16_t);    // bfloat16 = 2 bytes
        size_t output_value_size = M * k * sizeof(uint16_t);
        size_t output_index_size = M * k * sizeof(int32_t);
        
        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:A 
                                                        length:input_size 
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> valueBuffer = [device newBufferWithLength:output_value_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> indexBuffer = [device newBufferWithLength:output_index_size 
                                                        options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !valueBuffer || !indexBuffer) {
            std::cerr << "Failed to create Metal buffers" << std::endl;
            return -1;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline state and buffers
        [encoder setComputePipelineState:pipelineStateBfloat16];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:valueBuffer offset:0 atIndex:1];
        [encoder setBuffer:indexBuffer offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(unsigned int) atIndex:3];
        [encoder setBytes:&N length:sizeof(unsigned int) atIndex:4];
        [encoder setBytes:&k length:sizeof(unsigned int) atIndex:5];
        
        // Calculate thread groups - one threadgroup per row, 256 threads per group
        NSUInteger threadsPerThreadgroup = 256;
        NSUInteger threadgroupsPerGrid = M;
        
        // Allocate threadgroup memory for shared counter
        [encoder setThreadgroupMemoryLength:sizeof(int) atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroupsPerGrid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(V, valueBuffer.contents, output_value_size);
        memcpy(I, indexBuffer.contents, output_index_size);
        
        return 0;
    }
}

int metal_extract_k_values_float32(
    const float* A,
    float* V,
    int32_t* I,
    unsigned int M,
    unsigned int N,
    unsigned int k
) {
    if (!initialize_metal_extract_k_values()) {
        return -1;
    }
    
    @autoreleasepool {
        // Calculate buffer sizes
        size_t input_size = M * N * sizeof(float);
        size_t output_value_size = M * k * sizeof(float);
        size_t output_index_size = M * k * sizeof(int32_t);
        
        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:A 
                                                        length:input_size 
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> valueBuffer = [device newBufferWithLength:output_value_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> indexBuffer = [device newBufferWithLength:output_index_size 
                                                        options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !valueBuffer || !indexBuffer) {
            std::cerr << "Failed to create Metal buffers" << std::endl;
            return -1;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline state and buffers
        [encoder setComputePipelineState:pipelineStateFloat32];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:valueBuffer offset:0 atIndex:1];
        [encoder setBuffer:indexBuffer offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(unsigned int) atIndex:3];
        [encoder setBytes:&N length:sizeof(unsigned int) atIndex:4];
        [encoder setBytes:&k length:sizeof(unsigned int) atIndex:5];
        
        // Calculate thread groups - one threadgroup per row, 256 threads per group  
        NSUInteger threadsPerThreadgroup = 256;
        NSUInteger threadgroupsPerGrid = M;
        
        // Allocate threadgroup memory for shared counter
        [encoder setThreadgroupMemoryLength:sizeof(int) atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroupsPerGrid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(V, valueBuffer.contents, output_value_size);
        memcpy(I, indexBuffer.contents, output_index_size);
        
        return 0;
    }
}