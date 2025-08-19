#include "metal_softmax.hpp"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>
#include <vector>

// Static Metal device and library - initialized once
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_large_pipeline = nil;
static id<MTLCommandQueue> g_command_queue = nil;

static bool initialize_metal_softmax() {
    @autoreleasepool {
        if (g_device != nil) return true;
        
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "Failed to create Metal device for softmax" << std::endl;
            return false;
        }
        
        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "Failed to create command queue for softmax" << std::endl;
            return false;
        }
        
        // Load the Metal file from the same directory as this .mm file
        NSError* error = nil;
        NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
        NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
        NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_softmax.metal"];
        
        NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        
        if (error) {
            std::cerr << "Failed to load Metal source for softmax: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }
        
        g_library = [g_device newLibraryWithSource:source options:nil error:&error];
        if (error || !g_library) {
            std::cerr << "Failed to compile Metal library for softmax";
            if (error) std::cerr << ": " << error.localizedDescription.UTF8String;
            std::cerr << std::endl;
            return false;
        }
        
        // Create compute pipeline for standard softmax
        id<MTLFunction> softmax_function = [g_library newFunctionWithName:@"softmax_kernel"];
        if (!softmax_function) {
            std::cerr << "Failed to find softmax_kernel function" << std::endl;
            return false;
        }
        
        g_softmax_pipeline = [g_device newComputePipelineStateWithFunction:softmax_function error:&error];
        if (error || !g_softmax_pipeline) {
            std::cerr << "Failed to create softmax compute pipeline";
            if (error) std::cerr << ": " << error.localizedDescription.UTF8String;
            std::cerr << std::endl;
            return false;
        }
        
        // Large vocabulary pipeline commented out for now
        // TODO: Implement large vocabulary softmax pipeline when needed
        
        std::cout << "Metal softmax initialization successful" << std::endl;
        return true;
    }
}

int metal_softmax_float(
    const float* input,
    float* output,
    int batch_size,
    int vocab_size,
    float temperature
) {
    @autoreleasepool {
        if (!initialize_metal_softmax()) {
            return -1;
        }
        
        // Input validation
        if (!input || !output || batch_size <= 0 || vocab_size <= 0 || temperature <= 0.0f) {
            std::cerr << "Invalid parameters for metal_softmax_float" << std::endl;
            return -1;
        }
        
        const size_t input_size = static_cast<size_t>(batch_size) * vocab_size;
        
        // Create Metal buffers
        id<MTLBuffer> input_buffer = [g_device newBufferWithBytes:input
                                                           length:input_size * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> output_buffer = [g_device newBufferWithLength:input_size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> batch_size_buffer = [g_device newBufferWithBytes:&batch_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> vocab_size_buffer = [g_device newBufferWithBytes:&vocab_size
                                                               length:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> temperature_buffer = [g_device newBufferWithBytes:&temperature
                                                                length:sizeof(float)
                                                               options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !output_buffer || !batch_size_buffer || !vocab_size_buffer || !temperature_buffer) {
            std::cerr << "Failed to create Metal buffers for softmax" << std::endl;
            return -1;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        // Standard softmax kernel - handles any vocabulary size with loops
        [encoder setComputePipelineState:g_softmax_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:batch_size_buffer offset:0 atIndex:2];
        [encoder setBuffer:vocab_size_buffer offset:0 atIndex:3];
        [encoder setBuffer:temperature_buffer offset:0 atIndex:4];
        
        // Threadgroup size: min(vocab_size, max_threadgroup_size)
        const uint32_t max_threadgroup_size = 1024;
        uint32_t threadgroup_size = std::min(static_cast<uint32_t>(vocab_size), max_threadgroup_size);
        
        // Shared memory size: threadgroup_size floats
        [encoder setThreadgroupMemoryLength:threadgroup_size * sizeof(float) atIndex:0];
        
        MTLSize threads_per_threadgroup = MTLSizeMake(threadgroup_size, 1, 1);
        MTLSize threadgroups = MTLSizeMake(batch_size, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Check for Metal errors
        if (command_buffer.status == MTLCommandBufferStatusError) {
            std::cerr << "Metal command buffer error in softmax" << std::endl;
            if (command_buffer.error) {
                std::cerr << "Error: " << command_buffer.error.localizedDescription.UTF8String << std::endl;
            }
            return -1;
        }
        
        // Copy result back to output
        memcpy(output, output_buffer.contents, input_size * sizeof(float));
        
        return 0;
    }
}