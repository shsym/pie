#include "metal_softmax.hpp"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>
#include <vector>

// Static Metal device and library - initialized once
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLLibrary> g_tiled_library = nil;
static id<MTLComputePipelineState> g_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_large_tiled_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_large_online_tiled_pipeline = nil;
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
        
        // Load tiled softmax library for large vocabularies
        NSString* tiledMetalPath = [dirPath stringByAppendingPathComponent:@"metal_softmax_tiled.metal"];
        NSString* tiledSource = [NSString stringWithContentsOfFile:tiledMetalPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
        
        if (!error && tiledSource) {
            g_tiled_library = [g_device newLibraryWithSource:tiledSource options:nil error:&error];
            if (!error && g_tiled_library) {
                // Create tiled pipeline
                id<MTLFunction> tiled_function = [g_tiled_library newFunctionWithName:@"softmax_large_tiled"];
                if (tiled_function) {
                    g_softmax_large_tiled_pipeline = [g_device newComputePipelineStateWithFunction:tiled_function error:&error];
                    if (error) {
                        std::cerr << "Failed to create tiled softmax pipeline: " << error.localizedDescription.UTF8String << std::endl;
                        g_softmax_large_tiled_pipeline = nil;
                    }
                }
                
                // Create online tiled pipeline (optional advanced optimization)
                id<MTLFunction> online_tiled_function = [g_tiled_library newFunctionWithName:@"softmax_large_online_tiled"];
                if (online_tiled_function) {
                    g_softmax_large_online_tiled_pipeline = [g_device newComputePipelineStateWithFunction:online_tiled_function error:&error];
                    if (error) {
                        std::cerr << "Failed to create online tiled softmax pipeline: " << error.localizedDescription.UTF8String << std::endl;
                        g_softmax_large_online_tiled_pipeline = nil;
                    }
                }
                
                std::cout << "Tiled softmax kernels loaded: " 
                          << (g_softmax_large_tiled_pipeline ? "basic=✅" : "basic=❌") << " "
                          << (g_softmax_large_online_tiled_pipeline ? "online=✅" : "online=❌") << std::endl;
            }
        } else {
            std::cout << "Tiled softmax kernels not found - using standard kernel only" << std::endl;
        }
        
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
        
        // Intelligent kernel selection based on vocabulary size
        id<MTLComputePipelineState> chosen_pipeline = g_softmax_pipeline;  // Default fallback
        uint32_t threadgroup_size = 256;  // Default for tiled kernels
        size_t shared_mem_size = 0;
        
        // Use tiled kernels for large vocabularies (>16K) where cache thrashing dominates
        if (vocab_size > 16384 && g_softmax_large_tiled_pipeline) {
            chosen_pipeline = g_softmax_large_tiled_pipeline;
            threadgroup_size = 256;  // Optimized for tiling
            
            const uint32_t TILE_SIZE = 2048;
            // Shared memory: threadgroup_size floats for reduction + TILE_SIZE for cache
            shared_mem_size = (threadgroup_size + TILE_SIZE) * sizeof(float);
            
            std::cout << "Using tiled kernel for vocab_size=" << vocab_size 
                      << " (cache-friendly, target 5-9x speedup)" << std::endl;
        } else {
            // Use standard kernel for smaller vocabularies
            const uint32_t max_threadgroup_size = 1024;
            threadgroup_size = std::min(static_cast<uint32_t>(vocab_size), max_threadgroup_size);
            shared_mem_size = threadgroup_size * sizeof(float);
        }
        
        [encoder setComputePipelineState:chosen_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBuffer:batch_size_buffer offset:0 atIndex:2];
        [encoder setBuffer:vocab_size_buffer offset:0 atIndex:3];
        [encoder setBuffer:temperature_buffer offset:0 atIndex:4];
        
        // Configure shared memory and dispatch parameters
        [encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0];
        
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