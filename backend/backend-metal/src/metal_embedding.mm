#include "metal_embedding.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <iostream>

// Global Metal state for embedding operations
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLComputePipelineState> g_embeddingPipeline = nil;
static id<MTLComputePipelineState> g_embeddingVectorizedPipeline = nil;
static id<MTLComputePipelineState> g_embeddingF32Pipeline = nil;
static id<MTLLibrary> g_library = nil;

bool initialize_metal_embedding() {
    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "Failed to create Metal device for embedding" << std::endl;
            return false;
        }

        // Create command queue
        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) {
            std::cerr << "Failed to create Metal command queue for embedding" << std::endl;
            return false;
        }

        // Load Metal shader library from the same directory as this file
        NSError* error = nil;
        NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
        NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
        NSString* metalFilePath = [dirPath stringByAppendingPathComponent:@"metal_embedding.metal"];
        NSString* metalSource = [NSString stringWithContentsOfFile:metalFilePath
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];

        if (error || !metalSource) {
            std::cerr << "Failed to load Metal embedding shader source: " <<
                         (error ? error.localizedDescription.UTF8String : "unknown error") << std::endl;
            return false;
        }

        // Compile Metal library
        g_library = [g_device newLibraryWithSource:metalSource options:nil error:&error];
        if (error || !g_library) {
            std::cerr << "Failed to compile Metal embedding library: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

    // Get the embedding kernel functions
        id<MTLFunction> embeddingFunction = [g_library newFunctionWithName:@"metal_embedding_lookup_bfloat16"];
        if (!embeddingFunction) {
            std::cerr << "Failed to find metal_embedding_lookup_bfloat16 function" << std::endl;
            return false;
        }

        id<MTLFunction> embeddingVectorizedFunction = [g_library newFunctionWithName:@"metal_embedding_lookup_vectorized_bfloat16"];
        if (!embeddingVectorizedFunction) {
            std::cerr << "Failed to find metal_embedding_lookup_vectorized_bfloat16 function" << std::endl;
            return false;
        }

        id<MTLFunction> embeddingF32Function = [g_library newFunctionWithName:@"metal_embedding_lookup_float32"];
        if (!embeddingF32Function) {
            std::cerr << "Failed to find metal_embedding_lookup_float32 function" << std::endl;
            return false;
        }

        // Create compute pipeline states
        g_embeddingPipeline = [g_device newComputePipelineStateWithFunction:embeddingFunction error:&error];
        if (error || !g_embeddingPipeline) {
            std::cerr << "Failed to create embedding compute pipeline: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_embeddingVectorizedPipeline = [g_device newComputePipelineStateWithFunction:embeddingVectorizedFunction error:&error];
        if (error || !g_embeddingVectorizedPipeline) {
            std::cerr << "Failed to create vectorized embedding compute pipeline: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_embeddingF32Pipeline = [g_device newComputePipelineStateWithFunction:embeddingF32Function error:&error];
        if (error || !g_embeddingF32Pipeline) {
            std::cerr << "Failed to create f32 embedding compute pipeline: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        std::cout << "Metal Embedding initialized successfully" << std::endl;
        return true;
    }
}

void cleanup_metal_embedding() {
    g_embeddingPipeline = nil;
    g_embeddingVectorizedPipeline = nil;
    g_embeddingF32Pipeline = nil;
    g_library = nil;
    g_commandQueue = nil;
    g_device = nil;
}

void metal_embedding_lookup_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* embedding_matrix,
    size_t vocab_size,
    const int32_t* indices,
    size_t num_tokens,
    bfloat16_t* output,
    int hidden_size
) {
    @autoreleasepool {
        if (!g_device || !g_commandQueue || !g_embeddingVectorizedPipeline) {
            throw std::runtime_error("Metal Embedding not initialized. Call initialize_metal_embedding() first.");
        }

        // Calculate buffer sizes
        const size_t embedding_size = vocab_size * hidden_size * sizeof(bfloat16_t);
        const size_t indices_size = num_tokens * sizeof(int32_t);
        const size_t output_size = num_tokens * hidden_size * sizeof(bfloat16_t);

        // Create Metal buffers
        id<MTLBuffer> bufferEmbedding = [g_device newBufferWithBytes:embedding_matrix
                                                              length:embedding_size
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferIndices = [g_device newBufferWithBytes:indices
                                                            length:indices_size
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferOutput = [g_device newBufferWithLength:output_size
                                                           options:MTLResourceStorageModeShared];

        // Create parameters struct
        struct {
            uint32_t num_tokens;
            uint32_t hidden_size;
            uint32_t vocab_size;
        } params = {
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(hidden_size),
            static_cast<uint32_t>(vocab_size)
        };

        id<MTLBuffer> bufferParams = [g_device newBufferWithBytes:&params
                                                           length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set compute pipeline and buffers
        [encoder setComputePipelineState:g_embeddingVectorizedPipeline];  // Use vectorized version for better performance
        [encoder setBuffer:bufferEmbedding offset:0 atIndex:0];
        [encoder setBuffer:bufferIndices offset:0 atIndex:1];
        [encoder setBuffer:bufferOutput offset:0 atIndex:2];
        [encoder setBuffer:bufferParams offset:0 atIndex:3];

        // Configure threadgroup sizes
        // Each threadgroup processes one token lookup with 32 threads (SIMD group size)
        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);  // 32 threads per group (matches CUDA block size)
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_tokens, 1, 1);  // One group per token

        // Dispatch compute kernel
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.error) {
            NSString* errorDesc = [commandBuffer.error localizedDescription];
            throw std::runtime_error("Metal embedding compute failed: " + std::string([errorDesc UTF8String]));
        }

        // Copy result back to output buffer
        memcpy(output, [bufferOutput contents], output_size);
    }
}

void metal_embedding_lookup_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* embedding_matrix,
    size_t vocab_size,
    const int32_t* indices,
    size_t num_tokens,
    float* output,
    int hidden_size
) {
    @autoreleasepool {
        if (!g_device || !g_commandQueue || !g_embeddingF32Pipeline) {
            throw std::runtime_error("Metal Embedding not initialized. Call initialize_metal_embedding() first.");
        }

        const size_t embedding_size = vocab_size * hidden_size * sizeof(float);
        const size_t indices_size = num_tokens * sizeof(int32_t);
        const size_t output_size = num_tokens * hidden_size * sizeof(float);

        id<MTLBuffer> bufferEmbedding = [g_device newBufferWithBytes:embedding_matrix length:embedding_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferIndices = [g_device newBufferWithBytes:indices length:indices_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferOutput = [g_device newBufferWithLength:output_size options:MTLResourceStorageModeShared];

        struct {
            uint32_t num_tokens;
            uint32_t hidden_size;
            uint32_t vocab_size;
        } params = { static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(hidden_size), static_cast<uint32_t>(vocab_size) };
        id<MTLBuffer> bufferParams = [g_device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_embeddingF32Pipeline];
        [encoder setBuffer:bufferEmbedding offset:0 atIndex:0];
        [encoder setBuffer:bufferIndices offset:0 atIndex:1];
        [encoder setBuffer:bufferOutput offset:0 atIndex:2];
        [encoder setBuffer:bufferParams offset:0 atIndex:3];

        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_tokens, 1, 1);
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.error) {
            NSString* errorDesc = [commandBuffer.error localizedDescription];
            throw std::runtime_error("Metal embedding f32 compute failed: " + std::string([errorDesc UTF8String]));
        }
        memcpy(output, [bufferOutput contents], output_size);
    }
}