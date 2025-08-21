#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "metal_topk_mask_logits.hpp"
#include <iostream>
#include <cassert>

// Global Metal objects (initialized once)
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> topkF32PipelineState = nil;
static id<MTLComputePipelineState> topkBF16PipelineState = nil;
static dispatch_once_t onceToken;

static bool initialize_metal_topk_mask() {
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "Metal device creation failed" << std::endl;
                return;
            }

            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                std::cerr << "Metal command queue creation failed" << std::endl;
                return;
            }

            // Load the Metal file from the same directory as this .mm file
            NSError* error = nil;
            NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
            NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
            NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_topk_mask_logits.metal"];

            NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (error) {
                std::cerr << "Failed to read Metal source: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }

            library = [device newLibraryWithSource:source options:nil error:&error];
            if (error) {
                std::cerr << "Metal library compilation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }

            // Create pipeline states for both variants
            id<MTLFunction> topkF32Function = [library newFunctionWithName:@"metal_topk_mask_logits_float32"];
            id<MTLFunction> topkBF16Function = [library newFunctionWithName:@"metal_topk_mask_logits_bfloat16"];

            if (!topkF32Function) {
                std::cerr << "Failed to find metal_topk_mask_logits_float32 function" << std::endl;
                return;
            }

            if (!topkBF16Function) {
                std::cerr << "Failed to find metal_topk_mask_logits_bfloat16 function" << std::endl;
                return;
            }

            topkF32PipelineState = [device newComputePipelineStateWithFunction:topkF32Function error:&error];
            if (error) {
                std::cerr << "TopK F32 pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }

            topkBF16PipelineState = [device newComputePipelineStateWithFunction:topkBF16Function error:&error];
            if (error) {
                std::cerr << "TopK BF16 pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }
        }
    });

    return (device != nil && commandQueue != nil && library != nil &&
            topkF32PipelineState != nil && topkBF16PipelineState != nil);
}

struct TopKMaskParams {
    uint32_t num_tokens;     // Number of tokens (batch dimension)
    uint32_t vocab_size;     // Vocabulary size
    uint32_t k;              // Number of top-k values to keep per token
};

// Metal implementation of Top-K Mask Logits for float32
int metal_topk_mask_logits_float32(
    float* logits,                   // Input/output logits [num_tokens, vocab_size] (modified in-place)
    unsigned int num_tokens,         // Number of tokens (batch dimension)
    unsigned int vocab_size,         // Vocabulary size
    unsigned int k                   // Number of top-k values to keep per token
) {
    @autoreleasepool {
        if (!initialize_metal_topk_mask()) {
            std::cerr << "Metal TopK mask initialization failed" << std::endl;
            return -1;
        }

        // Validate inputs
        if (!logits || num_tokens == 0 || vocab_size == 0 || k == 0 || k > vocab_size) {
            std::cerr << "Invalid TopK mask parameters" << std::endl;
            return -2;
        }

        // Create Metal buffers
        const size_t logits_size = static_cast<size_t>(num_tokens) * vocab_size * sizeof(float);

        id<MTLBuffer> logitsBuffer = [device newBufferWithBytes:logits
                                                         length:logits_size
                                                        options:MTLResourceStorageModeShared];

        if (!logitsBuffer) {
            std::cerr << "Metal buffer allocation failed" << std::endl;
            return -3;
        }

        // Set up parameters
        TopKMaskParams params = {
            .num_tokens = num_tokens,
            .vocab_size = vocab_size,
            .k = k
        };

        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                         length:sizeof(TopKMaskParams)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:topkF32PipelineState];
        [encoder setBuffer:logitsBuffer offset:0 atIndex:0];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

        // Configure threadgroup and grid sizes
        // Each threadgroup processes one token
        NSUInteger threadsPerThreadgroup = 256;  // Match Metal kernel expectations
        NSUInteger threadgroupsPerGrid = num_tokens;

        MTLSize threadsPerThreadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize threadgroupsPerGridSize = MTLSizeMake(threadgroupsPerGrid, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGridSize
                threadsPerThreadgroup:threadsPerThreadgroupSize];

        [encoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for execution errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            std::cerr << "Metal command buffer execution failed" << std::endl;
            if (commandBuffer.error) {
                std::cerr << "Error: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
            }
            return -4;
        }

        // Copy result back to input buffer (in-place operation)
        memcpy(logits, logitsBuffer.contents, logits_size);

        return 0;  // Success
    }
}

// Metal implementation of Top-K Mask Logits for bfloat16
int metal_topk_mask_logits_bfloat16(
    void* logits,                    // Input/output logits [num_tokens, vocab_size] (modified in-place)
    unsigned int num_tokens,         // Number of tokens (batch dimension)
    unsigned int vocab_size,         // Vocabulary size
    unsigned int k                   // Number of top-k values to keep per token
) {
    @autoreleasepool {
        if (!initialize_metal_topk_mask()) {
            std::cerr << "Metal TopK mask initialization failed" << std::endl;
            return -1;
        }

        // Validate inputs
        if (!logits || num_tokens == 0 || vocab_size == 0 || k == 0 || k > vocab_size) {
            std::cerr << "Invalid TopK mask parameters" << std::endl;
            return -2;
        }

        // Create Metal buffers
        const size_t logits_size = static_cast<size_t>(num_tokens) * vocab_size * sizeof(uint16_t);

        id<MTLBuffer> logitsBuffer = [device newBufferWithBytes:logits
                                                         length:logits_size
                                                        options:MTLResourceStorageModeShared];

        if (!logitsBuffer) {
            std::cerr << "Metal buffer allocation failed" << std::endl;
            return -3;
        }

        // Set up parameters
        TopKMaskParams params = {
            .num_tokens = num_tokens,
            .vocab_size = vocab_size,
            .k = k
        };

        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                         length:sizeof(TopKMaskParams)
                                                        options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:topkBF16PipelineState];
        [encoder setBuffer:logitsBuffer offset:0 atIndex:0];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

        // Configure threadgroup and grid sizes
        // Each threadgroup processes one token
        NSUInteger threadsPerThreadgroup = 256;  // Match Metal kernel expectations
        NSUInteger threadgroupsPerGrid = num_tokens;

        MTLSize threadsPerThreadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize threadgroupsPerGridSize = MTLSizeMake(threadgroupsPerGrid, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGridSize
                threadsPerThreadgroup:threadsPerThreadgroupSize];

        [encoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for execution errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            std::cerr << "Metal command buffer execution failed" << std::endl;
            if (commandBuffer.error) {
                std::cerr << "Error: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
            }
            return -4;
        }

        // Copy result back to input buffer (in-place operation)
        memcpy(logits, logitsBuffer.contents, logits_size);

        return 0;  // Success
    }
}