#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "metal_rope.hpp"
#include <iostream>
#include <cassert>

// Global Metal objects (initialized once)
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> ropeBF16PipelineState = nil;
static id<MTLComputePipelineState> ropeF32PipelineState = nil;
static dispatch_once_t onceToken;

static bool initialize_metal_rope() {
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
            NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_rope.metal"];
            
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
            id<MTLFunction> ropeBF16Function = [library newFunctionWithName:@"metal_rope_bfloat16"];
            id<MTLFunction> ropeF32Function = [library newFunctionWithName:@"metal_rope_float32"];
            
            if (!ropeBF16Function) {
                std::cerr << "Failed to find metal_rope_bfloat16 function" << std::endl;
                return;
            }
            
            if (!ropeF32Function) {
                std::cerr << "Failed to find metal_rope_float32 function" << std::endl;
                return;
            }
            
            ropeBF16PipelineState = [device newComputePipelineStateWithFunction:ropeBF16Function error:&error];
            if (error) {
                std::cerr << "RoPE BF16 pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }
            
            ropeF32PipelineState = [device newComputePipelineStateWithFunction:ropeF32Function error:&error];
            if (error) {
                std::cerr << "RoPE F32 pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }
        }
    });
    
    return (device != nil && commandQueue != nil && library != nil && 
            ropeBF16PipelineState != nil && ropeF32PipelineState != nil);
}

struct RoPEParams {
    uint32_t num_tokens;     // Number of tokens in the sequence
    uint32_t num_heads;      // Number of attention heads
    uint32_t head_size;      // Size of each attention head
    float rope_theta;        // Base for rotary frequency computation (e.g., 10000.0)
    float rope_factor;       // Scaling factor for RoPE (e.g., 1.0)
};

// Metal implementation of RoPE for bfloat16
int metal_rope_bfloat16(
    void* input_qk,                  // Input/output tensor [num_tokens, num_heads, head_size]
    const int32_t* position_ids,     // Position IDs [num_tokens]
    unsigned int num_tokens,         // Number of tokens (sequence length)
    unsigned int num_heads,          // Number of attention heads
    unsigned int head_size,          // Size of each attention head
    float rope_theta,                // Base for rotary frequency (e.g., 10000.0)
    float rope_factor                // Scaling factor for RoPE (e.g., 1.0)
) {
    @autoreleasepool {
        if (!initialize_metal_rope()) {
            std::cerr << "Metal RoPE initialization failed" << std::endl;
            return -1;
        }
        
        // Validate inputs
        if (!input_qk || !position_ids || num_tokens == 0 || num_heads == 0 || head_size == 0) {
            std::cerr << "Invalid RoPE parameters" << std::endl;
            return -2;
        }
        
        // RoPE requires head_size to be even (pairs of elements)
        if (head_size % 2 != 0) {
            std::cerr << "RoPE requires even head_size, got: " << head_size << std::endl;
            return -2;
        }
        
        // Create Metal buffers
        const size_t tensor_size = static_cast<size_t>(num_tokens) * num_heads * head_size * sizeof(uint16_t);
        const size_t position_size = num_tokens * sizeof(int32_t);
        
        id<MTLBuffer> tensorBuffer = [device newBufferWithBytes:input_qk 
                                                         length:tensor_size 
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> positionBuffer = [device newBufferWithBytes:position_ids 
                                                           length:position_size 
                                                          options:MTLResourceStorageModeShared];
        
        if (!tensorBuffer || !positionBuffer) {
            std::cerr << "Metal buffer allocation failed" << std::endl;
            return -3;
        }
        
        // Set up parameters
        RoPEParams params = {
            .num_tokens = num_tokens,
            .num_heads = num_heads,
            .head_size = head_size,
            .rope_theta = rope_theta,
            .rope_factor = rope_factor
        };
        
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params 
                                                         length:sizeof(RoPEParams) 
                                                        options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:ropeBF16PipelineState];
        [encoder setBuffer:tensorBuffer offset:0 atIndex:0];
        [encoder setBuffer:positionBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
        
        // Configure dispatch: we need (num_tokens, num_heads, head_size/2) threads
        // Each thread processes one pair of elements
        MTLSize threadsPerGrid = MTLSizeMake(num_tokens, num_heads, head_size / 2);
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 4, 4);  // Adjust based on GPU capabilities
        
        // Ensure threadgroup size doesn't exceed limits
        NSUInteger maxThreadsPerThreadgroup = ropeBF16PipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger totalThreadsPerThreadgroup = threadsPerThreadgroup.width * 
                                                threadsPerThreadgroup.height * 
                                                threadsPerThreadgroup.depth;
        
        if (totalThreadsPerThreadgroup > maxThreadsPerThreadgroup) {
            // Fallback to smaller threadgroup
            threadsPerThreadgroup = MTLSizeMake(8, 4, 4);
            totalThreadsPerThreadgroup = 8 * 4 * 4;
            
            if (totalThreadsPerThreadgroup > maxThreadsPerThreadgroup) {
                threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
            }
        }
        
        [encoder dispatchThreads:threadsPerGrid 
           threadsPerThreadgroup:threadsPerThreadgroup];
        
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
        memcpy(input_qk, tensorBuffer.contents, tensor_size);
        
        return 0;  // Success
    }
}

// Metal implementation of RoPE for float32
int metal_rope_float32(
    float* input_qk,                 // Input/output tensor [num_tokens, num_heads, head_size]
    const int32_t* position_ids,     // Position IDs [num_tokens]
    unsigned int num_tokens,         // Number of tokens (sequence length)
    unsigned int num_heads,          // Number of attention heads
    unsigned int head_size,          // Size of each attention head
    float rope_theta,                // Base for rotary frequency (e.g., 10000.0)
    float rope_factor                // Scaling factor for RoPE (e.g., 1.0)
) {
    @autoreleasepool {
        if (!initialize_metal_rope()) {
            std::cerr << "Metal RoPE initialization failed" << std::endl;
            return -1;
        }
        
        // Validate inputs
        if (!input_qk || !position_ids || num_tokens == 0 || num_heads == 0 || head_size == 0) {
            std::cerr << "Invalid RoPE parameters" << std::endl;
            return -2;
        }
        
        // RoPE requires head_size to be even (pairs of elements)
        if (head_size % 2 != 0) {
            std::cerr << "RoPE requires even head_size, got: " << head_size << std::endl;
            return -2;
        }
        
        // Create Metal buffers
        const size_t tensor_size = static_cast<size_t>(num_tokens) * num_heads * head_size * sizeof(float);
        const size_t position_size = num_tokens * sizeof(int32_t);
        
        id<MTLBuffer> tensorBuffer = [device newBufferWithBytes:input_qk 
                                                         length:tensor_size 
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> positionBuffer = [device newBufferWithBytes:position_ids 
                                                           length:position_size 
                                                          options:MTLResourceStorageModeShared];
        
        if (!tensorBuffer || !positionBuffer) {
            std::cerr << "Metal buffer allocation failed" << std::endl;
            return -3;
        }
        
        // Set up parameters
        RoPEParams params = {
            .num_tokens = num_tokens,
            .num_heads = num_heads,
            .head_size = head_size,
            .rope_theta = rope_theta,
            .rope_factor = rope_factor
        };
        
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params 
                                                         length:sizeof(RoPEParams) 
                                                        options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:ropeF32PipelineState];
        [encoder setBuffer:tensorBuffer offset:0 atIndex:0];
        [encoder setBuffer:positionBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
        
        // Configure dispatch: we need (num_tokens, num_heads, head_size/2) threads
        // Each thread processes one pair of elements
        MTLSize threadsPerGrid = MTLSizeMake(num_tokens, num_heads, head_size / 2);
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 4, 4);  // Adjust based on GPU capabilities
        
        // Ensure threadgroup size doesn't exceed limits
        NSUInteger maxThreadsPerThreadgroup = ropeF32PipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger totalThreadsPerThreadgroup = threadsPerThreadgroup.width * 
                                                threadsPerThreadgroup.height * 
                                                threadsPerThreadgroup.depth;
        
        if (totalThreadsPerThreadgroup > maxThreadsPerThreadgroup) {
            // Fallback to smaller threadgroup
            threadsPerThreadgroup = MTLSizeMake(8, 4, 4);
            totalThreadsPerThreadgroup = 8 * 4 * 4;
            
            if (totalThreadsPerThreadgroup > maxThreadsPerThreadgroup) {
                threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
            }
        }
        
        [encoder dispatchThreads:threadsPerGrid 
           threadsPerThreadgroup:threadsPerThreadgroup];
        
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
        memcpy(input_qk, tensorBuffer.contents, tensor_size);
        
        return 0;  // Success
    }
}