#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "metal_rmsnorm.hpp"
#include <iostream>
#include <cassert>

// Global Metal objects (initialized once)
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static id<MTLComputePipelineState> rmsnormPipelineState = nil;
static id<MTLComputePipelineState> rmsnormSimdPipelineState = nil;
static dispatch_once_t onceToken;

static bool initialize_metal_rmsnorm() {
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
            NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_rmsnorm.metal"];
            
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
            
            // Create pipeline states for both kernel variants
            id<MTLFunction> rmsnormFunction = [library newFunctionWithName:@"metal_rmsnorm_bfloat16"];
            id<MTLFunction> rmsnormSimdFunction = [library newFunctionWithName:@"metal_rmsnorm_simd_bfloat16"];
            
            if (!rmsnormFunction) {
                std::cerr << "Failed to find metal_rmsnorm_bfloat16 function" << std::endl;
                return;
            }
            
            if (!rmsnormSimdFunction) {
                std::cerr << "Failed to find metal_rmsnorm_simd_bfloat16 function" << std::endl;
                return;
            }
            
            rmsnormPipelineState = [device newComputePipelineStateWithFunction:rmsnormFunction error:&error];
            if (error) {
                std::cerr << "RMSNorm pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }
            
            rmsnormSimdPipelineState = [device newComputePipelineStateWithFunction:rmsnormSimdFunction error:&error];
            if (error) {
                std::cerr << "RMSNorm SIMD pipeline state creation failed: " << error.localizedDescription.UTF8String << std::endl;
                return;
            }
        }
    });
    
    return (device != nil && commandQueue != nil && library != nil && 
            rmsnormPipelineState != nil && rmsnormSimdPipelineState != nil);
}

struct RMSNormParams {
    uint32_t num_tokens;     // Number of tokens (sequence length)
    uint32_t hidden_size;    // Hidden dimension size
    float eps;               // Epsilon for numerical stability (e.g., 1e-5)
};

// Metal implementation of RMS Normalization for bfloat16
int metal_rmsnorm_bfloat16(
    const void* input,           // Input tensor [num_tokens, hidden_size]
    const void* weight,          // Weight tensor [hidden_size]
    void* output,                // Output tensor [num_tokens, hidden_size]
    unsigned int num_tokens,     // Number of tokens (sequence length)
    unsigned int hidden_size,    // Hidden dimension size
    float eps                    // Epsilon for numerical stability (e.g., 1e-5)
) {
    @autoreleasepool {
        if (!initialize_metal_rmsnorm()) {
            std::cerr << "Metal RMSNorm initialization failed" << std::endl;
            return -1;
        }
        
        // Validate inputs
        if (!input || !weight || !output || num_tokens == 0 || hidden_size == 0) {
            std::cerr << "Invalid RMSNorm parameters" << std::endl;
            return -2;
        }
        
        // Create Metal buffers
        const size_t input_size = static_cast<size_t>(num_tokens) * hidden_size * sizeof(uint16_t);
        const size_t weight_size = hidden_size * sizeof(uint16_t);
        const size_t output_size = static_cast<size_t>(num_tokens) * hidden_size * sizeof(uint16_t);
        
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input 
                                                        length:input_size 
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> weightBuffer = [device newBufferWithBytes:weight 
                                                         length:weight_size 
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_size 
                                                         options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !weightBuffer || !outputBuffer) {
            std::cerr << "Metal buffer allocation failed" << std::endl;
            return -3;
        }
        
        // Set up parameters
        RMSNormParams params = {
            .num_tokens = num_tokens,
            .hidden_size = hidden_size,
            .eps = eps
        };
        
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params 
                                                         length:sizeof(RMSNormParams) 
                                                        options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Choose pipeline based on problem size
        // Use SIMD variant for larger problems for better efficiency
        id<MTLComputePipelineState> pipelineState = (hidden_size >= 512) ? 
            rmsnormSimdPipelineState : rmsnormPipelineState;
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
        
        // Configure threadgroup and grid sizes
        // Each threadgroup processes one token, threads cooperate for reduction
        NSUInteger threadsPerThreadgroup = 256;  // Match Metal kernel expectations
        NSUInteger threadgroupsPerGrid = num_tokens;
        
        // Set threadgroup memory size for reduction (256 floats)
        [encoder setThreadgroupMemoryLength:threadsPerThreadgroup * sizeof(float) atIndex:0];
        
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
        
        // Copy result back to output buffer
        memcpy(output, outputBuffer.contents, output_size);
        
        return 0;  // Success
    }
}

// Metal implementation of RMS Normalization for float32
int metal_rmsnorm_float32(
    const float* input,          // Input tensor [num_tokens, hidden_size]
    const float* weight,         // Weight tensor [hidden_size]
    float* output,               // Output tensor [num_tokens, hidden_size]
    unsigned int num_tokens,     // Number of tokens (sequence length)
    unsigned int hidden_size,    // Hidden dimension size
    float eps                    // Epsilon for numerical stability (e.g., 1e-5)
) {
    // Note: This function would require a separate float32 Metal kernel
    // For now, we'll convert to bfloat16 and back (with precision loss warning)
    std::cerr << "Warning: metal_rmsnorm_float32 not implemented, using bfloat16 conversion" << std::endl;
    
    // Convert float32 to bfloat16
    std::vector<uint16_t> input_bf16(static_cast<size_t>(num_tokens) * hidden_size);
    std::vector<uint16_t> weight_bf16(hidden_size);
    std::vector<uint16_t> output_bf16(static_cast<size_t>(num_tokens) * hidden_size);
    
    // Simple float32 to bfloat16 conversion (truncate mantissa)
    for (size_t i = 0; i < input_bf16.size(); ++i) {
        uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&input[i]);
        input_bf16[i] = static_cast<uint16_t>((f32_bits + 0x8000) >> 16);
    }
    
    for (size_t i = 0; i < weight_bf16.size(); ++i) {
        uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&weight[i]);
        weight_bf16[i] = static_cast<uint16_t>((f32_bits + 0x8000) >> 16);
    }
    
    // Call bfloat16 implementation
    int result = metal_rmsnorm_bfloat16(
        input_bf16.data(), weight_bf16.data(), output_bf16.data(),
        num_tokens, hidden_size, eps
    );
    
    if (result != 0) {
        return result;
    }
    
    // Convert bfloat16 back to float32
    for (size_t i = 0; i < output_bf16.size(); ++i) {
        uint32_t f32_bits = static_cast<uint32_t>(output_bf16[i]) << 16;
        output[i] = *reinterpret_cast<const float*>(&f32_bits);
    }
    
    return 0;
}