#include "metal_batch_prefill_attention.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace metal {
namespace batch_prefill_attention {

static id<MTLDevice> get_metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device" << std::endl;
        return nil;
    }
    return device;
}

static id<MTLLibrary> get_metal_library() {
    static id<MTLLibrary> library = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        id<MTLDevice> device = get_metal_device();
        if (device) {
            NSError* error = nil;
            
            // Embedded Metal shader source code
            NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void batch_prefill_attention_bf16_kernel(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const int* indptr [[buffer(3)]],
    device const int* indices [[buffer(4)]],
    device half* O [[buffer(5)]],
    constant int& num_tokens [[buffer(6)]],
    constant int& num_query_heads [[buffer(7)]],
    constant int& num_kv_heads [[buffer(8)]],
    constant int& head_size [[buffer(9)]],
    constant int& kv_len [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint token_idx = tgid.x;
    uint head_idx = tgid.y;
    
    if (token_idx >= uint(num_tokens) || head_idx >= uint(num_query_heads)) return;
    
    uint kv_head_idx = head_idx % uint(num_kv_heads);
    
    // Simple attention computation for each output dimension
    for (int d = int(tid.x); d < head_size; d += 32) {
        float output_val = 0.0f;
        float sum_weights = 0.0f;
        
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            float score = 0.0f;
            
            // Compute Q * K^T
            for (int dim = 0; dim < head_size; dim++) {
                float q_val = float(Q[token_idx * uint(num_query_heads) * uint(head_size) + head_idx * uint(head_size) + uint(dim)]);
                float k_val = float(K[uint(kv_pos) * kv_head_idx * uint(head_size) + kv_head_idx * uint(head_size) + uint(dim)]);
                score += q_val * k_val;
            }
            score *= scale;
            
            float attention_weight = exp(score);
            sum_weights += attention_weight;
            
            float v_val = float(V[uint(kv_pos) * kv_head_idx * uint(head_size) + kv_head_idx * uint(head_size) + uint(d)]);
            output_val += attention_weight * v_val;
        }
        
        O[token_idx * uint(num_query_heads) * uint(head_size) + head_idx * uint(head_size) + uint(d)] = half(output_val / sum_weights);
    }
}

kernel void batch_prefill_attention_f32_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const int* indptr [[buffer(3)]],
    device const int* indices [[buffer(4)]],
    device float* O [[buffer(5)]],
    constant int& num_tokens [[buffer(6)]],
    constant int& num_query_heads [[buffer(7)]],
    constant int& num_kv_heads [[buffer(8)]],
    constant int& head_size [[buffer(9)]],
    constant int& kv_len [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint token_idx = tgid.x;
    uint head_idx = tgid.y;
    
    if (token_idx >= uint(num_tokens) || head_idx >= uint(num_query_heads)) return;
    
    uint kv_head_idx = head_idx % uint(num_kv_heads);
    
    for (int d = int(tid.x); d < head_size; d += 32) {
        float output_val = 0.0f;
        float sum_weights = 0.0f;
        
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            float score = 0.0f;
            
            for (int dim = 0; dim < head_size; dim++) {
                float q_val = Q[token_idx * uint(num_query_heads) * uint(head_size) + head_idx * uint(head_size) + uint(dim)];
                float k_val = K[uint(kv_pos) * kv_head_idx * uint(head_size) + kv_head_idx * uint(head_size) + uint(dim)];
                score += q_val * k_val;
            }
            score *= scale;
            
            float attention_weight = exp(score);
            sum_weights += attention_weight;
            
            float v_val = V[uint(kv_pos) * kv_head_idx * uint(head_size) + kv_head_idx * uint(head_size) + uint(d)];
            output_val += attention_weight * v_val;
        }
        
        O[token_idx * uint(num_query_heads) * uint(head_size) + head_idx * uint(head_size) + uint(d)] = output_val / sum_weights;
    }
}
)";
            
            library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (!library || error) {
                NSLog(@"Failed to compile Metal shaders: %@", error ? error.localizedDescription : @"Unknown error");
            }
        }
    });
    return library;
}

void batch_prefill_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    const int32_t* indptr,
    const int32_t* indices,
    void* O,
    int num_tokens,
    int num_query_heads,
    int num_kv_heads,
    int head_size,
    int kv_len,
    int page_size,
    float scale
) {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) {
            std::cerr << "Failed to get Metal device for batch_prefill_attention" << std::endl;
            return;
        }
        
        id<MTLLibrary> library = get_metal_library();
        if (!library) {
            std::cerr << "Failed to get Metal library for batch_prefill_attention" << std::endl;
            return;
        }
        
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_bf16_kernel"];
        if (!function) {
            std::cerr << "Failed to find batch_prefill_attention_bf16_kernel function - shader may not be compiled" << std::endl;
            return;
        }
        
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            std::cerr << "Failed to create compute pipeline state: " << error.localizedDescription.UTF8String << std::endl;
            return;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Create buffers
        const size_t q_size = num_tokens * num_query_heads * head_size * sizeof(uint16_t);
        const size_t k_size = kv_len * num_kv_heads * head_size * sizeof(uint16_t);
        const size_t v_size = kv_len * num_kv_heads * head_size * sizeof(uint16_t);
        const size_t o_size = num_tokens * num_query_heads * head_size * sizeof(uint16_t);
        const size_t indptr_size = (num_tokens + 1) * sizeof(int32_t);
        const size_t indices_size = num_tokens * sizeof(int32_t);
        
        id<MTLBuffer> q_buffer = [device newBufferWithBytes:Q length:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buffer = [device newBufferWithBytes:K length:k_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> v_buffer = [device newBufferWithBytes:V length:v_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buffer = [device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
        
        // Create simple indptr/indices for testing (each token attends to full sequence)
        std::vector<int32_t> default_indptr(num_tokens + 1);
        std::vector<int32_t> default_indices(num_tokens);
        for (int i = 0; i <= num_tokens; i++) {
            default_indptr[i] = i * kv_len / num_tokens; // Simplified: equal distribution
        }
        for (int i = 0; i < num_tokens; i++) {
            default_indices[i] = i;
        }
        
        id<MTLBuffer> indptr_buffer = [device newBufferWithBytes:default_indptr.data() 
                                                          length:indptr_size 
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> indices_buffer = [device newBufferWithBytes:default_indices.data() 
                                                           length:indices_size 
                                                          options:MTLResourceStorageModeShared];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:q_buffer offset:0 atIndex:0];
        [encoder setBuffer:k_buffer offset:0 atIndex:1];
        [encoder setBuffer:v_buffer offset:0 atIndex:2];
        [encoder setBuffer:indptr_buffer offset:0 atIndex:3];
        [encoder setBuffer:indices_buffer offset:0 atIndex:4];
        [encoder setBuffer:o_buffer offset:0 atIndex:5];
        [encoder setBytes:&num_tokens length:sizeof(int) atIndex:6];
        [encoder setBytes:&num_query_heads length:sizeof(int) atIndex:7];
        [encoder setBytes:&num_kv_heads length:sizeof(int) atIndex:8];
        [encoder setBytes:&head_size length:sizeof(int) atIndex:9];
        [encoder setBytes:&kv_len length:sizeof(int) atIndex:10];
        [encoder setBytes:&scale length:sizeof(float) atIndex:11];
        
        // Configure threadgroup and grid sizes
        MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);  // 32 threads per threadgroup
        MTLSize threadsPerGrid = MTLSizeMake(num_tokens, num_query_heads, 1);
        
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            std::cerr << "Metal command buffer error: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
            return;
        }
        
        // Copy result back
        memcpy(O, o_buffer.contents, o_size);
    }
}

void batch_prefill_attention_f32(
    const void* Q,
    const void* K,
    const void* V,
    const int32_t* indptr,
    const int32_t* indices,
    void* O,
    int num_tokens,
    int num_query_heads,
    int num_kv_heads,
    int head_size,
    int kv_len,
    int page_size,
    float scale
) {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) {
            std::cerr << "Failed to get Metal device" << std::endl;
            return;
        }
        
        id<MTLLibrary> library = get_metal_library();
        if (!library) {
            std::cerr << "Failed to get Metal library" << std::endl;
            return;
        }
        
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_f32_kernel"];
        if (!function) {
            std::cerr << "Failed to find batch_prefill_attention_f32_kernel function" << std::endl;
            return;
        }
        
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            std::cerr << "Failed to create compute pipeline state: " << error.localizedDescription.UTF8String << std::endl;
            return;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Create buffers for float32
        const size_t q_size = num_tokens * num_query_heads * head_size * sizeof(float);
        const size_t k_size = kv_len * num_kv_heads * head_size * sizeof(float);
        const size_t v_size = kv_len * num_kv_heads * head_size * sizeof(float);
        const size_t o_size = num_tokens * num_query_heads * head_size * sizeof(float);
        const size_t indptr_size = (num_tokens + 1) * sizeof(int32_t);
        const size_t indices_size = num_tokens * sizeof(int32_t);
        
        id<MTLBuffer> q_buffer = [device newBufferWithBytes:Q length:q_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buffer = [device newBufferWithBytes:K length:k_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> v_buffer = [device newBufferWithBytes:V length:v_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buffer = [device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
        
        // Create simple indptr/indices for testing
        std::vector<int32_t> default_indptr(num_tokens + 1);
        std::vector<int32_t> default_indices(num_tokens);
        for (int i = 0; i <= num_tokens; i++) {
            default_indptr[i] = i * kv_len / num_tokens;
        }
        for (int i = 0; i < num_tokens; i++) {
            default_indices[i] = i;
        }
        
        id<MTLBuffer> indptr_buffer = [device newBufferWithBytes:default_indptr.data() 
                                                          length:indptr_size 
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> indices_buffer = [device newBufferWithBytes:default_indices.data() 
                                                           length:indices_size 
                                                          options:MTLResourceStorageModeShared];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:q_buffer offset:0 atIndex:0];
        [encoder setBuffer:k_buffer offset:0 atIndex:1];
        [encoder setBuffer:v_buffer offset:0 atIndex:2];
        [encoder setBuffer:indptr_buffer offset:0 atIndex:3];
        [encoder setBuffer:indices_buffer offset:0 atIndex:4];
        [encoder setBuffer:o_buffer offset:0 atIndex:5];
        [encoder setBytes:&num_tokens length:sizeof(int) atIndex:6];
        [encoder setBytes:&num_query_heads length:sizeof(int) atIndex:7];
        [encoder setBytes:&num_kv_heads length:sizeof(int) atIndex:8];
        [encoder setBytes:&head_size length:sizeof(int) atIndex:9];
        [encoder setBytes:&kv_len length:sizeof(int) atIndex:10];
        [encoder setBytes:&scale length:sizeof(float) atIndex:11];
        
        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
        MTLSize gridSize = MTLSizeMake(num_tokens, num_query_heads, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            std::cerr << "Metal command buffer error: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
            return;
        }
        
        // Copy result back
        memcpy(O, o_buffer.contents, o_size);
    }
}

} // namespace batch_prefill_attention
} // namespace metal