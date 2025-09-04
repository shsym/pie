#import "metal_append_paged_kv_cache.hpp"
#import "metal_common.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <iostream>
#import <stdexcept>

namespace {
    id<MTLDevice> g_device = nil;
    id<MTLLibrary> g_library = nil;
    id<MTLComputePipelineState> g_append_kv_cache_bfloat16_pipeline = nil;
    id<MTLComputePipelineState> g_append_kv_cache_float32_pipeline = nil;
}

bool initialize_metal_append_paged_kv_cache() {
    @autoreleasepool {
        // Get Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "Failed to create Metal device" << std::endl;
            return false;
        }

        // Find and load the Metal shader file
        NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
        NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
        NSString* metalFilePath = [dirPath stringByAppendingPathComponent:@"metal_append_paged_kv_cache.metal"];

        NSError *error = nil;
        NSString *metalSource = [NSString stringWithContentsOfFile:metalFilePath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
        if (!metalSource) {
            std::cerr << "Failed to load Metal shader: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        // Compile the Metal library
        g_library = [g_device newLibraryWithSource:metalSource
                                           options:nil
                                             error:&error];
        if (!g_library) {
            std::cerr << "Failed to compile Metal library: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        // Create compute pipeline states
        id<MTLFunction> bfloat16Kernel = [g_library newFunctionWithName:@"metal_append_paged_kv_cache_bfloat16"];
        if (!bfloat16Kernel) {
            std::cerr << "Failed to find bfloat16 kernel function" << std::endl;
            return false;
        }

        g_append_kv_cache_bfloat16_pipeline = [g_device newComputePipelineStateWithFunction:bfloat16Kernel error:&error];
        if (!g_append_kv_cache_bfloat16_pipeline) {
            std::cerr << "Failed to create bfloat16 pipeline: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        id<MTLFunction> float32Kernel = [g_library newFunctionWithName:@"metal_append_paged_kv_cache_float32"];
        if (!float32Kernel) {
            std::cerr << "Failed to find float32 kernel function" << std::endl;
            return false;
        }

        g_append_kv_cache_float32_pipeline = [g_device newComputePipelineStateWithFunction:float32Kernel error:&error];
        if (!g_append_kv_cache_float32_pipeline) {
            std::cerr << "Failed to create float32 pipeline: " << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        return true;
    }
}

void metal_append_paged_kv_cache_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const void* k_input,
    const void* v_input,
    void* paged_k_cache,
    void* paged_v_cache,
    const uint32_t* kv_batch_indices,
    const uint32_t* kv_positions,
    const uint32_t* kv_page_indices,
    const uint32_t* kv_page_indptr,
    const uint32_t* kv_last_page_lens,
    uint32_t num_tokens,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t page_size,
    uint32_t max_num_pages,
    uint32_t batch_size
) {
    @autoreleasepool {
        if (!g_append_kv_cache_bfloat16_pipeline) {
            if (!initialize_metal_append_paged_kv_cache()) {
                throw std::runtime_error("Failed to initialize Metal append paged KV cache");
            }
        }

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            throw std::runtime_error("Failed to create command buffer");
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Failed to create compute command encoder");
        }

        [encoder setComputePipelineState:g_append_kv_cache_bfloat16_pipeline];

        // Create Metal buffers
        size_t kv_input_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size * sizeof(uint16_t);
        size_t page_data_size = static_cast<size_t>(max_num_pages) * page_size * num_kv_heads * head_size * sizeof(uint16_t);

        id<MTLBuffer> k_input_buffer = [device newBufferWithBytes:k_input
                                                           length:kv_input_size
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> v_input_buffer = [device newBufferWithBytes:v_input
                                                           length:kv_input_size
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> paged_k_cache_buffer = [device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                                       length:page_data_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];

        id<MTLBuffer> paged_v_cache_buffer = [device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                                       length:page_data_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];

        id<MTLBuffer> kv_batch_indices_buffer = [device newBufferWithBytes:kv_batch_indices
                                                                    length:num_tokens * sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_positions_buffer = [device newBufferWithBytes:kv_positions
                                                                length:num_tokens * sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_page_indices_buffer = [device newBufferWithBytes:kv_page_indices
                                                                   length:max_num_pages * sizeof(uint32_t)
                                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_page_indptr_buffer = [device newBufferWithBytes:kv_page_indptr
                                                                  length:(batch_size + 1) * sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_last_page_lens_buffer = [device newBufferWithBytes:kv_last_page_lens
                                                                     length:batch_size * sizeof(uint32_t)
                                                                    options:MTLResourceStorageModeShared];

        // Setup parameters
        struct AppendPagedKVCacheParams {
            uint32_t num_tokens;
            uint32_t num_kv_heads;
            uint32_t head_size;
            uint32_t page_size;
            uint32_t max_num_pages;
            uint32_t batch_size;
            uint32_t k_stride_token;
            uint32_t k_stride_head;
            uint32_t v_stride_token;
            uint32_t v_stride_head;
        };

        AppendPagedKVCacheParams params = {
            .num_tokens = num_tokens,
            .num_kv_heads = num_kv_heads,
            .head_size = head_size,
            .page_size = page_size,
            .max_num_pages = max_num_pages,
            .batch_size = batch_size,
            .k_stride_token = num_kv_heads * head_size,
            .k_stride_head = head_size,
            .v_stride_token = num_kv_heads * head_size,
            .v_stride_head = head_size
        };

        id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];

        // Set buffers
        [encoder setBuffer:k_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:v_input_buffer offset:0 atIndex:1];
        [encoder setBuffer:paged_k_cache_buffer offset:0 atIndex:2];
        [encoder setBuffer:paged_v_cache_buffer offset:0 atIndex:3];
        [encoder setBuffer:kv_batch_indices_buffer offset:0 atIndex:4];
        [encoder setBuffer:kv_positions_buffer offset:0 atIndex:5];
        [encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:6];
        [encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:7];
        [encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:8];
        [encoder setBuffer:params_buffer offset:0 atIndex:9];

        // Calculate thread dimensions
        // Each thread processes one element (token, head, head_offset)
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 16);  // Balanced for most head sizes
        MTLSize numThreadgroups = MTLSizeMake(
            (num_tokens + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (num_kv_heads + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            (head_size + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth
        );

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            NSString *errorDesc = commandBuffer.error.localizedDescription;
            throw std::runtime_error("Metal command buffer failed: " + std::string(errorDesc.UTF8String));
        }
    }
}

void metal_append_paged_kv_cache_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const void* k_input,
    const void* v_input,
    void* paged_k_cache,
    void* paged_v_cache,
    const uint32_t* kv_batch_indices,
    const uint32_t* kv_positions,
    const uint32_t* kv_page_indices,
    const uint32_t* kv_page_indptr,
    const uint32_t* kv_last_page_lens,
    uint32_t num_tokens,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t page_size,
    uint32_t max_num_pages,
    uint32_t batch_size
) {
    @autoreleasepool {
        if (!g_append_kv_cache_float32_pipeline) {
            if (!initialize_metal_append_paged_kv_cache()) {
                throw std::runtime_error("Failed to initialize Metal append paged KV cache");
            }
        }

        // Implementation follows same pattern as above but uses float32 pipeline and 4-byte elements

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_append_kv_cache_float32_pipeline];

        // Create buffers with float32 sizes
        size_t kv_input_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size * sizeof(float);
        size_t page_data_size = static_cast<size_t>(max_num_pages) * page_size * num_kv_heads * head_size * sizeof(float);

        id<MTLBuffer> k_input_buffer = [device newBufferWithBytes:k_input
                                                           length:kv_input_size
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> v_input_buffer = [device newBufferWithBytes:v_input
                                                           length:kv_input_size
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> paged_k_cache_buffer = [device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                                       length:page_data_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];

        id<MTLBuffer> paged_v_cache_buffer = [device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                                       length:page_data_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];

        id<MTLBuffer> kv_batch_indices_buffer = [device newBufferWithBytes:kv_batch_indices
                                                                    length:num_tokens * sizeof(uint32_t)
                                                                   options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_positions_buffer = [device newBufferWithBytes:kv_positions
                                                                length:num_tokens * sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_page_indices_buffer = [device newBufferWithBytes:kv_page_indices
                                                                   length:max_num_pages * sizeof(uint32_t)
                                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_page_indptr_buffer = [device newBufferWithBytes:kv_page_indptr
                                                                  length:(batch_size + 1) * sizeof(uint32_t)
                                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> kv_last_page_lens_buffer = [device newBufferWithBytes:kv_last_page_lens
                                                                     length:batch_size * sizeof(uint32_t)
                                                                    options:MTLResourceStorageModeShared];

        // Setup parameters
        struct AppendPagedKVCacheParams {
            uint32_t num_tokens;
            uint32_t num_kv_heads;
            uint32_t head_size;
            uint32_t page_size;
            uint32_t max_num_pages;
            uint32_t batch_size;
            uint32_t k_stride_token;
            uint32_t k_stride_head;
            uint32_t v_stride_token;
            uint32_t v_stride_head;
        };

        AppendPagedKVCacheParams params = {
            .num_tokens = num_tokens,
            .num_kv_heads = num_kv_heads,
            .head_size = head_size,
            .page_size = page_size,
            .max_num_pages = max_num_pages,
            .batch_size = batch_size,
            .k_stride_token = num_kv_heads * head_size,
            .k_stride_head = head_size,
            .v_stride_token = num_kv_heads * head_size,
            .v_stride_head = head_size
        };

        id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];

        // Set buffers
        [encoder setBuffer:k_input_buffer offset:0 atIndex:0];
        [encoder setBuffer:v_input_buffer offset:0 atIndex:1];
        [encoder setBuffer:paged_k_cache_buffer offset:0 atIndex:2];
        [encoder setBuffer:paged_v_cache_buffer offset:0 atIndex:3];
        [encoder setBuffer:kv_batch_indices_buffer offset:0 atIndex:4];
        [encoder setBuffer:kv_positions_buffer offset:0 atIndex:5];
        [encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:6];
        [encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:7];
        [encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:8];
        [encoder setBuffer:params_buffer offset:0 atIndex:9];

        // Calculate thread dimensions
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 16);  // Balanced for most head sizes
        MTLSize numThreadgroups = MTLSizeMake(
            (num_tokens + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (num_kv_heads + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            (head_size + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth
        );

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            NSString *errorDesc = commandBuffer.error.localizedDescription;
            throw std::runtime_error("Metal command buffer failed: " + std::string(errorDesc.UTF8String));
        }
    }
}