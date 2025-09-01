#include "metal_attention_naive.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

// Reuse conversion utilities from optimized version
namespace {
    using bfloat16_t = uint16_t;

    inline float bf16_to_float(bfloat16_t bf16) {
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    inline uint16_t float_to_half(float f) {
        union { float f; uint32_t i; } u;
        u.f = f;

        if (f == 0.0f) return u.i >> 16;
        if (!std::isfinite(f)) {
            if (std::isnan(f)) return 0x7e00;
            return (u.i >> 16) | 0x7c00;
        }

        uint32_t sign = (u.i >> 16) & 0x8000;
        int32_t exp = ((u.i >> 23) & 0xff) - 127 + 15;
        uint32_t mantissa = (u.i >> 13) & 0x3ff;

        if (exp <= 0) {
            return static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00);
        } else {
            return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
        }
    }

    inline uint16_t bf16_to_half(bfloat16_t bf16) {
        return float_to_half(bf16_to_float(bf16));
    }

    inline float half_to_float(uint16_t h) {
        uint16_t h_exp = (h & 0x7C00u) >> 10;
        uint16_t h_sig = (h & 0x03FFu);
        uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;

        uint32_t f;
        if (h_exp == 0) {
            if (h_sig == 0) {
                f = sign;
            } else {
                int shift = 0;
                while ((h_sig & 0x0400u) == 0) { h_sig <<= 1; ++shift; }
                h_sig &= 0x03FFu;
                uint32_t exp = 127 - 15 - shift;
                uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
                f = sign | (exp << 23) | mant;
            }
        } else if (h_exp == 0x1Fu) {
            uint32_t exp = 0xFFu;
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        } else {
            uint32_t exp = static_cast<uint32_t>(h_exp) - 15 + 127;
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }

    inline bfloat16_t float_to_bf16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
    }

    std::vector<uint16_t> convert_bf16_to_half(const void* bf16_data, size_t count) {
        const bfloat16_t* src = static_cast<const bfloat16_t*>(bf16_data);
        std::vector<uint16_t> result(count);

        for (size_t i = 0; i < count; ++i) {
            result[i] = bf16_to_half(src[i]);
        }

        return result;
    }
}

namespace metal {
namespace naive_attention {

MetalNaiveAttentionHandle* metal_naive_attention_create_handle(
    int max_batch_size,
    int max_seq_length,
    int max_heads,
    int max_head_dim
) {
    MetalNaiveAttentionHandle* handle = new MetalNaiveAttentionHandle();

    // Initialize device
    handle->device = (__bridge_retained void*)MTLCreateSystemDefaultDevice();
    if (!handle->device) {
        std::cerr << "MetalNaiveAttentionHandle: Metal is not supported on this device" << std::endl;
        delete handle;
        return nullptr;
    }

    // Create command queue
    handle->commandQueue = (__bridge_retained void*)[(__bridge id<MTLDevice>)handle->device newCommandQueue];
    if (!handle->commandQueue) {
        std::cerr << "MetalNaiveAttentionHandle: Failed to create command queue" << std::endl;
        delete handle;
        return nullptr;
    }

    // Load Metal library with naive kernels
    NSError* error = nil;
    NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
    NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
    NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_attention_naive.metal"];
    NSString* metalSource = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];

    if (error || !metalSource) {
        std::cerr << "MetalNaiveAttentionHandle: Failed to load naive Metal source: "
                  << (error ? error.localizedDescription.UTF8String : "file not found") << std::endl;
        delete handle;
        return nullptr;
    }

    handle->library = (__bridge_retained void*)[(__bridge id<MTLDevice>)handle->device newLibraryWithSource:metalSource options:nil error:&error];
    if (!handle->library || error) {
        std::cerr << "MetalNaiveAttentionHandle: Failed to compile naive Metal library: "
                  << (error ? error.localizedDescription.UTF8String : "unknown error") << std::endl;
        delete handle;
        return nullptr;
    }

    // Create compute pipeline states for naive kernels
    id<MTLFunction> bf16_function = [(__bridge id<MTLLibrary>)handle->library newFunctionWithName:@"naive_attention_bf16_kernel"];
    if (bf16_function) {
        handle->pipeline_bf16 = (__bridge_retained void*)[(__bridge id<MTLDevice>)handle->device newComputePipelineStateWithFunction:bf16_function error:&error];
        if (error) {
            std::cerr << "MetalNaiveAttentionHandle: Failed to create naive BF16 pipeline: "
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }

    id<MTLFunction> f32_function = [(__bridge id<MTLLibrary>)handle->library newFunctionWithName:@"naive_attention_f32_kernel"];
    if (f32_function) {
        handle->pipeline_f32 = (__bridge_retained void*)[(__bridge id<MTLDevice>)handle->device newComputePipelineStateWithFunction:f32_function error:&error];
        if (error) {
            std::cerr << "MetalNaiveAttentionHandle: Failed to create naive F32 pipeline: "
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }

    // Set configuration bounds
    handle->max_batch_size = max_batch_size;
    handle->max_seq_length = max_seq_length;
    handle->max_heads = max_heads;
    handle->max_head_dim = max_head_dim;

    // Initialize statistics
    handle->total_calls = 0;
    handle->total_bytes_processed = 0;
    handle->initialized = true;

    std::cout << "MetalNaiveAttentionHandle: Successfully created NAIVE attention handle with bounds: "
              << "batch=" << max_batch_size << ", seq=" << max_seq_length
              << ", heads=" << max_heads << ", head_dim=" << max_head_dim << std::endl;
    std::cout << "                          *** THIS IS THE O(n²) BASELINE FOR COMPARISON ***" << std::endl;

    return handle;
}

void metal_naive_attention_destroy_handle(MetalNaiveAttentionHandle* handle) {
    if (!handle) return;

    std::cout << "MetalNaiveAttentionHandle: Destroying naive handle. Total calls: " << handle->total_calls
              << ", Total bytes processed: " << (handle->total_bytes_processed / 1024 / 1024) << " MB" << std::endl;

    // Release Metal objects (ARC will handle this automatically with __bridge_retained)
    if (handle->device) { (void)(__bridge_transfer id)handle->device; }
    if (handle->commandQueue) { (void)(__bridge_transfer id)handle->commandQueue; }
    if (handle->library) { (void)(__bridge_transfer id)handle->library; }
    if (handle->pipeline_bf16) { (void)(__bridge_transfer id)handle->pipeline_bf16; }
    if (handle->pipeline_f32) { (void)(__bridge_transfer id)handle->pipeline_f32; }

    delete handle;
}

MetalNaiveAttentionWorkspace metal_naive_attention_get_workspace(
    MetalNaiveAttentionHandle* handle,
    int num_tokens,
    int head_dim,
    int kv_head_dim,
    int page_size,
    int num_kv_pages
) {
    MetalNaiveAttentionWorkspace workspace = {0};

    if (!handle || !handle->initialized) {
        std::cerr << "MetalNaiveAttentionWorkspace: Invalid handle" << std::endl;
        return workspace;
    }

    // Same workspace layout as optimized version for fair comparison
    const size_t BUFFER_ALIGNMENT = 16;
    auto align = [](size_t size) {
        return (size + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    };

    size_t offset = 0;

    // Query buffer (converted from BF16 to Half)
    size_t q_count = static_cast<size_t>(num_tokens) * head_dim;
    workspace.q_buffer_offset = offset;
    workspace.q_buffer_size = align(q_count * sizeof(uint16_t));
    offset += workspace.q_buffer_size;

    // Key cache buffer
    size_t kv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;
    workspace.k_buffer_offset = offset;
    workspace.k_buffer_size = align(kv_count * sizeof(uint16_t));
    offset += workspace.k_buffer_size;

    // Value cache buffer
    workspace.v_buffer_offset = offset;
    workspace.v_buffer_size = align(kv_count * sizeof(uint16_t));
    offset += workspace.v_buffer_size;

    // Output buffer
    workspace.output_buffer_offset = offset;
    workspace.output_buffer_size = align(q_count * sizeof(uint16_t));
    offset += workspace.output_buffer_size;

    // Index buffers
    size_t total_index_elems = (num_tokens + 1) * 2 + num_kv_pages + num_tokens;
    workspace.index_buffer_offset = offset;
    workspace.index_buffer_size = align(total_index_elems * sizeof(int32_t));
    offset += workspace.index_buffer_size;

    // Parameters buffer
    workspace.params_buffer_offset = offset;
    workspace.params_buffer_size = align(256);
    offset += workspace.params_buffer_size;

    // Debug buffer
    workspace.debug_buffer_offset = offset;
    workspace.debug_buffer_size = align(64 * sizeof(float));
    offset += workspace.debug_buffer_size;

    // *** KEY DIFFERENCE: O(n²) ATTENTION MATRIX STORAGE ***
    // Calculate total sequence length from KV cache pages
    int total_kv_len = (num_kv_pages > 0) ? (num_kv_pages - 1) * page_size + page_size : 0;
    if (total_kv_len > 2048) total_kv_len = 2048; // Safety limit

    // Attention matrix: [num_qo * total_kv_len] floats
    size_t attention_matrix_elements = static_cast<size_t>(num_tokens) * total_kv_len;
    workspace.attention_matrix_offset = offset;
    workspace.attention_matrix_size = align(attention_matrix_elements * sizeof(float));
    offset += workspace.attention_matrix_size;

    workspace.alignment_padding = align(offset) - offset;
    workspace.total_size = align(offset);

    size_t matrix_mb = workspace.attention_matrix_size / 1024 / 1024;
    std::cout << "MetalNaiveAttentionWorkspace: Required NAIVE workspace size: "
              << (workspace.total_size / 1024 / 1024) << " MB for "
              << num_tokens << " tokens, " << num_kv_pages << " pages" << std::endl;
    std::cout << "                             *** WARNING: This uses O(n²) memory algorithm ***" << std::endl;
    std::cout << "                             Attention matrix alone: " << matrix_mb << " MB ("
              << num_tokens << " × " << total_kv_len << " elements)" << std::endl;

    return workspace;
}

int naive_batch_prefill_attention_unified_bf16(
    MetalNaiveAttentionHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const void* q_input,
    const void* paged_k_cache,
    const void* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    void* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
) {
    if (!handle || !handle->initialized || !workspace_buffer) {
        std::cerr << "naive_batch_prefill_attention_unified_bf16: Invalid handle or workspace" << std::endl;
        return -1;
    }

    // Get workspace layout
    MetalNaiveAttentionWorkspace workspace = metal_naive_attention_get_workspace(
        handle, num_qo, head_dim, kv_head_dim, page_size, num_kv_pages);

    if (workspace_size < workspace.total_size) {
        std::cerr << "naive_batch_prefill_attention_unified_bf16: Workspace too small. Required: "
                  << workspace.total_size << ", Provided: " << workspace_size << std::endl;
        return -1;
    }

    std::cout << "🔴 [NAIVE] naive_batch_prefill_attention_unified_bf16 called with O(n²) algorithm!" << std::endl;
    std::cout << "🔍 [NAIVE] Parameters: num_qo=" << num_qo << ", head_dim=" << head_dim
              << ", head_size=" << head_size << ", page_size=" << page_size
              << ", scale=" << scale << std::endl;

    @autoreleasepool {
        // Setup workspace regions
        char* workspace_base = static_cast<char*>(workspace_buffer);
        void* q_workspace = workspace_base + workspace.q_buffer_offset;
        void* k_workspace = workspace_base + workspace.k_buffer_offset;
        void* v_workspace = workspace_base + workspace.v_buffer_offset;
        void* output_workspace = workspace_base + workspace.output_buffer_offset;
        void* index_workspace = workspace_base + workspace.index_buffer_offset;
        void* params_workspace = workspace_base + workspace.params_buffer_offset;
        void* debug_workspace = workspace_base + workspace.debug_buffer_offset;

        // Convert input data directly into workspace
        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t kv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;

        // Convert BF16 to Half in workspace
        std::vector<uint16_t> q_half_data = convert_bf16_to_half(q_input, q_count);
        std::vector<uint16_t> k_half_data = convert_bf16_to_half(paged_k_cache, kv_count);
        std::vector<uint16_t> v_half_data = convert_bf16_to_half(paged_v_cache, kv_count);

        std::memcpy(q_workspace, q_half_data.data(), q_half_data.size() * sizeof(uint16_t));
        std::memcpy(k_workspace, k_half_data.data(), k_half_data.size() * sizeof(uint16_t));
        std::memcpy(v_workspace, v_half_data.data(), v_half_data.size() * sizeof(uint16_t));

        // Copy index data to workspace
        size_t index_offset = 0;
        size_t qo_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indices_size = num_kv_pages * sizeof(int32_t);
        size_t kv_last_page_lens_size = num_qo * sizeof(int32_t);

        std::memcpy((char*)index_workspace + index_offset, qo_indptr, qo_indptr_size);
        index_offset += qo_indptr_size;
        std::memcpy((char*)index_workspace + index_offset, kv_page_indptr, kv_page_indptr_size);
        index_offset += kv_page_indptr_size;
        std::memcpy((char*)index_workspace + index_offset, kv_page_indices, kv_page_indices_size);
        index_offset += kv_page_indices_size;
        std::memcpy((char*)index_workspace + index_offset, kv_last_page_lens, kv_last_page_lens_size);

        // Create Metal buffer views (no allocation!)
        id<MTLDevice> device = (__bridge id<MTLDevice>)handle->device;

        id<MTLBuffer> q_buf = [device newBufferWithBytesNoCopy:q_workspace
                                                        length:workspace.q_buffer_size
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

        id<MTLBuffer> k_buf = [device newBufferWithBytesNoCopy:k_workspace
                                                        length:workspace.k_buffer_size
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

        id<MTLBuffer> v_buf = [device newBufferWithBytesNoCopy:v_workspace
                                                        length:workspace.v_buffer_size
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

        id<MTLBuffer> out_buf = [device newBufferWithBytesNoCopy:output_workspace
                                                          length:workspace.output_buffer_size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        id<MTLBuffer> index_buf = [device newBufferWithBytesNoCopy:index_workspace
                                                            length:workspace.index_buffer_size
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];

        // *** ADD O(n²) ATTENTION MATRIX BUFFER ***
        void* attention_matrix_workspace = workspace_base + workspace.attention_matrix_offset;
        // Initialize attention matrix to zero
        std::memset(attention_matrix_workspace, 0, workspace.attention_matrix_size);

        id<MTLBuffer> attention_matrix_buf = [device newBufferWithBytesNoCopy:attention_matrix_workspace
                                                                       length:workspace.attention_matrix_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];

        // Create parameter buffer with updated structure
        struct Params {
            int num_qo;
            int head_dim;
            int kv_head_dim;
            int head_size;
            int page_size;
            int num_query_heads;
            int num_kv_heads;
            float scale;
            int total_kv_len;      // New parameter
            int num_partitions;    // New parameter
        };

        // Calculate total KV length for parameters
        int total_kv_len = (num_kv_pages > 0) ? (num_kv_pages - 1) * page_size + page_size : 0;
        if (total_kv_len > 2048) total_kv_len = 2048; // Safety limit
        int num_partitions = (total_kv_len + 255) / 256; // Using MAX_PARTITION_SIZE = 256

        Params* params = static_cast<Params*>(params_workspace);
        *params = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale, total_kv_len, num_partitions };

        id<MTLBuffer> params_buf = [device newBufferWithBytesNoCopy:params_workspace
                                                             length:workspace.params_buffer_size
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];

        // Create debug buffer
        id<MTLBuffer> debug_buf = [device newBufferWithBytesNoCopy:debug_workspace
                                                            length:workspace.debug_buffer_size
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];

        // Execute NAIVE Metal kernel
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)handle->commandQueue;
        id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)handle->pipeline_bf16;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:k_buf offset:0 atIndex:1];
        [enc setBuffer:v_buf offset:0 atIndex:2];

        // Set index buffers with proper offsets
        [enc setBuffer:index_buf offset:0 atIndex:3]; // qo_indptr
        [enc setBuffer:index_buf offset:qo_indptr_size atIndex:4]; // kv_page_indptr
        [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size atIndex:5]; // kv_page_indices
        [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size + kv_page_indices_size atIndex:6]; // kv_last_page_lens

        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        [enc setBuffer:debug_buf offset:0 atIndex:9];
        [enc setBuffer:attention_matrix_buf offset:0 atIndex:10]; // *** O(n²) ATTENTION MATRIX ***

        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];

        // Commit and wait
        [cmd commit];
        [cmd waitUntilCompleted];

        NSError* cmdError = cmd.error;
        if (cmdError) {
            std::cerr << "Naive Metal command buffer failed: " << cmdError.localizedDescription.UTF8String << std::endl;
            return -1;
        }

        // Convert output from Half to BF16
        if (output) {
            uint16_t* out_half = static_cast<uint16_t*>(output_workspace);
            bfloat16_t* out_bf16 = static_cast<bfloat16_t*>(output);
            for (size_t i = 0; i < q_count; ++i) {
                float f = half_to_float(out_half[i]);
                out_bf16[i] = float_to_bf16(f);
            }
        }

        // Update statistics
        handle->total_calls++;
        handle->total_bytes_processed += workspace.total_size;

        std::cout << "✅ [NAIVE] Naive O(n²) Metal attention completed!" << std::endl;
    }

    return 0;
}

int naive_batch_prefill_attention_unified_f32(
    MetalNaiveAttentionHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages
) {
    @autoreleasepool {
        if (!handle || !handle->device || !handle->pipeline_f32) {
            std::cerr << "❌ [NAIVE F32] Invalid handle or f32 pipeline not available" << std::endl;
            return -1;
        }

        std::cout << "🔴 [NAIVE F32] Processing with O(n²) memory algorithm" << std::endl;

        // Create Metal buffers as views
        id<MTLDevice> device = (__bridge id<MTLDevice>)handle->device;

        id<MTLBuffer> q_buf = [device newBufferWithBytesNoCopy:(void*)q_input
                                                        length:num_qo * head_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

        size_t kv_cache_size = num_kv_pages * page_size * kv_head_dim * sizeof(float);
        id<MTLBuffer> pk_buf = [device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                         length:kv_cache_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];

        id<MTLBuffer> pv_buf = [device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                         length:kv_cache_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];

        size_t indptr_size = (num_qo + 1) * sizeof(int32_t);
        id<MTLBuffer> qo_indptr_buf = [device newBufferWithBytesNoCopy:(void*)qo_indptr
                                                                length:indptr_size
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];

        id<MTLBuffer> kv_page_indptr_buf = [device newBufferWithBytesNoCopy:(void*)kv_page_indptr
                                                                     length:indptr_size
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];

        id<MTLBuffer> kv_page_indices_buf = [device newBufferWithBytesNoCopy:(void*)kv_page_indices
                                                                      length:num_kv_pages * sizeof(int32_t)
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];

        id<MTLBuffer> kv_last_page_lens_buf = [device newBufferWithBytesNoCopy:(void*)kv_last_page_lens
                                                                         length:num_qo * sizeof(int32_t)
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];

        id<MTLBuffer> out_buf = [device newBufferWithBytesNoCopy:output
                                                          length:num_qo * head_dim * sizeof(float)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        // Create parameter buffer
        struct Params {
            int num_qo;
            int head_dim;
            int kv_head_dim;
            int head_size;
            int page_size;
            int num_query_heads;
            int num_kv_heads;
            float scale;
        };

        Params params = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };

        id<MTLBuffer> params_buf = [device newBufferWithBytes:&params
                                                       length:sizeof(params)
                                                      options:MTLResourceStorageModeShared];

        // Create command buffer and encoder
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)handle->commandQueue;
        id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)handle->pipeline_f32;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:pk_buf offset:0 atIndex:1];
        [enc setBuffer:pv_buf offset:0 atIndex:2];
        [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
        [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
        [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
        [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:params_buf offset:0 atIndex:8];

        // Dispatch computation
        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];

        // Execute and wait
        [cmd commit];
        [cmd waitUntilCompleted];

        // Update handle statistics
        handle->total_calls++;
        handle->total_bytes_processed += workspace_size;

        std::cout << "✅ [NAIVE F32] Naive O(n²) Metal attention completed!" << std::endl;
    }

    return 0;
}

} // namespace naive_attention
} // namespace metal