#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstring>

// F16 conversion utilities
static uint16_t float_to_f16_bits(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;
    
    if (exp == 0xFF) {  // Infinity or NaN
        return static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0));
    }
    
    if (exp == 0) {  // Zero or subnormal
        return static_cast<uint16_t>(sign << 15);
    }
    
    int32_t f16_exp = static_cast<int32_t>(exp) - 127 + 15;
    
    if (f16_exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);  // Infinity
    }
    
    if (f16_exp <= 0) {
        return static_cast<uint16_t>(sign << 15);  // Zero
    }
    
    uint32_t f16_mantissa = mantissa >> 13;
    return static_cast<uint16_t>((sign << 15) | (f16_exp << 10) | f16_mantissa);
}

static float f16_bits_to_float(uint16_t f16_bits) {
    uint32_t sign = (f16_bits >> 15) & 0x1;
    uint32_t exp = (f16_bits >> 10) & 0x1F;
    uint32_t mantissa = f16_bits & 0x3FF;
    
    uint32_t f32_bits;
    
    if (exp == 0x1F) {  // Infinity or NaN
        f32_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else if (exp == 0) {  // Zero or subnormal
        f32_bits = sign << 31;  // Zero
    } else {  // Normal case
        uint32_t f32_exp = exp - 15 + 127;  // Adjust bias
        f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
    }
    
    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

void float_to_half_vector(const std::vector<float>& input, std::vector<uint16_t>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = float_to_f16_bits(input[i]);
    }
}

struct FlashAttentionParams {
    uint32_t head_dim;
    uint32_t head_size;
    uint32_t seq_len_q;
    uint32_t seq_len_kv;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t group_size;
    float scale;
    bool causal_mask;
    uint32_t batch_size;
    uint32_t page_size;
    uint32_t tokens_per_page;
};

int main() {
    std::cout << "\n=== FlashAttention Tiled F16 Test ===\n";
    
    // Initialize Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to create Metal device\n";
        return -1;
    }
    
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        std::cerr << "Failed to create command queue\n";
        return -1;
    }
    
    // Load FlashAttention tiled kernel
    NSError* error = nil;
    NSString* shaderPath = @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_flash_attention_tiled.metal";
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath 
                                                      encoding:NSUTF8StringEncoding 
                                                         error:&error];
    if (!shaderSource) {
        std::cerr << "Failed to load FlashAttention tiled shader: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile FlashAttention tiled library: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLFunction> flashFunction = [library newFunctionWithName:@"flash_attention_f16_tiled"];
    if (!flashFunction) {
        std::cerr << "Failed to find FlashAttention tiled function\n";
        return -1;
    }
    
    id<MTLComputePipelineState> flashPipeline = [device newComputePipelineStateWithFunction:flashFunction error:&error];
    if (!flashPipeline) {
        std::cerr << "Failed to create FlashAttention tiled pipeline: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    std::cout << "✅ FlashAttention tiled kernel loaded successfully\n";
    
    // Test configuration
    uint32_t batch_size = 1;
    uint32_t seq_len_q = 128;   // Longer sequence to test tiling
    uint32_t seq_len_kv = 128;
    uint32_t num_heads = 1;
    uint32_t head_size = 64;
    uint32_t head_dim = 64;
    uint32_t num_kv_heads = 1;
    uint32_t tokens_per_page = 16;
    
    std::cout << "Test configuration:\n";
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length (Q): " << seq_len_q << std::endl;
    std::cout << "  Sequence length (KV): " << seq_len_kv << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head size: " << head_size << std::endl;
    
    // Generate test data
    std::vector<float> q_data_f32(batch_size * seq_len_q * num_heads * head_size);
    std::vector<float> k_data_f32(seq_len_kv * head_size);
    std::vector<float> v_data_f32(seq_len_kv * head_size);
    
    // Initialize with small random values suitable for F16
    for (size_t i = 0; i < q_data_f32.size(); i++) {
        q_data_f32[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.2f;  // Range [-0.1, 0.1]
    }
    
    for (size_t i = 0; i < k_data_f32.size(); i++) {
        k_data_f32[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    
    for (size_t i = 0; i < v_data_f32.size(); i++) {
        v_data_f32[i] = (float(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    
    // Convert to F16
    std::vector<uint16_t> q_data_f16, k_data_f16, v_data_f16;
    float_to_half_vector(q_data_f32, q_data_f16);
    float_to_half_vector(k_data_f32, k_data_f16);
    float_to_half_vector(v_data_f32, v_data_f16);
    
    // Setup FlashAttention parameters
    FlashAttentionParams params = {};
    params.head_dim = head_dim;
    params.head_size = head_size;
    params.seq_len_q = seq_len_q;
    params.seq_len_kv = seq_len_kv;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_heads / num_kv_heads;
    params.scale = 1.0f / sqrtf(static_cast<float>(head_size));
    params.causal_mask = false; // Test without causal mask first
    params.batch_size = batch_size;
    params.page_size = tokens_per_page * head_dim;
    params.tokens_per_page = tokens_per_page;
    
    // Setup indices for simple sequential layout
    std::vector<uint32_t> qo_indptr = {0, seq_len_q};
    std::vector<uint32_t> kv_indptr = {0, seq_len_kv};
    
    // Setup paging (simple sequential pages)
    uint32_t num_pages = (seq_len_kv + tokens_per_page - 1) / tokens_per_page;
    std::vector<uint32_t> kv_page_indptr = {0, num_pages};
    std::vector<uint32_t> kv_page_indices;
    std::vector<uint32_t> kv_last_page_lens = {seq_len_kv % tokens_per_page};
    if (kv_last_page_lens[0] == 0) kv_last_page_lens[0] = tokens_per_page;
    
    for (uint32_t i = 0; i < num_pages; i++) {
        kv_page_indices.push_back(i);
    }
    
    // Create paged K,V cache
    size_t total_page_size = num_pages * tokens_per_page * head_dim;
    std::vector<uint16_t> paged_k_f16(total_page_size, 0);
    std::vector<uint16_t> paged_v_f16(total_page_size, 0);
    
    // Fill paged cache
    for (uint32_t token_idx = 0; token_idx < seq_len_kv; token_idx++) {
        uint32_t page_idx = token_idx / tokens_per_page;
        uint32_t pos_in_page = token_idx % tokens_per_page;
        
        for (uint32_t d = 0; d < head_size; d++) {
            size_t src_idx = token_idx * head_size + d;
            size_t dst_idx = page_idx * tokens_per_page * head_dim + pos_in_page * head_dim + d;
            
            paged_k_f16[dst_idx] = k_data_f16[src_idx];
            paged_v_f16[dst_idx] = v_data_f16[src_idx];
        }
    }
    
    // Create Metal buffers
    id<MTLBuffer> q_buffer = [device newBufferWithBytes:q_data_f16.data()
                                                  length:q_data_f16.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> k_buffer = [device newBufferWithBytes:paged_k_f16.data()
                                                  length:paged_k_f16.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> v_buffer = [device newBufferWithBytes:paged_v_f16.data()
                                                  length:paged_v_f16.size() * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> qo_indptr_buffer = [device newBufferWithBytes:qo_indptr.data()
                                                         length:qo_indptr.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_indptr_buffer = [device newBufferWithBytes:kv_indptr.data()
                                                         length:kv_indptr.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_page_indptr_buffer = [device newBufferWithBytes:kv_page_indptr.data()
                                                              length:kv_page_indptr.size() * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_page_indices_buffer = [device newBufferWithBytes:kv_page_indices.data()
                                                               length:kv_page_indices.size() * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_last_page_lens_buffer = [device newBufferWithBytes:kv_last_page_lens.data()
                                                                 length:kv_last_page_lens.size() * sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params
                                                      length:sizeof(FlashAttentionParams)
                                                     options:MTLResourceStorageModeShared];
    
    // Create output buffer
    size_t output_size = batch_size * seq_len_q * num_heads * head_size;
    id<MTLBuffer> output_buffer = [device newBufferWithLength:output_size * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
    
    // Calculate shared memory size needed
    size_t q_smem_size = 64 * head_size * sizeof(uint16_t);      // TILE_SIZE_Q * head_size
    size_t k_smem_size = 64 * head_size * sizeof(uint16_t);      // TILE_SIZE_KV * head_size  
    size_t v_smem_size = 64 * head_size * sizeof(uint16_t);      // TILE_SIZE_KV * head_size
    size_t s_smem_size = 64 * 64 * sizeof(uint16_t);            // TILE_SIZE_Q * TILE_SIZE_KV
    size_t total_smem_size = q_smem_size + k_smem_size + v_smem_size + s_smem_size;
    
    std::cout << "Shared memory required: " << total_smem_size << " bytes\n";
    
    // Calculate grid dimensions
    uint32_t num_q_tiles = (seq_len_q + 63) / 64;  // TILE_SIZE_Q = 64
    MTLSize gridSize = MTLSizeMake(batch_size, num_heads, num_q_tiles);
    MTLSize threadgroupSize = MTLSizeMake(128, 1, 1); // 4 SIMD groups of 32 threads
    
    std::cout << "Grid size: [" << gridSize.width << ", " << gridSize.height << ", " << gridSize.depth << "]\n";
    std::cout << "Threadgroup size: [" << threadgroupSize.width << ", " << threadgroupSize.height << ", " << threadgroupSize.depth << "]\n";
    
    // Run FlashAttention tiled kernel
    std::cout << "\n--- Running FlashAttention Tiled Kernel ---\n";
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:flashPipeline];
    [encoder setBuffer:q_buffer offset:0 atIndex:0];
    [encoder setBuffer:k_buffer offset:0 atIndex:1];
    [encoder setBuffer:v_buffer offset:0 atIndex:2];
    [encoder setBuffer:qo_indptr_buffer offset:0 atIndex:3];
    [encoder setBuffer:kv_indptr_buffer offset:0 atIndex:4];
    [encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:5];
    [encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:6];
    [encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:7];
    [encoder setBuffer:params_buffer offset:0 atIndex:8];
    [encoder setBuffer:output_buffer offset:0 atIndex:9];
    [encoder setThreadgroupMemoryLength:total_smem_size atIndex:0];
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        std::cerr << "❌ FlashAttention kernel failed with status: " << (int)commandBuffer.status << std::endl;
        if (commandBuffer.error) {
            std::cerr << "Error: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
        }
        return -1;
    }
    
    // Read results
    uint16_t* results = (uint16_t*)[output_buffer contents];
    
    // Convert first few results to float for display
    std::cout << "\n--- FlashAttention Tiled Results ---\n";
    std::cout << "First 8 output values (F32 for display):\n";
    for (int i = 0; i < std::min(8, (int)head_size); i++) {
        float result_f32 = f16_bits_to_float(results[i]);
        std::cout << result_f32 << " (F16: 0x" << std::hex << results[i] << std::dec << ") ";
    }
    std::cout << std::endl;
    
    // Basic sanity checks
    bool has_nonzero = false;
    bool has_reasonable_values = true;
    
    for (size_t i = 0; i < std::min(output_size, size_t(64)); i++) {
        float val = f16_bits_to_float(results[i]);
        if (val != 0.0f) has_nonzero = true;
        if (std::abs(val) > 10.0f || std::isnan(val) || std::isinf(val)) {
            has_reasonable_values = false;
            std::cout << "Unreasonable value at [" << i << "]: " << val << std::endl;
        }
    }
    
    std::cout << "\n--- Basic Validation ---\n";
    std::cout << "Has non-zero outputs: " << (has_nonzero ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Values in reasonable range: " << (has_reasonable_values ? "✅ YES" : "❌ NO") << std::endl;
    
    bool success = has_nonzero && has_reasonable_values;
    
    std::cout << "\n--- Final Result ---\n";
    std::cout << "FlashAttention Tiled Kernel: " << (success ? "✅ PASSED" : "❌ FAILED") << std::endl;
    
    if (success) {
        std::cout << "✅ Tiled implementation shows reasonable outputs and completes successfully\n";
        std::cout << "✅ Ready for more comprehensive accuracy testing and optimization\n";
    }
    
    return success ? 0 : 1;
}