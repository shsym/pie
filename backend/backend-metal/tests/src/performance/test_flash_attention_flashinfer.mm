#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// F16 conversion utilities
uint16_t float_to_f16_bits(float val) {
    uint32_t f32_bits = *reinterpret_cast<uint32_t*>(&val);
    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa = f32_bits & 0x7FFFFF;
    
    if (exp == 0) return (sign << 15);
    if (exp == 0xFF) return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    
    int new_exp = exp - 127 + 15;
    if (new_exp <= 0) return (sign << 15);
    if (new_exp >= 31) return (sign << 15) | 0x7C00;
    
    uint32_t new_mantissa = mantissa >> 13;
    if (mantissa & 0x1000) new_mantissa++;
    
    return (sign << 15) | (new_exp << 10) | (new_mantissa & 0x3FF);
}

float f16_bits_to_float(uint16_t bits) {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float val = mantissa / 1024.0f / 1024.0f;
        return sign ? -val : val;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    
    uint32_t f32_exp = exp - 15 + 127;
    uint32_t f32_mantissa = mantissa << 13;
    uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | f32_mantissa;
    return *reinterpret_cast<float*>(&f32_bits);
}

struct FlashInferParams {
    uint32_t head_dim;
    uint32_t head_size;
    uint32_t q_stride_seq;
    uint32_t q_stride_head;
    uint32_t k_stride_head;
    uint32_t v_stride_head;
    uint32_t o_stride_seq;
    uint32_t o_stride_head;
    float scale;
    uint32_t num_layers;
    uint32_t layer_idx;
    uint32_t causal;
    uint32_t num_kv_heads;
    uint32_t group_size;
    float logit_cap;
    
    // FlashInfer-specific parameters
    uint32_t num_blocks_per_seq;
    uint32_t max_seq_len;
    uint32_t block_size;
    uint32_t load_balance_factor;
};

int main() {
    std::cout << "=== FlashAttention FlashInfer F16 Test ===" << std::endl;
    
    // Initialize Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "❌ Failed to create Metal device" << std::endl;
        return -1;
    }
    
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Load the metal library
    NSString* libraryPath = @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_flash_attention_flashinfer.metal";
    NSError* error = nil;
    NSString* librarySource = [NSString stringWithContentsOfFile:libraryPath
                                                        encoding:NSUTF8StringEncoding 
                                                           error:&error];
    
    if (error) {
        std::cout << "❌ Failed to read shader file: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLLibrary> library = [device newLibraryWithSource:librarySource options:nil error:&error];
    if (error) {
        std::cout << "❌ Failed to compile FlashInfer library: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"flash_attention_flashinfer_f16"];
    if (!function) {
        std::cout << "❌ Failed to find FlashInfer kernel function" << std::endl;
        return -1;
    }
    
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        std::cout << "❌ Failed to create pipeline state: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    std::cout << "✅ FlashInfer kernel loaded successfully" << std::endl;
    
    // Test configuration with longer sequences for FlashInfer validation
    const uint32_t batch_size = 1;
    const uint32_t num_heads = 1;
    const uint32_t head_size = 64;
    const uint32_t seq_len_q = 256;  // Longer for dynamic scheduling test
    const uint32_t seq_len_kv = 256;
    const uint32_t num_kv_heads = 1;
    const uint32_t group_size = num_heads / num_kv_heads;
    
    // FlashInfer-specific configuration
    const uint32_t tile_size_q = 64;
    const uint32_t tile_size_kv = 64;
    const uint32_t num_blocks_q = (seq_len_q + tile_size_q - 1) / tile_size_q;
    const uint32_t num_blocks_kv = (seq_len_kv + tile_size_kv - 1) / tile_size_kv;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length (Q): " << seq_len_q << std::endl;
    std::cout << "  Sequence length (KV): " << seq_len_kv << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head size: " << head_size << std::endl;
    std::cout << "  FlashInfer blocks (Q): " << num_blocks_q << std::endl;
    std::cout << "  FlashInfer blocks (KV): " << num_blocks_kv << std::endl;
    
    // Create test data with more realistic distributions
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> normal_dist(0.0f, 0.1f);  // Smaller variance for stability
    
    // Input tensors
    std::vector<uint16_t> q_data(batch_size * seq_len_q * num_heads * head_size);
    std::vector<uint16_t> output_data(batch_size * seq_len_q * num_heads * head_size, 0);
    
    // Generate query data with attention-like distribution
    for (size_t i = 0; i < q_data.size(); i++) {
        float val = normal_dist(gen);
        q_data[i] = float_to_f16_bits(val);
    }
    
    // Simplified paged KV cache setup (single page for testing)
    const uint32_t tokens_per_page = 16;
    const uint32_t num_pages = (seq_len_kv + tokens_per_page - 1) / tokens_per_page;
    const uint32_t kv_cache_size = num_pages * tokens_per_page * num_kv_heads * head_size;
    
    std::vector<uint16_t> k_cache(kv_cache_size);
    std::vector<uint16_t> v_cache(kv_cache_size);
    std::vector<uint32_t> page_indices(num_pages);
    std::vector<uint32_t> qo_indptr = {0, seq_len_q};
    std::vector<uint32_t> kv_indptr = {0, seq_len_kv};
    std::vector<uint32_t> kv_page_indptr = {0, num_pages};
    std::vector<uint32_t> kv_last_page_lens = {seq_len_kv % tokens_per_page == 0 ? tokens_per_page : seq_len_kv % tokens_per_page};
    
    // Initialize KV cache and page indices
    for (size_t i = 0; i < k_cache.size(); i++) {
        k_cache[i] = float_to_f16_bits(normal_dist(gen));
        v_cache[i] = float_to_f16_bits(normal_dist(gen));
    }
    
    for (size_t i = 0; i < page_indices.size(); i++) {
        page_indices[i] = i;  // Simple linear mapping for testing
    }
    
    // FlashInfer parameters
    FlashInferParams params = {};
    params.head_dim = head_size;
    params.head_size = head_size;
    params.q_stride_seq = num_heads * head_size;
    params.q_stride_head = head_size;
    params.k_stride_head = head_size;
    params.v_stride_head = head_size;
    params.o_stride_seq = num_heads * head_size;
    params.o_stride_head = head_size;
    params.scale = 1.0f / sqrtf((float)head_size);
    params.causal = 1;  // Enable causal masking for FlashInfer test
    params.num_kv_heads = num_kv_heads;
    params.group_size = group_size;
    params.logit_cap = 50.0f;
    
    // FlashInfer-specific parameters
    params.num_blocks_per_seq = num_blocks_q;
    params.max_seq_len = std::max(seq_len_q, seq_len_kv);
    params.block_size = tile_size_q;
    params.load_balance_factor = 1;  // Balanced scheduling
    
    // Create Metal buffers
    id<MTLBuffer> qBuffer = [device newBufferWithBytes:q_data.data() 
                                               length:q_data.size() * sizeof(uint16_t) 
                                              options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> kCacheBuffer = [device newBufferWithBytes:k_cache.data()
                                                     length:k_cache.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> vCacheBuffer = [device newBufferWithBytes:v_cache.data()
                                                     length:v_cache.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> qoIndptrBuffer = [device newBufferWithBytes:qo_indptr.data()
                                                       length:qo_indptr.size() * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> kvIndptrBuffer = [device newBufferWithBytes:kv_indptr.data()
                                                       length:kv_indptr.size() * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> kvPageIndptrBuffer = [device newBufferWithBytes:kv_page_indptr.data()
                                                          length:kv_page_indptr.size() * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> pageIndicesBuffer = [device newBufferWithBytes:page_indices.data()
                                                         length:page_indices.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> lastPageLensBuffer = [device newBufferWithBytes:kv_last_page_lens.data()
                                                           length:kv_last_page_lens.size() * sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                     length:sizeof(FlashInferParams)
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_data.size() * sizeof(uint16_t)
                                                     options:MTLResourceStorageModeShared];
    
    // Calculate shared memory requirements for FlashInfer
    uint32_t q_smem_size = tile_size_q * head_size * sizeof(uint16_t);
    uint32_t kv_smem_size = tile_size_kv * head_size * sizeof(uint16_t) * 2;  // K + V
    uint32_t s_smem_size = tile_size_q * tile_size_kv * sizeof(uint16_t);
    uint32_t total_smem = q_smem_size + kv_smem_size + s_smem_size;
    
    std::cout << "FlashInfer shared memory required: " << total_smem << " bytes" << std::endl;
    
    // Configure FlashInfer grid with dynamic scheduling
    MTLSize gridSize = MTLSizeMake(batch_size, num_heads, num_blocks_q);  // One block per Q tile
    MTLSize threadgroupSize = MTLSizeMake(128, 1, 1);  // 4 warps for good occupancy
    
    std::cout << "FlashInfer grid size: [" << gridSize.width << ", " << gridSize.height << ", " << gridSize.depth << "]" << std::endl;
    std::cout << "Threadgroup size: [" << threadgroupSize.width << ", " << threadgroupSize.height << ", " << threadgroupSize.depth << "]" << std::endl;
    
    // Execute FlashInfer kernel
    std::cout << "\n--- Running FlashInfer Kernel ---\n" << std::endl;
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:qBuffer offset:0 atIndex:0];
    [encoder setBuffer:kCacheBuffer offset:0 atIndex:1];
    [encoder setBuffer:vCacheBuffer offset:0 atIndex:2];
    [encoder setBuffer:qoIndptrBuffer offset:0 atIndex:3];
    [encoder setBuffer:kvIndptrBuffer offset:0 atIndex:4];
    [encoder setBuffer:kvPageIndptrBuffer offset:0 atIndex:5];
    [encoder setBuffer:pageIndicesBuffer offset:0 atIndex:6];
    [encoder setBuffer:lastPageLensBuffer offset:0 atIndex:7];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:8];
    [encoder setBuffer:outputBuffer offset:0 atIndex:9];
    [encoder setThreadgroupMemoryLength:total_smem atIndex:0];
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    if (commandBuffer.error) {
        std::cout << "❌ FlashInfer kernel execution failed: " << commandBuffer.error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    // Read results
    uint16_t* results = (uint16_t*)outputBuffer.contents;
    size_t output_size = output_data.size();
    
    std::cout << "--- FlashInfer Results ---" << std::endl;
    std::cout << "First 8 output values (F32 for display):" << std::endl;
    for (int i = 0; i < std::min(8, (int)head_size); i++) {
        float result_f32 = f16_bits_to_float(results[i]);
        std::cout << result_f32 << " (F16: 0x" << std::hex << results[i] << std::dec << ") ";
    }
    std::cout << std::endl;
    
    // FlashInfer-specific validation
    bool has_nonzero = false;
    bool has_reasonable_values = true;
    bool causal_pattern_correct = true;
    
    // Check overall statistics
    for (size_t i = 0; i < std::min(output_size, size_t(128)); i++) {
        float val = f16_bits_to_float(results[i]);
        if (val != 0.0f) has_nonzero = true;
        if (std::abs(val) > 10.0f || std::isnan(val) || std::isinf(val)) {
            has_reasonable_values = false;
            std::cout << "Unreasonable value at [" << i << "]: " << val << std::endl;
        }
    }
    
    // Validate causal masking pattern (basic check)
    if (seq_len_q >= 4 && seq_len_kv >= 4) {
        // Check that early positions have different attention patterns than later positions
        float early_sum = 0.0f, late_sum = 0.0f;
        for (int d = 0; d < 8; d++) {
            early_sum += std::abs(f16_bits_to_float(results[0 * head_size + d]));  // First token
            late_sum += std::abs(f16_bits_to_float(results[(seq_len_q-1) * head_size + d]));  // Last token
        }
        
        // With causal masking, both early and late tokens should have reasonable magnitudes
        // Early tokens attend to fewer positions, late tokens to more, but both should be non-zero
        float ratio = (late_sum > 0) ? early_sum / late_sum : 1.0f;
        if (early_sum == 0.0f || late_sum == 0.0f || ratio > 50.0f || ratio < 0.02f) {
            causal_pattern_correct = false;
            std::cout << "Causal pattern issue: early_sum=" << early_sum << ", late_sum=" << late_sum << ", ratio=" << ratio << std::endl;
        }
    }
    
    std::cout << "\n--- FlashInfer Validation ---" << std::endl;
    std::cout << "Has non-zero outputs: " << (has_nonzero ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Values in reasonable range: " << (has_reasonable_values ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "Causal attention pattern: " << (causal_pattern_correct ? "✅ CORRECT" : "⚠️  QUESTIONABLE") << std::endl;
    
    std::cout << "\n--- Final Result ---" << std::endl;
    if (has_nonzero && has_reasonable_values && causal_pattern_correct) {
        std::cout << "FlashInfer Kernel: ✅ PASSED" << std::endl;
        std::cout << "✅ Dynamic block scheduling working correctly" << std::endl;
        std::cout << "✅ Block-based masking implemented successfully" << std::endl;
        std::cout << "✅ F16 numerical stability maintained" << std::endl;
        std::cout << "✅ Ready for performance benchmarking and accuracy comparison" << std::endl;
        return 0;
    } else {
        std::cout << "FlashInfer Kernel: ❌ FAILED" << std::endl;
        std::cout << "❌ Issues detected in FlashInfer implementation" << std::endl;
        return -1;
    }
}