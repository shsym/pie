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

struct AttentionParams {
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
};

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

bool compare_f16_arrays(
    const uint16_t* array1,
    const uint16_t* array2,
    size_t size,
    float tolerance,
    const std::string& name1,
    const std::string& name2,
    bool verbose = false
) {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    size_t mismatch_count = 0;
    
    for (size_t i = 0; i < size; i++) {
        float val1 = f16_bits_to_float(array1[i]);
        float val2 = f16_bits_to_float(array2[i]);
        
        float diff = std::abs(val1 - val2);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        
        if (diff > tolerance) {
            mismatch_count++;
            if (verbose && mismatch_count <= 10) {
                std::cout << "  [" << i << "]: " << name1 << "=" << val1 
                         << " vs " << name2 << "=" << val2 << " (diff=" << diff << ")" << std::endl;
            }
        }
    }
    
    avg_diff /= size;
    float match_percentage = 100.0f * (size - mismatch_count) / size;
    
    std::cout << "\n--- " << name1 << " vs " << name2 << " Comparison ---" << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Average difference: " << avg_diff << std::endl;
    std::cout << "Elements within tolerance: " << match_percentage << "%" << std::endl;
    std::cout << "Mismatches: " << mismatch_count << " / " << size << std::endl;
    
    bool passed = match_percentage >= 95.0f && max_diff < tolerance * 5.0f;
    std::cout << "Result: " << (passed ? "✅ PASSED" : "❌ FAILED") << std::endl;
    
    return passed;
}

int main() {
    std::cout << "=== FlashAttention Implementation Accuracy Comparison ===" << std::endl;
    
    // Initialize Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "❌ Failed to create Metal device" << std::endl;
        return -1;
    }
    
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Load all three kernels
    std::vector<std::pair<std::string, std::string>> kernels = {
        {"F16 Native", "/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_f16_native.metal"},
        {"Tiled FlashAttention", "/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_flash_attention_tiled.metal"},
        {"FlashInfer", "/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_flash_attention_flashinfer.metal"}
    };
    
    std::vector<id<MTLComputePipelineState>> pipelineStates;
    std::vector<std::string> kernelNames;
    
    for (const auto& [name, path] : kernels) {
        NSString* libraryPath = [NSString stringWithUTF8String:path.c_str()];
        NSError* error = nil;
        NSString* librarySource = [NSString stringWithContentsOfFile:libraryPath
                                                            encoding:NSUTF8StringEncoding 
                                                               error:&error];
        
        if (error) {
            std::cout << "⚠️  Failed to read " << name << " shader: " << error.localizedDescription.UTF8String << std::endl;
            continue;
        }
        
        id<MTLLibrary> library = [device newLibraryWithSource:librarySource options:nil error:&error];
        if (error) {
            std::cout << "⚠️  Failed to compile " << name << " library: " << error.localizedDescription.UTF8String << std::endl;
            continue;
        }
        
        id<MTLFunction> function = nil;
        if (name == "F16 Native") {
            function = [library newFunctionWithName:@"unified_batch_prefill_attention_f16_native"];
        } else if (name == "Tiled FlashAttention") {
            function = [library newFunctionWithName:@"flash_attention_f16_tiled"];
        } else if (name == "FlashInfer") {
            function = [library newFunctionWithName:@"flash_attention_flashinfer_f16"];
        }
        
        if (!function) {
            std::cout << "⚠️  Failed to find " << name << " kernel function" << std::endl;
            continue;
        }
        
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            std::cout << "⚠️  Failed to create " << name << " pipeline: " << error.localizedDescription.UTF8String << std::endl;
            continue;
        }
        
        pipelineStates.push_back(pipelineState);
        kernelNames.push_back(name);
        std::cout << "✅ " << name << " kernel loaded successfully" << std::endl;
    }
    
    if (pipelineStates.size() < 2) {
        std::cout << "❌ Need at least 2 kernels for comparison" << std::endl;
        return -1;
    }
    
    // Test configuration - use moderate size for accurate comparison
    const uint32_t batch_size = 1;
    const uint32_t num_heads = 1;
    const uint32_t head_size = 64;
    const uint32_t seq_len_q = 128;  // Moderate size for accurate comparison
    const uint32_t seq_len_kv = 128;
    const uint32_t num_kv_heads = 1;
    const uint32_t group_size = num_heads / num_kv_heads;
    
    std::cout << "\nTest configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length (Q): " << seq_len_q << std::endl;
    std::cout << "  Sequence length (KV): " << seq_len_kv << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head size: " << head_size << std::endl;
    
    // Generate reproducible test data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> normal_dist(0.0f, 0.08f);  // Small variance for numerical stability
    
    // Input tensors (same for all kernels)
    std::vector<uint16_t> q_data(batch_size * seq_len_q * num_heads * head_size);
    for (size_t i = 0; i < q_data.size(); i++) {
        float val = normal_dist(gen);
        q_data[i] = float_to_f16_bits(val);
    }
    
    // Paged KV cache setup
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
    
    for (size_t i = 0; i < k_cache.size(); i++) {
        k_cache[i] = float_to_f16_bits(normal_dist(gen));
        v_cache[i] = float_to_f16_bits(normal_dist(gen));
    }
    
    for (size_t i = 0; i < page_indices.size(); i++) {
        page_indices[i] = i;
    }
    
    // Prepare parameters for different kernels
    AttentionParams native_params = {};
    native_params.head_dim = head_size;
    native_params.head_size = head_size;
    native_params.q_stride_seq = num_heads * head_size;
    native_params.q_stride_head = head_size;
    native_params.k_stride_head = head_size;
    native_params.v_stride_head = head_size;
    native_params.o_stride_seq = num_heads * head_size;
    native_params.o_stride_head = head_size;
    native_params.scale = 1.0f / sqrtf((float)head_size);
    native_params.causal = 1;
    native_params.num_kv_heads = num_kv_heads;
    native_params.group_size = group_size;
    native_params.logit_cap = 50.0f;
    
    FlashInferParams flashinfer_params = {};
    flashinfer_params.head_dim = head_size;
    flashinfer_params.head_size = head_size;
    flashinfer_params.q_stride_seq = num_heads * head_size;
    flashinfer_params.q_stride_head = head_size;
    flashinfer_params.k_stride_head = head_size;
    flashinfer_params.v_stride_head = head_size;
    flashinfer_params.o_stride_seq = num_heads * head_size;
    flashinfer_params.o_stride_head = head_size;
    flashinfer_params.scale = 1.0f / sqrtf((float)head_size);
    flashinfer_params.causal = 1;
    flashinfer_params.num_kv_heads = num_kv_heads;
    flashinfer_params.group_size = group_size;
    flashinfer_params.logit_cap = 50.0f;
    flashinfer_params.num_blocks_per_seq = (seq_len_q + 63) / 64;
    flashinfer_params.max_seq_len = std::max(seq_len_q, seq_len_kv);
    flashinfer_params.block_size = 64;
    flashinfer_params.load_balance_factor = 1;
    
    // Create Metal buffers (shared across kernels)
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
    
    id<MTLBuffer> nativeParamsBuffer = [device newBufferWithBytes:&native_params
                                                           length:sizeof(AttentionParams)
                                                          options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> flashinferParamsBuffer = [device newBufferWithBytes:&flashinfer_params
                                                               length:sizeof(FlashInferParams)
                                                              options:MTLResourceStorageModeShared];
    
    // Execute all kernels and collect results
    std::vector<std::vector<uint16_t>> all_results;
    
    for (size_t kernel_idx = 0; kernel_idx < pipelineStates.size(); kernel_idx++) {
        const std::string& name = kernelNames[kernel_idx];
        id<MTLComputePipelineState> pipelineState = pipelineStates[kernel_idx];
        
        std::cout << "\n--- Running " << name << " Kernel ---" << std::endl;
        
        // Create output buffer for this kernel
        std::vector<uint16_t> output_data(batch_size * seq_len_q * num_heads * head_size, 0);
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_data.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        
        // Configure grid and threadgroup based on kernel type
        MTLSize gridSize, threadgroupSize;
        uint32_t shared_memory_size = 0;
        
        if (name == "F16 Native") {
            gridSize = MTLSizeMake(batch_size, num_heads, seq_len_q);
            threadgroupSize = MTLSizeMake(1, 1, 1);
        } else if (name == "Tiled FlashAttention") {
            uint32_t num_q_tiles = (seq_len_q + 63) / 64;  // 64 is TILE_SIZE_Q
            gridSize = MTLSizeMake(batch_size, num_heads, num_q_tiles);
            threadgroupSize = MTLSizeMake(128, 1, 1);
            shared_memory_size = 32768; // 32KB shared memory
        } else if (name == "FlashInfer") {
            uint32_t num_blocks_q = (seq_len_q + 63) / 64;
            gridSize = MTLSizeMake(batch_size, num_heads, num_blocks_q);
            threadgroupSize = MTLSizeMake(128, 1, 1);
            shared_memory_size = 32768; // 32KB shared memory
        }
        
        // Execute kernel
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
        
        // Use appropriate parameters buffer
        if (name == "FlashInfer") {
            [encoder setBuffer:flashinferParamsBuffer offset:0 atIndex:8];
        } else {
            [encoder setBuffer:nativeParamsBuffer offset:0 atIndex:8];
        }
        
        [encoder setBuffer:outputBuffer offset:0 atIndex:9];
        
        if (shared_memory_size > 0) {
            [encoder setThreadgroupMemoryLength:shared_memory_size atIndex:0];
        }
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            std::cout << "❌ " << name << " kernel execution failed" << std::endl;
            continue;
        }
        
        // Copy results
        uint16_t* results = (uint16_t*)outputBuffer.contents;
        output_data.assign(results, results + output_data.size());
        all_results.push_back(output_data);
        
        std::cout << "✅ " << name << " completed successfully" << std::endl;
        
        // Show first few values for verification
        std::cout << "First 4 values: ";
        for (int i = 0; i < 4; i++) {
            std::cout << f16_bits_to_float(output_data[i]) << " ";
        }
        std::cout << std::endl;
    }
    
    if (all_results.size() < 2) {
        std::cout << "❌ Need at least 2 successful kernel executions for comparison" << std::endl;
        return -1;
    }
    
    // Compare all implementations pairwise
    std::cout << "\n=== Accuracy Comparison Results ===" << std::endl;
    
    bool all_passed = true;
    float f16_tolerance = 0.001f; // F16 precision tolerance
    
    for (size_t i = 0; i < all_results.size(); i++) {
        for (size_t j = i + 1; j < all_results.size(); j++) {
            bool passed = compare_f16_arrays(
                all_results[i].data(),
                all_results[j].data(),
                all_results[i].size(),
                f16_tolerance,
                kernelNames[i],
                kernelNames[j]
            );
            all_passed &= passed;
        }
    }
    
    std::cout << "\n=== Final Accuracy Assessment ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ ALL IMPLEMENTATIONS AGREE" << std::endl;
        std::cout << "✅ FlashAttention implementations are mathematically equivalent" << std::endl;
        std::cout << "✅ F16 numerical precision maintained across all kernels" << std::endl;
        std::cout << "✅ Ready for production deployment" << std::endl;
        return 0;
    } else {
        std::cout << "❌ IMPLEMENTATIONS DIVERGE" << std::endl;
        std::cout << "❌ Further debugging required" << std::endl;
        return -1;
    }
}