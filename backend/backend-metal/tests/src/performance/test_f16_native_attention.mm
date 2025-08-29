#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstring>

// F16 conversion utilities for host-side data preparation
static uint16_t float_to_f16_bits(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 0xFF) {  // Infinity or NaN
        return static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0));
    }
    
    if (exp == 0) {  // Zero or subnormal
        return static_cast<uint16_t>(sign << 15);
    }
    
    // Adjust exponent for F16 bias (15 vs 127)
    int32_t f16_exp = static_cast<int32_t>(exp) - 127 + 15;
    
    // Handle overflow
    if (f16_exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);  // Infinity
    }
    
    // Handle underflow
    if (f16_exp <= 0) {
        return static_cast<uint16_t>(sign << 15);  // Zero
    }
    
    // Normal case
    uint32_t f16_mantissa = mantissa >> 13;  // Truncate to 10 bits
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

// Convert vector to F16 (using Metal's internal half representation)
void float_to_half_vector(const std::vector<float>& input, std::vector<uint16_t>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = float_to_f16_bits(input[i]);
    }
}

void half_to_float_vector(const std::vector<uint16_t>& input, std::vector<float>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = f16_bits_to_float(input[i]);
    }
}

// F16 comparison with appropriate tolerance
bool compare_f16_results(const uint16_t* expected, const uint16_t* actual, size_t size, float tolerance) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float exp_f32 = f16_bits_to_float(expected[i]);
        float act_f32 = f16_bits_to_float(actual[i]);
        
        float abs_diff = std::abs(exp_f32 - act_f32);
        max_diff = std::max(max_diff, abs_diff);
        sum_diff += abs_diff;
        
        if (abs_diff > tolerance) {
            std::cout << "  Mismatch at [" << i << "]: expected " << exp_f32 
                      << " (F16: 0x" << std::hex << expected[i] << "), got " << act_f32
                      << " (F16: 0x" << actual[i] << std::dec << "), diff: " << abs_diff << std::endl;
            return false;
        }
    }
    
    std::cout << "F16 Comparison: max_diff=" << max_diff << ", avg_diff=" << (sum_diff / size) << std::endl;
    return true;
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

int main() {
    std::cout << "\n=== F16 Native Attention Test ===\n";
    
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
    
    // Load F16 native kernel
    NSError* error = nil;
    NSString* shaderPath = @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_f16_native.metal";
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath 
                                                      encoding:NSUTF8StringEncoding 
                                                         error:&error];
    if (!shaderSource) {
        std::cerr << "Failed to load F16 native shader source: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile F16 native shader library: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    id<MTLFunction> f16Function = [library newFunctionWithName:@"unified_batch_prefill_attention_f16_native"];
    if (!f16Function) {
        std::cerr << "Failed to find F16 native function\n";
        return -1;
    }
    
    id<MTLComputePipelineState> f16Pipeline = [device newComputePipelineStateWithFunction:f16Function error:&error];
    if (!f16Pipeline) {
        std::cerr << "Failed to create F16 native pipeline: " << error.localizedDescription.UTF8String << std::endl;
        return -1;
    }
    
    std::cout << "✅ F16 native kernel loaded successfully\n";
    
    // Test configuration: 1 sequence, 4 tokens, 1 head, head_size=4
    uint32_t batch_size = 1;
    uint32_t seq_len = 4;
    uint32_t num_heads = 1;
    uint32_t head_size = 4;
    uint32_t head_dim = 128;
    uint32_t num_kv_heads = 1;
    uint32_t tokens_per_page = 16;
    
    // Test data (use smaller values suitable for F16 precision)
    std::vector<float> q_data_f32 = {
        0.1f, 0.2f, 0.3f, 0.4f,      // Q0
        1.1f, 1.2f, 1.3f, 1.4f,      // Q1  
        2.1f, 2.2f, 2.3f, 2.4f,      // Q2
        3.1f, 3.2f, 3.3f, 3.4f       // Q3
    };
    
    std::vector<float> k_data_f32 = {
        1.0f, 0.0f, 0.0f, 0.0f,      // K0
        0.0f, 1.0f, 0.0f, 0.0f,      // K1
        0.0f, 0.0f, 1.0f, 0.0f,      // K2
        0.0f, 0.0f, 0.0f, 1.0f       // K3
    };
    
    std::vector<float> v_data_f32 = {
        0.1f, 0.2f, 0.3f, 0.4f,      // V0
        0.5f, 0.6f, 0.7f, 0.8f,      // V1
        0.9f, 1.0f, 1.1f, 1.2f,      // V2
        1.3f, 1.4f, 1.5f, 1.6f       // V3
    };
    
    std::cout << "Converting test data to F16...\n";
    
    // Convert to F16
    std::vector<uint16_t> q_data_f16, k_data_f16, v_data_f16;
    float_to_half_vector(q_data_f32, q_data_f16);
    float_to_half_vector(k_data_f32, k_data_f16);
    float_to_half_vector(v_data_f32, v_data_f16);
    
    // Setup parameters
    AttentionParams params = {};
    params.head_dim = head_dim;
    params.head_size = head_size;
    params.q_stride_seq = num_heads * head_size;
    params.q_stride_head = head_size;
    params.k_stride_head = head_size;
    params.v_stride_head = head_size;
    params.o_stride_seq = num_heads * head_size;
    params.o_stride_head = head_size;
    params.scale = 1.0f / sqrtf(static_cast<float>(head_size));
    params.num_layers = 1;
    params.layer_idx = 0;
    params.causal = 0;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_heads / num_kv_heads;
    params.logit_cap = 0.0f;
    
    // Setup indices
    std::vector<uint32_t> qo_indptr = {0, seq_len};
    std::vector<uint32_t> kv_indptr = {0, seq_len};
    std::vector<uint32_t> kv_page_indptr = {0, 1};
    std::vector<uint32_t> kv_page_indices = {0};
    std::vector<uint32_t> kv_last_page_lens = {seq_len};
    
    // Create paged cache layout (F16)
    size_t page_size = tokens_per_page * num_kv_heads * head_dim;
    std::vector<uint16_t> paged_k_f16(page_size, 0);
    std::vector<uint16_t> paged_v_f16(page_size, 0);
    
    // Fill paged cache
    for (uint32_t token_idx = 0; token_idx < seq_len; token_idx++) {
        for (uint32_t head_idx = 0; head_idx < num_kv_heads; head_idx++) {
            for (uint32_t d = 0; d < head_size; d++) {
                size_t src_idx = token_idx * head_size + d;
                size_t dst_idx = token_idx * num_kv_heads * head_dim + head_idx * head_dim + d;
                
                paged_k_f16[dst_idx] = k_data_f16[src_idx];
                paged_v_f16[dst_idx] = v_data_f16[src_idx];
            }
        }
    }
    
    // Create Metal buffers (F16)
    id<MTLBuffer> q_buffer_f16 = [device newBufferWithBytes:q_data_f16.data()
                                                      length:q_data_f16.size() * sizeof(uint16_t)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> k_buffer_f16 = [device newBufferWithBytes:paged_k_f16.data()
                                                      length:paged_k_f16.size() * sizeof(uint16_t)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> v_buffer_f16 = [device newBufferWithBytes:paged_v_f16.data()
                                                      length:paged_v_f16.size() * sizeof(uint16_t)
                                                     options:MTLResourceStorageModeShared];
    
    // Create index buffers
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
                                                      length:sizeof(AttentionParams)
                                                     options:MTLResourceStorageModeShared];
    
    // Create output buffer (F16)
    size_t output_size = batch_size * seq_len * num_heads * head_size;
    id<MTLBuffer> output_f16_buffer = [device newBufferWithLength:output_size * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
    
    // Run F16 native kernel
    std::cout << "\n--- Running F16 Native Kernel ---\n";
    
    id<MTLCommandBuffer> f16CommandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> f16Encoder = [f16CommandBuffer computeCommandEncoder];
    [f16Encoder setComputePipelineState:f16Pipeline];
    [f16Encoder setBuffer:q_buffer_f16 offset:0 atIndex:0];
    [f16Encoder setBuffer:k_buffer_f16 offset:0 atIndex:1];
    [f16Encoder setBuffer:v_buffer_f16 offset:0 atIndex:2];
    [f16Encoder setBuffer:qo_indptr_buffer offset:0 atIndex:3];
    [f16Encoder setBuffer:kv_indptr_buffer offset:0 atIndex:4];
    [f16Encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:5];
    [f16Encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:6];
    [f16Encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:7];
    [f16Encoder setBuffer:params_buffer offset:0 atIndex:8];
    [f16Encoder setBuffer:output_f16_buffer offset:0 atIndex:9];
    
    MTLSize f16GridSize = MTLSizeMake(batch_size, num_heads, seq_len);
    MTLSize f16ThreadgroupSize = MTLSizeMake(1, 1, 1);
    [f16Encoder dispatchThreads:f16GridSize threadsPerThreadgroup:f16ThreadgroupSize];
    [f16Encoder endEncoding];
    [f16CommandBuffer commit];
    [f16CommandBuffer waitUntilCompleted];
    
    // Read results (F16)
    uint16_t* f16_results = (uint16_t*)[output_f16_buffer contents];
    
    // Convert to F32 for display only
    std::vector<float> f16_results_f32(head_size);
    for (size_t i = 0; i < head_size; i++) {
        f16_results_f32[i] = f16_bits_to_float(f16_results[i]);
    }
    
    // Display results
    std::cout << "\n--- F16 Native Results ---\n";
    std::cout << "First 4 elements (token 0, head 0) in F32 for display:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << f16_results_f32[i] << " (F16 bits: 0x" << std::hex << f16_results[i] << std::dec << ") ";
    }
    std::cout << std::endl;
    
    // Compute expected results using reference computation (F32 precision for reference)
    std::vector<float> expected_f32(head_size);
    
    // Simple reference computation for first token
    float scores_f32[4];
    float max_score_f32 = -INFINITY;
    
    // Compute scores
    for (int kv_pos = 0; kv_pos < 4; kv_pos++) {
        float score = 0.0f;
        for (int d = 0; d < 4; d++) {
            score += q_data_f32[d] * k_data_f32[kv_pos * 4 + d];
        }
        score *= params.scale;
        scores_f32[kv_pos] = score;
        max_score_f32 = std::max(max_score_f32, score);
    }
    
    // Apply softmax
    float sum_exp_f32 = 0.0f;
    for (int kv_pos = 0; kv_pos < 4; kv_pos++) {
        float exp_score = expf(scores_f32[kv_pos] - max_score_f32);
        scores_f32[kv_pos] = exp_score;
        sum_exp_f32 += exp_score;
    }
    
    // Normalize
    for (int kv_pos = 0; kv_pos < 4; kv_pos++) {
        scores_f32[kv_pos] /= sum_exp_f32;
    }
    
    // Compute output
    for (int d = 0; d < 4; d++) {
        expected_f32[d] = 0.0f;
        for (int kv_pos = 0; kv_pos < 4; kv_pos++) {
            expected_f32[d] += scores_f32[kv_pos] * v_data_f32[kv_pos * 4 + d];
        }
    }
    
    std::cout << "\nExpected results (F32 reference): ";
    for (int i = 0; i < 4; i++) {
        std::cout << expected_f32[i] << " ";
    }
    std::cout << std::endl;
    
    // Convert expected to F16 for comparison
    std::vector<uint16_t> expected_f16(head_size);
    for (size_t i = 0; i < head_size; i++) {
        expected_f16[i] = float_to_f16_bits(expected_f32[i]);
    }
    
    // Compare in F16 space with appropriate tolerance
    float f16_tolerance = 1e-2f; // More relaxed tolerance for F16
    bool accuracy_ok = compare_f16_results(expected_f16.data(), f16_results, head_size, f16_tolerance);
    
    std::cout << "\n--- Final Result ---\n";
    std::cout << "F16 Native Kernel: " << (accuracy_ok ? "✅ PASSED" : "❌ FAILED") << std::endl;
    std::cout << "F16 tolerance used: " << f16_tolerance << std::endl;
    
    return accuracy_ok ? 0 : 1;
}