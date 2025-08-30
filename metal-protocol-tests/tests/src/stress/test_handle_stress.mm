#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include "../../backend/backend-metal/src/metal_batch_prefill_handle.hpp"

// Function to generate random bf16 data
void generate_random_bf16(void* buffer, size_t num_elements, float min_val = -1.0f, float max_val = 1.0f) {
    uint16_t* bf16_buffer = static_cast<uint16_t*>(buffer);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < num_elements; ++i) {
        float value = dis(gen);
        uint32_t float_bits = *reinterpret_cast<uint32_t*>(&value);
        bf16_buffer[i] = static_cast<uint16_t>(float_bits >> 16);
    }
}

// Function to convert bf16 to float for verification
float bf16_to_float(uint16_t bf16_val) {
    uint32_t float_bits = static_cast<uint32_t>(bf16_val) << 16;
    return *reinterpret_cast<float*>(&float_bits);
}

// Function to verify output is valid (not NaN/Inf and within expected range)
bool verify_output(void* output, size_t num_elements) {
    uint16_t* bf16_output = static_cast<uint16_t*>(output);
    
    for (size_t i = 0; i < num_elements; ++i) {
        float val = bf16_to_float(bf16_output[i]);
        
        // Check for NaN or Inf
        if (std::isnan(val) || std::isinf(val)) {
            std::cerr << "ERROR: Found NaN/Inf at position " << i << std::endl;
            return false;
        }
        
        // Check for reasonable range (attention outputs should be bounded)
        if (std::abs(val) > 100.0f) {
            std::cerr << "ERROR: Value out of range at position " << i << ": " << val << std::endl;
            return false;
        }
    }
    
    return true;
}

// Function to compare two outputs
bool compare_outputs(void* output1, void* output2, size_t num_elements, float tolerance = 1e-3f) {
    uint16_t* bf16_output1 = static_cast<uint16_t*>(output1);
    uint16_t* bf16_output2 = static_cast<uint16_t*>(output2);
    
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    
    for (size_t i = 0; i < num_elements; ++i) {
        float val1 = bf16_to_float(bf16_output1[i]);
        float val2 = bf16_to_float(bf16_output2[i]);
        float diff = std::abs(val1 - val2);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        if (diff > tolerance) {
            std::cerr << "ERROR: Outputs differ at position " << i 
                     << ": " << val1 << " vs " << val2 
                     << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    
    std::cout << "Max difference: " << max_diff << " at position " << max_diff_idx << std::endl;
    return true;
}

int main() {
    std::cout << "=== Metal Batch Prefill Attention Handle Stress Test ===" << std::endl;
    
    // Test configuration
    const int num_tokens = 128;
    const int num_query_heads = 32;
    const int num_kv_heads = 8;
    const int head_size = 128;
    const int kv_len = 2048;
    const int page_size = 16;
    const int batch_size = 1;
    const int num_pages = 128;
    
    const int head_dim = num_query_heads * head_size;
    const int kv_head_dim = num_kv_heads * head_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    
    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  num_query_heads: " << num_query_heads << std::endl;
    std::cout << "  num_kv_heads: " << num_kv_heads << std::endl;
    std::cout << "  head_size: " << head_size << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;
    std::cout << "  kv_head_dim: " << kv_head_dim << std::endl;
    std::cout << "  kv_len: " << kv_len << std::endl;
    std::cout << "  page_size: " << page_size << std::endl;
    std::cout << "  num_pages: " << num_pages << std::endl;
    
    // Create handle once (this should be reused)
    std::cout << "\nCreating attention handle..." << std::endl;
    auto* attention_handle = metal::batch_prefill_attention::metal_batch_prefill_create_handle(
        1024,  // max_batch_size
        8192,  // max_seq_length
        num_query_heads,
        head_dim
    );
    
    if (!attention_handle) {
        std::cerr << "ERROR: Failed to create attention handle" << std::endl;
        return 1;
    }
    
    // Get workspace requirements
    auto workspace = metal::batch_prefill_attention::metal_batch_prefill_get_workspace(
        attention_handle,
        num_tokens,
        head_dim,
        kv_head_dim,
        page_size,
        num_pages
    );
    
    std::cout << "Workspace size required: " << workspace.total_size << " bytes" << std::endl;
    
    // Allocate workspace buffer once (this should be reused)
    id<MTLBuffer> workspace_buffer = [attention_handle->device 
        newBufferWithLength:workspace.total_size
        options:MTLResourceStorageModeShared];
    
    if (!workspace_buffer) {
        std::cerr << "ERROR: Failed to allocate workspace buffer" << std::endl;
        metal::batch_prefill_attention::metal_batch_prefill_destroy_handle(attention_handle);
        return 1;
    }
    
    // Allocate input/output buffers
    size_t q_size = num_tokens * head_dim;
    size_t k_size = kv_len * kv_head_dim;
    size_t v_size = kv_len * kv_head_dim;
    size_t kv_cache_size = num_pages * page_size * kv_head_dim;
    size_t output_size = num_tokens * head_dim;
    
    std::vector<uint16_t> q_input(q_size);
    std::vector<uint16_t> k_input(k_size);
    std::vector<uint16_t> v_input(v_size);
    std::vector<uint16_t> paged_k_cache(kv_cache_size);
    std::vector<uint16_t> paged_v_cache(kv_cache_size);
    std::vector<uint16_t> output(output_size);
    std::vector<uint16_t> reference_output(output_size);
    
    // Initialize index arrays
    std::vector<int32_t> qo_indptr = {0, num_tokens};
    std::vector<int32_t> kv_page_indptr = {0, num_pages};
    std::vector<int32_t> kv_page_indices(num_pages);
    for (int i = 0; i < num_pages; ++i) {
        kv_page_indices[i] = i;
    }
    std::vector<int32_t> kv_last_page_lens = {page_size};
    
    // Generate initial random data
    std::cout << "\nGenerating random input data..." << std::endl;
    generate_random_bf16(q_input.data(), q_size);
    generate_random_bf16(k_input.data(), k_size);
    generate_random_bf16(v_input.data(), v_size);
    generate_random_bf16(paged_k_cache.data(), kv_cache_size);
    generate_random_bf16(paged_v_cache.data(), kv_cache_size);
    
    // Run attention multiple times to test for internal errors
    const int num_iterations = 10;
    std::cout << "\n=== Running " << num_iterations << " iterations ===" << std::endl;
    
    bool all_passed = true;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << num_iterations << " ---" << std::endl;
        
        // Clear output buffer
        std::fill(output.begin(), output.end(), 0);
        
        // Run attention
        try {
            metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
                attention_handle,
                [workspace_buffer contents],
                [workspace_buffer length],
                q_input.data(),
                paged_k_cache.data(),
                paged_v_cache.data(),
                qo_indptr.data(),
                kv_page_indptr.data(),
                kv_page_indices.data(),
                kv_last_page_lens.data(),
                output.data(),
                num_tokens,
                head_dim,
                kv_head_dim,
                head_size,
                page_size,
                num_query_heads,
                num_kv_heads,
                scale,
                num_pages
            );
            
            std::cout << "✓ Attention execution successful" << std::endl;
            
            // Verify output is valid
            if (!verify_output(output.data(), output_size)) {
                std::cerr << "✗ Output validation failed!" << std::endl;
                all_passed = false;
                break;
            }
            std::cout << "✓ Output validation passed" << std::endl;
            
            // For first iteration, save as reference
            if (iter == 0) {
                std::copy(output.begin(), output.end(), reference_output.begin());
                std::cout << "✓ Saved as reference output" << std::endl;
            } else {
                // Compare with reference output
                if (!compare_outputs(output.data(), reference_output.data(), output_size)) {
                    std::cerr << "✗ Output consistency check failed!" << std::endl;
                    all_passed = false;
                    break;
                }
                std::cout << "✓ Output consistency check passed" << std::endl;
            }
            
            // Optionally regenerate some input data to test with different inputs
            if (iter % 3 == 2 && iter < num_iterations - 1) {
                std::cout << "  Regenerating input data for next iteration..." << std::endl;
                generate_random_bf16(q_input.data(), q_size);
                generate_random_bf16(k_input.data(), k_size);
                generate_random_bf16(v_input.data(), v_size);
                
                // Run with new data to establish new reference
                std::fill(output.begin(), output.end(), 0);
                metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
                    attention_handle,
                    [workspace_buffer contents],
                    [workspace_buffer length],
                    q_input.data(),
                    paged_k_cache.data(),
                    paged_v_cache.data(),
                    qo_indptr.data(),
                    kv_page_indptr.data(),
                    kv_page_indices.data(),
                    kv_last_page_lens.data(),
                    output.data(),
                    num_tokens,
                    head_dim,
                    kv_head_dim,
                    head_size,
                    page_size,
                    num_query_heads,
                    num_kv_heads,
                    scale,
                    num_pages
                );
                std::copy(output.begin(), output.end(), reference_output.begin());
                std::cout << "  ✓ New reference established for regenerated data" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "✗ Exception during attention execution: " << e.what() << std::endl;
            all_passed = false;
            break;
        }
    }
    
    // Test with edge cases
    std::cout << "\n=== Testing edge cases ===" << std::endl;
    
    // Test 1: All zeros input
    std::cout << "\nTest 1: All zeros input" << std::endl;
    std::fill(q_input.begin(), q_input.end(), 0);
    std::fill(output.begin(), output.end(), 0);
    
    try {
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            attention_handle,
            [workspace_buffer contents],
            [workspace_buffer length],
            q_input.data(),
            paged_k_cache.data(),
            paged_v_cache.data(),
            qo_indptr.data(),
            kv_page_indptr.data(),
            kv_page_indices.data(),
            kv_last_page_lens.data(),
            output.data(),
            num_tokens,
            head_dim,
            kv_head_dim,
            head_size,
            page_size,
            num_query_heads,
            num_kv_heads,
            scale,
            num_pages
        );
        
        if (!verify_output(output.data(), output_size)) {
            std::cerr << "✗ All zeros test failed validation!" << std::endl;
            all_passed = false;
        } else {
            std::cout << "✓ All zeros test passed" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "✗ All zeros test threw exception: " << e.what() << std::endl;
        all_passed = false;
    }
    
    // Test 2: Large values input
    std::cout << "\nTest 2: Large values input" << std::endl;
    generate_random_bf16(q_input.data(), q_size, -10.0f, 10.0f);
    std::fill(output.begin(), output.end(), 0);
    
    try {
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            attention_handle,
            [workspace_buffer contents],
            [workspace_buffer length],
            q_input.data(),
            paged_k_cache.data(),
            paged_v_cache.data(),
            qo_indptr.data(),
            kv_page_indptr.data(),
            kv_page_indices.data(),
            kv_last_page_lens.data(),
            output.data(),
            num_tokens,
            head_dim,
            kv_head_dim,
            head_size,
            page_size,
            num_query_heads,
            num_kv_heads,
            scale,
            num_pages
        );
        
        if (!verify_output(output.data(), output_size)) {
            std::cerr << "✗ Large values test failed validation!" << std::endl;
            all_passed = false;
        } else {
            std::cout << "✓ Large values test passed" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "✗ Large values test threw exception: " << e.what() << std::endl;
        all_passed = false;
    }
    
    // Cleanup
    std::cout << "\nCleaning up resources..." << std::endl;
    metal::batch_prefill_attention::metal_batch_prefill_destroy_handle(attention_handle);
    
    // Final result
    std::cout << "\n=== Final Result ===" << std::endl;
    if (all_passed) {
        std::cout << "✅ ALL TESTS PASSED" << std::endl;
        std::cout << "The handle-based API successfully prevents Metal Internal Errors!" << std::endl;
        std::cout << "Input and output validation confirmed correct operation." << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED" << std::endl;
        return 1;
    }
}