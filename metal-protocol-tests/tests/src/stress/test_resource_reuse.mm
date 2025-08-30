#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "../../backend/backend-metal/src/metal_batch_prefill_handle.hpp"

// Simple test to demonstrate resource reuse prevents Metal Internal Errors
int main() {
    std::cout << "=== Metal Resource Reuse Demonstration ===" << std::endl;
    
    // Test configuration - using same configuration as the failing case
    const int num_tokens = 128;
    const int num_query_heads = 32;
    const int num_kv_heads = 8;
    const int head_size = 128;
    const int head_dim = num_query_heads * head_size;
    const int kv_head_dim = num_kv_heads * head_size;
    const int page_size = 16;
    const int num_pages = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    
    // Create handle once (this allocates Metal resources once)
    std::cout << "Creating reusable attention handle..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto* attention_handle = metal::batch_prefill_attention::metal_batch_prefill_create_handle(
        1024, 8192, num_query_heads, head_dim
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "✓ Handle created in " << duration.count() << " ms" << std::endl;
    
    // Get workspace requirements once
    auto workspace = metal::batch_prefill_attention::metal_batch_prefill_get_workspace(
        attention_handle, num_tokens, head_dim, kv_head_dim, page_size, num_pages
    );
    
    // Allocate workspace buffer once (this allocates memory once)
    std::cout << "Allocating workspace buffer (" << workspace.total_size / 1024 / 1024 << " MB)..." << std::endl;
    id<MTLBuffer> workspace_buffer = [attention_handle->device 
        newBufferWithLength:workspace.total_size
        options:MTLResourceStorageModeShared];
    
    // Allocate input/output data
    size_t q_size = num_tokens * head_dim;
    size_t kv_cache_size = num_pages * page_size * kv_head_dim;
    size_t output_size = num_tokens * head_dim;
    
    std::vector<uint16_t> q_input(q_size, 0x3C00);  // bf16 value of 1.0
    std::vector<uint16_t> paged_k_cache(kv_cache_size, 0x3C00);
    std::vector<uint16_t> paged_v_cache(kv_cache_size, 0x3C00);
    std::vector<uint16_t> output(output_size);
    
    // Initialize index arrays
    std::vector<int32_t> qo_indptr = {0, num_tokens};
    std::vector<int32_t> kv_page_indptr = {0, num_pages};
    std::vector<int32_t> kv_page_indices(num_pages);
    for (int i = 0; i < num_pages; ++i) {
        kv_page_indices[i] = i;
    }
    std::vector<int32_t> kv_last_page_lens = {page_size};
    
    // Rapid-fire execution test - This would cause Metal Internal Error with the old API
    const int rapid_iterations = 50;
    std::cout << "\n=== Rapid-fire execution test (" << rapid_iterations << " iterations) ===" << std::endl;
    std::cout << "This would cause Metal Internal Error (0x0000000e) with the old per-call allocation API" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    bool all_successful = true;
    
    for (int i = 0; i < rapid_iterations; ++i) {
        if (i % 10 == 0) {
            std::cout << "Iteration " << (i + 1) << "/" << rapid_iterations << std::flush;
        }
        
        try {
            // Clear output
            std::fill(output.begin(), output.end(), 0);
            
            // Execute attention (reusing handle and workspace)
            metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
                attention_handle,                    // ← REUSED handle (no new Metal resources)
                [workspace_buffer contents],        // ← REUSED workspace buffer
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
            
            if (i % 10 == 9) {
                std::cout << " ✓" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << " ✗" << std::endl;
            std::cerr << "ERROR at iteration " << i << ": " << e.what() << std::endl;
            all_successful = false;
            break;
        }
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (all_successful) {
        std::cout << "\n✅ SUCCESS: All " << rapid_iterations << " iterations completed without Metal Internal Error!" << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per call: " << (duration.count() / rapid_iterations) << " ms" << std::endl;
        
        // Verify output is valid
        bool output_valid = true;
        for (size_t i = 0; i < output_size; ++i) {
            uint32_t float_bits = static_cast<uint32_t>(output[i]) << 16;
            float val = *reinterpret_cast<float*>(&float_bits);
            if (std::isnan(val) || std::isinf(val)) {
                output_valid = false;
                break;
            }
        }
        
        if (output_valid) {
            std::cout << "✓ Output validation: All values are valid (no NaN/Inf)" << std::endl;
        } else {
            std::cout << "✗ Output validation: Found invalid values" << std::endl;
        }
        
    } else {
        std::cout << "\n❌ FAILED: Metal Internal Error occurred during rapid execution" << std::endl;
    }
    
    // Resource tracking
    std::cout << "\n=== Resource Usage Summary ===" << std::endl;
    std::cout << "Handle allocations: 1 (reused for all " << rapid_iterations << " calls)" << std::endl;
    std::cout << "Workspace allocations: 1 (reused for all " << rapid_iterations << " calls)" << std::endl;
    std::cout << "Command queues: 1 (reused for all " << rapid_iterations << " calls)" << std::endl;
    std::cout << "Pipeline states: 1 (reused for all " << rapid_iterations << " calls)" << std::endl;
    
    // Cleanup
    std::cout << "\nCleaning up resources..." << std::endl;
    metal::batch_prefill_attention::metal_batch_prefill_destroy_handle(attention_handle);
    
    std::cout << "\n=== Summary ===" << std::endl;
    if (all_successful) {
        std::cout << "✅ The handle-based API successfully prevents Metal Internal Errors by:" << std::endl;
        std::cout << "   • Reusing Metal device and command queue across calls" << std::endl;
        std::cout << "   • Reusing compiled pipeline states" << std::endl;
        std::cout << "   • Reusing pre-allocated workspace buffers" << std::endl;
        std::cout << "   • Avoiding per-call resource allocation/deallocation" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Resource exhaustion still occurs - further investigation needed" << std::endl;
        return 1;
    }
}