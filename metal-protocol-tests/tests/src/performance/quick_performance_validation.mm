#include "metal_batch_prefill_handle.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace metal::batch_prefill_attention;
using namespace std::chrono;

struct QuickBenchmarkResult {
    const char* config_name;
    double baseline_time_ms;
    double simdgroup_time_ms;
    double per_head_time_ms;
    double simdgroup_speedup;
    double per_head_speedup;
    bool success;
};

int main() {
    std::cout << "⚡ Quick Metal FlashAttention Priority 0 & 2 Performance Validation" << std::endl;
    std::cout << "=================================================================" << std::endl;
    
    // Create handle
    MetalBatchPrefillHandle* handle = metal_batch_prefill_create_handle(512, 2048, 32, 1024);
    if (!handle) {
        std::cerr << "❌ Failed to create Metal handle" << std::endl;
        return 1;
    }
    
    // Quick test configurations focused on Priority 0 targets
    struct TestConfig {
        const char* name;
        int num_tokens;
        int head_dim; 
        int head_size;
        int pages;
    } configs[] = {
        {"Small: 128 tokens, 64 head_size", 128, 2048, 64, 8},
        {"Small: 256 tokens, 64 head_size", 256, 4096, 64, 16},
        {"Medium: 512 tokens, 128 head_size", 512, 4096, 128, 32}
    };
    
    std::vector<QuickBenchmarkResult> results;
    std::mt19937 rng(12345);
    
    for (const auto& config : configs) {
        std::cout << "\n🔄 Testing: " << config.name << std::endl;
        
        // Generate minimal test data
        std::vector<uint16_t> q_data(config.num_tokens * config.head_dim, 0x3F80); // 1.0 in bf16
        std::vector<uint16_t> k_data(config.pages * 16 * config.head_dim, 0x3F80);
        std::vector<uint16_t> v_data(config.pages * 16 * config.head_dim, 0x3F80);
        std::vector<uint16_t> output_data(config.num_tokens * config.head_dim, 0);
        
        std::vector<int32_t> qo_indptr = {0, config.num_tokens};
        std::vector<int32_t> kv_page_indptr = {0, config.pages};
        std::vector<int32_t> kv_page_indices(config.pages);
        std::iota(kv_page_indices.begin(), kv_page_indices.end(), 0);
        std::vector<int32_t> kv_last_page_lens = {16};
        
        // Get workspace
        MetalBatchPrefillWorkspace workspace = metal_batch_prefill_get_workspace(
            handle, config.num_tokens, config.head_dim, config.head_dim, 16, config.pages);
        std::vector<uint8_t> workspace_buffer(workspace.total_size);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
        int num_heads = config.head_dim / config.head_size;
        
        QuickBenchmarkResult result = {};
        result.config_name = config.name;
        
        // Test baseline (3 iterations for speed)
        auto start = high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            batch_prefill_attention_unified_bf16(
                handle, workspace_buffer.data(), workspace_buffer.size(),
                q_data.data(), k_data.data(), v_data.data(),
                qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(),
                kv_last_page_lens.data(), output_data.data(),
                config.num_tokens, config.head_dim, config.head_dim, config.head_size, 16,
                num_heads, num_heads, scale, config.pages, KernelOptimizationLevel::BASELINE
            );
        }
        auto end = high_resolution_clock::now();
        result.baseline_time_ms = duration_cast<microseconds>(end - start).count() / (1000.0 * 3);
        
        // Test simdgroup optimized
        start = high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            batch_prefill_attention_unified_bf16(
                handle, workspace_buffer.data(), workspace_buffer.size(),
                q_data.data(), k_data.data(), v_data.data(),
                qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(),
                kv_last_page_lens.data(), output_data.data(),
                config.num_tokens, config.head_dim, config.head_dim, config.head_size, 16,
                num_heads, num_heads, scale, config.pages, KernelOptimizationLevel::SIMDGROUP_OPT
            );
        }
        end = high_resolution_clock::now();
        result.simdgroup_time_ms = duration_cast<microseconds>(end - start).count() / (1000.0 * 3);
        
        // Test per-head optimized (Priority 2)
        start = high_resolution_clock::now();
        for (int i = 0; i < 3; i++) {
            batch_prefill_attention_unified_bf16(
                handle, workspace_buffer.data(), workspace_buffer.size(),
                q_data.data(), k_data.data(), v_data.data(),
                qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(),
                kv_last_page_lens.data(), output_data.data(),
                config.num_tokens, config.head_dim, config.head_dim, config.head_size, 16,
                num_heads, num_heads, scale, config.pages, KernelOptimizationLevel::PER_HEAD_OPT
            );
        }
        end = high_resolution_clock::now();
        result.per_head_time_ms = duration_cast<microseconds>(end - start).count() / (1000.0 * 3);
        
        result.simdgroup_speedup = result.baseline_time_ms / result.simdgroup_time_ms;
        result.per_head_speedup = result.baseline_time_ms / result.per_head_time_ms;
        result.success = true;
        
        std::cout << "  📋 Baseline: " << std::fixed << std::setprecision(2) << result.baseline_time_ms << "ms" << std::endl;
        std::cout << "  ⚡ Simdgroup: " << std::fixed << std::setprecision(2) << result.simdgroup_time_ms << "ms" << std::endl;
        std::cout << "  🎯 Per-head: " << std::fixed << std::setprecision(2) << result.per_head_time_ms << "ms" << std::endl;
        
        std::cout << "  🚀 Simdgroup speedup: " << std::fixed << std::setprecision(2) << result.simdgroup_speedup << "x";
        if (result.simdgroup_speedup > 1.3) {
            std::cout << " ✅ EXCELLENT";
        } else if (result.simdgroup_speedup > 1.1) {
            std::cout << " ✅ GOOD";
        } else if (result.simdgroup_speedup > 0.9) {
            std::cout << " ➖ NEUTRAL";
        } else {
            std::cout << " ❌ SLOWER";
        }
        std::cout << std::endl;
        
        std::cout << "  🎯 Per-head speedup: " << std::fixed << std::setprecision(2) << result.per_head_speedup << "x";
        if (result.per_head_speedup > 1.3) {
            std::cout << " ✅ EXCELLENT";
        } else if (result.per_head_speedup > 1.1) {
            std::cout << " ✅ GOOD";
        } else if (result.per_head_speedup > 0.9) {
            std::cout << " ➖ NEUTRAL";
        } else {
            std::cout << " ❌ SLOWER";
        }
        std::cout << std::endl;
        
        results.push_back(result);
    }
    
    // Summary
    std::cout << "\n📊 PRIORITY 0 & 2 PERFORMANCE VALIDATION SUMMARY" << std::endl;
    std::cout << "================================================" << std::endl;
    
    double total_simdgroup_speedup = 0.0;
    double total_per_head_speedup = 0.0;
    int simdgroup_good_results = 0;
    int per_head_good_results = 0;
    
    for (const auto& result : results) {
        std::cout << "📝 " << result.config_name << ":" << std::endl;
        std::cout << "  ⚡ Simdgroup: " << std::fixed << std::setprecision(2) << result.simdgroup_speedup << "x";
        if (result.simdgroup_speedup > 1.1) {
            std::cout << " ✅";
            simdgroup_good_results++;
        } else if (result.simdgroup_speedup < 0.9) {
            std::cout << " ❌";
        }
        std::cout << std::endl;
        
        std::cout << "  🎯 Per-head: " << std::fixed << std::setprecision(2) << result.per_head_speedup << "x";
        if (result.per_head_speedup > 1.1) {
            std::cout << " ✅";
            per_head_good_results++;
        } else if (result.per_head_speedup < 0.9) {
            std::cout << " ❌";
        }
        std::cout << std::endl;
        
        total_simdgroup_speedup += result.simdgroup_speedup;
        total_per_head_speedup += result.per_head_speedup;
    }
    
    double avg_simdgroup_speedup = total_simdgroup_speedup / results.size();
    double avg_per_head_speedup = total_per_head_speedup / results.size();
    
    std::cout << "\n🎯 FINAL RESULTS:" << std::endl;
    std::cout << "   📈 Average simdgroup speedup: " << std::fixed << std::setprecision(2) << avg_simdgroup_speedup << "x" << std::endl;
    std::cout << "   🎯 Average per-head speedup: " << std::fixed << std::setprecision(2) << avg_per_head_speedup << "x" << std::endl;
    std::cout << "   ✅ Simdgroup good results: " << simdgroup_good_results << "/" << results.size() << std::endl;
    std::cout << "   ✅ Per-head good results: " << per_head_good_results << "/" << results.size() << std::endl;
    
    bool priority0_passed = avg_simdgroup_speedup >= 1.3;
    bool priority2_passed = avg_per_head_speedup >= 1.2; // Slightly lower target for Priority 2
    
    if (priority0_passed && priority2_passed) {
        std::cout << "   🏆 ALL VALIDATIONS: PASSED ✅" << std::endl;
        std::cout << "   🎉 Both Priority 0 and Priority 2 optimizations successful!" << std::endl;
    } else if (priority0_passed) {
        std::cout << "   ✅ PRIORITY 0 VALIDATION: PASSED" << std::endl;
        std::cout << "   ⚠️ PRIORITY 2 VALIDATION: NEEDS TUNING" << std::endl;
    } else if (priority2_passed) {
        std::cout << "   ✅ PRIORITY 2 VALIDATION: PASSED" << std::endl;
        std::cout << "   ⚠️ PRIORITY 0 VALIDATION: NEEDS TUNING" << std::endl;
    } else {
        std::cout << "   ⚠️ BOTH VALIDATIONS: NEED TUNING" << std::endl;
        std::cout << "   🔍 Performance below expectations for both optimization levels" << std::endl;
    }
    
    metal_batch_prefill_destroy_handle(handle);
    return 0;
}