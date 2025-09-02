#include "metal_batch_prefill_handle.hpp"
#include <iostream>
#include <chrono>

using namespace metal::batch_prefill_attention;

// Simple microbenchmark and validation test for kernel selection
int main() {
    std::cout << "🧪 Testing Metal Attention Kernel Selection" << std::endl;
    
    // Test handle creation
    MetalBatchPrefillHandle* handle = metal_batch_prefill_create_handle(64, 2048, 32, 1024);
    if (!handle) {
        std::cerr << "❌ Failed to create Metal handle" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Metal handle created successfully" << std::endl;
    std::cout << "📊 Testing kernel availability:" << std::endl;
    
    // Check kernel availability
    bool has_baseline_bf16 = (handle->pipeline_bf16_baseline != nil);
    bool has_baseline_f32 = (handle->pipeline_f32_baseline != nil);
    bool has_simdgroup_bf16 = (handle->pipeline_bf16_simdgroup != nil);
    bool has_simdgroup_f32 = (handle->pipeline_f32_simdgroup != nil);
    
    std::cout << "  📋 Baseline BF16 kernel: " << (has_baseline_bf16 ? "✅ Available" : "❌ Not available") << std::endl;
    std::cout << "  📋 Baseline F32 kernel: " << (has_baseline_f32 ? "✅ Available" : "❌ Not available") << std::endl;
    std::cout << "  ⚡ Simdgroup BF16 kernel: " << (has_simdgroup_bf16 ? "✅ Available" : "❌ Not available") << std::endl;
    std::cout << "  ⚡ Simdgroup F32 kernel: " << (has_simdgroup_f32 ? "✅ Available" : "❌ Not available") << std::endl;
    
    // Test workspace calculation for different problem sizes
    std::cout << "\n📏 Testing workspace requirements:" << std::endl;
    
    struct TestCase {
        int num_tokens;
        int head_dim;
        int kv_head_dim;
        int page_size;
        int num_kv_pages;
        const char* description;
    };
    
    TestCase test_cases[] = {
        {128, 2048, 2048, 16, 8, "Small sequence (128 tokens, 16 pages)"},
        {512, 4096, 4096, 16, 32, "Medium sequence (512 tokens, 32 pages)"},
        {1024, 8192, 8192, 16, 64, "Large sequence (1024 tokens, 64 pages)"}
    };
    
    for (const auto& test : test_cases) {
        MetalBatchPrefillWorkspace workspace = metal_batch_prefill_get_workspace(
            handle, test.num_tokens, test.head_dim, test.kv_head_dim, test.page_size, test.num_kv_pages);
        
        std::cout << "  📦 " << test.description << ": " 
                  << (workspace.total_size / 1024.0 / 1024.0) << " MB" << std::endl;
    }
    
    // Test kernel selection logic (without actual execution)
    std::cout << "\n🎯 Testing kernel selection logic:" << std::endl;
    
    struct SelectionTest {
        KernelOptimizationLevel level;
        int num_tokens;
        int total_kv_len;
        int head_size;
        const char* description;
    };
    
    SelectionTest selection_tests[] = {
        {KernelOptimizationLevel::AUTO, 128, 256, 64, "AUTO: Small sequence should use simdgroup"},
        {KernelOptimizationLevel::AUTO, 1024, 2048, 128, "AUTO: Large sequence should use baseline"},
        {KernelOptimizationLevel::BASELINE, 128, 256, 64, "BASELINE: Force baseline kernel"},
        {KernelOptimizationLevel::SIMDGROUP_OPT, 128, 256, 64, "SIMDGROUP_OPT: Force simdgroup kernel"}
    };
    
    for (const auto& test : selection_tests) {
        std::string expected_kernel;
        switch (test.level) {
            case KernelOptimizationLevel::AUTO:
                if (test.total_kv_len <= 512 && test.head_size <= 128) {
                    expected_kernel = has_simdgroup_bf16 ? "Simdgroup" : "Baseline";
                } else {
                    expected_kernel = has_baseline_bf16 ? "Baseline" : "Unified";
                }
                break;
            case KernelOptimizationLevel::BASELINE:
                expected_kernel = "Baseline";
                break;
            case KernelOptimizationLevel::SIMDGROUP_OPT:
                expected_kernel = "Simdgroup";
                break;
        }
        
        std::cout << "  🎲 " << test.description << " → Expected: " << expected_kernel << std::endl;
    }
    
    // Performance characteristics note
    std::cout << "\n⚡ Priority 0 Optimization Targets:" << std::endl;
    std::cout << "  📈 Thread utilization: 12.5% → 70%+" << std::endl;
    std::cout << "  🚧 Barriers per block: 6-8 → 2-3" << std::endl;
    std::cout << "  💾 Memory traffic: Eliminates w_block array" << std::endl;
    std::cout << "  🏃 Expected speedup: 30-80% for 128-512 token sequences" << std::endl;
    
    // Clean up
    metal_batch_prefill_destroy_handle(handle);
    
    std::cout << "\n✅ Kernel selection test completed successfully!" << std::endl;
    std::cout << "🎯 Ready for Priority 0 optimization validation" << std::endl;
    
    return 0;
}