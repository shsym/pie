#include "metal_batch_prefill_handle.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace metal::batch_prefill_attention;
using namespace std::chrono;

struct BenchmarkConfig {
    int num_tokens;
    int head_dim;
    int kv_head_dim;
    int head_size;
    int page_size;
    int num_kv_pages;
    int num_query_heads;
    int num_kv_heads;
    const char* description;
};

struct BenchmarkResult {
    double baseline_time_ms;
    double simdgroup_time_ms;
    double speedup_factor;
    bool baseline_success;
    bool simdgroup_success;
    size_t memory_usage_mb;
};

class AttentionBenchmark {
private:
    MetalBatchPrefillHandle* handle;
    std::mt19937 rng;
    
    // Generate random test data
    std::vector<uint16_t> generate_random_bf16_data(size_t size) {
        std::vector<uint16_t> data(size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t i = 0; i < size; i++) {
            float val = dist(rng);
            // Convert to bfloat16 (truncate to 16 bits)
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            data[i] = static_cast<uint16_t>((bits + 0x8000u) >> 16);
        }
        return data;
    }
    
    std::vector<int32_t> generate_sequential_indices(int start, int count) {
        std::vector<int32_t> indices(count);
        for (int i = 0; i < count; i++) {
            indices[i] = start + i;
        }
        return indices;
    }
    
    double benchmark_kernel(const BenchmarkConfig& config, 
                           KernelOptimizationLevel opt_level, 
                           int iterations = 10) {
        
        // Generate test data
        size_t q_size = config.num_tokens * config.head_dim;
        size_t kv_size = config.num_kv_pages * config.page_size * config.kv_head_dim;
        
        auto q_data = generate_random_bf16_data(q_size);
        auto k_data = generate_random_bf16_data(kv_size);
        auto v_data = generate_random_bf16_data(kv_size);
        auto output_data = std::vector<uint16_t>(q_size, 0);
        
        // Generate index data
        auto qo_indptr = generate_sequential_indices(0, 2); // Single sequence
        qo_indptr[1] = config.num_tokens;
        
        auto kv_page_indptr = generate_sequential_indices(0, 2);
        kv_page_indptr[1] = config.num_kv_pages;
        
        auto kv_page_indices = generate_sequential_indices(0, config.num_kv_pages);
        auto kv_last_page_lens = std::vector<int32_t>(1, config.page_size);
        
        // Get workspace
        MetalBatchPrefillWorkspace workspace = metal_batch_prefill_get_workspace(
            handle, config.num_tokens, config.head_dim, config.kv_head_dim, 
            config.page_size, config.num_kv_pages);
        
        // Allocate workspace buffer
        std::vector<uint8_t> workspace_buffer(workspace.total_size);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
        
        // Warm up (1 iteration)
        try {
            batch_prefill_attention_unified_bf16(
                handle, workspace_buffer.data(), workspace_buffer.size(),
                q_data.data(), k_data.data(), v_data.data(),
                qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(),
                kv_last_page_lens.data(), output_data.data(),
                config.num_tokens, config.head_dim, config.kv_head_dim,
                config.head_size, config.page_size, config.num_query_heads,
                config.num_kv_heads, scale, config.num_kv_pages, opt_level
            );
        } catch (...) {
            return -1.0; // Indicate failure
        }
        
        // Benchmark iterations
        auto start_time = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            batch_prefill_attention_unified_bf16(
                handle, workspace_buffer.data(), workspace_buffer.size(),
                q_data.data(), k_data.data(), v_data.data(),
                qo_indptr.data(), kv_page_indptr.data(), kv_page_indices.data(),
                kv_last_page_lens.data(), output_data.data(),
                config.num_tokens, config.head_dim, config.kv_head_dim,
                config.head_size, config.page_size, config.num_query_heads,
                config.num_kv_heads, scale, config.num_kv_pages, opt_level
            );
        }
        
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        
        return duration.count() / (1000.0 * iterations); // Convert to ms per iteration
    }
    
public:
    AttentionBenchmark() : rng(12345) {
        handle = metal_batch_prefill_create_handle(1024, 4096, 64, 2048);
        if (!handle) {
            throw std::runtime_error("Failed to create Metal handle");
        }
    }
    
    ~AttentionBenchmark() {
        if (handle) {
            metal_batch_prefill_destroy_handle(handle);
        }
    }
    
    BenchmarkResult run_benchmark(const BenchmarkConfig& config, int iterations = 10) {
        BenchmarkResult result = {};
        
        // Calculate memory usage
        MetalBatchPrefillWorkspace workspace = metal_batch_prefill_get_workspace(
            handle, config.num_tokens, config.head_dim, config.kv_head_dim,
            config.page_size, config.num_kv_pages);
        result.memory_usage_mb = workspace.total_size / (1024 * 1024);
        
        std::cout << "🔄 Benchmarking: " << config.description << std::endl;
        std::cout << "  📊 Parameters: " << config.num_tokens << " tokens, "
                  << config.head_size << " head_size, " << config.num_kv_pages << " pages" << std::endl;
        std::cout << "  💾 Memory: " << result.memory_usage_mb << " MB workspace" << std::endl;
        
        // Benchmark baseline kernel
        std::cout << "  📋 Testing baseline kernel..." << std::flush;
        result.baseline_time_ms = benchmark_kernel(config, KernelOptimizationLevel::BASELINE, iterations);
        result.baseline_success = (result.baseline_time_ms >= 0);
        
        if (result.baseline_success) {
            std::cout << " " << std::fixed << std::setprecision(3) << result.baseline_time_ms << "ms" << std::endl;
        } else {
            std::cout << " FAILED" << std::endl;
        }
        
        // Benchmark simdgroup optimized kernel
        std::cout << "  ⚡ Testing simdgroup kernel..." << std::flush;
        result.simdgroup_time_ms = benchmark_kernel(config, KernelOptimizationLevel::SIMDGROUP_OPT, iterations);
        result.simdgroup_success = (result.simdgroup_time_ms >= 0);
        
        if (result.simdgroup_success) {
            std::cout << " " << std::fixed << std::setprecision(3) << result.simdgroup_time_ms << "ms" << std::endl;
        } else {
            std::cout << " FAILED" << std::endl;
        }
        
        // Calculate speedup
        if (result.baseline_success && result.simdgroup_success && result.baseline_time_ms > 0) {
            result.speedup_factor = result.baseline_time_ms / result.simdgroup_time_ms;
            std::cout << "  🚀 Speedup: " << std::fixed << std::setprecision(2) << result.speedup_factor << "x";
            
            if (result.speedup_factor > 1.3) {
                std::cout << " ✅ GOOD";
            } else if (result.speedup_factor > 1.1) {
                std::cout << " ⚠️ MODEST";  
            } else if (result.speedup_factor < 0.9) {
                std::cout << " ❌ SLOWER";
            } else {
                std::cout << " ➖ NEUTRAL";
            }
            std::cout << std::endl;
        } else {
            result.speedup_factor = 0.0;
            std::cout << "  🚀 Speedup: N/A (one or both kernels failed)" << std::endl;
        }
        
        std::cout << std::endl;
        return result;
    }
};

int main() {
    std::cout << "🏁 Metal FlashAttention Priority 0 Performance Validation" << std::endl;
    std::cout << "==========================================================" << std::endl << std::endl;
    
    try {
        AttentionBenchmark benchmark;
        
        // Define benchmark configurations - focus on Priority 0 target scenarios
        std::vector<BenchmarkConfig> configs = {
            // Small sequences (Priority 0 optimization targets)
            {128, 2048, 2048, 64, 16, 8, 32, 32, "Small: 128 tokens, 64 head_size"},
            {256, 4096, 4096, 64, 16, 16, 64, 64, "Small: 256 tokens, 64 head_size"},
            {512, 4096, 4096, 128, 16, 32, 32, 32, "Medium: 512 tokens, 128 head_size"},
            
            // Edge of optimization benefit
            {512, 8192, 8192, 128, 16, 32, 64, 64, "Medium: 512 tokens, 128 head_size (large head_dim)"},
            
            // Large sequences (should favor baseline)
            {1024, 8192, 8192, 128, 16, 64, 64, 64, "Large: 1024 tokens, 128 head_size"},
            {2048, 16384, 16384, 256, 16, 128, 64, 64, "Large: 2048 tokens, 256 head_size"},
            
            // Memory stress tests
            {128, 16384, 16384, 256, 16, 8, 64, 64, "Memory stress: 128 tokens, 256 head_size"},
        };
        
        std::vector<BenchmarkResult> results;
        
        for (const auto& config : configs) {
            results.push_back(benchmark.run_benchmark(config));
        }
        
        // Summary report
        std::cout << "📊 PERFORMANCE VALIDATION SUMMARY" << std::endl;
        std::cout << "==================================" << std::endl;
        
        int successful_tests = 0;
        int improved_tests = 0;
        double total_speedup = 0.0;
        int speedup_count = 0;
        
        for (size_t i = 0; i < results.size(); i++) {
            const auto& result = results[i];
            const auto& config = configs[i];
            
            std::cout << "📝 " << config.description << std::endl;
            std::cout << "   Baseline: " << (result.baseline_success ? "✅" : "❌");
            std::cout << "  Simdgroup: " << (result.simdgroup_success ? "✅" : "❌");
            
            if (result.baseline_success && result.simdgroup_success) {
                successful_tests++;
                std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << result.speedup_factor << "x";
                
                if (result.speedup_factor > 1.1) {
                    improved_tests++;
                    std::cout << " 🚀";
                } else if (result.speedup_factor < 0.9) {
                    std::cout << " ⚠️";
                }
                
                total_speedup += result.speedup_factor;
                speedup_count++;
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "🎯 PRIORITY 0 VALIDATION RESULTS:" << std::endl;
        std::cout << "   ✅ Successful tests: " << successful_tests << "/" << configs.size() << std::endl;
        std::cout << "   🚀 Improved performance: " << improved_tests << "/" << successful_tests << std::endl;
        
        if (speedup_count > 0) {
            double avg_speedup = total_speedup / speedup_count;
            std::cout << "   📈 Average speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;
            
            if (avg_speedup >= 1.3 && improved_tests >= successful_tests / 2) {
                std::cout << "   🏆 PRIORITY 0 VALIDATION: PASSED ✅" << std::endl;
                std::cout << "   🎉 Expected 30-80% speedup achieved!" << std::endl;
            } else if (avg_speedup >= 1.1) {
                std::cout << "   ✅ PRIORITY 0 VALIDATION: PARTIAL SUCCESS" << std::endl;
                std::cout << "   📊 Some improvement observed, may need tuning" << std::endl;
            } else {
                std::cout << "   ⚠️ PRIORITY 0 VALIDATION: NEEDS INVESTIGATION" << std::endl;
                std::cout << "   🔍 Speedup below expectations, check implementation" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}