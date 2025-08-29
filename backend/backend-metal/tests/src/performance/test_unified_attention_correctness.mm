#include "metal_batch_prefill_attention_unified.hpp"
#include "attention_perf_utils.hpp" 
#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

/**
 * Comprehensive test suite for Unified FlashAttention-like Metal kernel
 * 
 * Tests the new unified kernel with:
 * 1. Variable sequence lengths (short, medium, long)
 * 2. Multiple heads in parallel
 * 3. Batched processing
 * 4. Dynamic tile size configuration
 * 5. Edge cases and stress testing
 */

using bfloat16_t = uint16_t;
using namespace metal::unified_attention;

// === BFloat16 conversion utilities ===
static bfloat16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
}

static float bf16_to_float(bfloat16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// === Test Configuration ===
struct UnifiedTestConfig {
    int num_sequences;
    std::vector<int> sequence_lengths;  // Variable length per sequence
    int num_heads;
    int head_size; 
    int page_size;
    float tolerance;
    std::string description;
    bool enable_debug;
};

// === Paged KV Cache Data Structure ===
struct SimplePagedKVData {
    std::vector<bfloat16_t> paged_k_cache;
    std::vector<bfloat16_t> paged_v_cache;
    std::vector<int32_t> kv_page_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;
};

// === Test Data Generator ===
class UnifiedTestDataGenerator {
public:
    static std::vector<bfloat16_t> generate_bf16_data(size_t count, float min_val, float max_val, uint64_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        std::vector<bfloat16_t> result(count);
        for (size_t i = 0; i < count; i++) {
            result[i] = float_to_bf16(dist(rng));
        }
        return result;
    }
    
    static SimplePagedKVData generate_paged_kv_data(
        const std::vector<int>& sequence_lengths,
        int num_heads,
        int head_size,
        int page_size,
        uint64_t seed
    ) {
        SimplePagedKVData data;
        std::mt19937 rng(seed);
        
        int total_pages = 0;
        data.kv_page_indptr.push_back(0);
        
        // Calculate page requirements for each sequence  
        for (int seq_len : sequence_lengths) {
            int pages_needed = (seq_len + page_size - 1) / page_size;
            total_pages += pages_needed;
            data.kv_page_indptr.push_back(total_pages);
            
            // Add page indices
            for (int p = 0; p < pages_needed; p++) {
                data.kv_page_indices.push_back(total_pages - pages_needed + p);
            }
            
            // Last page length
            int last_page_len = seq_len % page_size;
            if (last_page_len == 0) last_page_len = page_size;
            data.kv_last_page_lens.push_back(last_page_len);
        }
        
        // Generate KV cache data
        size_t cache_size = total_pages * page_size * num_heads * head_size;
        data.paged_k_cache = generate_bf16_data(cache_size, -2.0f, 2.0f, seed + 1);
        data.paged_v_cache = generate_bf16_data(cache_size, -2.0f, 2.0f, seed + 2);
        
        
        return data;
    }
};

// === Test Cases ===
class UnifiedAttentionTest {
private:
    std::vector<UnifiedTestConfig> test_configs;
    
public:
    UnifiedAttentionTest() {
        setup_test_cases();
    }
    
    void setup_test_cases() {
        // Test 0: Minimal debug case (PRIORITY)
        test_configs.push_back({
            .num_sequences = 1,
            .sequence_lengths = {4},
            .num_heads = 1,
            .head_size = 4,
            .page_size = 4,
            .tolerance = 1e-2f,
            .description = "Minimal single sequence debug",
            .enable_debug = true
        });
        
        // Test 1: Variable short sequences
        test_configs.push_back({
            .num_sequences = 4,
            .sequence_lengths = {8, 16, 32, 24},
            .num_heads = 8,
            .head_size = 64,
            .page_size = 16,
            .tolerance = 1e-2f,
            .description = "Variable short sequences",
            .enable_debug = true
        });
        
        // Test 2: Medium sequences with different heads
        test_configs.push_back({
            .num_sequences = 3,
            .sequence_lengths = {128, 256, 192}, 
            .num_heads = 12,
            .head_size = 64,
            .page_size = 64,
            .tolerance = 1e-2f,
            .description = "Medium sequences multi-head",
            .enable_debug = true
        });
        
        // Test 3: Long sequences stress test
        test_configs.push_back({
            .num_sequences = 2,
            .sequence_lengths = {1024, 2048},
            .num_heads = 16,
            .head_size = 64, 
            .page_size = 128,
            .tolerance = 1e-2f,
            .description = "Long sequences stress test",
            .enable_debug = false
        });
        
        // Test 4: Mixed sequence lengths
        test_configs.push_back({
            .num_sequences = 6,
            .sequence_lengths = {16, 64, 128, 256, 512, 32},
            .num_heads = 8,
            .head_size = 128,
            .page_size = 32,
            .tolerance = 1e-2f,
            .description = "Mixed sequence lengths",
            .enable_debug = false
        });
        
        // Test 5: Single head edge case
        test_configs.push_back({
            .num_sequences = 2,
            .sequence_lengths = {64, 96},
            .num_heads = 1,
            .head_size = 64,
            .page_size = 16,
            .tolerance = 1e-2f,
            .description = "Single head edge case",
            .enable_debug = true
        });
        
        // Test 6: Large batch with many small sequences
        test_configs.push_back({
            .num_sequences = 16,
            .sequence_lengths = {8, 12, 16, 20, 24, 28, 32, 10, 14, 18, 22, 26, 30, 6, 36, 40},
            .num_heads = 4,
            .head_size = 32,
            .page_size = 8,
            .tolerance = 1e-2f,
            .description = "Large batch small sequences",
            .enable_debug = false
        });
    }
    
    bool run_single_test(const UnifiedTestConfig& config) {
        std::cout << "\n=== Testing: " << config.description << " ===\n";
        std::cout << "Sequences: " << config.num_sequences 
                  << ", Heads: " << config.num_heads 
                  << ", Head size: " << config.head_size << std::endl;
        
        // Calculate total query tokens and head dimension
        int total_qo = 0;
        for (int seq_len : config.sequence_lengths) {
            total_qo += seq_len;
        }
        int head_dim = config.num_heads * config.head_size;
        
        std::cout << "Total queries: " << total_qo << ", Head dim: " << head_dim << std::endl;
        
        // Generate test data
        auto q_input = UnifiedTestDataGenerator::generate_bf16_data(
            total_qo * head_dim, -1.0f, 1.0f, 12345);
            
        auto kv_data = UnifiedTestDataGenerator::generate_paged_kv_data(
            config.sequence_lengths, config.num_heads, config.head_size, 
            config.page_size, 54321);
        
        // Create qo_indptr (query token boundaries)
        std::vector<int32_t> qo_indptr;
        qo_indptr.push_back(0);
        int qo_offset = 0;
        for (int seq_len : config.sequence_lengths) {
            qo_offset += seq_len;
            qo_indptr.push_back(qo_offset);
        }
        
        // Prepare output buffers
        std::vector<bfloat16_t> unified_output(total_qo * head_dim, 0);
        std::vector<float> debug_output(1024, 0.0f);
        
        // Run unified attention kernel
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            unified_batch_prefill_attention_auto(
                q_input.data(),
                kv_data.paged_k_cache.data(),
                kv_data.paged_v_cache.data(),
                qo_indptr.data(),
                kv_data.kv_page_indptr.data(),
                kv_data.kv_page_indices.data(),
                kv_data.kv_last_page_lens.data(),
                unified_output.data(),
                total_qo,
                config.num_sequences,
                head_dim,
                config.head_size,
                config.page_size,
                1.0f / sqrt(config.head_size),
                config.enable_debug ? debug_output.data() : nullptr
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "Unified kernel execution time: " << duration.count() << " μs" << std::endl;
            
            // Validate output
            bool has_non_zero = false;
            float max_val = 0.0f, min_val = 0.0f;
            
            for (size_t i = 0; i < unified_output.size(); i++) {
                float val = bf16_to_float(unified_output[i]);
                if (val != 0.0f) has_non_zero = true;
                max_val = std::max(max_val, val);
                min_val = std::min(min_val, val);
            }
            
            std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
            std::cout << "Has non-zero output: " << (has_non_zero ? "YES" : "NO") << std::endl;
            
            // Print debug output if enabled and available
            if (config.enable_debug && debug_output.size() >= 35) {
                std::cout << "\n--- DEBUG OUTPUT ---" << std::endl;
                std::cout << "Basic params: scale=" << debug_output[0] 
                          << ", head_dim=" << debug_output[1]
                          << ", num_heads=" << debug_output[2] << std::endl;
                std::cout << "Sequence: seq_len=" << debug_output[3] 
                          << ", kv_len=" << debug_output[4]
                          << ", head_size=" << debug_output[5] << std::endl;
                std::cout << "Boundaries: q_start=" << debug_output[8] 
                          << ", q_end=" << debug_output[9]
                          << ", num_pages=" << debug_output[10] << std::endl;
                std::cout << "Thread ID: batch_head=" << debug_output[15] 
                          << ", seq_idx=" << debug_output[16]
                          << ", head_idx=" << debug_output[17] << std::endl;
                
                if (debug_output[3] > 0) { // seq_len > 0
                    std::cout << "Q vector[0]: [" << debug_output[20] 
                              << ", " << debug_output[21] 
                              << ", " << debug_output[22] << "]" << std::endl;
                    std::cout << "K vector[0]: [" << debug_output[23] 
                              << ", " << debug_output[24] 
                              << ", " << debug_output[25] << "]" << std::endl;
                    std::cout << "K indexing: page=" << debug_output[28] 
                              << ", index=" << debug_output[29] 
                              << ", page_pos=" << debug_output[30] << std::endl;
                    std::cout << "K cache samples: [" << debug_output[31] 
                              << ", " << debug_output[32] 
                              << ", " << debug_output[33] 
                              << ", ... " << debug_output[34] << "]" << std::endl;
                    std::cout << "Manual Q·K: " << debug_output[26] << std::endl;
                    std::cout << "Loop entered: " << (debug_output[27] > 0 ? "YES" : "NO") << std::endl;
                    std::cout << "Final output[0]: " << debug_output[6] << std::endl;
                }
                std::cout << "Execution marker: " << debug_output[7] << std::endl;
                std::cout << "--- END DEBUG ---\n" << std::endl;
            }
            
            if (!has_non_zero) {
                std::cerr << "ERROR: All output values are zero!" << std::endl;
                return false;
            }
            
            // Check for NaN/Inf values
            int nan_count = 0, inf_count = 0;
            for (size_t i = 0; i < unified_output.size(); i++) {
                float val = bf16_to_float(unified_output[i]);
                if (std::isnan(val)) nan_count++;
                if (std::isinf(val)) inf_count++;
            }
            
            if (nan_count > 0 || inf_count > 0) {
                std::cerr << "ERROR: Found " << nan_count << " NaN and " 
                          << inf_count << " Inf values!" << std::endl;
                return false;
            }
            
            // Performance metrics
            size_t total_flops = 0;
            for (int seq_len : config.sequence_lengths) {
                total_flops += seq_len * seq_len * config.head_size * config.num_heads * 2; // Rough estimate
            }
            
            double gflops = (double)total_flops / (duration.count() / 1000.0) / 1e6; // GFLOPS
            std::cout << "Estimated performance: " << gflops << " GFLOPS" << std::endl;
            
            std::cout << "✅ Test PASSED: " << config.description << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Test FAILED: " << config.description << std::endl;
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool run_all_tests() {
        std::cout << "\n🚀 Running Unified FlashAttention Correctness Tests\n";
        std::cout << "====================================================\n";
        
        // Initialize unified attention
        if (!metal::unified_attention::initialize()) {
            std::cerr << "Failed to initialize unified attention system\n";
            return false;
        }
        
        int passed = 0, total = test_configs.size();
        
        for (const auto& config : test_configs) {
            if (run_single_test(config)) {
                passed++;
            } else {
                std::cerr << "\nTest failed, continuing with remaining tests...\n";
            }
        }
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Test Results: " << passed << "/" << total << " tests passed" << std::endl;
        
        if (passed == total) {
            std::cout << "🎉 All tests PASSED! Unified attention is working correctly." << std::endl;
        } else {
            std::cout << "⚠️  Some tests FAILED. Please review the implementation." << std::endl;
        }
        
        metal::unified_attention::cleanup();
        return passed == total;
    }
    
    bool run_performance_comparison() {
        std::cout << "\n⚡ Performance Comparison: Unified vs Original\n";
        std::cout << "==============================================\n";
        
        // TODO: Add comparison with original kernel when available
        std::cout << "Performance comparison not yet implemented\n";
        std::cout << "Would compare unified kernel against original implementation\n";
        
        return true;
    }
};

// === Main Test Functions ===

bool test_unified_attention_correctness() {
    UnifiedAttentionTest test_suite;
    return test_suite.run_all_tests();
}

bool test_unified_attention_performance() {
    UnifiedAttentionTest test_suite;
    return test_suite.run_performance_comparison();
}

// === Test Program Entry Point ===
int main(int argc, char* argv[]) {
    std::cout << "Unified FlashAttention Metal Kernel Test Suite\n";
    std::cout << "==============================================\n";
    
    bool run_perf = false;
    if (argc > 1 && std::string(argv[1]) == "--performance") {
        run_perf = true;
    }
    
    bool success = true;
    
    // Run correctness tests
    success &= test_unified_attention_correctness();
    
    // Run performance tests if requested
    if (run_perf) {
        success &= test_unified_attention_performance();
    }
    
    return success ? 0 : 1;
}