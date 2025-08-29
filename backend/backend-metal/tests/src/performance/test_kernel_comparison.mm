#include "metal_batch_prefill_attention_unified.hpp"
#include "metal_batch_prefill_attention.hpp" 
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
 * Kernel Comparison Test
 * 
 * Compare outputs between:
 * 1. Our working Step 1 single-thread kernel
 * 2. Original batch_prefill_attention kernel
 * 
 * Validates mathematical correctness and accuracy.
 */

using bfloat16_t = uint16_t;

// BFloat16 conversion utilities
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

// Simple test data generator
class ComparisonTestDataGenerator {
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
};

// Paged KV cache structure
struct ComparisonPagedKVData {
    std::vector<bfloat16_t> paged_k_cache;
    std::vector<bfloat16_t> paged_v_cache;
    std::vector<int32_t> kv_page_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;
};

ComparisonPagedKVData generate_comparison_kv_data(
    const std::vector<int>& sequence_lengths,
    int num_heads,
    int head_size,
    int page_size,
    uint64_t seed
) {
    ComparisonPagedKVData data;
    
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
    data.paged_k_cache = ComparisonTestDataGenerator::generate_bf16_data(cache_size, -1.0f, 1.0f, seed + 1);
    data.paged_v_cache = ComparisonTestDataGenerator::generate_bf16_data(cache_size, -1.0f, 1.0f, seed + 2);
    
    return data;
}

bool run_kernel_comparison_test() {
    std::cout << "\\n🔍 Kernel Comparison Test: Step 1 vs Original\\n";
    std::cout << "============================================\\n";
    
    // Test configuration (minimal for precise comparison)
    const int num_sequences = 1;
    const std::vector<int> sequence_lengths = {8};
    const int num_heads = 2;
    const int head_size = 16;
    const int page_size = 8;
    
    // Calculate dimensions
    int total_qo = 0;
    for (int seq_len : sequence_lengths) {
        total_qo += seq_len;
    }
    int head_dim = num_heads * head_size;
    float scale = 1.0f / sqrt(head_size);
    
    std::cout << "Configuration:\\n";
    std::cout << "  Sequences: " << num_sequences << "\\n";
    std::cout << "  Total queries: " << total_qo << "\\n";
    std::cout << "  Heads: " << num_heads << ", Head size: " << head_size << "\\n";
    std::cout << "  Page size: " << page_size << ", Scale: " << scale << "\\n\\n";
    
    // Generate test data (same seed for both kernels)
    auto q_input = ComparisonTestDataGenerator::generate_bf16_data(
        total_qo * head_dim, -1.0f, 1.0f, 12345);
        
    auto kv_data = generate_comparison_kv_data(
        sequence_lengths, num_heads, head_size, page_size, 54321);
    
    // Create qo_indptr
    std::vector<int32_t> qo_indptr;
    qo_indptr.push_back(0);
    int qo_offset = 0;
    for (int seq_len : sequence_lengths) {
        qo_offset += seq_len;
        qo_indptr.push_back(qo_offset);
    }
    
    // Prepare output buffers
    std::vector<bfloat16_t> step1_output(total_qo * head_dim, 0);
    std::vector<bfloat16_t> original_output(total_qo * head_dim, 0);
    std::vector<float> debug_output(1024, 0.0f);
    
    bool step1_success = false;
    bool original_success = false;
    
    // Initialize systems
    if (!metal::unified_attention::initialize()) {
        std::cerr << "Failed to initialize unified attention system\\n";
        return false;
    }
    
    // Test Step 1 kernel (our working implementation)
    try {
        std::cout << "Testing Step 1 kernel...\\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        metal::unified_attention::unified_batch_prefill_attention_auto(
            q_input.data(),
            kv_data.paged_k_cache.data(),
            kv_data.paged_v_cache.data(),
            qo_indptr.data(),
            kv_data.kv_page_indptr.data(),
            kv_data.kv_page_indices.data(),
            kv_data.kv_last_page_lens.data(),
            step1_output.data(),
            total_qo,
            num_sequences,
            head_dim,
            head_size,
            page_size,
            scale,
            debug_output.data()
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  ✅ Step 1 execution time: " << duration.count() << " μs\\n";
        step1_success = true;
        
    } catch (const std::exception& e) {
        std::cerr << "  ❌ Step 1 kernel failed: " << e.what() << "\\n";
    }
    
    // Test original kernel
    try {
        std::cout << "Testing original kernel...\\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int num_kv_pages = kv_data.kv_page_indptr.back();
        
        const int num_query_heads = num_heads;
        const int num_kv_heads = num_heads;  // Same as query for now (no MQA/GQA)
        const int kv_head_dim = num_kv_heads * head_size;
        
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input.data(),
            kv_data.paged_k_cache.data(),
            kv_data.paged_v_cache.data(),
            qo_indptr.data(),
            kv_data.kv_page_indptr.data(),
            kv_data.kv_page_indices.data(),
            kv_data.kv_last_page_lens.data(),
            original_output.data(),
            total_qo,
            head_dim,
            kv_head_dim,
            head_size,
            page_size,
            num_query_heads,
            num_kv_heads,
            scale,
            num_kv_pages
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  ✅ Original execution time: " << duration.count() << " μs\\n";
        original_success = true;
        
    } catch (const std::exception& e) {
        std::cerr << "  ❌ Original kernel failed: " << e.what() << "\\n";
    }
    
    // Compare outputs if both succeeded
    if (step1_success && original_success) {
        std::cout << "\\nComparing outputs...\\n";
        
        float max_diff = 0.0f;
        float avg_diff = 0.0f;
        int zero_count_step1 = 0, zero_count_original = 0;
        
        for (size_t i = 0; i < step1_output.size(); i++) {
            float val1 = bf16_to_float(step1_output[i]);
            float val2 = bf16_to_float(original_output[i]);
            
            float diff = std::abs(val1 - val2);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (val1 == 0.0f) zero_count_step1++;
            if (val2 == 0.0f) zero_count_original++;
        }
        
        avg_diff /= step1_output.size();
        
        std::cout << "  Max difference: " << max_diff << "\\n";
        std::cout << "  Average difference: " << avg_diff << "\\n";
        std::cout << "  Zero count - Step 1: " << zero_count_step1 
                  << ", Original: " << zero_count_original << "\\n";
        
        // Show sample values for detailed comparison
        std::cout << "\\nSample value comparison (first 8 elements):\\n";
        for (int i = 0; i < 8 && i < (int)step1_output.size(); i++) {
            float val1 = bf16_to_float(step1_output[i]);
            float val2 = bf16_to_float(original_output[i]);
            std::cout << "  [" << i << "] Step 1: " << val1 << ", Original: " << val2 
                      << ", Diff: " << std::abs(val1 - val2) << "\\n";
        }
        
        // Determine success criteria
        const float tolerance = 1e-3f; // 0.001 as specified in plan
        bool accuracy_ok = max_diff < tolerance;
        bool both_non_zero = (zero_count_step1 < (int)step1_output.size() / 2) && 
                            (zero_count_original < (int)original_output.size() / 2);
        
        std::cout << "\\n" << std::string(50, '=') << "\\n";
        if (accuracy_ok && both_non_zero) {
            std::cout << "✅ COMPARISON SUCCESS: Max diff " << max_diff 
                      << " < tolerance " << tolerance << "\\n";
            std::cout << "   Both kernels produce meaningful non-zero outputs\\n";
        } else {
            std::cout << "⚠️  COMPARISON ISSUES:\\n";
            if (!accuracy_ok) {
                std::cout << "   - Max diff " << max_diff << " exceeds tolerance " << tolerance << "\\n";
            }
            if (!both_non_zero) {
                std::cout << "   - One or both kernels producing mostly zeros\\n";
            }
        }
        
        metal::unified_attention::cleanup();
        return accuracy_ok && both_non_zero;
        
    } else {
        std::cout << "\\n❌ Cannot compare - one or both kernels failed\\n";
        metal::unified_attention::cleanup();
        return false;
    }
}

int main() {
    std::cout << "Kernel Comparison Test Suite\\n";
    std::cout << "===========================\\n";
    
    bool success = run_kernel_comparison_test();
    
    return success ? 0 : 1;
}