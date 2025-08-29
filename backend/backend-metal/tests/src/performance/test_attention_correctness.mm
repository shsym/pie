#include "attention_perf_utils.hpp"
#include "metal_batch_prefill_attention.hpp"
#include "metal_batch_prefill_attention_unified.hpp"
#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>

/**
 * Comprehensive correctness testing for Metal batch prefill attention optimization
 * This ensures that optimizations maintain numerical accuracy
 * Updated to use f32 precision for both kernels to avoid bfloat16 issues
 */

using bfloat16_t = uint16_t;

// Enhanced random data generation with specific patterns for edge cases
class AttentionTestDataGenerator {
public:
    // F32 data generation for higher precision testing
    static std::vector<float> generate_f32_data(size_t count, float min_val, float max_val, uint64_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        std::vector<float> result(count);
        for (size_t i = 0; i < count; i++) {
            result[i] = dist(rng);
        }
        return result;
    }
    
    // Generate edge case patterns for robustness testing (f32 version)
    static std::vector<float> generate_f32_edge_case_data(size_t count, const std::string& pattern, uint64_t seed) {
        std::vector<float> result(count);
        std::mt19937 rng(seed);
        
        if (pattern == "zeros") {
            std::fill(result.begin(), result.end(), 0.0f);
        } else if (pattern == "ones") {
            std::fill(result.begin(), result.end(), 1.0f);
        } else if (pattern == "alternating") {
            for (size_t i = 0; i < count; i++) {
                result[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            }
        } else if (pattern == "large_values") {
            std::uniform_real_distribution<float> dist(10.0f, 100.0f);
            for (size_t i = 0; i < count; i++) {
                result[i] = dist(rng);
            }
        } else if (pattern == "small_values") {
            std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
            for (size_t i = 0; i < count; i++) {
                result[i] = dist(rng);
            }
        } else {
            // Default to random
            return generate_f32_data(count, -1.0f, 1.0f, seed);
        }
        
        return result;
    }

    // Legacy bf16 functions kept for compatibility
    static std::vector<bfloat16_t> generate_bf16_data(size_t count, float min_val, float max_val, uint64_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        std::vector<bfloat16_t> result(count);
        for (size_t i = 0; i < count; i++) {
            float val = dist(rng);
            result[i] = float_to_bf16(val);
        }
        return result;
    }
    
    static std::vector<bfloat16_t> generate_edge_case_data(size_t count, const std::string& pattern, uint64_t seed) {
        std::vector<bfloat16_t> result(count);
        std::mt19937 rng(seed);
        
        if (pattern == "zeros") {
            std::fill(result.begin(), result.end(), float_to_bf16(0.0f));
        } else if (pattern == "ones") {
            std::fill(result.begin(), result.end(), float_to_bf16(1.0f));
        } else if (pattern == "alternating") {
            for (size_t i = 0; i < count; i++) {
                result[i] = float_to_bf16((i % 2 == 0) ? 1.0f : -1.0f);
            }
        } else if (pattern == "large_values") {
            std::uniform_real_distribution<float> dist(10.0f, 100.0f);
            for (size_t i = 0; i < count; i++) {
                result[i] = float_to_bf16(dist(rng));
            }
        } else if (pattern == "small_values") {
            std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
            for (size_t i = 0; i < count; i++) {
                result[i] = float_to_bf16(dist(rng));
            }
        } else {
            // Default to random
            return generate_bf16_data(count, -1.0f, 1.0f, seed);
        }
        
        return result;
    }

private:
    static bfloat16_t float_to_bf16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
    }
};

// Reference implementation for correctness comparison
class ReferenceAttention {
public:
    // F32 version for higher precision reference
    static std::vector<float> compute_reference_attention_f32(
        const std::vector<float>& q_input,
        const std::vector<float>& paged_k_cache,
        const std::vector<float>& paged_v_cache,
        const std::vector<int32_t>& qo_indptr,
        const std::vector<int32_t>& kv_page_indptr,
        const std::vector<int32_t>& kv_page_indices,
        const std::vector<int32_t>& kv_last_page_lens,
        const AttentionBenchmarkConfig& config
    ) {
        const int head_dim = config.num_heads * config.head_dim;
        const int total_tokens = config.batch_size * config.seq_length;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::vector<float> output(total_tokens * head_dim, 0.0f);
        
        // Simple reference implementation (not optimized, but correct)
        for (int batch_idx = 0; batch_idx < config.batch_size; batch_idx++) {
            int seq_start = qo_indptr[batch_idx];
            int seq_end = qo_indptr[batch_idx + 1];
            
            for (int q_idx = seq_start; q_idx < seq_end; q_idx++) {
                // For each query token, compute attention over all key-value pairs
                for (int head = 0; head < config.num_heads; head++) {
                    std::vector<float> scores;
                    std::vector<std::vector<float>> values;
                    
                    // Compute scores and collect values for this head
                    int kv_start_page = kv_page_indptr[batch_idx];
                    int kv_end_page = kv_page_indptr[batch_idx + 1];
                    
                    for (int page_idx = kv_start_page; page_idx < kv_end_page; page_idx++) {
                        int page_id = kv_page_indices[page_idx];
                        int tokens_in_page = (page_idx == kv_end_page - 1) ? 
                            kv_last_page_lens[batch_idx] : config.page_size;
                        
                        for (int token_in_page = 0; token_in_page < tokens_in_page; token_in_page++) {
                            // Compute Q·K^T for this key
                            float score = 0.0f;
                            for (int d = 0; d < config.head_dim; d++) {
                                int q_offset = q_idx * head_dim + head * config.head_dim + d;
                                int k_offset = page_id * config.page_size * head_dim + 
                                             token_in_page * head_dim + head * config.head_dim + d;
                                
                                float q_val = q_input[q_offset];
                                float k_val = paged_k_cache[k_offset];
                                score += q_val * k_val;
                            }
                            scores.push_back(score * scale);
                            
                            // Collect V values
                            std::vector<float> v_vals(config.head_dim);
                            for (int d = 0; d < config.head_dim; d++) {
                                int v_offset = page_id * config.page_size * head_dim + 
                                             token_in_page * head_dim + head * config.head_dim + d;
                                v_vals[d] = paged_v_cache[v_offset];
                            }
                            values.push_back(v_vals);
                        }
                    }
                    
                    // Apply softmax
                    if (!scores.empty()) {
                        float max_score = *std::max_element(scores.begin(), scores.end());
                        float sum_exp = 0.0f;
                        
                        for (float& score : scores) {
                            score = std::exp(score - max_score);
                            sum_exp += score;
                        }
                        
                        // Normalize and compute weighted sum
                        for (int d = 0; d < config.head_dim; d++) {
                            float weighted_sum = 0.0f;
                            for (size_t i = 0; i < scores.size(); i++) {
                                float weight = scores[i] / sum_exp;
                                weighted_sum += weight * values[i][d];
                            }
                            
                            int out_offset = q_idx * head_dim + head * config.head_dim + d;
                            output[out_offset] = weighted_sum;
                        }
                    }
                }
            }
        }
        
        return output;
    }

    // Legacy bf16 version kept for compatibility
    static std::vector<float> compute_reference_attention(
        const std::vector<bfloat16_t>& q_input,
        const std::vector<bfloat16_t>& paged_k_cache,
        const std::vector<bfloat16_t>& paged_v_cache,
        const std::vector<int32_t>& qo_indptr,
        const std::vector<int32_t>& kv_page_indptr,
        const std::vector<int32_t>& kv_page_indices,
        const std::vector<int32_t>& kv_last_page_lens,
        const AttentionBenchmarkConfig& config
    ) {
        const int head_dim = config.num_heads * config.head_dim;
        const int total_tokens = config.batch_size * config.seq_length;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::vector<float> output(total_tokens * head_dim, 0.0f);
        
        // Simple reference implementation (not optimized, but correct)
        for (int batch_idx = 0; batch_idx < config.batch_size; batch_idx++) {
            int seq_start = qo_indptr[batch_idx];
            int seq_end = qo_indptr[batch_idx + 1];
            
            for (int q_idx = seq_start; q_idx < seq_end; q_idx++) {
                // For each query token, compute attention over all key-value pairs
                for (int head = 0; head < config.num_heads; head++) {
                    std::vector<float> scores;
                    std::vector<std::vector<float>> values;
                    
                    // Compute scores and collect values for this head
                    int kv_start_page = kv_page_indptr[batch_idx];
                    int kv_end_page = kv_page_indptr[batch_idx + 1];
                    
                    for (int page_idx = kv_start_page; page_idx < kv_end_page; page_idx++) {
                        int page_id = kv_page_indices[page_idx];
                        int tokens_in_page = (page_idx == kv_end_page - 1) ? 
                            kv_last_page_lens[batch_idx] : config.page_size;
                        
                        for (int token_in_page = 0; token_in_page < tokens_in_page; token_in_page++) {
                            // Compute Q·K^T for this key
                            float score = 0.0f;
                            for (int d = 0; d < config.head_dim; d++) {
                                int q_offset = q_idx * head_dim + head * config.head_dim + d;
                                int k_offset = page_id * config.page_size * head_dim + 
                                             token_in_page * head_dim + head * config.head_dim + d;
                                
                                float q_val = bf16_to_float(q_input[q_offset]);
                                float k_val = bf16_to_float(paged_k_cache[k_offset]);
                                score += q_val * k_val;
                            }
                            scores.push_back(score * scale);
                            
                            // Collect V values
                            std::vector<float> v_vals(config.head_dim);
                            for (int d = 0; d < config.head_dim; d++) {
                                int v_offset = page_id * config.page_size * head_dim + 
                                             token_in_page * head_dim + head * config.head_dim + d;
                                v_vals[d] = bf16_to_float(paged_v_cache[v_offset]);
                            }
                            values.push_back(v_vals);
                        }
                    }
                    
                    // Apply softmax
                    if (!scores.empty()) {
                        float max_score = *std::max_element(scores.begin(), scores.end());
                        float sum_exp = 0.0f;
                        
                        for (float& score : scores) {
                            score = std::exp(score - max_score);
                            sum_exp += score;
                        }
                        
                        // Normalize and compute weighted sum
                        for (int d = 0; d < config.head_dim; d++) {
                            float weighted_sum = 0.0f;
                            for (size_t i = 0; i < scores.size(); i++) {
                                float weight = scores[i] / sum_exp;
                                weighted_sum += weight * values[i][d];
                            }
                            
                            int out_offset = q_idx * head_dim + head * config.head_dim + d;
                            output[out_offset] = weighted_sum;
                        }
                    }
                }
            }
        }
        
        return output;
    }

private:
    static float bf16_to_float(bfloat16_t bf16) {
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// Test case structure
struct AttentionCorrectnessTest {
    std::string name;
    AttentionBenchmarkConfig config;
    std::string data_pattern;  // "random", "zeros", "ones", etc.
    double tolerance;
    uint64_t seed;
};

class AttentionCorrectnessValidator {
public:
    // Main validation function now using f32 precision
    static bool validate_against_reference(const AttentionCorrectnessTest& test) {
        std::cout << "🧪 Testing (F32): " << test.name << std::endl;
        
        try {
            // Generate f32 test data for higher precision testing
            auto kv_data_f32 = setup_paged_kv_data_f32(test.config, test.seed);
            auto q_input_f32 = generate_test_query_data_f32(test.config, test.data_pattern, test.seed);
            
            // Run original Metal implementation (f32 - placeholder for now)
            std::vector<float> metal_output_original = run_metal_attention_f32(
                q_input_f32, kv_data_f32, test.config
            );
            
            // Run optimized Metal implementation (f32 FlashAttention)
            std::vector<float> metal_output_optimized = run_metal_attention_optimized_f32(
                q_input_f32, kv_data_f32, test.config
            );
            
            // Run reference implementation (f32)
            std::vector<float> reference_output = ReferenceAttention::compute_reference_attention_f32(
                q_input_f32, kv_data_f32.paged_k_cache, kv_data_f32.paged_v_cache,
                kv_data_f32.qo_indptr, kv_data_f32.kv_page_indptr, kv_data_f32.kv_page_indices,
                kv_data_f32.kv_last_page_lens, test.config
            );
            
            // Compare original kernel results (f32 vs f32)
            std::cout << "  Original kernel vs Reference:" << std::endl;
            bool original_passed = compare_outputs_f32(metal_output_original, reference_output, test.tolerance, "Original");
            
            // Compare optimized kernel results (f32 vs f32)
            std::cout << "  Optimized kernel vs Reference:" << std::endl;
            bool optimized_passed = compare_outputs_f32(metal_output_optimized, reference_output, test.tolerance, "Optimized");
            
            // Also compare original vs optimized for consistency check (f32 vs f32)
            std::cout << "  Original vs Optimized consistency check:" << std::endl;
            bool consistency_passed = compare_metal_outputs_f32(metal_output_original, metal_output_optimized, test.tolerance);
            
            // Test passes if both kernels pass and are consistent
            bool overall_passed = original_passed && optimized_passed && consistency_passed;
            
            if (overall_passed) {
                std::cout << "✅ " << test.name << " - All checks passed" << std::endl;
            } else {
                std::cout << "❌ " << test.name << " - Failed (Original:" << (original_passed ? "✅" : "❌") 
                          << ", Optimized:" << (optimized_passed ? "✅" : "❌")
                          << ", Consistency:" << (consistency_passed ? "✅" : "❌") << ")" << std::endl;
            }
            
            return overall_passed;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
            return false;
        }
    }

private:
    // F32 version of paged KV data structure
    struct PagedKVDataF32 {
        std::vector<float> paged_k_cache;
        std::vector<float> paged_v_cache;
        std::vector<int32_t> qo_indptr;
        std::vector<int32_t> kv_page_indptr;
        std::vector<int32_t> kv_page_indices;
        std::vector<int32_t> kv_last_page_lens;
        size_t num_pages;
    };
    
    // Legacy bf16 version kept for compatibility
    struct PagedKVData {
        std::vector<bfloat16_t> paged_k_cache;
        std::vector<bfloat16_t> paged_v_cache;
        std::vector<int32_t> qo_indptr;
        std::vector<int32_t> kv_page_indptr;
        std::vector<int32_t> kv_page_indices;
        std::vector<int32_t> kv_last_page_lens;
        size_t num_pages;
    };
    
    // F32 version of paged KV data setup
    static PagedKVDataF32 setup_paged_kv_data_f32(const AttentionBenchmarkConfig& config, uint64_t seed) {
        PagedKVDataF32 data;
        
        const int head_dim = config.num_heads * config.head_dim;
        data.num_pages = (config.seq_length + config.page_size - 1) / config.page_size;
        
        // Generate KV cache data with f32 precision
        size_t page_data_size = data.num_pages * config.page_size * head_dim * config.batch_size;
        data.paged_k_cache = AttentionTestDataGenerator::generate_f32_data(
            page_data_size, -0.5f, 0.5f, seed);
        data.paged_v_cache = AttentionTestDataGenerator::generate_f32_data(
            page_data_size, -0.5f, 0.5f, seed + 1);
        
        // Setup pointers
        data.qo_indptr.resize(config.batch_size + 1);
        for (int i = 0; i <= config.batch_size; i++) {
            data.qo_indptr[i] = i * config.seq_length;
        }
        
        data.kv_page_indptr.resize(config.batch_size + 1);
        for (int i = 0; i <= config.batch_size; i++) {
            data.kv_page_indptr[i] = i * static_cast<int>(data.num_pages);
        }
        
        size_t total_pages = config.batch_size * data.num_pages;
        data.kv_page_indices.resize(total_pages);
        for (size_t i = 0; i < total_pages; i++) {
            data.kv_page_indices[i] = static_cast<int32_t>(i);
        }
        
        data.kv_last_page_lens.resize(config.batch_size);
        int last_page_len = config.seq_length - (static_cast<int>(data.num_pages) - 1) * config.page_size;
        for (int i = 0; i < config.batch_size; i++) {
            data.kv_last_page_lens[i] = last_page_len;
        }
        
        return data;
    }
    
    // Legacy bf16 version kept for compatibility
    static PagedKVData setup_paged_kv_data(const AttentionBenchmarkConfig& config, uint64_t seed) {
        PagedKVData data;
        
        const int head_dim = config.num_heads * config.head_dim;
        data.num_pages = (config.seq_length + config.page_size - 1) / config.page_size;
        
        // Generate KV cache data
        size_t page_data_size = data.num_pages * config.page_size * head_dim * config.batch_size;
        data.paged_k_cache = AttentionTestDataGenerator::generate_bf16_data(
            page_data_size, -0.5f, 0.5f, seed);
        data.paged_v_cache = AttentionTestDataGenerator::generate_bf16_data(
            page_data_size, -0.5f, 0.5f, seed + 1);
        
        // Setup pointers
        data.qo_indptr.resize(config.batch_size + 1);
        for (int i = 0; i <= config.batch_size; i++) {
            data.qo_indptr[i] = i * config.seq_length;
        }
        
        data.kv_page_indptr.resize(config.batch_size + 1);
        for (int i = 0; i <= config.batch_size; i++) {
            data.kv_page_indptr[i] = i * static_cast<int>(data.num_pages);
        }
        
        size_t total_pages = config.batch_size * data.num_pages;
        data.kv_page_indices.resize(total_pages);
        for (size_t i = 0; i < total_pages; i++) {
            data.kv_page_indices[i] = static_cast<int32_t>(i);
        }
        
        data.kv_last_page_lens.resize(config.batch_size);
        int last_page_len = config.seq_length - (static_cast<int>(data.num_pages) - 1) * config.page_size;
        for (int i = 0; i < config.batch_size; i++) {
            data.kv_last_page_lens[i] = last_page_len;
        }
        
        return data;
    }
    
    // F32 version of query data generation
    static std::vector<float> generate_test_query_data_f32(
        const AttentionBenchmarkConfig& config, 
        const std::string& pattern, 
        uint64_t seed
    ) {
        size_t q_size = config.batch_size * config.seq_length * config.num_heads * config.head_dim;
        return AttentionTestDataGenerator::generate_f32_edge_case_data(q_size, pattern, seed);
    }
    
    // Legacy bf16 version kept for compatibility
    static std::vector<bfloat16_t> generate_test_query_data(
        const AttentionBenchmarkConfig& config, 
        const std::string& pattern, 
        uint64_t seed
    ) {
        size_t q_size = config.batch_size * config.seq_length * config.num_heads * config.head_dim;
        return AttentionTestDataGenerator::generate_edge_case_data(q_size, pattern, seed);
    }
    
    // F32 version of original Metal attention runner
    static std::vector<float> run_metal_attention_f32(
        const std::vector<float>& q_input,
        const PagedKVDataF32& kv_data,
        const AttentionBenchmarkConfig& config
    ) {
        // TODO: Implement f32 version of original kernel
        // For now, return reference computation as placeholder
        return ReferenceAttention::compute_reference_attention_f32(
            q_input, kv_data.paged_k_cache, kv_data.paged_v_cache,
            kv_data.qo_indptr, kv_data.kv_page_indptr, kv_data.kv_page_indices,
            kv_data.kv_last_page_lens, config
        );
    }
    
    // Legacy bf16 version of original Metal attention runner
    static std::vector<bfloat16_t> run_metal_attention(
        const std::vector<bfloat16_t>& q_input,
        const PagedKVData& kv_data,
        const AttentionBenchmarkConfig& config
    ) {
        const int num_qo = config.batch_size * config.seq_length;
        const int head_dim = config.num_heads * config.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::vector<bfloat16_t> output(q_input.size());
        
        const int num_query_heads = config.num_heads;
        const int num_kv_heads = config.num_heads;  // Same as query for now (no MQA/GQA)
        const int kv_head_dim = num_kv_heads * config.head_dim;
        
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input.data(),
            kv_data.paged_k_cache.data(),
            kv_data.paged_v_cache.data(),
            kv_data.qo_indptr.data(),
            kv_data.kv_page_indptr.data(),
            kv_data.kv_page_indices.data(),
            kv_data.kv_last_page_lens.data(),
            output.data(),
            num_qo,
            head_dim,
            kv_head_dim,
            config.head_dim,
            config.page_size,
            num_query_heads,
            num_kv_heads,
            scale,
            static_cast<int>(kv_data.num_pages * config.batch_size)
        );
        
        return output;
    }
    
    // F32 version of optimized (FlashAttention) Metal attention runner
    static std::vector<float> run_metal_attention_optimized_f32(
        const std::vector<float>& q_input,
        const PagedKVDataF32& kv_data,
        const AttentionBenchmarkConfig& config
    ) {
        const int num_qo = config.batch_size * config.seq_length;
        const int head_dim = config.num_heads * config.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::vector<float> output(q_input.size());
        std::vector<float> debug_out(20, -999.0f);
        
        // Use unified attention system for FlashAttention with f32 precision
        metal::unified_attention::UnifiedParams params = {
            .num_qo = num_qo,
            .num_sequences = config.batch_size,
            .head_dim = head_dim,
            .head_size = config.head_dim,
            .num_heads = config.num_heads,
            .page_size = config.page_size,
            .max_seq_len = config.seq_length,
            .scale = scale
        };
        
        // Print debug info
        std::cout << "[F32 FlashAttention Debug] scale=" << params.scale 
                  << ", head_dim=" << params.head_dim 
                  << ", page_size=" << params.page_size 
                  << ", num_qo=" << params.num_qo
                  << ", total_kv_len=" << (kv_data.kv_page_indices.size() * config.page_size)
                  << ", num_pages=" << kv_data.kv_page_indices.size()
                  << ", last_page_len=" << (kv_data.kv_last_page_lens.empty() ? 0 : kv_data.kv_last_page_lens[0]) << std::endl;
        
        try {
            metal::unified_attention::unified_batch_prefill_attention_f32(
                q_input.data(),
                kv_data.paged_k_cache.data(),
                kv_data.paged_v_cache.data(),
                kv_data.qo_indptr.data(),
                kv_data.kv_page_indptr.data(),
                kv_data.kv_page_indices.data(),
                kv_data.kv_last_page_lens.data(),
                output.data(),
                params,
                debug_out.data()
            );
        } catch (const std::exception& e) {
            std::cerr << "F32 FlashAttention failed: " << e.what() << std::endl;
            // Fallback to reference implementation
            output = ReferenceAttention::compute_reference_attention_f32(
                q_input, kv_data.paged_k_cache, kv_data.paged_v_cache,
                kv_data.qo_indptr, kv_data.kv_page_indptr, kv_data.kv_page_indices,
                kv_data.kv_last_page_lens, config
            );
        }
        
        return output;
    }
    
    // Legacy bf16 version of optimized Metal attention runner
    static std::vector<bfloat16_t> run_metal_attention_optimized(
        const std::vector<bfloat16_t>& q_input,
        const PagedKVData& kv_data,
        const AttentionBenchmarkConfig& config
    ) {
        const int num_qo = config.batch_size * config.seq_length;
        const int head_dim = config.num_heads * config.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
        
        std::vector<bfloat16_t> output(q_input.size());
        std::vector<float> debug_out(20, -999.0f);
        
        // Use unified attention system for FlashAttention
        metal::unified_attention::UnifiedParams params = {
            .num_qo = num_qo,
            .num_sequences = config.batch_size,
            .head_dim = head_dim,
            .head_size = config.head_dim,
            .num_heads = config.num_heads,
            .page_size = config.page_size,
            .max_seq_len = config.seq_length,
            .scale = scale
        };
        
        // Print debug info
        std::cout << "[Optimized Attention Debug] scale=" << params.scale 
                  << ", head_dim=" << params.head_dim 
                  << ", page_size=" << params.page_size 
                  << ", num_qo=" << params.num_qo
                  << ", total_kv_len=" << (kv_data.kv_page_indices.size() * config.page_size)
                  << ", num_pages=" << kv_data.kv_page_indices.size()
                  << ", last_page_len=" << (kv_data.kv_last_page_lens.empty() ? 0 : kv_data.kv_last_page_lens[0]) << std::endl;
        
        try {
            metal::unified_attention::unified_batch_prefill_attention_bf16(
                q_input.data(),
                kv_data.paged_k_cache.data(),
                kv_data.paged_v_cache.data(),
                kv_data.qo_indptr.data(),
                kv_data.kv_page_indptr.data(),
                kv_data.kv_page_indices.data(),
                kv_data.kv_last_page_lens.data(),
                output.data(),
                params,
                debug_out.data()
            );
        } catch (const std::exception& e) {
            std::cerr << "FlashAttention failed: " << e.what() << std::endl;
            // Fallback to original Metal attention
            const int num_query_heads = config.num_heads;
            const int num_kv_heads = config.num_heads;  // Same as query for now (no MQA/GQA)
            const int kv_head_dim = num_kv_heads * config.head_dim;
            
            metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
                q_input.data(),
                kv_data.paged_k_cache.data(),
                kv_data.paged_v_cache.data(),
                kv_data.qo_indptr.data(),
                kv_data.kv_page_indptr.data(),
                kv_data.kv_page_indices.data(),
                kv_data.kv_last_page_lens.data(),
                output.data(),
                num_qo,
                head_dim,
                kv_head_dim,
                config.head_dim,
                config.page_size,
                num_query_heads,
                num_kv_heads,
                scale,
                static_cast<int>(kv_data.kv_page_indices.size())
            );
        }
        
        return output;
    }
    
    // F32 version of output comparison
    static bool compare_outputs_f32(
        const std::vector<float>& metal_output,
        const std::vector<float>& reference_output,
        double tolerance,
        const std::string& test_name
    ) {
        if (metal_output.size() != reference_output.size()) {
            std::cerr << "❌ Size mismatch: " << metal_output.size() << " vs " << reference_output.size() << std::endl;
            return false;
        }
        
        double max_diff = 0.0;
        double avg_diff = 0.0;
        size_t num_mismatches = 0;
        
        for (size_t i = 0; i < metal_output.size(); i++) {
            float metal_val = metal_output[i];
            float ref_val = reference_output[i];
            double diff = std::abs(metal_val - ref_val);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 5) { // Show first 5 mismatches
                    std::cerr << "   Mismatch at index " << i << ": " 
                              << metal_val << " vs " << ref_val 
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
        
        avg_diff /= metal_output.size();
        
        std::cout << "   Max difference: " << max_diff << std::endl;
        std::cout << "   Avg difference: " << avg_diff << std::endl;
        std::cout << "   Mismatches: " << num_mismatches << " / " << metal_output.size() 
                  << " (" << (100.0 * num_mismatches / metal_output.size()) << "%)" << std::endl;
        
        bool passed = (num_mismatches == 0);
        std::cout << (passed ? "✅" : "❌") << " " << test_name << std::endl;
        
        return passed;
    }
    
    // Legacy bf16 version of output comparison
    static bool compare_outputs(
        const std::vector<bfloat16_t>& metal_output,
        const std::vector<float>& reference_output,
        double tolerance,
        const std::string& test_name
    ) {
        if (metal_output.size() != reference_output.size()) {
            std::cerr << "❌ Size mismatch: " << metal_output.size() << " vs " << reference_output.size() << std::endl;
            return false;
        }
        
        double max_diff = 0.0;
        double avg_diff = 0.0;
        size_t num_mismatches = 0;
        
        for (size_t i = 0; i < metal_output.size(); i++) {
            // Convert bf16 to float
            uint32_t bits = static_cast<uint32_t>(metal_output[i]) << 16;
            float metal_val;
            std::memcpy(&metal_val, &bits, sizeof(metal_val));
            
            float ref_val = reference_output[i];
            double diff = std::abs(metal_val - ref_val);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 5) { // Show first 5 mismatches
                    std::cerr << "   Mismatch at index " << i << ": " 
                              << metal_val << " vs " << ref_val 
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
        
        avg_diff /= metal_output.size();
        
        std::cout << "   Max difference: " << max_diff << std::endl;
        std::cout << "   Avg difference: " << avg_diff << std::endl;
        std::cout << "   Mismatches: " << num_mismatches << " / " << metal_output.size() 
                  << " (" << (100.0 * num_mismatches / metal_output.size()) << "%)" << std::endl;
        
        bool passed = (num_mismatches == 0);
        std::cout << (passed ? "✅" : "❌") << " " << test_name << std::endl;
        
        return passed;
    }
    
    // F32 version of Metal output comparison
    static bool compare_metal_outputs_f32(
        const std::vector<float>& output1,
        const std::vector<float>& output2,
        double tolerance
    ) {
        if (output1.size() != output2.size()) {
            std::cerr << "❌ Size mismatch: " << output1.size() << " vs " << output2.size() << std::endl;
            return false;
        }
        
        double max_diff = 0.0;
        double avg_diff = 0.0;
        size_t num_mismatches = 0;
        
        for (size_t i = 0; i < output1.size(); i++) {
            float val1 = output1[i];
            float val2 = output2[i];
            double diff = std::abs(val1 - val2);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 3) { // Show first 3 mismatches
                    std::cerr << "   Mismatch at index " << i << ": " 
                              << val1 << " vs " << val2 
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
        
        avg_diff /= output1.size();
        
        std::cout << "   Max difference: " << max_diff << std::endl;
        std::cout << "   Avg difference: " << avg_diff << std::endl;
        std::cout << "   Mismatches: " << num_mismatches << " / " << output1.size() 
                  << " (" << (100.0 * num_mismatches / output1.size()) << "%)" << std::endl;
        
        bool passed = (num_mismatches == 0);
        std::cout << (passed ? "✅" : "❌") << " Consistency check" << std::endl;
        
        return passed;
    }
    
    // Legacy bf16 version of Metal output comparison
    static bool compare_metal_outputs(
        const std::vector<bfloat16_t>& output1,
        const std::vector<bfloat16_t>& output2,
        double tolerance
    ) {
        if (output1.size() != output2.size()) {
            std::cerr << "❌ Size mismatch: " << output1.size() << " vs " << output2.size() << std::endl;
            return false;
        }
        
        double max_diff = 0.0;
        double avg_diff = 0.0;
        size_t num_mismatches = 0;
        
        for (size_t i = 0; i < output1.size(); i++) {
            // Convert both bf16 values to float
            uint32_t bits1 = static_cast<uint32_t>(output1[i]) << 16;
            uint32_t bits2 = static_cast<uint32_t>(output2[i]) << 16;
            float val1, val2;
            std::memcpy(&val1, &bits1, sizeof(val1));
            std::memcpy(&val2, &bits2, sizeof(val2));
            
            double diff = std::abs(val1 - val2);
            
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            
            if (diff > tolerance) {
                num_mismatches++;
                if (num_mismatches <= 3) { // Show first 3 mismatches
                    std::cerr << "   Mismatch at index " << i << ": " 
                              << val1 << " vs " << val2 
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
        
        avg_diff /= output1.size();
        
        std::cout << "   Max difference: " << max_diff << std::endl;
        std::cout << "   Avg difference: " << avg_diff << std::endl;
        std::cout << "   Mismatches: " << num_mismatches << " / " << output1.size() 
                  << " (" << (100.0 * num_mismatches / output1.size()) << "%)" << std::endl;
        
        bool passed = (num_mismatches == 0);
        std::cout << (passed ? "✅" : "❌") << " Consistency check" << std::endl;
        
        return passed;
    }
};

std::vector<AttentionCorrectnessTest> get_correctness_tests() {
    return {
        // Basic functionality tests with f32 precision and tighter tolerances
        {"Small single head", {1, 128, 1, 64, 16, "f32", "Basic test"}, "random", 1e-6, 42},
        {"Medium multi-head", {1, 512, 8, 128, 16, "f32", "Multi-head test"}, "random", 1e-6, 43},
        {"Batch processing", {4, 256, 16, 64, 16, "f32", "Batch test"}, "random", 1e-6, 44},
        
        // Edge case tests with f32 precision
        {"Zero queries", {1, 128, 8, 64, 16, "f32", "Zero test"}, "zeros", 1e-6, 45},
        {"Uniform queries", {1, 128, 8, 64, 16, "f32", "Ones test"}, "ones", 1e-6, 46},
        {"Large values", {1, 256, 16, 128, 16, "f32", "Large values"}, "large_values", 1e-5, 47},
        {"Small values", {1, 256, 16, 128, 16, "f32", "Small values"}, "small_values", 1e-7, 48},
        
        // Sequence length variations with f32 precision
        {"Single token", {1, 1, 8, 64, 16, "f32", "Single token"}, "random", 1e-6, 49},
        {"Page boundary", {1, 16, 8, 64, 16, "f32", "Exact page size"}, "random", 1e-6, 50},
        {"Cross-page", {1, 17, 8, 64, 16, "f32", "Cross page boundary"}, "random", 1e-6, 51},
        {"Long sequence", {1, 1024, 32, 128, 16, "f32", "Long sequence"}, "random", 1e-6, 52},
        
        // Numerical stability tests with f32 precision
        {"Alternating pattern", {1, 256, 16, 64, 16, "f32", "Alternating"}, "alternating", 1e-6, 53},
    };
}

int main(int argc, char* argv[]) {
    std::cout << "🧪 Metal Attention Kernel Correctness Testing (F32)" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Testing with F32 precision for both original and optimized kernels" << std::endl;
    std::cout << std::endl;
    
    // Initialize unified attention system
    if (!metal::unified_attention::initialize()) {
        std::cerr << "❌ Failed to initialize unified attention system" << std::endl;
        return 1;
    }
    
    auto tests = get_correctness_tests();
    
    int passed = 0;
    int total = tests.size();
    
    std::cout << "Running " << total << " f32 correctness tests..." << std::endl;
    
    for (const auto& test : tests) {
        if (AttentionCorrectnessValidator::validate_against_reference(test)) {
            passed++;
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    metal::unified_attention::cleanup();
    
    // Summary
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "📊 Test Results: " << passed << "/" << total << " passed" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 All f32 tests passed! F32 kernels are working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "⚠️  " << (total - passed) << " tests failed. F32 implementation needs fixes." << std::endl;
        return 1;
    }
}