#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "metal_softmax.hpp"
#include "benchmark_utils.hpp"

/**
 * Performance benchmark for Metal Softmax focusing on large vocabularies (>16K)
 * 
 * This benchmark specifically targets the tile optimization use case:
 * - Large vocabularies where cache thrashing dominates performance
 * - Measures baseline performance before tile optimization
 * - Tracks memory bandwidth utilization and cache efficiency
 */

// Large vocabulary test configurations (focus on >16K)
const std::vector<int> LARGE_VOCAB_SIZES = {16384, 32000, 65536, 131072};
const std::vector<int> BATCH_SIZES = {1, 4, 16, 32};
const int ITERATIONS = 100;
const float TEMPERATURE = 1.0f;

/**
 * CPU reference implementation for validation
 */
void cpu_softmax_reference(const float* input, float* output, int batch_size, int vocab_size, float temperature) {
    for (int b = 0; b < batch_size; b++) {
        const float* row = input + b * vocab_size;
        float* out_row = output + b * vocab_size;
        
        // Find max for numerical stability
        float max_val = *std::max_element(row, row + vocab_size);
        
        // Apply temperature scaling and exp
        double sum = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            float scaled = (row[i] - max_val) / temperature;
            out_row[i] = std::exp(scaled);
            sum += out_row[i];
        }
        
        // Normalize
        for (int i = 0; i < vocab_size; i++) {
            out_row[i] /= sum;
        }
    }
}

/**
 * Generate realistic test data for large vocabulary scenarios
 */
std::vector<float> generate_large_vocab_data(int batch_size, int vocab_size) {
    std::vector<float> data(batch_size * vocab_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Use more realistic distribution for large vocabularies
    // Most values small with some large outliers (like in real transformer outputs)
    std::normal_distribution<float> small_dist(0.0f, 1.0f);
    std::normal_distribution<float> large_dist(5.0f, 2.0f);
    std::uniform_real_distribution<float> outlier_prob(0.0f, 1.0f);
    
    for (int b = 0; b < batch_size; b++) {
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            
            // 90% normal distribution, 10% outliers
            if (outlier_prob(gen) < 0.9f) {
                data[idx] = small_dist(gen);
            } else {
                data[idx] = large_dist(gen);
            }
        }
    }
    
    return data;
}

/**
 * Validate correctness of Metal implementation against CPU reference
 */
bool validate_correctness(const float* metal_output, const float* cpu_output, 
                         int batch_size, int vocab_size, float tolerance = 1e-5f) {
    bool pass = true;
    
    for (int b = 0; b < batch_size; b++) {
        double metal_sum = 0.0;
        double cpu_sum = 0.0;
        
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            metal_sum += metal_output[idx];
            cpu_sum += cpu_output[idx];
            
            // Check individual values
            if (std::abs(metal_output[idx] - cpu_output[idx]) > tolerance) {
                if (pass) {  // Only print first few errors
                    std::cerr << "Mismatch at batch " << b << ", vocab " << v 
                              << ": Metal=" << metal_output[idx] 
                              << ", CPU=" << cpu_output[idx] << std::endl;
                }
                pass = false;
            }
            
            // Check non-negative
            if (metal_output[idx] < 0.0f) {
                std::cerr << "Negative probability at batch " << b << ", vocab " << v 
                          << ": " << metal_output[idx] << std::endl;
                pass = false;
            }
        }
        
        // Check normalization
        if (std::abs(metal_sum - 1.0) > tolerance) {
            std::cerr << "Batch " << b << " sum = " << metal_sum << ", expected 1.0" << std::endl;
            pass = false;
        }
    }
    
    return pass;
}

/**
 * Run comprehensive benchmark for a specific configuration
 */
BenchmarkStats benchmark_configuration(int batch_size, int vocab_size) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing vocabulary size: " << vocab_size << " (batch_size: " << batch_size << ")" << std::endl;
    
    // Generate test data
    auto input_data = generate_large_vocab_data(batch_size, vocab_size);
    std::vector<float> metal_output(batch_size * vocab_size);
    std::vector<float> cpu_output(batch_size * vocab_size);
    
    // Calculate memory usage (input + output)
    size_t memory_bytes = batch_size * vocab_size * sizeof(float) * 2;
    
    // CPU reference for validation
    auto cpu_benchmark = [&]() {
        cpu_softmax_reference(input_data.data(), cpu_output.data(), batch_size, vocab_size, TEMPERATURE);
    };
    
    auto cpu_stats = BenchmarkRunner::run_benchmark(
        "CPU Reference", cpu_benchmark, ITERATIONS, vocab_size, batch_size, memory_bytes
    );
    
    // Metal implementation
    auto metal_benchmark = [&]() {
        int result = metal_softmax_float(input_data.data(), metal_output.data(), 
                                       batch_size, vocab_size, TEMPERATURE);
        if (result != 0) {
            throw std::runtime_error("Metal softmax failed with code: " + std::to_string(result));
        }
    };
    
    auto metal_stats = BenchmarkRunner::run_benchmark(
        "Metal Standard", metal_benchmark, ITERATIONS, vocab_size, batch_size, memory_bytes
    );
    
    // Validation
    std::cout << "\n--- Validation ---" << std::endl;
    bool valid = validate_correctness(metal_output.data(), cpu_output.data(), batch_size, vocab_size);
    std::cout << "Metal Standard:  " << (valid ? "✅ PASS" : "❌ FAIL") << std::endl;
    
    // Results
    BenchmarkRunner::print_results(cpu_stats, "CPU Reference");
    BenchmarkRunner::print_results(metal_stats, "Metal Standard");
    BenchmarkRunner::compare_results(cpu_stats, metal_stats, "CPU", "Metal");
    
    // Cache efficiency analysis for large vocabularies
    double data_size_kb = memory_bytes / 1024.0;
    double l1_cache_kb = 16.0;  // M2 Pro L1 cache size
    
    std::cout << "\n--- Cache Analysis ---" << std::endl;
    std::cout << "Data size: " << data_size_kb << " KB" << std::endl;
    std::cout << "L1 cache: " << l1_cache_kb << " KB" << std::endl;
    std::cout << "Cache thrashing ratio: " << (data_size_kb / l1_cache_kb) << "x" << std::endl;
    
    if (data_size_kb > l1_cache_kb * 4) {
        std::cout << "⚠️  Severe cache thrashing expected - prime candidate for tiling" << std::endl;
    }
    
    return metal_stats;
}

int main() {
    std::cout << "Metal Softmax Large Vocabulary Performance Benchmark" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << "Target: Large vocabularies (>16K) optimization" << std::endl;
    std::cout << "Focus: Cache thrashing mitigation through tiling" << std::endl;
    
    std::vector<BenchmarkStats> all_results;
    std::vector<std::string> result_names;
    
    try {
        // Test each large vocabulary configuration
        for (int vocab_size : LARGE_VOCAB_SIZES) {
            for (int batch_size : BATCH_SIZES) {
                // Skip extremely large configurations that might cause memory issues
                size_t total_elements = static_cast<size_t>(batch_size) * vocab_size;
                if (total_elements > 32 * 131072) {  // 32 * 128K limit
                    continue;
                }
                
                auto stats = benchmark_configuration(batch_size, vocab_size);
                all_results.push_back(stats);
                result_names.push_back("metal_standard_" + std::to_string(vocab_size) + "_" + std::to_string(batch_size));
            }
        }
        
        // Export results for tracking
        CSVExporter::export_results("softmax_large_vocab_baseline.csv", all_results, result_names);
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "SUMMARY: Large Vocabulary Performance Analysis" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Find worst and best performing configurations
        auto worst_perf = std::min_element(all_results.begin(), all_results.end(),
            [](const BenchmarkStats& a, const BenchmarkStats& b) {
                return a.memory_bandwidth_gb_per_sec < b.memory_bandwidth_gb_per_sec;
            });
            
        auto best_perf = std::max_element(all_results.begin(), all_results.end(),
            [](const BenchmarkStats& a, const BenchmarkStats& b) {
                return a.memory_bandwidth_gb_per_sec < b.memory_bandwidth_gb_per_sec;
            });
        
        if (worst_perf != all_results.end() && best_perf != all_results.end()) {
            std::cout << "Worst performance: " << worst_perf->memory_bandwidth_gb_per_sec 
                      << " GB/s (vocab=" << worst_perf->vocab_size << ")" << std::endl;
            std::cout << "Best performance:  " << best_perf->memory_bandwidth_gb_per_sec 
                      << " GB/s (vocab=" << best_perf->vocab_size << ")" << std::endl;
        }
        
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "1. Implement tiled softmax kernel for cache-friendly access" << std::endl;
        std::cout << "2. Target 5-9x speedup for configurations with <10% cache efficiency" << std::endl;
        std::cout << "3. Re-run this benchmark with tiled implementation" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}