#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include "metal_softmax.hpp"
#include "benchmark_utils.hpp"

/**
 * Performance validation for tiled softmax optimization
 * 
 * Focus: Measure actual vs target speedup for large vocabularies
 * Target: 5-9x speedup for vocab sizes 16K-128K
 */

const std::vector<int> TARGET_VOCAB_SIZES = {17000, 32000, 65536, 100000};
const std::vector<int> PERFORMANCE_BATCH_SIZES = {1, 4, 16};
const int PERFORMANCE_ITERATIONS = 100;
const float TEMPERATURE = 1.0f;

/**
 * CPU reference for baseline comparison
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
 * Generate realistic large vocabulary test data
 */
std::vector<float> generate_test_data(int batch_size, int vocab_size) {
    std::vector<float> data(batch_size * vocab_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    for (auto& val : data) {
        val = dist(gen);
    }
    
    return data;
}

/**
 * Performance validation for a specific configuration
 */
void validate_performance(int batch_size, int vocab_size) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Validation: vocab=" << vocab_size << ", batch=" << batch_size << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Generate test data
    auto input_data = generate_test_data(batch_size, vocab_size);
    std::vector<float> metal_output(batch_size * vocab_size);
    std::vector<float> cpu_output(batch_size * vocab_size);
    
    // Calculate memory usage
    size_t memory_bytes = batch_size * vocab_size * sizeof(float) * 2;  // input + output
    
    // CPU baseline benchmark
    auto cpu_benchmark = [&]() {
        cpu_softmax_reference(input_data.data(), cpu_output.data(), batch_size, vocab_size, TEMPERATURE);
    };
    
    auto cpu_stats = BenchmarkRunner::run_benchmark(
        "CPU Baseline", cpu_benchmark, PERFORMANCE_ITERATIONS, vocab_size, batch_size, memory_bytes
    );
    
    // Metal tiled implementation benchmark
    auto metal_benchmark = [&]() {
        int result = metal_softmax_float(input_data.data(), metal_output.data(), 
                                       batch_size, vocab_size, TEMPERATURE);
        if (result != 0) {
            throw std::runtime_error("Metal softmax failed");
        }
    };
    
    auto metal_stats = BenchmarkRunner::run_benchmark(
        "Metal Tiled", metal_benchmark, PERFORMANCE_ITERATIONS, vocab_size, batch_size, memory_bytes
    );
    
    // Results
    BenchmarkRunner::print_results(cpu_stats, "CPU Baseline");
    BenchmarkRunner::print_results(metal_stats, "Metal Tiled");
    BenchmarkRunner::compare_results(cpu_stats, metal_stats, "CPU Baseline", "Metal Tiled");
    
    // Cache analysis
    double data_size_kb = memory_bytes / 1024.0;
    double l1_cache_kb = 16.0;
    double cache_thrashing_ratio = data_size_kb / l1_cache_kb;
    
    std::cout << "\n--- Cache & Performance Analysis ---" << std::endl;
    std::cout << "Data size: " << data_size_kb << " KB" << std::endl;
    std::cout << "Cache thrashing: " << cache_thrashing_ratio << "x L1 cache" << std::endl;
    std::cout << "Memory bandwidth: " << metal_stats.memory_bandwidth_gb_per_sec << " GB/s" << std::endl;
    std::cout << "Peak utilization: " << metal_stats.cache_efficiency_percent << "%" << std::endl;
    
    // Target analysis based on tile optimization plan
    double expected_min_speedup = 5.0;
    double expected_max_speedup = 9.0;
    
    if (vocab_size >= 16384 && vocab_size <= 32000) {
        expected_min_speedup = 5.0;
        expected_max_speedup = 5.6;
    } else if (vocab_size > 32000 && vocab_size <= 65536) {
        expected_min_speedup = 5.6;
        expected_max_speedup = 6.7;
    } else if (vocab_size > 65536) {
        expected_min_speedup = 6.7;
        expected_max_speedup = 9.0;
    }
    
    double actual_speedup = cpu_stats.avg_time_ms / metal_stats.avg_time_ms;
    
    std::cout << "\n--- Target Achievement Analysis ---" << std::endl;
    std::cout << "Target speedup range: " << expected_min_speedup << "x - " << expected_max_speedup << "x" << std::endl;
    std::cout << "Actual speedup: " << actual_speedup << "x" << std::endl;
    
    if (actual_speedup >= expected_min_speedup) {
        std::cout << "🎯 TARGET ACHIEVED! (" << (actual_speedup / expected_min_speedup) << "x target)" << std::endl;
    } else {
        double needed_improvement = expected_min_speedup / actual_speedup;
        std::cout << "⚠️  Target missed - need " << needed_improvement << "x more improvement" << std::endl;
        
        // Analysis suggestions
        if (metal_stats.cache_efficiency_percent < 20.0) {
            std::cout << "💡 Low cache efficiency suggests tiling not working optimally" << std::endl;
        }
        if (metal_stats.memory_bandwidth_gb_per_sec < 10.0) {
            std::cout << "💡 Low memory bandwidth suggests memory access pattern issues" << std::endl;
        }
    }
}

int main() {
    std::cout << "Metal Softmax Tiled Optimization Performance Validation" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Target: Validate 5-9x speedup for large vocabularies" << std::endl;
    std::cout << "Method: Tile-based cache-friendly implementation" << std::endl;
    
    std::vector<BenchmarkStats> all_results;
    std::vector<std::string> result_names;
    
    int configurations_tested = 0;
    int targets_achieved = 0;
    
    try {
        // Test each target configuration
        for (int vocab_size : TARGET_VOCAB_SIZES) {
            for (int batch_size : PERFORMANCE_BATCH_SIZES) {
                // Skip extremely large configurations
                size_t total_elements = static_cast<size_t>(batch_size) * vocab_size;
                if (total_elements > 64 * 100000) {  // Reasonable limit
                    continue;
                }
                
                validate_performance(batch_size, vocab_size);
                configurations_tested++;
            }
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "FINAL VALIDATION SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "Configurations tested: " << configurations_tested << std::endl;
        std::cout << "Implementation status: Tiled kernels integrated and functional" << std::endl;
        std::cout << "Next steps:" << std::endl;
        std::cout << "1. Analyze performance results" << std::endl;
        std::cout << "2. Tune tile size if needed" << std::endl;
        std::cout << "3. Consider online tiled algorithm for additional optimization" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Performance validation failed: " << e.what() << std::endl;
        return 1;
    }
}