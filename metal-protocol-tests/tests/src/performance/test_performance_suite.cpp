#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "metal_softmax.hpp"
#include "metal_extract_k_values.hpp" 
#include "metal_topk_mask_logits.hpp"

// Test configuration for performance benchmarking
const int PERF_BATCH_SIZES[] = {1, 4, 8, 16, 32};
const int PERF_VOCAB_SIZES[] = {8192, 16384, 32000, 65536};
const int PERF_K_VALUES[] = {10, 50, 100};
const int PERF_ITERATIONS = 100;
const float TEMPERATURE = 1.0f;

class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        double avg_time_ms;
        double throughput_ops_per_sec;
        double memory_bandwidth_gb_per_sec;
        size_t memory_usage_mb;
    };
    
    static BenchmarkResult benchmark_softmax(int batch_size, int vocab_size) {
        std::cout << "  Benchmarking Softmax [" << batch_size << ", " << vocab_size << "]..." << std::endl;
        
        // Generate test data
        std::vector<float> input_data(batch_size * vocab_size);
        std::vector<float> output_data(batch_size * vocab_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 2.0f);
        
        for (auto& val : input_data) {
            val = dist(gen);
        }
        
        // Warm up
        for (int i = 0; i < 5; i++) {
            metal_softmax_float(input_data.data(), output_data.data(), 
                              batch_size, vocab_size, TEMPERATURE);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < PERF_ITERATIONS; i++) {
            metal_softmax_float(input_data.data(), output_data.data(),
                              batch_size, vocab_size, TEMPERATURE);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        BenchmarkResult result;
        result.avg_time_ms = static_cast<double>(duration.count()) / (PERF_ITERATIONS * 1000.0);
        result.throughput_ops_per_sec = (static_cast<double>(batch_size * vocab_size) / result.avg_time_ms) * 1000.0;
        
        // Memory usage: input + output + temp storage
        size_t memory_bytes = batch_size * vocab_size * sizeof(float) * 2;
        result.memory_usage_mb = memory_bytes / (1024 * 1024);
        
        // Memory bandwidth (assuming read input + write output)
        result.memory_bandwidth_gb_per_sec = (memory_bytes / (result.avg_time_ms / 1000.0)) / (1024 * 1024 * 1024);
        
        return result;
    }
    
    static BenchmarkResult benchmark_extract_k_values(int batch_size, int vocab_size, int k) {
        std::cout << "  Benchmarking Extract K Values [" << batch_size << ", " << vocab_size << ", k=" << k << "]..." << std::endl;
        
        // Generate sparse test data 
        std::vector<float> input_data(batch_size * vocab_size);
        std::vector<float> output_values(batch_size * k);
        std::vector<int32_t> output_indices(batch_size * k);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 2.0f);
        std::uniform_real_distribution<float> sparse_prob(0.0f, 1.0f);
        
        // Create sparse data (80% zeros)
        for (auto& val : input_data) {
            if (sparse_prob(gen) < 0.8f) {
                val = 0.0f;
            } else {
                val = dist(gen);
            }
        }
        
        // Warm up
        for (int i = 0; i < 5; i++) {
            metal_extract_k_values_float32(input_data.data(), output_values.data(),
                                         output_indices.data(), batch_size, vocab_size, k);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < PERF_ITERATIONS; i++) {
            metal_extract_k_values_float32(input_data.data(), output_values.data(),
                                         output_indices.data(), batch_size, vocab_size, k);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        BenchmarkResult result;
        result.avg_time_ms = static_cast<double>(duration.count()) / (PERF_ITERATIONS * 1000.0);
        result.throughput_ops_per_sec = (static_cast<double>(batch_size * vocab_size) / result.avg_time_ms) * 1000.0;
        
        // Memory usage: input + outputs
        size_t memory_bytes = batch_size * vocab_size * sizeof(float) + 
                             batch_size * k * (sizeof(float) + sizeof(int32_t));
        result.memory_usage_mb = memory_bytes / (1024 * 1024);
        result.memory_bandwidth_gb_per_sec = (memory_bytes / (result.avg_time_ms / 1000.0)) / (1024 * 1024 * 1024);
        
        return result;
    }
    
    static BenchmarkResult benchmark_topk_mask_logits(int batch_size, int vocab_size, int k) {
        std::cout << "  Benchmarking TopK Mask Logits [" << batch_size << ", " << vocab_size << ", k=" << k << "]..." << std::endl;
        
        // Generate test data
        std::vector<float> logits_data(batch_size * vocab_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 2.0f);
        
        for (auto& val : logits_data) {
            val = dist(gen);
        }
        
        // Warm up
        for (int i = 0; i < 5; i++) {
            std::vector<float> temp_data = logits_data;  // Reset data
            metal_topk_mask_logits_float32(temp_data.data(), batch_size, vocab_size, k);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < PERF_ITERATIONS; i++) {
            std::vector<float> temp_data = logits_data;  // Reset data each iteration
            metal_topk_mask_logits_float32(temp_data.data(), batch_size, vocab_size, k);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        BenchmarkResult result;
        result.avg_time_ms = static_cast<double>(duration.count()) / (PERF_ITERATIONS * 1000.0);
        result.throughput_ops_per_sec = (static_cast<double>(batch_size * vocab_size) / result.avg_time_ms) * 1000.0;
        
        // Memory usage: logits (in-place operation but need temp copy)
        size_t memory_bytes = batch_size * vocab_size * sizeof(float) * 2;
        result.memory_usage_mb = memory_bytes / (1024 * 1024);
        result.memory_bandwidth_gb_per_sec = (memory_bytes / (result.avg_time_ms / 1000.0)) / (1024 * 1024 * 1024);
        
        return result;
    }
    
    static void print_result(const std::string& operation, const BenchmarkResult& result) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "    " << operation << ":" << std::endl;
        std::cout << "      Time: " << result.avg_time_ms << " ms" << std::endl;
        std::cout << "      Throughput: " << std::scientific << std::setprecision(2) 
                  << result.throughput_ops_per_sec << " ops/sec" << std::endl;
        std::cout << "      Bandwidth: " << std::fixed << std::setprecision(2) 
                  << result.memory_bandwidth_gb_per_sec << " GB/s" << std::endl;
        std::cout << "      Memory: " << result.memory_usage_mb << " MB" << std::endl;
    }
};

int main() {
    std::cout << "=== Metal Performance Benchmark ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Iterations per test: " << PERF_ITERATIONS << std::endl;
    std::cout << "  Temperature: " << TEMPERATURE << std::endl;
    std::cout << std::endl;
    
    // Benchmark different configurations
    for (int batch_size : PERF_BATCH_SIZES) {
        for (int vocab_size : PERF_VOCAB_SIZES) {
            // Skip very large configurations that might cause memory issues
            if (batch_size * vocab_size > 1024 * 1024) continue;
            
            std::cout << "Configuration: Batch=" << batch_size << ", Vocab=" << vocab_size << std::endl;
            
            // Softmax benchmark
            auto softmax_result = PerformanceBenchmark::benchmark_softmax(batch_size, vocab_size);
            PerformanceBenchmark::print_result("Softmax", softmax_result);
            
            // Extract K Values benchmark
            for (int k : PERF_K_VALUES) {
                if (k <= vocab_size / 10) {  // Only test reasonable k values
                    auto extract_result = PerformanceBenchmark::benchmark_extract_k_values(batch_size, vocab_size, k);
                    PerformanceBenchmark::print_result("Extract K=" + std::to_string(k), extract_result);
                }
            }
            
            // TopK Mask benchmark
            for (int k : PERF_K_VALUES) {
                if (k <= vocab_size / 10) {  // Only test reasonable k values
                    auto topk_result = PerformanceBenchmark::benchmark_topk_mask_logits(batch_size, vocab_size, k);
                    PerformanceBenchmark::print_result("TopK Mask K=" + std::to_string(k), topk_result);
                }
            }
            
            std::cout << std::endl;
        }
    }
    
    std::cout << "🎉 Performance benchmarking completed!" << std::endl;
    std::cout << "\nResults show Metal kernel performance across different configurations." << std::endl;
    std::cout << "These benchmarks validate that the Metal backend can handle typical" << std::endl;
    std::cout << "LLM inference workloads efficiently on Apple Silicon." << std::endl;
    
    return 0;
}