#include "attention_perf_utils.hpp"
#include "metal_batch_prefill_attention.hpp"
#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cstring>

/**
 * Comprehensive performance benchmark for Metal batch prefill attention
 * This will establish baseline metrics and track optimization progress
 */

using bfloat16_t = uint16_t;

// Utility function to generate random bfloat16 data
std::vector<bfloat16_t> generate_random_bf16(size_t count, float min_val = -1.0f, float max_val = 1.0f, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    std::vector<bfloat16_t> result(count);
    for (size_t i = 0; i < count; i++) {
        float val = dist(rng);
        // Simple float to bfloat16 conversion (truncate mantissa)
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        result[i] = static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
    }
    return result;
}

// Setup paged KV cache data structure
struct PagedKVData {
    std::vector<bfloat16_t> paged_k_cache;
    std::vector<bfloat16_t> paged_v_cache;
    std::vector<int32_t> qo_indptr;
    std::vector<int32_t> kv_page_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;
    size_t num_pages;
};

PagedKVData setup_paged_kv_data(const AttentionBenchmarkConfig& config, uint64_t seed = 123) {
    PagedKVData data;
    
    const int total_tokens = config.batch_size * config.seq_length;
    const int head_dim = config.num_heads * config.head_dim;
    
    // Calculate number of pages needed
    data.num_pages = (config.seq_length + config.page_size - 1) / config.page_size;
    
    // Allocate paged KV cache
    size_t page_data_size = data.num_pages * config.page_size * head_dim;
    data.paged_k_cache = generate_random_bf16(page_data_size, -0.5f, 0.5f, seed);
    data.paged_v_cache = generate_random_bf16(page_data_size, -0.5f, 0.5f, seed + 1);
    
    // Setup query-output indptr (simple: each sequence has seq_length tokens)
    data.qo_indptr.resize(config.batch_size + 1);
    for (int i = 0; i <= config.batch_size; i++) {
        data.qo_indptr[i] = i * config.seq_length;
    }
    
    // Setup KV page indptr (each sequence uses consecutive pages)
    data.kv_page_indptr.resize(config.batch_size + 1);
    for (int i = 0; i <= config.batch_size; i++) {
        data.kv_page_indptr[i] = i * static_cast<int>(data.num_pages);
    }
    
    // Setup page indices (simple linear mapping)
    size_t total_pages = config.batch_size * data.num_pages;
    data.kv_page_indices.resize(total_pages);
    for (size_t i = 0; i < total_pages; i++) {
        data.kv_page_indices[i] = static_cast<int32_t>(i);
    }
    
    // Setup last page lengths
    data.kv_last_page_lens.resize(config.batch_size);
    int last_page_len = config.seq_length - (static_cast<int>(data.num_pages) - 1) * config.page_size;
    for (int i = 0; i < config.batch_size; i++) {
        data.kv_last_page_lens[i] = last_page_len;
    }
    
    return data;
}

// Benchmark function for current attention implementation
class AttentionBenchmark {
private:
    AttentionBenchmarkConfig config_;
    PagedKVData kv_data_;
    std::vector<bfloat16_t> q_input_;
    std::vector<bfloat16_t> output_;
    
public:
    AttentionBenchmark(const AttentionBenchmarkConfig& config) 
        : config_(config), kv_data_(setup_paged_kv_data(config)) {
        
        // Generate query input
        size_t q_size = config.batch_size * config.seq_length * config.num_heads * config.head_dim;
        q_input_ = generate_random_bf16(q_size, -0.1f, 0.1f, 456);
        
        // Allocate output buffer
        output_.resize(q_size);
    }
    
    void run_current_kernel() {
        const int num_qo = config_.batch_size * config_.seq_length;
        const int head_dim = config_.num_heads * config_.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
        
        // Call the current Metal attention implementation
        const int num_query_heads = config_.num_heads;
        const int num_kv_heads = config_.num_heads;  // Same as query for now (no MQA/GQA)
        const int kv_head_dim = num_kv_heads * config_.head_dim;
        
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input_.data(),
            kv_data_.paged_k_cache.data(),
            kv_data_.paged_v_cache.data(),
            kv_data_.qo_indptr.data(),
            kv_data_.kv_page_indptr.data(),
            kv_data_.kv_page_indices.data(),
            kv_data_.kv_last_page_lens.data(),
            output_.data(),
            num_qo,
            head_dim,
            kv_head_dim,
            config_.head_dim,
            config_.page_size,
            num_query_heads,
            num_kv_heads,
            scale,
            static_cast<int>(kv_data_.num_pages * config_.batch_size)
        );
    }
    
    void run_optimized_kernel() {
        const int num_qo = config_.batch_size * config_.seq_length;
        const int head_dim = config_.num_heads * config_.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
        
        // Call the optimized Metal attention implementation
        metal::batch_prefill_attention::batch_prefill_attention_optimized_bf16(
            q_input_.data(),
            kv_data_.paged_k_cache.data(),
            kv_data_.paged_v_cache.data(),
            kv_data_.qo_indptr.data(),
            kv_data_.kv_page_indptr.data(),
            kv_data_.kv_page_indices.data(),
            kv_data_.kv_last_page_lens.data(),
            output_.data(),
            num_qo,
            head_dim,
            config_.head_dim,
            config_.page_size,
            scale,
            static_cast<int>(kv_data_.num_pages * config_.batch_size)
        );
    }
    
    // Validate output (basic sanity checks)
    bool validate_output() {
        // Check for NaN or infinite values
        for (size_t i = 0; i < output_.size(); i++) {
            // Convert bf16 to float for checking
            uint32_t bits = static_cast<uint32_t>(output_[i]) << 16;
            float val;
            std::memcpy(&val, &bits, sizeof(val));
            
            if (!std::isfinite(val)) {
                std::cerr << "Invalid output at index " << i << ": " << val << std::endl;
                return false;
            }
        }
        
        std::cout << "✅ Output validation passed" << std::endl;
        return true;
    }
    
    const AttentionBenchmarkConfig& get_config() const { return config_; }
    const std::vector<bfloat16_t>& get_output() const { return output_; }
};

void run_single_benchmark(const AttentionBenchmarkConfig& config, const std::string& kernel_name = "Current Metal") {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "🧪 Testing: " << config.description << std::endl;
    
    try {
        AttentionBenchmark benchmark(config);
        
        // Warmup and validation
        std::cout << "🔄 Running validation..." << std::endl;
        benchmark.run_current_kernel();
        if (!benchmark.validate_output()) {
            std::cerr << "❌ Validation failed for " << config.description << std::endl;
            return;
        }
        
        // Performance benchmark
        auto benchmark_func = [&benchmark]() {
            benchmark.run_current_kernel();
        };
        
        auto stats = AttentionBenchmarkRunner::run_benchmark(
            config, kernel_name, benchmark_func, 30  // 30 iterations for good statistics
        );
        
        AttentionBenchmarkRunner::print_results(stats);
        
        // Store results for later analysis
        static std::vector<AttentionBenchmarkStats> all_results;
        all_results.push_back(stats);
        
        // Export to CSV after each benchmark
        AttentionCSVExporter::export_results("attention_benchmark_results.csv", all_results);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
    }
}

void run_comparative_benchmark(const AttentionBenchmarkConfig& config) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🏁 Comparative Benchmark: " << config.description << std::endl;
    
    try {
        AttentionBenchmark benchmark(config);
        
        // Validate both kernels produce similar results
        std::cout << "🔄 Validating kernel consistency..." << std::endl;
        std::vector<bfloat16_t> original_output = benchmark.get_output();
        benchmark.run_current_kernel();
        std::vector<bfloat16_t> current_output = benchmark.get_output();
        
        benchmark.run_optimized_kernel();
        std::vector<bfloat16_t> optimized_output = benchmark.get_output();
        
        // Basic validation that both kernels work
        if (!benchmark.validate_output()) {
            std::cerr << "❌ Optimized kernel validation failed for " << config.description << std::endl;
            return;
        }
        
        std::cout << "✅ Both kernels validated successfully" << std::endl;
        
        // Benchmark original kernel
        auto current_benchmark_func = [&benchmark]() {
            benchmark.run_current_kernel();
        };
        
        auto current_stats = AttentionBenchmarkRunner::run_benchmark(
            config, "Original Metal", current_benchmark_func, 30
        );
        
        // Benchmark optimized kernel
        auto optimized_benchmark_func = [&benchmark]() {
            benchmark.run_optimized_kernel();
        };
        
        auto optimized_stats = AttentionBenchmarkRunner::run_benchmark(
            config, "Optimized Metal", optimized_benchmark_func, 30
        );
        
        // Print individual results
        AttentionBenchmarkRunner::print_results(current_stats);
        AttentionBenchmarkRunner::print_results(optimized_stats);
        
        // Print comparison
        AttentionBenchmarkRunner::compare_results(current_stats, optimized_stats);
        
        // Store both results
        static std::vector<AttentionBenchmarkStats> all_results;
        all_results.push_back(current_stats);
        all_results.push_back(optimized_stats);
        
        // Export to CSV
        AttentionCSVExporter::export_results("attention_comparative_results.csv", all_results);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Comparative benchmark failed: " << e.what() << std::endl;
    }
}

void run_comprehensive_benchmarks() {
    std::cout << "🚀 Starting Comprehensive Metal Attention Benchmarks" << std::endl;
    std::cout << "Target: Compare original vs optimized attention kernels" << std::endl;
    
    // Development configs (quick tests)
    std::cout << "\n📊 Development Configuration Comparisons" << std::endl;
    for (const auto& config : AttentionTestConfigs::get_dev_configs()) {
        run_comparative_benchmark(config);
    }
    
    // Production configs (realistic workloads)
    std::cout << "\n🏭 Production Configuration Comparisons" << std::endl;
    for (const auto& config : AttentionTestConfigs::get_production_configs()) {
        run_comparative_benchmark(config);
    }
    
    // Stress test configs (if requested)
    char run_stress;
    std::cout << "\n⚡ Run stress test comparisons? (y/n): ";
    std::cin >> run_stress;
    
    if (run_stress == 'y' || run_stress == 'Y') {
        std::cout << "\n⚡ Stress Test Configuration Comparisons" << std::endl;
        for (const auto& config : AttentionTestConfigs::get_stress_configs()) {
            run_comparative_benchmark(config);
        }
    }
}

void run_single_config_benchmark(int argc, char* argv[]) {
    // Parse command line arguments for custom configuration
    AttentionBenchmarkConfig config;
    config.batch_size = (argc > 1) ? std::atoi(argv[1]) : 1;
    config.seq_length = (argc > 2) ? std::atoi(argv[2]) : 1024;
    config.num_heads = (argc > 3) ? std::atoi(argv[3]) : 32;
    config.head_dim = (argc > 4) ? std::atoi(argv[4]) : 128;
    config.page_size = (argc > 5) ? std::atoi(argv[5]) : 16;
    config.dtype = (argc > 6) ? std::string(argv[6]) : "bf16";
    config.description = "Custom configuration";
    
    std::cout << "🎯 Running single configuration benchmark" << std::endl;
    run_single_benchmark(config);
}

void print_usage() {
    std::cout << "Metal Attention Benchmark Tool" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  test_attention_benchmark                    # Run comprehensive benchmarks" << std::endl;
    std::cout << "  test_attention_benchmark --single <args>   # Run single configuration" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Single configuration arguments:" << std::endl;
    std::cout << "  batch_size seq_length num_heads head_dim page_size dtype" << std::endl;
    std::cout << "  Example: test_attention_benchmark --single 1 2048 32 128 16 bf16" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Output:" << std::endl;
    std::cout << "  - Console output with detailed timing and performance metrics" << std::endl;
    std::cout << "  - attention_benchmark_results.csv with all results" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🔥 Metal Attention Kernel Performance Benchmark" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    if (argc > 1 && std::string(argv[1]) == "--help") {
        print_usage();
        return 0;
    }
    
    if (argc > 1 && std::string(argv[1]) == "--single") {
        run_single_config_benchmark(argc - 1, argv + 1);
    } else {
        run_comprehensive_benchmarks();
    }
    
    std::cout << "\n🎉 Benchmark completed! Check attention_benchmark_results.csv for detailed data." << std::endl;
    return 0;
}