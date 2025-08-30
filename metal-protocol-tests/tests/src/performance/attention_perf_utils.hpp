#pragma once

#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <map>

/**
 * Performance benchmarking utilities for Metal Attention kernel optimization
 * Focus on attention-specific metrics and patterns
 */

struct AttentionBenchmarkConfig {
    int batch_size;
    int seq_length;
    int num_heads;
    int head_dim;
    int page_size;
    std::string dtype;
    std::string description;
};

struct AttentionBenchmarkStats {
    double min_time_ms;
    double max_time_ms; 
    double avg_time_ms;
    double median_time_ms;
    double p95_time_ms;
    double p99_time_ms;
    double stddev_ms;
    
    // Memory metrics
    double memory_bandwidth_gb_per_sec;
    double theoretical_peak_gb_per_sec = 200.0; // M2 Pro theoretical peak
    double memory_efficiency_percent;
    size_t total_memory_bytes;
    
    // Attention-specific metrics
    double tokens_per_second;
    double flops_per_second; // FLOPs for attention computation
    double cache_efficiency_percent;
    
    // Configuration
    AttentionBenchmarkConfig config;
    int iterations;
    std::string kernel_name;
};

class AttentionBenchmarkRunner {
public:
    /**
     * Run attention performance benchmark with statistical analysis
     */
    template<typename Func>
    static AttentionBenchmarkStats run_benchmark(
        const AttentionBenchmarkConfig& config,
        const std::string& kernel_name,
        Func benchmark_func,
        int iterations = 50
    ) {
        std::cout << "🚀 Running " << kernel_name << " benchmark..." << std::endl;
        std::cout << "   Config: batch=" << config.batch_size 
                  << ", seq=" << config.seq_length 
                  << ", heads=" << config.num_heads
                  << ", head_dim=" << config.head_dim
                  << ", page_size=" << config.page_size
                  << ", dtype=" << config.dtype << std::endl;
        
        std::vector<double> times_ms;
        times_ms.reserve(iterations);
        
        // Warm-up runs (important for Metal GPU)
        std::cout << "   Warming up..." << std::flush;
        for (int i = 0; i < 10; i++) {
            benchmark_func();
        }
        std::cout << " done" << std::endl;
        
        // Actual benchmark runs
        std::cout << "   Benchmarking..." << std::flush;
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_func();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times_ms.push_back(duration.count() / 1000.0);
            
            if ((i + 1) % 10 == 0) {
                std::cout << "." << std::flush;
            }
        }
        std::cout << " done" << std::endl;
        
        return calculate_stats(times_ms, config, kernel_name, iterations);
    }
    
    /**
     * Calculate comprehensive statistics from timing data
     */
    static AttentionBenchmarkStats calculate_stats(
        const std::vector<double>& times_ms,
        const AttentionBenchmarkConfig& config,
        const std::string& kernel_name,
        int iterations
    ) {
        AttentionBenchmarkStats stats;
        stats.config = config;
        stats.kernel_name = kernel_name;
        stats.iterations = iterations;
        
        // Sort for percentile calculations
        auto sorted_times = times_ms;
        std::sort(sorted_times.begin(), sorted_times.end());
        
        stats.min_time_ms = sorted_times.front();
        stats.max_time_ms = sorted_times.back();
        stats.avg_time_ms = std::accumulate(sorted_times.begin(), sorted_times.end(), 0.0) / sorted_times.size();
        
        // Median
        size_t n = sorted_times.size();
        if (n % 2 == 0) {
            stats.median_time_ms = (sorted_times[n/2 - 1] + sorted_times[n/2]) / 2.0;
        } else {
            stats.median_time_ms = sorted_times[n/2];
        }
        
        // Percentiles
        stats.p95_time_ms = sorted_times[static_cast<size_t>(n * 0.95)];
        stats.p99_time_ms = sorted_times[static_cast<size_t>(n * 0.99)];
        
        // Standard deviation
        double variance = 0.0;
        for (double time : sorted_times) {
            variance += (time - stats.avg_time_ms) * (time - stats.avg_time_ms);
        }
        stats.stddev_ms = std::sqrt(variance / sorted_times.size());
        
        // Calculate memory usage and bandwidth
        calculate_memory_metrics(stats);
        
        return stats;
    }
    
    /**
     * Calculate memory-related metrics
     */
    static void calculate_memory_metrics(AttentionBenchmarkStats& stats) {
        const auto& cfg = stats.config;
        
        // Estimate memory usage for attention computation
        // Q: [batch, seq_len, num_heads, head_dim]
        // K, V: [batch, seq_len, num_heads, head_dim] (in paged format)
        // O: [batch, seq_len, num_heads, head_dim]
        
        size_t element_size = (cfg.dtype == "bf16" || cfg.dtype == "f16") ? 2 : 4;
        
        size_t q_size = cfg.batch_size * cfg.seq_length * cfg.num_heads * cfg.head_dim * element_size;
        size_t kv_size = cfg.batch_size * cfg.seq_length * cfg.num_heads * cfg.head_dim * element_size * 2; // K + V
        size_t o_size = cfg.batch_size * cfg.seq_length * cfg.num_heads * cfg.head_dim * element_size;
        
        stats.total_memory_bytes = q_size + kv_size + o_size;
        
        // Memory bandwidth (GB/s) - using average time
        stats.memory_bandwidth_gb_per_sec = (stats.total_memory_bytes / (stats.avg_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        
        // Memory efficiency (compared to theoretical peak)
        stats.memory_efficiency_percent = (stats.memory_bandwidth_gb_per_sec / stats.theoretical_peak_gb_per_sec) * 100.0;
        
        // Tokens per second
        int total_tokens = cfg.batch_size * cfg.seq_length;
        stats.tokens_per_second = total_tokens / (stats.avg_time_ms / 1000.0);
        
        // Estimate FLOPs for attention computation
        // Attention: Q @ K^T (seq * seq * head_dim) + Softmax + @ V (seq * seq * head_dim)
        // Per head: 2 * seq^2 * head_dim FLOPs
        int64_t flops_per_head = 2LL * cfg.seq_length * cfg.seq_length * cfg.head_dim;
        int64_t total_flops = cfg.batch_size * cfg.num_heads * flops_per_head;
        stats.flops_per_second = total_flops / (stats.avg_time_ms / 1000.0);
        
        // Estimate cache efficiency based on memory access patterns
        // Higher bandwidth utilization often correlates with better cache usage
        stats.cache_efficiency_percent = std::min(100.0, stats.memory_efficiency_percent * 1.2);
    }
    
    /**
     * Print detailed benchmark results
     */
    static void print_results(const AttentionBenchmarkStats& stats) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n========== " << stats.kernel_name << " Benchmark Results ==========" << std::endl;
        
        // Configuration
        std::cout << "📋 Configuration:" << std::endl;
        std::cout << "   Batch: " << stats.config.batch_size 
                  << ", Seq Length: " << stats.config.seq_length
                  << ", Heads: " << stats.config.num_heads
                  << ", Head Dim: " << stats.config.head_dim << std::endl;
        std::cout << "   Page Size: " << stats.config.page_size
                  << ", DType: " << stats.config.dtype
                  << ", Iterations: " << stats.iterations << std::endl;
        
        // Timing statistics
        std::cout << "\n⏱️  Timing Statistics:" << std::endl;
        std::cout << "   Min:    " << stats.min_time_ms << " ms" << std::endl;
        std::cout << "   Avg:    " << stats.avg_time_ms << " ms" << std::endl;
        std::cout << "   Median: " << stats.median_time_ms << " ms" << std::endl;
        std::cout << "   P95:    " << stats.p95_time_ms << " ms" << std::endl;
        std::cout << "   Max:    " << stats.max_time_ms << " ms" << std::endl;
        std::cout << "   StdDev: " << stats.stddev_ms << " ms" << std::endl;
        
        // Performance metrics
        std::cout << "\n🚀 Performance Metrics:" << std::endl;
        std::cout << "   Memory BW:      " << stats.memory_bandwidth_gb_per_sec << " GB/s" << std::endl;
        std::cout << "   Memory Eff:     " << stats.memory_efficiency_percent << "% of peak" << std::endl;
        std::cout << "   Tokens/sec:     " << static_cast<int64_t>(stats.tokens_per_second) << std::endl;
        std::cout << "   GFLOPs/sec:     " << (stats.flops_per_second / 1e9) << std::endl;
        std::cout << "   Cache Eff:      " << stats.cache_efficiency_percent << "%" << std::endl;
        
        // Memory usage
        std::cout << "\n💾 Memory Usage:" << std::endl;
        std::cout << "   Total Memory:   " << (stats.total_memory_bytes / (1024 * 1024)) << " MB" << std::endl;
        
        std::cout << "=================================================" << std::endl;
    }
    
    /**
     * Compare two benchmark results and show improvement
     */
    static void compare_results(
        const AttentionBenchmarkStats& baseline, 
        const AttentionBenchmarkStats& optimized
    ) {
        double speedup = baseline.avg_time_ms / optimized.avg_time_ms;
        double bandwidth_improvement = optimized.memory_bandwidth_gb_per_sec / baseline.memory_bandwidth_gb_per_sec;
        double tokens_improvement = optimized.tokens_per_second / baseline.tokens_per_second;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n🔥 Performance Comparison: " << optimized.kernel_name << " vs " << baseline.kernel_name << std::endl;
        std::cout << "===========================================" << std::endl;
        
        // Main metrics
        std::cout << "⚡ Speedup:        " << speedup << "x ";
        if (speedup >= 2.0) std::cout << "🎯";
        std::cout << std::endl;
        
        std::cout << "🌊 Bandwidth:      " << bandwidth_improvement << "x ";
        if (bandwidth_improvement >= 1.5) std::cout << "✨";
        std::cout << std::endl;
        
        std::cout << "🚀 Tokens/sec:     " << tokens_improvement << "x ";
        if (tokens_improvement >= 2.0) std::cout << "🏆";
        std::cout << std::endl;
        
        // Detailed comparison
        std::cout << "\nDetailed Comparison:" << std::endl;
        std::cout << "   Latency:     " << baseline.avg_time_ms << " ms → " << optimized.avg_time_ms << " ms" << std::endl;
        std::cout << "   Memory BW:   " << baseline.memory_bandwidth_gb_per_sec << " → " << optimized.memory_bandwidth_gb_per_sec << " GB/s" << std::endl;
        std::cout << "   Memory Eff:  " << baseline.memory_efficiency_percent << "% → " << optimized.memory_efficiency_percent << "%" << std::endl;
        std::cout << "   Cache Eff:   " << baseline.cache_efficiency_percent << "% → " << optimized.cache_efficiency_percent << "%" << std::endl;
        
        // Achievement status
        std::cout << "\n🎯 Optimization Targets:" << std::endl;
        if (speedup >= 2.0) {
            std::cout << "   ✅ 2x+ Speedup Achieved!" << std::endl;
        } else {
            std::cout << "   ⚠️  Need " << (2.0 / speedup) << "x more for 2x target" << std::endl;
        }
        
        if (optimized.memory_efficiency_percent >= 30.0) {
            std::cout << "   ✅ 30%+ Memory Efficiency Achieved!" << std::endl;
        } else {
            std::cout << "   ⚠️  Need " << (30.0 - optimized.memory_efficiency_percent) << "% more memory efficiency" << std::endl;
        }
        
        std::cout << "===========================================" << std::endl;
    }
};

/**
 * CSV Export for tracking optimization progress over time
 */
class AttentionCSVExporter {
public:
    static void export_results(
        const std::string& filename, 
        const std::vector<AttentionBenchmarkStats>& results
    ) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        // Header
        file << "timestamp,kernel_name,batch_size,seq_length,num_heads,head_dim,page_size,dtype,";
        file << "min_ms,avg_ms,median_ms,p95_ms,max_ms,stddev_ms,";
        file << "memory_bandwidth_gb_per_s,memory_efficiency_percent,tokens_per_second,gflops_per_second,";
        file << "cache_efficiency_percent,total_memory_mb,iterations" << std::endl;
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // Data rows
        for (const auto& stats : results) {
            const auto& cfg = stats.config;
            
            file << time_t << "," << stats.kernel_name << ","
                 << cfg.batch_size << "," << cfg.seq_length << "," << cfg.num_heads << "," 
                 << cfg.head_dim << "," << cfg.page_size << "," << cfg.dtype << ","
                 << stats.min_time_ms << "," << stats.avg_time_ms << "," << stats.median_time_ms << ","
                 << stats.p95_time_ms << "," << stats.max_time_ms << "," << stats.stddev_ms << ","
                 << stats.memory_bandwidth_gb_per_sec << "," << stats.memory_efficiency_percent << ","
                 << stats.tokens_per_second << "," << (stats.flops_per_second / 1e9) << ","
                 << stats.cache_efficiency_percent << "," << (stats.total_memory_bytes / (1024 * 1024)) << ","
                 << stats.iterations
                 << std::endl;
        }
        
        file.close();
        std::cout << "📊 Results exported to " << filename << std::endl;
    }
};

/**
 * Common test configurations for attention benchmarking
 */
namespace AttentionTestConfigs {
    // Small configurations for development
    static std::vector<AttentionBenchmarkConfig> get_dev_configs() {
        return {
            {1, 128, 8, 64, 16, "bf16", "Small dev test"},
            {1, 512, 16, 128, 16, "bf16", "Medium dev test"},
        };
    }
    
    // Production-like configurations
    static std::vector<AttentionBenchmarkConfig> get_production_configs() {
        return {
            {1, 1024, 32, 128, 16, "bf16", "Llama-7B style"},
            {1, 2048, 32, 128, 16, "bf16", "Llama-7B long context"},
            {4, 512, 32, 128, 16, "bf16", "Batch inference"},
            {1, 4096, 40, 128, 16, "bf16", "Llama-13B style"},
            {1, 8192, 32, 128, 16, "bf16", "Long context"},
        };
    }
    
    // Stress test configurations
    static std::vector<AttentionBenchmarkConfig> get_stress_configs() {
        return {
            {8, 2048, 32, 128, 16, "bf16", "Large batch stress test"},
            {1, 16384, 32, 128, 16, "bf16", "Very long context"},
            {1, 1024, 64, 128, 16, "bf16", "Many heads stress test"},
        };
    }
}