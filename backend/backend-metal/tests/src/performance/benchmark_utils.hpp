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

/**
 * Performance benchmarking utilities for Metal operations
 * Focus on large vocabulary optimization tracking
 */

struct BenchmarkStats {
    double min_time_ms;
    double max_time_ms; 
    double avg_time_ms;
    double median_time_ms;
    double p95_time_ms;
    double p99_time_ms;
    double memory_bandwidth_gb_per_sec;
    double theoretical_peak_gb_per_sec = 200.0; // M2 Pro theoretical peak
    double cache_efficiency_percent;
    size_t memory_bytes;
    int vocab_size;
    int batch_size;
};

class BenchmarkRunner {
public:
    /**
     * Run performance benchmark with statistical analysis
     */
    template<typename Func>
    static BenchmarkStats run_benchmark(
        const std::string& name,
        Func benchmark_func,
        int iterations,
        int vocab_size,
        int batch_size,
        size_t memory_bytes
    ) {
        std::cout << "Running " << name << " [batch=" << batch_size 
                  << ", vocab=" << vocab_size << "]..." << std::endl;
        
        std::vector<double> times_ms;
        times_ms.reserve(iterations);
        
        // Warm-up runs
        for (int i = 0; i < 5; i++) {
            benchmark_func();
        }
        
        // Actual benchmark runs
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            benchmark_func();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times_ms.push_back(duration.count() / 1000.0);
        }
        
        return calculate_stats(times_ms, vocab_size, batch_size, memory_bytes);
    }
    
    /**
     * Calculate comprehensive statistics from timing data
     */
    static BenchmarkStats calculate_stats(
        const std::vector<double>& times_ms,
        int vocab_size,
        int batch_size, 
        size_t memory_bytes
    ) {
        BenchmarkStats stats;
        
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
        
        // Memory bandwidth (GB/s) - using average time
        stats.memory_bandwidth_gb_per_sec = (memory_bytes / (stats.avg_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        
        // Cache efficiency estimation (compared to theoretical peak)
        stats.cache_efficiency_percent = (stats.memory_bandwidth_gb_per_sec / stats.theoretical_peak_gb_per_sec) * 100.0;
        
        stats.memory_bytes = memory_bytes;
        stats.vocab_size = vocab_size;
        stats.batch_size = batch_size;
        
        return stats;
    }
    
    /**
     * Print detailed benchmark results
     */
    static void print_results(const BenchmarkStats& stats, const std::string& kernel_name) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== " << kernel_name << " ===" << std::endl;
        std::cout << "Time (ms)    - Min: " << stats.min_time_ms 
                  << ", Avg: " << stats.avg_time_ms
                  << ", Median: " << stats.median_time_ms
                  << ", P95: " << stats.p95_time_ms
                  << ", Max: " << stats.max_time_ms << std::endl;
        std::cout << "Memory BW    - " << stats.memory_bandwidth_gb_per_sec << " GB/s" << std::endl;
        std::cout << "Cache Eff.   - " << stats.cache_efficiency_percent << "% of peak" << std::endl;
        std::cout << "Data Size    - " << (stats.memory_bytes / (1024 * 1024)) << " MB" << std::endl;
    }
    
    /**
     * Compare two benchmark results and show improvement
     */
    static void compare_results(const BenchmarkStats& baseline, const BenchmarkStats& optimized, 
                              const std::string& baseline_name, const std::string& optimized_name) {
        double speedup = baseline.avg_time_ms / optimized.avg_time_ms;
        double bandwidth_improvement = optimized.memory_bandwidth_gb_per_sec / baseline.memory_bandwidth_gb_per_sec;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n--- Performance Comparison ---" << std::endl;
        std::cout << optimized_name << " vs " << baseline_name << ":" << std::endl;
        std::cout << "Speedup:     " << speedup << "x" << std::endl;
        std::cout << "Bandwidth:   " << bandwidth_improvement << "x improvement" << std::endl;
        std::cout << "Cache Eff:   " << baseline.cache_efficiency_percent << "% -> " 
                  << optimized.cache_efficiency_percent << "%" << std::endl;
        
        if (speedup >= 5.0) {
            std::cout << "🎯 Target achieved: 5x+ speedup" << std::endl;
        } else {
            std::cout << "⚠️  Target missed: Need " << (5.0 / speedup) << "x more improvement" << std::endl;
        }
    }
};

/**
 * CSV Export for tracking optimization progress over time
 */
class CSVExporter {
public:
    static void export_results(const std::string& filename, 
                             const std::vector<BenchmarkStats>& results,
                             const std::vector<std::string>& kernel_names) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << " for writing" << std::endl;
            return;
        }
        
        // Header
        file << "timestamp,kernel,batch_size,vocab_size,memory_mb,";
        file << "min_ms,avg_ms,median_ms,p95_ms,max_ms,";
        file << "bandwidth_gb_per_s,cache_efficiency_percent" << std::endl;
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // Data rows
        for (size_t i = 0; i < results.size(); i++) {
            const auto& stats = results[i];
            const auto& name = (i < kernel_names.size()) ? kernel_names[i] : "unknown";
            
            file << time_t << "," << name << ","
                 << stats.batch_size << "," << stats.vocab_size << ","
                 << (stats.memory_bytes / (1024 * 1024)) << ","
                 << stats.min_time_ms << "," << stats.avg_time_ms << ","
                 << stats.median_time_ms << "," << stats.p95_time_ms << "," << stats.max_time_ms << ","
                 << stats.memory_bandwidth_gb_per_sec << "," << stats.cache_efficiency_percent
                 << std::endl;
        }
        
        file.close();
        std::cout << "Results exported to " << filename << std::endl;
    }
};