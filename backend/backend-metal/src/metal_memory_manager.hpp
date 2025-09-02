#pragma once

#include <cstdint>
#include <iostream>
#include <iomanip>

namespace metal {
namespace memory {

// Metal GPU memory limits on Apple Silicon
constexpr size_t METAL_THREADGROUP_MEMORY_LIMIT = 32768; // 32KB
constexpr size_t METAL_SIMD_SIZE = 32;
constexpr size_t METAL_MAX_THREADGROUP_SIZE = 1024;

/**
 * @brief Memory usage calculation for attention kernels
 */
struct AttentionMemoryLayout {
    size_t q_shared_memory;           // Query shared memory (MAX_HEAD_DIM * sizeof(data_type))
    size_t k_block_memory;            // Key block memory (KERNEL_BLOCK_SIZE * MAX_HEAD_DIM * sizeof(data_type))
    size_t v_block_memory;            // Value block memory (KERNEL_BLOCK_SIZE * MAX_HEAD_DIM * sizeof(data_type))
    size_t w_block_memory;            // Weight block memory (KERNEL_BLOCK_SIZE * sizeof(float)) - only baseline
    size_t accumulator_memory;        // Accumulator memory (MAX_HEAD_DIM * sizeof(float))
    size_t scratch_memory;            // Scratch/reduction memory
    size_t total_memory;              // Total threadgroup memory required
    
    bool exceeds_limit() const {
        return total_memory > METAL_THREADGROUP_MEMORY_LIMIT;
    }
    
    double utilization() const {
        return static_cast<double>(total_memory) / METAL_THREADGROUP_MEMORY_LIMIT;
    }
};

/**
 * @brief Kernel configuration optimized for memory constraints
 */
struct OptimalKernelConfig {
    int max_head_size;                // Maximum head size that fits in memory
    int kernel_block_size;            // Optimal block size for this configuration
    int num_threadgroups_per_head;    // Split large heads across multiple threadgroups
    bool use_staging;                 // Use staging buffers for large heads
    bool force_baseline;              // Force baseline due to memory constraints
    const char* strategy;             // Description of memory management strategy
};

/**
 * @brief Memory manager for Metal attention kernels
 */
class AttentionMemoryManager {
public:
    /**
     * @brief Calculate memory usage for given parameters
     */
    static AttentionMemoryLayout calculate_memory_usage(
        int head_size, 
        int kernel_block_size,
        bool is_f32,
        bool is_baseline_kernel
    ) {
        AttentionMemoryLayout layout = {};
        
        size_t data_type_size = is_f32 ? sizeof(float) : sizeof(uint16_t); // BF16/F16
        size_t float_size = sizeof(float);
        
        // Shared memory arrays
        layout.q_shared_memory = head_size * data_type_size;
        layout.k_block_memory = kernel_block_size * head_size * data_type_size;
        layout.v_block_memory = kernel_block_size * head_size * data_type_size;
        layout.accumulator_memory = head_size * float_size; // Always float for accumulation
        
        // Weight block only needed for baseline kernel
        if (is_baseline_kernel) {
            layout.w_block_memory = kernel_block_size * float_size;
        }
        
        // Scratch memory for reductions and temporary values
        layout.scratch_memory = 512; // Conservative estimate for temp values, barriers, etc.
        
        layout.total_memory = layout.q_shared_memory + 
                             layout.k_block_memory + 
                             layout.v_block_memory + 
                             layout.w_block_memory +
                             layout.accumulator_memory + 
                             layout.scratch_memory;
        
        return layout;
    }
    
    /**
     * @brief Get optimal kernel configuration for given parameters
     */
    static OptimalKernelConfig get_optimal_config(
        int requested_head_size,
        int page_size,
        bool is_f32
    ) {
        OptimalKernelConfig config = {};
        
        // Start with requested parameters
        int kernel_block_size = page_size; // Prefer page-aligned access
        
        // Test different configurations to find optimal one
        
        // 1. Try simdgroup optimized first (no w_block needed)
        AttentionMemoryLayout simdgroup_layout = calculate_memory_usage(
            requested_head_size, kernel_block_size, is_f32, false);
        
        if (!simdgroup_layout.exceeds_limit()) {
            config.max_head_size = requested_head_size;
            config.kernel_block_size = kernel_block_size;
            config.num_threadgroups_per_head = 1;
            config.use_staging = false;
            config.force_baseline = false;
            config.strategy = "Simdgroup optimized - fits in memory";
            return config;
        }
        
        // 2. Try baseline (has w_block but might still fit)
        AttentionMemoryLayout baseline_layout = calculate_memory_usage(
            requested_head_size, kernel_block_size, is_f32, true);
        
        if (!baseline_layout.exceeds_limit()) {
            config.max_head_size = requested_head_size;
            config.kernel_block_size = kernel_block_size;
            config.num_threadgroups_per_head = 1;
            config.use_staging = false;
            config.force_baseline = true;
            config.strategy = "Baseline kernel - simdgroup too large";
            return config;
        }
        
        // 3. Reduce block size to fit memory
        for (int reduced_block_size = kernel_block_size / 2; reduced_block_size >= 8; reduced_block_size /= 2) {
            AttentionMemoryLayout reduced_layout = calculate_memory_usage(
                requested_head_size, reduced_block_size, is_f32, false);
            
            if (!reduced_layout.exceeds_limit()) {
                config.max_head_size = requested_head_size;
                config.kernel_block_size = reduced_block_size;
                config.num_threadgroups_per_head = 1;
                config.use_staging = false;
                config.force_baseline = false;
                config.strategy = "Reduced block size to fit memory";
                return config;
            }
        }
        
        // 4. Split head across multiple threadgroups (head dimension partitioning)
        int max_safe_head_size = find_max_head_size_for_memory(kernel_block_size, is_f32);
        if (max_safe_head_size > 0) {
            config.max_head_size = max_safe_head_size;
            config.kernel_block_size = kernel_block_size;
            config.num_threadgroups_per_head = (requested_head_size + max_safe_head_size - 1) / max_safe_head_size;
            config.use_staging = true;
            config.force_baseline = false;
            config.strategy = "Multi-threadgroup head partitioning";
            return config;
        }
        
        // 5. Emergency fallback - very small head size
        config.max_head_size = 64; // Conservative minimum
        config.kernel_block_size = 8;  // Minimum block size
        config.num_threadgroups_per_head = (requested_head_size + 63) / 64;
        config.use_staging = true;
        config.force_baseline = true;
        config.strategy = "Emergency fallback - minimal memory usage";
        
        return config;
    }
    
    /**
     * @brief Find maximum head size that fits in memory for given block size
     */
    static int find_max_head_size_for_memory(int kernel_block_size, bool is_f32) {
        // Binary search for maximum head size
        int low = 32, high = 512; // Reasonable bounds
        int max_head_size = 0;
        
        while (low <= high) {
            int mid = (low + high) / 2;
            AttentionMemoryLayout layout = calculate_memory_usage(mid, kernel_block_size, is_f32, false);
            
            if (!layout.exceeds_limit()) {
                max_head_size = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return max_head_size;
    }
    
    /**
     * @brief Print memory analysis for debugging
     */
    static void print_memory_analysis(
        int head_size, 
        int kernel_block_size, 
        bool is_f32,
        bool is_baseline
    ) {
        AttentionMemoryLayout layout = calculate_memory_usage(head_size, kernel_block_size, is_f32, is_baseline);
        
        std::cout << "🧮 Memory Analysis (head_size=" << head_size << ", block_size=" << kernel_block_size 
                  << ", " << (is_f32 ? "F32" : "BF16") << ", " << (is_baseline ? "baseline" : "simdgroup") << ")" << std::endl;
        std::cout << "   📊 Q shared: " << layout.q_shared_memory << " bytes" << std::endl;
        std::cout << "   📊 K block: " << layout.k_block_memory << " bytes" << std::endl;
        std::cout << "   📊 V block: " << layout.v_block_memory << " bytes" << std::endl;
        if (layout.w_block_memory > 0) {
            std::cout << "   📊 W block: " << layout.w_block_memory << " bytes" << std::endl;
        }
        std::cout << "   📊 Accumulators: " << layout.accumulator_memory << " bytes" << std::endl;
        std::cout << "   📊 Scratch: " << layout.scratch_memory << " bytes" << std::endl;
        std::cout << "   💾 Total: " << layout.total_memory << " bytes ("
                  << std::fixed << std::setprecision(1) << (layout.utilization() * 100) << "% of 32KB)" << std::endl;
        
        if (layout.exceeds_limit()) {
            std::cout << "   ❌ EXCEEDS LIMIT by " << (layout.total_memory - METAL_THREADGROUP_MEMORY_LIMIT) 
                      << " bytes" << std::endl;
        } else {
            std::cout << "   ✅ Within limit" << std::endl;
        }
        std::cout << std::endl;
    }
};

} // namespace memory
} // namespace metal