#include "metal_batch_prefill_attention.hpp"
#include "metal_batch_prefill_handle.hpp"
#include "metal_memory_manager.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>

// Conversion utilities for bfloat16 to IEEE half
namespace {
    using bfloat16_t = uint16_t;

    // Convert bfloat16 to float32
    inline float bf16_to_float(bfloat16_t bf16) {
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    // Convert float32 to IEEE half precision (16-bit)
    inline uint16_t float_to_half(float f) {
        union { float f; uint32_t i; } u;
        u.f = f;

        // Handle special cases first
        if (f == 0.0f) return u.i >> 16; // Preserve sign of zero
        if (!std::isfinite(f)) {
            if (std::isnan(f)) return 0x7e00; // NaN
            return (u.i >> 16) | 0x7c00; // Infinity with correct sign
        }

        uint32_t sign = (u.i >> 16) & 0x8000;
        int32_t exp = ((u.i >> 23) & 0xff) - 127 + 15;
        uint32_t mantissa = (u.i >> 13) & 0x3ff;

        if (exp <= 0) {
            // Underflow to zero
            return static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            // Overflow to infinity
            return static_cast<uint16_t>(sign | 0x7c00);
        } else {
            return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
        }
    }

    // Convert bfloat16 to IEEE half via float32
    inline uint16_t bf16_to_half(bfloat16_t bf16) {
        return float_to_half(bf16_to_float(bf16));
    }

    // Convert IEEE half precision (16-bit) to float32
    inline float half_to_float(uint16_t h) {
        uint16_t h_exp = (h & 0x7C00u) >> 10;
        uint16_t h_sig = (h & 0x03FFu);
        uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;

        uint32_t f;
        if (h_exp == 0) {
            // Zero or subnormal
            if (h_sig == 0) {
                f = sign; // +/- 0
            } else {
                // Normalize the subnormal number
                int shift = 0;
                while ((h_sig & 0x0400u) == 0) { h_sig <<= 1; ++shift; }
                h_sig &= 0x03FFu;
                uint32_t exp = 127 - 15 - shift;
                uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
                f = sign | (exp << 23) | mant;
            }
        } else if (h_exp == 0x1Fu) {
            // Inf or NaN
            uint32_t exp = 0xFFu;
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        } else {
            // Normalized number
            uint32_t exp = static_cast<uint32_t>(h_exp) - 15 + 127;
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }

    // Convert float32 to bfloat16 (truncate with round-to-nearest)
    inline bfloat16_t float_to_bf16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round to nearest even by adding 0x8000 before truncation
        return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
    }

    // Convert a vector of bfloat16 data to IEEE half format
    std::vector<uint16_t> convert_bf16_to_half(const void* bf16_data, size_t count) {
        const bfloat16_t* src = static_cast<const bfloat16_t*>(bf16_data);
        std::vector<uint16_t> result(count);

        for (size_t i = 0; i < count; ++i) {
            result[i] = bf16_to_half(src[i]);
        }

        return result;
    }
}

namespace metal {
namespace batch_prefill_attention {

// ============================================================================
// Handle Management Implementation
// ============================================================================

MetalBatchPrefillHandle* metal_batch_prefill_create_handle(
    int max_batch_size,
    int max_seq_length, 
    int max_heads,
    int max_head_dim
) {
    MetalBatchPrefillHandle* handle = new MetalBatchPrefillHandle();
    
    // Initialize device
    handle->device = MTLCreateSystemDefaultDevice();
    if (!handle->device) {
        std::cerr << "MetalBatchPrefillHandle: Metal is not supported on this device" << std::endl;
        delete handle;
        return nullptr;
    }
    
    // Create command queue
    handle->commandQueue = [handle->device newCommandQueue];
    if (!handle->commandQueue) {
        std::cerr << "MetalBatchPrefillHandle: Failed to create command queue" << std::endl;
        delete handle;
        return nullptr;
    }
    
    // Load Metal library - combine all Metal source files
    NSError* error = nil;
    NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
    NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
    
    // Load all Metal source files and combine them
    NSMutableString* combinedSource = [[NSMutableString alloc] init];
    
    // List of Metal files to load (in dependency order)
    NSArray* metalFiles = @[
        @"metal_attention_common.metal",        // Common utilities (must be first)
        @"metal_attention_baseline.metal",      // Baseline kernels
        @"metal_attention_simdgroup_opt.metal", // Optimized kernels  
        @"metal_batch_prefill_attention.metal"  // Original kernels (backward compatibility)
    ];
    
    for (NSString* filename in metalFiles) {
        NSString* metalPath = [dirPath stringByAppendingPathComponent:filename];
        NSString* metalSource = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
        
        if (error || !metalSource) {
            std::cerr << "MetalBatchPrefillHandle: Failed to load Metal source (" 
                      << filename.UTF8String << "): " 
                      << (error ? error.localizedDescription.UTF8String : "file not found") << std::endl;
            delete handle;
            return nullptr;
        }
        
        [combinedSource appendString:metalSource];
        [combinedSource appendString:@"\n\n"]; // Add spacing between files
        error = nil; // Reset error for next iteration
    }
    
    if (combinedSource.length == 0) {
        std::cerr << "MetalBatchPrefillHandle: No Metal source loaded" << std::endl;
        delete handle;
        return nullptr;
    }
    
    handle->library = [handle->device newLibraryWithSource:combinedSource options:nil error:&error];
    if (!handle->library || error) {
        std::cerr << "MetalBatchPrefillHandle: Failed to compile Metal library: " 
                  << (error ? error.localizedDescription.UTF8String : "unknown error") << std::endl;
        delete handle;
        return nullptr;
    }
    
    // Create compute pipeline states for all kernel variants
    
    // Original unified kernels (maintained for backward compatibility)
    id<MTLFunction> bf16_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_bf16_kernel"];
    if (bf16_function) {
        handle->pipeline_bf16 = [handle->device newComputePipelineStateWithFunction:bf16_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create BF16 pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    id<MTLFunction> f32_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_f32_kernel"];
    if (f32_function) {
        handle->pipeline_f32 = [handle->device newComputePipelineStateWithFunction:f32_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create F32 pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    // Baseline reference kernels
    id<MTLFunction> bf16_baseline_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_bf16_baseline_kernel"];
    if (bf16_baseline_function) {
        handle->pipeline_bf16_baseline = [handle->device newComputePipelineStateWithFunction:bf16_baseline_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create BF16 baseline pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    id<MTLFunction> f32_baseline_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_f32_baseline_kernel"];
    if (f32_baseline_function) {
        handle->pipeline_f32_baseline = [handle->device newComputePipelineStateWithFunction:f32_baseline_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create F32 baseline pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    // Simdgroup optimized kernels (Priority 0)
    id<MTLFunction> bf16_simdgroup_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_bf16_simdgroup_kernel"];
    if (bf16_simdgroup_function) {
        handle->pipeline_bf16_simdgroup = [handle->device newComputePipelineStateWithFunction:bf16_simdgroup_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create BF16 simdgroup pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    id<MTLFunction> f32_simdgroup_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_f32_simdgroup_kernel"];
    if (f32_simdgroup_function) {
        handle->pipeline_f32_simdgroup = [handle->device newComputePipelineStateWithFunction:f32_simdgroup_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create F32 simdgroup pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    // Per-head mapping kernels (Priority 2)
    id<MTLFunction> bf16_per_head_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_bf16_per_head_kernel"];
    if (bf16_per_head_function) {
        handle->pipeline_bf16_per_head = [handle->device newComputePipelineStateWithFunction:bf16_per_head_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create BF16 per-head pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }
    
    // Note: F32 per-head kernel not implemented yet - would be similar to BF16 version
    handle->pipeline_f32_per_head = nil;
    
    // Set configuration bounds
    handle->max_batch_size = max_batch_size;
    handle->max_seq_length = max_seq_length;
    handle->max_heads = max_heads;
    handle->max_head_dim = max_head_dim;
    
    // Initialize statistics
    handle->total_calls = 0;
    handle->total_bytes_processed = 0;
    handle->initialized = true;
    
    std::cout << "MetalBatchPrefillHandle: Successfully created handle with bounds: "
              << "batch=" << max_batch_size << ", seq=" << max_seq_length 
              << ", heads=" << max_heads << ", head_dim=" << max_head_dim << std::endl;
    
    return handle;
}

void metal_batch_prefill_destroy_handle(MetalBatchPrefillHandle* handle) {
    if (!handle) return;
    
    std::cout << "MetalBatchPrefillHandle: Destroying handle. Total calls: " << handle->total_calls
              << ", Total bytes processed: " << (handle->total_bytes_processed / 1024 / 1024) << " MB" << std::endl;
    
    // Metal objects will be automatically released by ARC
    handle->device = nil;
    handle->commandQueue = nil;
    handle->library = nil;
    handle->pipeline_bf16 = nil;
    handle->pipeline_f32 = nil;
    
    delete handle;
}

MetalBatchPrefillWorkspace metal_batch_prefill_get_workspace(
    MetalBatchPrefillHandle* handle,
    int num_tokens,
    int head_dim,
    int kv_head_dim,
    int page_size,
    int num_kv_pages
) {
    MetalBatchPrefillWorkspace workspace = {0};
    
    if (!handle || !handle->initialized) {
        std::cerr << "MetalBatchPrefillWorkspace: Invalid handle" << std::endl;
        return workspace;
    }
    
    // Buffer alignment for Metal (16-byte alignment)
    const size_t BUFFER_ALIGNMENT = 16;
    auto align = [](size_t size) {
        return (size + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    };
    
    size_t offset = 0;
    
    // Query buffer (converted from BF16 to Half)
    size_t q_count = static_cast<size_t>(num_tokens) * head_dim;
    workspace.q_buffer_offset = offset;
    workspace.q_buffer_size = align(q_count * sizeof(uint16_t));
    offset += workspace.q_buffer_size;
    
    // Key cache buffer (converted from BF16 to Half)
    size_t kv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;
    workspace.k_buffer_offset = offset;
    workspace.k_buffer_size = align(kv_count * sizeof(uint16_t));
    offset += workspace.k_buffer_size;
    
    // Value cache buffer (converted from BF16 to Half)
    workspace.v_buffer_offset = offset;
    workspace.v_buffer_size = align(kv_count * sizeof(uint16_t));
    offset += workspace.v_buffer_size;
    
    // Output buffer
    workspace.output_buffer_offset = offset;
    workspace.output_buffer_size = align(q_count * sizeof(uint16_t));
    offset += workspace.output_buffer_size;
    
    // Index buffers (combined for all index arrays)
    size_t total_index_elems = (num_tokens + 1) * 2 + num_kv_pages + num_tokens; // indptr arrays + indices + lens
    workspace.index_buffer_offset = offset;
    workspace.index_buffer_size = align(total_index_elems * sizeof(int32_t));
    offset += workspace.index_buffer_size;
    
    // Parameters buffer
    workspace.params_buffer_offset = offset;
    workspace.params_buffer_size = align(256); // Generous size for parameters struct
    offset += workspace.params_buffer_size;
    
    // Debug buffer
    workspace.debug_buffer_offset = offset; 
    workspace.debug_buffer_size = align(64 * sizeof(float)); // Debug data
    offset += workspace.debug_buffer_size;
    
    // Final alignment
    workspace.alignment_padding = align(offset) - offset;
    workspace.total_size = align(offset);
    
    std::cout << "MetalBatchPrefillWorkspace: Required workspace size: " 
              << (workspace.total_size / 1024 / 1024) << " MB for "
              << num_tokens << " tokens, " << num_kv_pages << " pages" << std::endl;
    
    return workspace;
}

// ============================================================================
// Kernel Selection Logic
// ============================================================================

namespace {
    using namespace metal::memory;
    
    // Helper function to select optimal kernel based on problem size, memory constraints, and device capabilities
    id<MTLComputePipelineState> select_bf16_kernel(
        MetalBatchPrefillHandle* handle,
        KernelOptimizationLevel opt_level,
        int num_tokens,
        int total_kv_len,
        int head_size,
        int page_size,
        bool debug_memory = false
    ) {
        // Get optimal memory configuration
        OptimalKernelConfig memory_config = AttentionMemoryManager::get_optimal_config(
            head_size, page_size, false); // BF16
        
        if (debug_memory) {
            std::cout << "🧮 Memory-aware kernel selection for BF16:" << std::endl;
            std::cout << "   📊 Requested head_size: " << head_size << std::endl;
            std::cout << "   🎯 Strategy: " << memory_config.strategy << std::endl;
            AttentionMemoryManager::print_memory_analysis(head_size, page_size, false, false);
            AttentionMemoryManager::print_memory_analysis(head_size, page_size, false, true);
        }
        
        // Handle memory constraints
        if (memory_config.force_baseline) {
            if (debug_memory) {
                std::cout << "   🔄 Forced to use baseline due to memory constraints" << std::endl;
            }
            return handle->pipeline_bf16_baseline ? handle->pipeline_bf16_baseline : handle->pipeline_bf16;
        }
        
        // Handle head size that requires partitioning
        if (memory_config.num_threadgroups_per_head > 1) {
            if (debug_memory) {
                std::cout << "   ⚠️ Large head size requires " << memory_config.num_threadgroups_per_head 
                          << " threadgroups - using baseline for safety" << std::endl;
            }
            // For now, fall back to baseline for multi-threadgroup scenarios
            // TODO: Implement head partitioning in future optimization
            return handle->pipeline_bf16_baseline ? handle->pipeline_bf16_baseline : handle->pipeline_bf16;
        }
        
        // Memory-optimized selection logic
        switch (opt_level) {
            case KernelOptimizationLevel::BASELINE:
                return handle->pipeline_bf16_baseline ? handle->pipeline_bf16_baseline : handle->pipeline_bf16;
                
            case KernelOptimizationLevel::SIMDGROUP_OPT:
                return handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline;
                
            case KernelOptimizationLevel::PER_HEAD_OPT:
                return handle->pipeline_bf16_per_head ? handle->pipeline_bf16_per_head : 
                       (handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline);
                
            case KernelOptimizationLevel::AUTO:
            default:
                // Auto selection based on memory constraints and problem characteristics
                // Priority 2 optimizations (per-head) are most effective for small batch sizes
                // where GPU occupancy is low due to insufficient threadgroups
                if (num_tokens <= 16 && total_kv_len <= 512 && head_size <= 128) {
                    // Very small batches: use per-head mapping for better occupancy
                    return handle->pipeline_bf16_per_head ? handle->pipeline_bf16_per_head : 
                           (handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline);
                } else if (total_kv_len <= 512 && head_size <= 128) {
                    // Small sequences: prefer simdgroup for thread utilization
                    return handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline;
                } else {
                    // Large sequences: use baseline (optimization overhead not worth it)
                    return handle->pipeline_bf16_baseline ? handle->pipeline_bf16_baseline : handle->pipeline_bf16;
                }
        }
    }
    
    id<MTLComputePipelineState> select_f32_kernel(
        MetalBatchPrefillHandle* handle,
        KernelOptimizationLevel opt_level,
        int num_tokens,
        int total_kv_len,
        int head_size,
        int page_size,
        bool debug_memory = false
    ) {
        // F32 uses 4x more memory than BF16, so memory constraints are more severe
        OptimalKernelConfig memory_config = AttentionMemoryManager::get_optimal_config(
            head_size, page_size, true); // F32
        
        if (debug_memory) {
            std::cout << "🧮 Memory-aware kernel selection for F32:" << std::endl;
            std::cout << "   📊 Requested head_size: " << head_size << std::endl;
            std::cout << "   🎯 Strategy: " << memory_config.strategy << std::endl;
            AttentionMemoryManager::print_memory_analysis(head_size, page_size, true, false);
            AttentionMemoryManager::print_memory_analysis(head_size, page_size, true, true);
        }
        
        // F32 kernels are much more memory constrained - be conservative
        if (memory_config.force_baseline || memory_config.num_threadgroups_per_head > 1) {
            if (debug_memory) {
                std::cout << "   🔄 Using baseline due to F32 memory constraints" << std::endl;
            }
            return handle->pipeline_f32_baseline ? handle->pipeline_f32_baseline : handle->pipeline_f32;
        }
        
        // F32 memory-optimized selection logic
        switch (opt_level) {
            case KernelOptimizationLevel::BASELINE:
                return handle->pipeline_f32_baseline ? handle->pipeline_f32_baseline : handle->pipeline_f32;
                
            case KernelOptimizationLevel::SIMDGROUP_OPT:
                // Check if simdgroup kernel fits in memory
                if (handle->pipeline_f32_simdgroup) {
                    AttentionMemoryLayout simdgroup_layout = AttentionMemoryManager::calculate_memory_usage(
                        head_size, page_size, true, false);
                    if (!simdgroup_layout.exceeds_limit()) {
                        return handle->pipeline_f32_simdgroup;
                    }
                }
                // Fall back to baseline if simdgroup doesn't fit
                return handle->pipeline_f32_baseline ? handle->pipeline_f32_baseline : handle->pipeline_f32;
                
            case KernelOptimizationLevel::PER_HEAD_OPT:
                // F32 per-head kernel not implemented yet, fall back to simdgroup or baseline
                if (handle->pipeline_f32_simdgroup) {
                    AttentionMemoryLayout simdgroup_layout = AttentionMemoryManager::calculate_memory_usage(
                        head_size, page_size, true, false);
                    if (!simdgroup_layout.exceeds_limit()) {
                        return handle->pipeline_f32_simdgroup;
                    }
                }
                return handle->pipeline_f32_baseline ? handle->pipeline_f32_baseline : handle->pipeline_f32;
                
            case KernelOptimizationLevel::AUTO:
            default:
                // For F32, be more conservative due to memory constraints
                if (total_kv_len <= 256 && head_size <= 64) { // Much smaller thresholds for F32
                    // Only use simdgroup for very small problems with F32
                    if (handle->pipeline_f32_simdgroup) {
                        AttentionMemoryLayout simdgroup_layout = AttentionMemoryManager::calculate_memory_usage(
                            head_size, page_size, true, false);
                        if (!simdgroup_layout.exceeds_limit()) {
                            return handle->pipeline_f32_simdgroup;
                        }
                    }
                }
                // Default to baseline for F32 due to memory constraints
                return handle->pipeline_f32_baseline ? handle->pipeline_f32_baseline : handle->pipeline_f32;
        }
    }
}

// ============================================================================
// New Handle-Based Attention Functions
// ============================================================================

void batch_prefill_attention_unified_bf16(
    MetalBatchPrefillHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const void* q_input,
    const void* paged_k_cache,
    const void* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    void* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages,
    KernelOptimizationLevel opt_level
) {
    if (!handle || !handle->initialized || !workspace_buffer) {
        std::cerr << "batch_prefill_attention_unified_bf16: Invalid handle or workspace" << std::endl;
        return;
    }
    
    // Get workspace layout
    MetalBatchPrefillWorkspace workspace = metal_batch_prefill_get_workspace(
        handle, num_qo, head_dim, kv_head_dim, page_size, num_kv_pages);
    
    if (workspace_size < workspace.total_size) {
        std::cerr << "batch_prefill_attention_unified_bf16: Workspace too small. Required: " 
                  << workspace.total_size << ", Provided: " << workspace_size << std::endl;
        return;
    }
    
    std::cout << "🟢 [HANDLE] batch_prefill_attention_unified_bf16 called with workspace!" << std::endl;
    std::cout << "🔍 [HANDLE] Parameters: num_qo=" << num_qo << ", head_dim=" << head_dim 
              << ", head_size=" << head_size << ", page_size=" << page_size 
              << ", scale=" << scale << std::endl;
    
    @autoreleasepool {
        // Get workspace regions
        char* workspace_base = static_cast<char*>(workspace_buffer);
        void* q_workspace = workspace_base + workspace.q_buffer_offset;
        void* k_workspace = workspace_base + workspace.k_buffer_offset;
        void* v_workspace = workspace_base + workspace.v_buffer_offset;
        void* output_workspace = workspace_base + workspace.output_buffer_offset;
        void* index_workspace = workspace_base + workspace.index_buffer_offset;
        void* params_workspace = workspace_base + workspace.params_buffer_offset;
        void* debug_workspace = workspace_base + workspace.debug_buffer_offset;
        
        // Convert input data directly into workspace
        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t kv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;
        
        // Convert BF16 to Half in workspace
        std::vector<uint16_t> q_half_data = convert_bf16_to_half(q_input, q_count);
        std::vector<uint16_t> k_half_data = convert_bf16_to_half(paged_k_cache, kv_count);
        std::vector<uint16_t> v_half_data = convert_bf16_to_half(paged_v_cache, kv_count);
        
        std::memcpy(q_workspace, q_half_data.data(), q_half_data.size() * sizeof(uint16_t));
        std::memcpy(k_workspace, k_half_data.data(), k_half_data.size() * sizeof(uint16_t));
        std::memcpy(v_workspace, v_half_data.data(), v_half_data.size() * sizeof(uint16_t));
        
        // Copy index data to workspace
        size_t index_offset = 0;
        size_t qo_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indices_size = num_kv_pages * sizeof(int32_t);
        size_t kv_last_page_lens_size = num_qo * sizeof(int32_t);
        
        std::memcpy((char*)index_workspace + index_offset, qo_indptr, qo_indptr_size);
        index_offset += qo_indptr_size;
        std::memcpy((char*)index_workspace + index_offset, kv_page_indptr, kv_page_indptr_size);
        index_offset += kv_page_indptr_size;
        std::memcpy((char*)index_workspace + index_offset, kv_page_indices, kv_page_indices_size);
        index_offset += kv_page_indices_size;
        std::memcpy((char*)index_workspace + index_offset, kv_last_page_lens, kv_last_page_lens_size);
        
        // Create Metal buffer views (no allocation!)
        id<MTLBuffer> q_buf = [handle->device newBufferWithBytesNoCopy:q_workspace
                                                               length:workspace.q_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        
        id<MTLBuffer> k_buf = [handle->device newBufferWithBytesNoCopy:k_workspace
                                                               length:workspace.k_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        
        id<MTLBuffer> v_buf = [handle->device newBufferWithBytesNoCopy:v_workspace
                                                               length:workspace.v_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        
        id<MTLBuffer> out_buf = [handle->device newBufferWithBytesNoCopy:output_workspace
                                                                 length:workspace.output_buffer_size
                                                                options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        
        id<MTLBuffer> index_buf = [handle->device newBufferWithBytesNoCopy:index_workspace
                                                                   length:workspace.index_buffer_size
                                                                  options:MTLResourceStorageModeShared
                                                             deallocator:nil];
        
        // Create parameter buffer
        struct Params {
            int num_qo;
            int head_dim;
            int kv_head_dim;
            int head_size;
            int page_size;
            int num_query_heads;
            int num_kv_heads;
            float scale;
        };
        
        Params* params = static_cast<Params*>(params_workspace);
        *params = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
        
        id<MTLBuffer> params_buf = [handle->device newBufferWithBytesNoCopy:params_workspace
                                                                     length:workspace.params_buffer_size
                                                                    options:MTLResourceStorageModeShared
                                                               deallocator:nil];
        
        // Create debug buffer
        id<MTLBuffer> debug_buf = [handle->device newBufferWithBytesNoCopy:debug_workspace
                                                                    length:workspace.debug_buffer_size
                                                                   options:MTLResourceStorageModeShared
                                                              deallocator:nil];
        
        // Select optimal kernel based on problem characteristics and memory constraints
        // Estimate total_kv_len (this is a rough estimate - exact calculation happens in kernel)
        int estimated_total_kv_len = (num_kv_pages > 0) ? (num_kv_pages - 1) * page_size + page_size : 0;
        id<MTLComputePipelineState> selected_pipeline = select_bf16_kernel(
            handle, opt_level, num_qo, estimated_total_kv_len, head_size, page_size, true); // Enable debug for large heads
        
        if (!selected_pipeline) {
            std::cerr << "batch_prefill_attention_unified_bf16: No suitable kernel found" << std::endl;
            return;
        }
        
        // Execute Metal kernel
        id<MTLCommandBuffer> cmd = [handle->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        
        [enc setComputePipelineState:selected_pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:k_buf offset:0 atIndex:1];
        [enc setBuffer:v_buf offset:0 atIndex:2];
        
        // Set index buffers with proper offsets
        [enc setBuffer:index_buf offset:0 atIndex:3]; // qo_indptr
        [enc setBuffer:index_buf offset:qo_indptr_size atIndex:4]; // kv_page_indptr
        [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size atIndex:5]; // kv_page_indices
        [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size + kv_page_indices_size atIndex:6]; // kv_last_page_lens
        
        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        [enc setBuffer:debug_buf offset:0 atIndex:9];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid;
        
        // Determine threadgroup grid size based on kernel type
        if (selected_pipeline == handle->pipeline_bf16_per_head) {
            // Per-head kernel: one threadgroup per (qo, head) pair
            threadgroupsPerGrid = MTLSizeMake(num_qo * num_query_heads, 1, 1);
        } else {
            // Standard kernels: one threadgroup per qo (loops over heads internally)
            threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        }
        
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        
        // Commit and wait (with retry logic preserved)
        int retries = 3;
        NSError* cmdError = nil;
        while (retries > 0) {
            [cmd commit];
            [cmd waitUntilCompleted];
            cmdError = cmd.error;
            
            if (!cmdError) {
                break; // Success
            }
            
            retries--;
            if (retries > 0) {
                cmd = [handle->commandQueue commandBuffer];
                enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:selected_pipeline];
                [enc setBuffer:q_buf offset:0 atIndex:0];
                [enc setBuffer:k_buf offset:0 atIndex:1];
                [enc setBuffer:v_buf offset:0 atIndex:2];
                [enc setBuffer:index_buf offset:0 atIndex:3];
                [enc setBuffer:index_buf offset:qo_indptr_size atIndex:4];
                [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size atIndex:5];
                [enc setBuffer:index_buf offset:qo_indptr_size + kv_page_indptr_size + kv_page_indices_size atIndex:6];
                [enc setBuffer:out_buf offset:0 atIndex:7];
                [enc setBuffer:params_buf offset:0 atIndex:8];
                [enc setBuffer:debug_buf offset:0 atIndex:9];
                MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
                MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
                [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [enc endEncoding];
            } else {
                std::cerr << "Metal command buffer failed after retries: " << cmdError.localizedDescription.UTF8String << std::endl;
                return;
            }
        }
        
        // Convert output from Half to BF16
        if (output) {
            uint16_t* out_half = static_cast<uint16_t*>(output_workspace);
            bfloat16_t* out_bf16 = static_cast<bfloat16_t*>(output);
            for (size_t i = 0; i < q_count; ++i) {
                float f = half_to_float(out_half[i]);
                out_bf16[i] = float_to_bf16(f);
            }
        }
        
        // Update statistics
        handle->total_calls++;
        handle->total_bytes_processed += workspace.total_size;
        
        std::cout << "✅ [HANDLE] Metal batch attention completed successfully using workspace!" << std::endl;
    }
}

void batch_prefill_attention_unified_f32(
    MetalBatchPrefillHandle* handle,
    void* workspace_buffer,
    size_t workspace_size,
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    int num_qo,
    int head_dim,
    int kv_head_dim,
    int head_size,
    int page_size,
    int num_query_heads,
    int num_kv_heads,
    float scale,
    int num_kv_pages,
    KernelOptimizationLevel opt_level
) {
    @autoreleasepool {
        if (!handle || !handle->device || !handle->pipeline_f32) {
            std::cerr << "❌ [HANDLE] Invalid handle or f32 pipeline not available" << std::endl;
            return;
        }
        
        // Validate workspace
        if (!workspace_buffer || workspace_size == 0) {
            std::cerr << "❌ [HANDLE] Invalid workspace buffer" << std::endl;
            return;
        }
        
        // Calculate workspace layout
        auto workspace = metal_batch_prefill_get_workspace(
            handle, num_qo, head_dim, kv_head_dim, page_size, num_kv_pages
        );
        
        if (workspace_size < workspace.total_size) {
            std::cerr << "❌ [HANDLE] Workspace too small: " << workspace_size 
                      << " < " << workspace.total_size << " required" << std::endl;
            return;
        }
        
        std::cout << "🟢 [HANDLE F32] Processing with workspace: " << workspace.total_size << " bytes" << std::endl;
        
        // Setup workspace pointers
        char* workspace_ptr = static_cast<char*>(workspace_buffer);
        void* params_workspace = workspace_ptr;
        void* output_workspace = workspace_ptr + workspace.params_buffer_size;
        void* index_workspace = workspace_ptr + workspace.params_buffer_size + workspace.output_buffer_size;
        
        // Create Metal buffers as views into workspace
        id<MTLBuffer> q_buf = [handle->device newBufferWithBytesNoCopy:(void*)q_input
                                                                length:num_qo * head_dim * sizeof(float)
                                                               options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        
        size_t kv_cache_size = num_kv_pages * page_size * kv_head_dim * sizeof(float);
        id<MTLBuffer> pk_buf = [handle->device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                                 length:kv_cache_size
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        id<MTLBuffer> pv_buf = [handle->device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                                 length:kv_cache_size
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        
        size_t indptr_size = (num_qo + 1) * sizeof(int32_t);
        id<MTLBuffer> qo_indptr_buf = [handle->device newBufferWithBytesNoCopy:(void*)qo_indptr
                                                                        length:indptr_size
                                                                       options:MTLResourceStorageModeShared
                                                                   deallocator:nil];
        
        id<MTLBuffer> kv_page_indptr_buf = [handle->device newBufferWithBytesNoCopy:(void*)kv_page_indptr
                                                                             length:indptr_size
                                                                            options:MTLResourceStorageModeShared
                                                                        deallocator:nil];
        
        id<MTLBuffer> kv_page_indices_buf = [handle->device newBufferWithBytesNoCopy:(void*)kv_page_indices
                                                                              length:num_kv_pages * sizeof(int32_t)
                                                                             options:MTLResourceStorageModeShared
                                                                         deallocator:nil];
        
        id<MTLBuffer> kv_last_page_lens_buf = [handle->device newBufferWithBytesNoCopy:(void*)kv_last_page_lens
                                                                               length:num_qo * sizeof(int32_t)
                                                                              options:MTLResourceStorageModeShared
                                                                          deallocator:nil];
        
        id<MTLBuffer> out_buf = [handle->device newBufferWithBytesNoCopy:output
                                                                 length:num_qo * head_dim * sizeof(float)
                                                                options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        
        // Create parameter buffer
        struct Params {
            int num_qo;
            int head_dim;
            int kv_head_dim;
            int head_size;
            int page_size;
            int num_query_heads;
            int num_kv_heads;
            float scale;
        };
        
        Params* params = static_cast<Params*>(params_workspace);
        *params = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
        
        id<MTLBuffer> params_buf = [handle->device newBufferWithBytesNoCopy:params_workspace
                                                                     length:workspace.params_buffer_size
                                                                    options:MTLResourceStorageModeShared
                                                               deallocator:nil];
        
        // Select optimal kernel based on problem characteristics and memory constraints  
        int estimated_total_kv_len = (num_kv_pages > 0) ? (num_kv_pages - 1) * page_size + page_size : 0;
        id<MTLComputePipelineState> selected_pipeline = select_f32_kernel(
            handle, opt_level, num_qo, estimated_total_kv_len, head_size, page_size, true); // Enable debug for F32 memory analysis
        
        if (!selected_pipeline) {
            std::cerr << "batch_prefill_attention_unified_f32: No suitable kernel found" << std::endl;
            return;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> cmd = [handle->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        
        [enc setComputePipelineState:selected_pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:pk_buf offset:0 atIndex:1];
        [enc setBuffer:pv_buf offset:0 atIndex:2];
        [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
        [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
        [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
        [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        
        // Dispatch computation
        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        
        // Execute and wait
        [cmd commit];
        [cmd waitUntilCompleted];
        
        // Update handle statistics
        handle->total_calls++;
        handle->total_bytes_processed += workspace.total_size;
        
        std::cout << "✅ [HANDLE F32] Metal batch attention completed successfully using workspace!" << std::endl;
    }
}

// Forward declaration for library loader (legacy support)
static id<MTLLibrary> get_metal_library();

// ============================================================================
// End of new handle-based implementation
// Old functions have been removed - use handle-based API only
// ============================================================================



} // namespace batch_prefill_attention
} // namespace metal