#include "metal_batch_prefill_attention.hpp"
#include "metal_batch_prefill_handle.hpp"
#include "metal_memory_manager.hpp"
#include "metal_common.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <thread>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include "workspace_utils.hpp"

// GPU Configuration Loading
namespace {

    struct GPUConfig {
        std::string gpu_name;
        int max_concurrent_threads;
        int max_buffer_size_mb;
        int max_total_workspace_mb;
        // Threadgroup parameters
        int max_threads_per_threadgroup;
        int max_threadgroups_per_grid;  // We'll use the [0] value from array
        int head_dim_threshold;
        int min_tokens_for_chunking;
        int max_tokens_per_chunk;
        bool enable_adaptive_chunking;
    };

    GPUConfig loadGPUConfiguration(id<MTLDevice> device) {
        GPUConfig config = {};

        // Determine GPU family for configuration selection
        NSString* deviceName = device.name;
        std::string gpu_key = "default";  // Fallback

        if ([deviceName containsString:@"M3"]) {
            if ([deviceName containsString:@"Max"]) gpu_key = "M3_Max";
            else if ([deviceName containsString:@"Pro"]) gpu_key = "M3_Pro";
            else gpu_key = "M3";
        } else if ([deviceName containsString:@"M2"]) {
            if ([deviceName containsString:@"Max"]) gpu_key = "M2_Max";
            else if ([deviceName containsString:@"Pro"]) gpu_key = "M2_Pro";
            else gpu_key = "M2";
        } else if ([deviceName containsString:@"M1"]) {
            if ([deviceName containsString:@"Max"]) gpu_key = "M1_Max";
            else if ([deviceName containsString:@"Pro"]) gpu_key = "M1_Pro";
            else gpu_key = "M1";
        }

        // Try to load configuration from JSON file
        auto workspace_root = workspace_utils::find_workspace_root();
        NSString* configPath = nil;
        if (!workspace_root.empty()) {
            auto config_file_path = workspace_root / "metal-protocol-tests" / "apple_gpu_configs.json";
            configPath = [NSString stringWithUTF8String:config_file_path.c_str()];
        }
        NSData* jsonData = configPath ? [NSData dataWithContentsOfFile:configPath] : nil;

        if (jsonData) {
            NSError* error = nil;
            NSDictionary* configDict = [NSJSONSerialization JSONObjectWithData:jsonData
                                                                       options:NSJSONReadingAllowFragments
                                                                         error:&error];

            if (!error && configDict[@"apple_gpu_configs"]) {
                NSDictionary* gpuConfigs = configDict[@"apple_gpu_configs"];
                NSDictionary* selectedConfig = gpuConfigs[@(gpu_key.c_str())];

                if (!selectedConfig) {
                    selectedConfig = gpuConfigs[@"default"];  // Fallback to default
                    gpu_key = "default";
                }

                if (selectedConfig) {
                    config.gpu_name = std::string([selectedConfig[@"name"] UTF8String]);
                    config.max_concurrent_threads = [selectedConfig[@"max_concurrent_threads"] intValue];
                    config.max_buffer_size_mb = [selectedConfig[@"max_buffer_size_mb"] intValue];
                    config.max_total_workspace_mb = [selectedConfig[@"max_total_workspace_mb"] intValue];
                    // Threadgroup parameters
                    config.max_threads_per_threadgroup = [selectedConfig[@"max_threads_per_threadgroup"] intValue];
                    NSArray* threadgroupsArray = selectedConfig[@"max_threadgroups_per_grid"];
                    if (threadgroupsArray && threadgroupsArray.count > 0) {
                        config.max_threadgroups_per_grid = [threadgroupsArray[0] intValue];  // Use X dimension
                    }

                    // Chunking configuration
                    NSDictionary* chunkingConfig = selectedConfig[@"chunking"];
                    if (chunkingConfig) {
                        config.head_dim_threshold = [chunkingConfig[@"head_dim_threshold"] intValue];
                        config.min_tokens_for_chunking = [chunkingConfig[@"min_tokens_for_chunking"] intValue];
                        config.max_tokens_per_chunk = [chunkingConfig[@"max_tokens_per_chunk"] intValue];
                        config.enable_adaptive_chunking = [chunkingConfig[@"enable_adaptive_chunking"] boolValue];
                    }

                    std::cout << "🔧 [GPU CONFIG] Loaded configuration for " << config.gpu_name
                              << " (" << gpu_key << ")" << std::endl;
                    std::cout << "   📊 Chunking: head_dim_threshold=" << config.head_dim_threshold
                              << ", max_tokens_per_chunk=" << config.max_tokens_per_chunk << std::endl;
                }
            }
        } else {
            std::cout << "⚠️ [GPU CONFIG] Could not load configuration file, using defaults" << std::endl;
        }

        // Fallback to conservative defaults if loading failed
        // TODO: Default should be also coming from the configuration
        if (config.gpu_name.empty()) {
            config.gpu_name = "Unknown Apple Silicon";
            config.max_concurrent_threads = 32768;
            config.max_buffer_size_mb = 64;
            config.max_total_workspace_mb = 200;
            // Threadgroup defaults
            config.max_threads_per_threadgroup = 1024;
            config.max_threadgroups_per_grid = 65535;
            config.head_dim_threshold = 4096;
            config.min_tokens_for_chunking = 256;
            config.max_tokens_per_chunk = 256;
            config.enable_adaptive_chunking = true;
            std::cout << "🔧 [GPU CONFIG] Using conservative defaults for " << config.gpu_name << std::endl;
        }

        return config;
    }
}

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
        @"metal_attention_naive.metal"          // Naive O(n²) attention for comparison
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
    id<MTLFunction> bf16_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_bf16_baseline_kernel"];
    if (bf16_function) {
        handle->pipeline_bf16 = [handle->device newComputePipelineStateWithFunction:bf16_function error:&error];
        if (error) {
            std::cerr << "MetalBatchPrefillHandle: Failed to create BF16 pipeline: "
                      << error.localizedDescription.UTF8String << std::endl;
        }
    }

    id<MTLFunction> f32_function = [handle->library newFunctionWithName:@"batch_prefill_attention_unified_f32_baseline_kernel"];
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

    // Load GPU-specific configuration
    GPUConfig gpuConfig = loadGPUConfiguration(handle->device);
    handle->gpu_config.gpu_name = gpuConfig.gpu_name;
    handle->gpu_config.max_concurrent_threads = gpuConfig.max_concurrent_threads;
    handle->gpu_config.max_buffer_size_mb = gpuConfig.max_buffer_size_mb;
    handle->gpu_config.max_total_workspace_mb = gpuConfig.max_total_workspace_mb;
    // Threadgroup parameters
    handle->gpu_config.max_threads_per_threadgroup = gpuConfig.max_threads_per_threadgroup;
    handle->gpu_config.max_threadgroups_per_grid = gpuConfig.max_threadgroups_per_grid;
    handle->gpu_config.head_dim_threshold = gpuConfig.head_dim_threshold;
    handle->gpu_config.min_tokens_for_chunking = gpuConfig.min_tokens_for_chunking;
    handle->gpu_config.max_tokens_per_chunk = gpuConfig.max_tokens_per_chunk;
    handle->gpu_config.enable_adaptive_chunking = gpuConfig.enable_adaptive_chunking;

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
    std::cout << "DEBUG WORKSPACE: total_size=" << workspace.total_size << " bytes, offset=" << offset << std::endl;
    std::cout << "DEBUG WORKSPACE: q_buffer_size=" << workspace.q_buffer_size
              << ", k_buffer_size=" << workspace.k_buffer_size
              << ", v_buffer_size=" << workspace.v_buffer_size << std::endl;

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
                // Auto selection based on GPU occupancy and memory constraints
                // The optimizations are beneficial across a wide range of problem sizes

                // Optimized AUTO selection with fixed simdgroup kernel
                // Priority 2 (per-head) works well for very small batches with extreme performance needs
                if (num_tokens <= 16 && total_kv_len <= 512 && head_size <= 128) {
                    return handle->pipeline_bf16_per_head ? handle->pipeline_bf16_per_head :
                           (handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline);
                }
                else if (head_size <= 256) { // Use optimized kernel for all supported sizes (MAX_HEAD_DIM=256)
                    return handle->pipeline_bf16_simdgroup ? handle->pipeline_bf16_simdgroup : handle->pipeline_bf16_baseline;
                }
                // Fallback to baseline only if head_size exceeds hardware limits
                else {
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

    // GPU Memory and Buffer Size Validation - Use GPU-specific configuration
    const size_t MAX_BUFFER_SIZE = handle->gpu_config.max_buffer_size_mb * 1024 * 1024;
    const size_t MAX_TOTAL_WORKSPACE = handle->gpu_config.max_total_workspace_mb * 1024 * 1024;

    if (workspace.total_size > MAX_TOTAL_WORKSPACE) {
        std::cerr << "❌ [HANDLE] Workspace size exceeds GPU memory limits: "
                  << (workspace.total_size / 1024 / 1024) << " MB > "
                  << (MAX_TOTAL_WORKSPACE / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   💡 Try reducing batch size, sequence length, or head dimension" << std::endl;
        std::cerr << "   💡 Current params: " << num_qo << " tokens, " << head_size << " head_size" << std::endl;
        return;
    }

    // Check individual buffer sizes - These are the actual problem causing Metal Internal Errors
    size_t max_individual_buffer = std::max({
        workspace.q_buffer_size,
        workspace.k_buffer_size,
        workspace.v_buffer_size,
        workspace.output_buffer_size
    });

    if (max_individual_buffer > MAX_BUFFER_SIZE) {
        std::cerr << "❌ [HANDLE] Individual buffer exceeds Apple Silicon GPU limits:" << std::endl;
        std::cerr << "   Q buffer: " << (workspace.q_buffer_size / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   K buffer: " << (workspace.k_buffer_size / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   V buffer: " << (workspace.v_buffer_size / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   Output buffer: " << (workspace.output_buffer_size / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   💡 Apple Silicon GPU limit: " << (MAX_BUFFER_SIZE / 1024 / 1024) << " MB per buffer" << std::endl;
        std::cerr << "   💡 Largest buffer: " << (max_individual_buffer / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   💡 Reduce: num_qo(" << num_qo << "), head_dim(" << head_dim << "), or head_size(" << head_size << ")" << std::endl;
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

        // BUFFER VALIDATION: Verify each workspace region is accessible
        std::cout << "🔍 [BUFFER CHECK] Validating workspace regions..." << std::endl;
        std::cout << "   Workspace base: " << (void*)workspace_base << std::endl;
        std::cout << "   Q workspace: " << q_workspace << " (size: " << workspace.q_buffer_size << ")" << std::endl;
        std::cout << "   K workspace: " << k_workspace << " (size: " << workspace.k_buffer_size << ")" << std::endl;
        std::cout << "   V workspace: " << v_workspace << " (size: " << workspace.v_buffer_size << ")" << std::endl;
        std::cout << "   Counts: q_count=" << q_count << ", kv_count=" << kv_count << std::endl;

        // Convert BF16 to Half in workspace
        std::cout << "🔄 [DATA CONVERSION] Converting BF16 to Half..." << std::endl;
        std::cout << "   Q input ptr: " << q_input << ", count: " << q_count << std::endl;
        std::cout << "   K cache ptr: " << paged_k_cache << ", count: " << kv_count << std::endl;
        std::cout << "   V cache ptr: " << paged_v_cache << ", count: " << kv_count << std::endl;

        std::vector<uint16_t> q_half_data = convert_bf16_to_half(q_input, q_count);
        std::vector<uint16_t> k_half_data = convert_bf16_to_half(paged_k_cache, kv_count);
        std::vector<uint16_t> v_half_data = convert_bf16_to_half(paged_v_cache, kv_count);

        // Copy converted data to workspace with validation
        std::cout << "🔄 [MEMORY COPY] Copying converted data to workspace..." << std::endl;
        std::cout << "   Q: " << q_half_data.size() << " elements -> " << q_workspace << std::endl;
        std::memcpy(q_workspace, q_half_data.data(), q_half_data.size() * sizeof(uint16_t));

        std::cout << "   K: " << k_half_data.size() << " elements -> " << k_workspace << std::endl;
        std::memcpy(k_workspace, k_half_data.data(), k_half_data.size() * sizeof(uint16_t));

        std::cout << "   V: " << v_half_data.size() << " elements -> " << v_workspace << std::endl;
        std::memcpy(v_workspace, v_half_data.data(), v_half_data.size() * sizeof(uint16_t));

        // Copy index data to workspace
        size_t index_offset = 0;
        size_t qo_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indptr_size = (num_qo + 1) * sizeof(int32_t);
        size_t kv_page_indices_size = num_kv_pages * sizeof(int32_t);
        size_t kv_last_page_lens_size = num_qo * sizeof(int32_t);

        std::cout << "🔄 [INDEX COPY] Copying index data to workspace..." << std::endl;
        std::cout << "   Index workspace: " << index_workspace << ", total buffer size: " << workspace.index_buffer_size << std::endl;
        std::cout << "   Sizes: qo=" << qo_indptr_size << ", kv_page=" << kv_page_indptr_size
                  << ", indices=" << kv_page_indices_size << ", lens=" << kv_last_page_lens_size << std::endl;
        std::cout << "   Total needed: " << (qo_indptr_size + kv_page_indptr_size + kv_page_indices_size + kv_last_page_lens_size) << std::endl;

        // Validate pointers before memcpy
        if (!qo_indptr) {
            std::cerr << "❌ qo_indptr is null!" << std::endl;
        }
        if (!kv_page_indptr) {
            std::cerr << "❌ kv_page_indptr is null!" << std::endl;
        }
        if (!kv_page_indices && num_kv_pages > 0) {
            std::cerr << "❌ kv_page_indices is null but num_kv_pages=" << num_kv_pages << std::endl;
        }
        if (!kv_last_page_lens) {
            std::cerr << "❌ kv_last_page_lens is null!" << std::endl;
        }

        // Test index workspace accessibility
        volatile uint8_t* test_index = (volatile uint8_t*)index_workspace;
        test_index[0] = 0xFF;
        if (workspace.index_buffer_size > 1) {
            test_index[workspace.index_buffer_size-1] = 0xFF;
        }
        std::cout << "   ✅ Index workspace accessible" << std::endl;

        std::cout << "   Copying qo_indptr: " << (void*)qo_indptr << " -> " << ((char*)index_workspace + index_offset) << std::endl;
        std::memcpy((char*)index_workspace + index_offset, qo_indptr, qo_indptr_size);
        index_offset += qo_indptr_size;
        std::cout << "   ✅ qo_indptr copied, new offset: " << index_offset << std::endl;

        std::cout << "   Copying kv_page_indptr: " << (void*)kv_page_indptr << " -> " << ((char*)index_workspace + index_offset) << std::endl;
        std::memcpy((char*)index_workspace + index_offset, kv_page_indptr, kv_page_indptr_size);
        index_offset += kv_page_indptr_size;
        std::cout << "   ✅ kv_page_indptr copied, new offset: " << index_offset << std::endl;

        if (num_kv_pages > 0 && kv_page_indices) {
            std::cout << "   Copying kv_page_indices: " << (void*)kv_page_indices << " -> " << ((char*)index_workspace + index_offset) << std::endl;
            std::memcpy((char*)index_workspace + index_offset, kv_page_indices, kv_page_indices_size);
            index_offset += kv_page_indices_size;
            std::cout << "   ✅ kv_page_indices copied, new offset: " << index_offset << std::endl;
        } else {
            std::cout << "   Skipping kv_page_indices (num_kv_pages=" << num_kv_pages << ")" << std::endl;
        }

        std::cout << "   Copying kv_last_page_lens: " << (void*)kv_last_page_lens << " -> " << ((char*)index_workspace + index_offset) << std::endl;
        std::memcpy((char*)index_workspace + index_offset, kv_last_page_lens, kv_last_page_lens_size);
        std::cout << "   ✅ kv_last_page_lens copied" << std::endl;

        std::cout << "🔄 [METAL BUFFERS] Creating Metal buffer views..." << std::endl;

        // Create Metal buffer views (no allocation!)
        std::cout << "   Creating Q buffer: " << q_workspace << ", size=" << workspace.q_buffer_size << std::endl;
        id<MTLBuffer> q_buf = [handle->device newBufferWithBytesNoCopy:q_workspace
                                                               length:workspace.q_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        if (!q_buf) {
            std::cerr << "❌ Failed to create Q buffer!" << std::endl;
            return;
        }
        std::cout << "   ✅ Q buffer created" << std::endl;

        std::cout << "   Creating K buffer: " << k_workspace << ", size=" << workspace.k_buffer_size << std::endl;
        id<MTLBuffer> k_buf = [handle->device newBufferWithBytesNoCopy:k_workspace
                                                               length:workspace.k_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        if (!k_buf) {
            std::cerr << "❌ Failed to create K buffer!" << std::endl;
            return;
        }
        std::cout << "   ✅ K buffer created" << std::endl;

        std::cout << "   Creating V buffer: " << v_workspace << ", size=" << workspace.v_buffer_size << std::endl;
        id<MTLBuffer> v_buf = [handle->device newBufferWithBytesNoCopy:v_workspace
                                                               length:workspace.v_buffer_size
                                                              options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        if (!v_buf) {
            std::cerr << "❌ Failed to create V buffer!" << std::endl;
            return;
        }
        std::cout << "   ✅ V buffer created" << std::endl;

        std::cout << "   Creating output buffer: " << output_workspace << ", size=" << workspace.output_buffer_size << std::endl;
        id<MTLBuffer> out_buf = [handle->device newBufferWithBytesNoCopy:output_workspace
                                                                 length:workspace.output_buffer_size
                                                                options:MTLResourceStorageModeShared
                                                           deallocator:nil];
        if (!out_buf) {
            std::cerr << "❌ Failed to create output buffer!" << std::endl;
            return;
        }
        std::cout << "   ✅ Output buffer created" << std::endl;

        // OPTIMIZATION: Combine small buffers to reduce buffer object count
        // This reduces Metal Internal Error (0x0E) risk from too many concurrent buffer objects
        size_t combined_small_buffers_size = workspace.index_buffer_size + workspace.params_buffer_size + workspace.debug_buffer_size;
        std::cout << "   Creating combined buffer: " << index_workspace << ", size=" << combined_small_buffers_size << std::endl;
        id<MTLBuffer> combined_buf = [handle->device newBufferWithBytesNoCopy:index_workspace
                                                                       length:combined_small_buffers_size
                                                                      options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
        if (!combined_buf) {
            std::cerr << "❌ Failed to create combined buffer!" << std::endl;
            return;
        }
        std::cout << "   ✅ Combined buffer created" << std::endl;

        // Create parameter buffer data in workspace
        Params* params = static_cast<Params*>(params_workspace);
        *params = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };

        // Select optimal kernel based on problem characteristics and memory constraints
        // Estimate total_kv_len (this is a rough estimate - exact calculation happens in kernel)
        int estimated_total_kv_len = (num_kv_pages > 0) ? (num_kv_pages - 1) * page_size + page_size : 0;
        id<MTLComputePipelineState> selected_pipeline = select_bf16_kernel(
            handle, opt_level, num_qo, estimated_total_kv_len, head_size, page_size, false);

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

        // Set index buffers with proper offsets using combined buffer
        [enc setBuffer:combined_buf offset:0 atIndex:3]; // qo_indptr
        [enc setBuffer:combined_buf offset:qo_indptr_size atIndex:4]; // kv_page_indptr
        [enc setBuffer:combined_buf offset:qo_indptr_size + kv_page_indptr_size atIndex:5]; // kv_page_indices
        [enc setBuffer:combined_buf offset:qo_indptr_size + kv_page_indptr_size + kv_page_indices_size atIndex:6]; // kv_last_page_lens

        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:combined_buf offset:workspace.index_buffer_size atIndex:8]; // params buffer
        [enc setBuffer:combined_buf offset:workspace.index_buffer_size + workspace.params_buffer_size atIndex:9]; // debug buffer

        // ⚡ ADAPTIVE KERNEL PARALLELIZATION - Prevent Metal Internal Error (0x0E)
        // Apple Silicon GPU has limits on total parallel threads: ~65,536 concurrent threads
        // Dynamically adjust parallelization based on workload size to prevent GPU overwhelm

        NSUInteger maxThreadsPerGroup = selected_pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger recommendedThreadsPerGroup = MIN(128, maxThreadsPerGroup);

        // Calculate total work (threads needed)
        NSUInteger totalWork;
        if (selected_pipeline == handle->pipeline_bf16_per_head) {
            totalWork = num_qo * num_query_heads;
        } else {
            totalWork = num_qo;
        }

        // Apple Silicon GPU limits from configuration
        const NSUInteger MAX_CONCURRENT_THREADS = handle->gpu_config.max_concurrent_threads;
        const NSUInteger MAX_THREADGROUPS = handle->gpu_config.max_threadgroups_per_grid;
        const size_t MAX_BUFFER_SIZE = handle->gpu_config.max_buffer_size_mb * 1024 * 1024;

        // Declare parallelization variables outside conditional blocks for retry access
        NSUInteger threadsPerGroup = recommendedThreadsPerGroup;
        NSUInteger threadgroupCount = totalWork;

        // ⚡ CONFIGURATION-BASED ADAPTIVE PARALLELIZATION - Prevent Metal Internal Error (0x0E)
        // Use GPU configuration limits instead of hardcoded values
        // Dynamically adjust parallelization based on workload size to prevent GPU overwhelm

        if (totalWork * threadsPerGroup > MAX_CONCURRENT_THREADS) {
            // 🔧 WORKLOAD TOO LARGE: Reduce ONLY threads per threadgroup (keep all threadgroups)
            // Each threadgroup MUST process one query token (qo_idx = tgid.x in kernel)
            std::cout << "🔧 [CONFIG] Large workload detected: " << totalWork << " x " << threadsPerGroup
                      << " = " << (totalWork * threadsPerGroup) << " threads (>" << MAX_CONCURRENT_THREADS << ")" << std::endl;

            // CRITICAL: Metal attention kernels are hardcoded with TGP_SIZE = 128
            // Reducing threadsPerGroup breaks kernel correctness assumptions
            // Instead, we must process in smaller batches to stay within GPU limits

            NSUInteger maxBatchSize = MAX_CONCURRENT_THREADS / recommendedThreadsPerGroup;
            if (maxBatchSize < 1) maxBatchSize = 1;

            if (totalWork <= maxBatchSize) {
                // Can process all tokens in one batch
                threadsPerGroup = recommendedThreadsPerGroup;  // Keep 128
                threadgroupCount = totalWork;
            } else {
                // Process in smaller batches - this would require kernel chunking
                // For now, use original settings and accept potential Metal errors for very large workloads
                threadsPerGroup = recommendedThreadsPerGroup;  // Keep 128 for correctness
                threadgroupCount = totalWork;

                std::cout << "   ⚠️  WARNING: Large workload may cause Metal Internal Error" << std::endl;
                std::cout << "   💡 Consider reducing batch_size or sequence_length in model config" << std::endl;
            }

            std::cout << "   ✅ Configuration-based sizing: " << threadsPerGroup << " threads/group, "
                      << threadgroupCount << " groups (total: " << (threadsPerGroup * threadgroupCount) << " threads)" << std::endl;
        } else {
            // 🚀 WORKLOAD MANAGEABLE: Use optimal parallelization
            std::cout << "🚀 [CONFIG] Optimal workload: " << threadsPerGroup << " threads/group, "
                      << threadgroupCount << " groups (total: " << (threadsPerGroup * threadgroupCount) << " threads)" << std::endl;
        }

        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroupCount, 1, 1);

        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];

        // Error handling with detailed logging
        using namespace MetalErrorHandling;

        // Log GPU memory status before execution
        GPUMemoryInfo memInfo = getGPUMemoryInfo(handle->device);
        std::cout << "GPU Memory Status: " << memInfo.current_allocated / (1024 * 1024) << " MB / "
                  << memInfo.recommended_max / (1024 * 1024) << " MB ("
                  << static_cast<int>(memInfo.usage_ratio * 100) << "%)" << std::endl;

        int retries = 3;
        NSError* cmdError = nil;
        bool success = false;

        while (retries > 0) {
            [cmd commit];
            [cmd waitUntilCompleted];
            cmdError = cmd.error;
            if (!cmdError) {
                success = true;
                break; // Success
            }

            // Enhanced error logging
            std::cerr << "❌ Metal command buffer failed (attempt " << (4-retries) << "/3):" << std::endl;
            std::cerr << "   Error: " << cmdError.localizedDescription.UTF8String << std::endl;
            std::cerr << "   Code: " << cmdError.code << std::endl;
            std::cerr << "   Domain: " << cmdError.domain.UTF8String << std::endl;
            std::cerr << "   Parameters: num_qo=" << num_qo << ", head_dim=" << head_dim << ", head_size=" << head_size
                      << ", workspace=" << workspace.total_size / (1024 * 1024) << "MB" << std::endl;

            retries--;
            if (retries > 0) {
                std::cout << "🔄 Retrying... (" << retries << " attempts remaining)" << std::endl;

                // 🔧 RESOURCE CLEANUP: Proper Metal resource management to prevent buildup
                // Wait a bit to let GPU recover from potential resource contention
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                // Create fresh command buffer for retry - old one is auto-released by ARC
                cmd = [handle->commandQueue commandBuffer];
                enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:selected_pipeline];
                [enc setBuffer:q_buf offset:0 atIndex:0];
                [enc setBuffer:k_buf offset:0 atIndex:1];
                [enc setBuffer:v_buf offset:0 atIndex:2];
                [enc setBuffer:combined_buf offset:0 atIndex:3]; // qo_indptr
                [enc setBuffer:combined_buf offset:qo_indptr_size atIndex:4]; // kv_page_indptr
                [enc setBuffer:combined_buf offset:qo_indptr_size + kv_page_indptr_size atIndex:5]; // kv_page_indices
                [enc setBuffer:combined_buf offset:qo_indptr_size + kv_page_indptr_size + kv_page_indices_size atIndex:6]; // kv_last_page_lens
                [enc setBuffer:out_buf offset:0 atIndex:7];
                [enc setBuffer:combined_buf offset:workspace.index_buffer_size atIndex:8]; // params buffer
                [enc setBuffer:combined_buf offset:workspace.index_buffer_size + workspace.params_buffer_size atIndex:9]; // debug buffer
                // Use same adaptive parallelization for retry
                MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
                MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroupCount, 1, 1);
                [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [enc endEncoding];
            } else {
                // Final failure with comprehensive error report
                std::cerr << "💥 FINAL ERROR: Metal command buffer failed after 3 retries" << std::endl;
                std::cerr << "   Last error: " << cmdError.localizedDescription.UTF8String << std::endl;
                std::cerr << "   Error code: " << cmdError.code << std::endl;

                // Provide helpful suggestions based on error type
                if (cmdError.code == 14) { // Internal Error
                    std::cerr << "   💡 This is an Internal Metal error - may be caused by GPU memory pressure" << std::endl;
                    std::cerr << "      Try reducing batch size or sequence length" << std::endl;
                } else if (cmdError.code == 5) { // Innocent Victim
                    std::cerr << "   💡 GPU error recovery occurred - another process may have caused GPU instability" << std::endl;
                    std::cerr << "      Try reducing GPU workload or restarting the application" << std::endl;
                }

                std::cerr << "   Current GPU usage: " << static_cast<int>(memInfo.usage_ratio * 100) << "%" << std::endl;

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

        // GPU Memory and Buffer Size Validation - Use GPU-specific configuration
        const size_t MAX_BUFFER_SIZE = handle->gpu_config.max_buffer_size_mb * 1024 * 1024;
        const size_t MAX_TOTAL_WORKSPACE = handle->gpu_config.max_total_workspace_mb * 1024 * 1024;

        if (workspace.total_size > MAX_TOTAL_WORKSPACE) {
            std::cerr << "❌ [HANDLE F32] Workspace size exceeds GPU memory limits: "
                      << (workspace.total_size / 1024 / 1024) << " MB > "
                      << (MAX_TOTAL_WORKSPACE / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   💡 Try reducing batch size, sequence length, or head dimension" << std::endl;
            std::cerr << "   💡 Current params: " << num_qo << " tokens, " << head_size << " head_size" << std::endl;
            return;
        }

        // Check individual buffer sizes for f32 (2x larger than bf16)
        size_t q_buffer_size = num_qo * head_dim * sizeof(float);
        size_t kv_total_size = num_kv_pages * page_size * kv_head_dim * sizeof(float);
        size_t output_buffer_size = num_qo * head_dim * sizeof(float);

        size_t max_buffer = std::max({q_buffer_size, kv_total_size, output_buffer_size});

        if (max_buffer > MAX_BUFFER_SIZE) {
            std::cerr << "❌ [HANDLE F32] Individual buffer exceeds Apple Silicon GPU limits:" << std::endl;
            std::cerr << "   Q buffer: " << (q_buffer_size / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   KV cache: " << (kv_total_size / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Output buffer: " << (output_buffer_size / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   💡 Apple Silicon GPU limit: " << (MAX_BUFFER_SIZE / 1024 / 1024) << " MB per buffer" << std::endl;
            std::cerr << "   💡 Largest buffer: " << (max_buffer / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   💡 F32 uses 2x memory vs BF16 - consider using BF16 instead" << std::endl;
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

        // ⚡ ADAPTIVE KERNEL PARALLELIZATION - Apply same logic for F32
        NSUInteger maxThreadsPerGroup = selected_pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger recommendedThreadsPerGroup = MIN(128, maxThreadsPerGroup);
        NSUInteger totalWork = num_qo;

        const NSUInteger MAX_CONCURRENT_THREADS = handle->gpu_config.max_concurrent_threads;

        NSUInteger threadsPerGroup;
        NSUInteger threadgroupCount;

        if (totalWork * recommendedThreadsPerGroup > MAX_CONCURRENT_THREADS) {
            // F32 kernels may be more flexible, but apply same logic for consistency
            NSUInteger maxBatchSize = MAX_CONCURRENT_THREADS / recommendedThreadsPerGroup;
            if (maxBatchSize < 1) maxBatchSize = 1;

            if (totalWork <= maxBatchSize) {
                threadsPerGroup = recommendedThreadsPerGroup;
                threadgroupCount = totalWork;
            } else {
                // Keep original settings for correctness
                threadsPerGroup = recommendedThreadsPerGroup;
                threadgroupCount = totalWork;
                std::cout << "   ⚠️  WARNING: Large F32 workload may cause Metal Internal Error" << std::endl;
            }
            std::cout << "🔧 [ADAPTIVE F32] Reducing parallelization: " << threadsPerGroup
                      << " threads/group, " << threadgroupCount << " groups" << std::endl;
        } else {
            threadsPerGroup = recommendedThreadsPerGroup;
            threadgroupCount = totalWork;
            std::cout << "🚀 [ADAPTIVE F32] Optimal parallelization: " << threadsPerGroup
                      << " threads/group, " << threadgroupCount << " groups" << std::endl;
        }

        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroupCount, 1, 1);
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

// ============================================================================
// Diagnostic function for buffer validation
// ============================================================================

int batch_prefill_attention_diagnostic(
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
    float* debug_output,
    int debug_size
) {
    @autoreleasepool {
        if (!handle || debug_size < 20) {
            std::cerr << "❌ [DIAGNOSTIC] Invalid handle or debug buffer too small" << std::endl;
            return -1;
        }

        std::cout << "🔍 [DIAGNOSTIC] Starting buffer accessibility test..." << std::endl;

        // Initialize debug output
        for (int i = 0; i < debug_size; ++i) {
            debug_output[i] = -99.0f; // Uninitialized marker
        }

        // Create command buffer
        id<MTLCommandBuffer> cmd = [handle->commandQueue commandBuffer];
        cmd.label = @"BufferDiagnosticTest";

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        enc.label = @"BufferDiagnosticEncoder";

        // Use baseline pipeline for diagnostics
        [enc setComputePipelineState:handle->pipeline_bf16_baseline];

        // Create debug buffer for GPU output
        id<MTLBuffer> debug_buffer = [handle->device newBufferWithBytes:debug_output
                                                                length:debug_size * sizeof(float)
                                                               options:MTLResourceStorageModeShared];

        // Set all buffers exactly as they would be in normal operation
        size_t kv_cache_size = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim * sizeof(uint16_t);
        [enc setBuffer:[handle->device newBufferWithBytes:q_input length:num_qo * head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:0];  // q_input
        [enc setBuffer:[handle->device newBufferWithBytes:paged_k_cache length:kv_cache_size options:MTLResourceStorageModeShared]
                offset:0 atIndex:1];  // paged_k_cache
        [enc setBuffer:[handle->device newBufferWithBytes:paged_v_cache length:kv_cache_size options:MTLResourceStorageModeShared]
                offset:0 atIndex:2];  // paged_v_cache
        [enc setBuffer:[handle->device newBufferWithBytes:qo_indptr length:(num_qo + 1) * sizeof(int32_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:3];  // qo_indptr
        [enc setBuffer:[handle->device newBufferWithBytes:kv_page_indptr length:(num_qo + 1) * sizeof(int32_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:4];  // kv_page_indptr
        [enc setBuffer:[handle->device newBufferWithBytes:kv_page_indices length:128 * sizeof(int32_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:5];  // kv_page_indices
        [enc setBuffer:[handle->device newBufferWithBytes:kv_last_page_lens length:num_qo * sizeof(int32_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:6];  // kv_last_page_lens
        [enc setBuffer:[handle->device newBufferWithBytes:output length:num_qo * head_dim * sizeof(uint16_t) options:MTLResourceStorageModeShared]
                offset:0 atIndex:7];  // output

        // Create params struct
        Params params = {
            .num_qo = num_qo,
            .head_dim = head_dim,
            .kv_head_dim = kv_head_dim,
            .head_size = head_size,
            .page_size = page_size,
            .num_query_heads = num_query_heads,
            .num_kv_heads = num_kv_heads,
            .scale = scale
        };

        [enc setBytes:&params length:sizeof(params) atIndex:8];  // params
        [enc setBuffer:debug_buffer offset:0 atIndex:9];         // debug_out

        // Dispatch minimal grid - just one threadgroup with one thread
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];

        std::cout << "🚀 [DIAGNOSTIC] Executing diagnostic kernel..." << std::endl;

        // Execute and wait
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "❌ [DIAGNOSTIC] Command buffer failed with status: " << (int)cmd.status << std::endl;
            if (cmd.error) {
                NSLog(@"Command buffer error: %@", cmd.error.localizedDescription);
            }
            return -2;
        }

        // Copy results back
        memcpy(debug_output, debug_buffer.contents, debug_size * sizeof(float));

        std::cout << "✅ [DIAGNOSTIC] Kernel completed, analyzing results..." << std::endl;

        // Analyze diagnostic results
        for (int i = 0; i < debug_size && i < 12; ++i) {
            float value = debug_output[i];
            if (value == -99.0f) {
                std::cout << "   debug[" << i << "] = " << value << " (uninitialized)" << std::endl;
            } else if (value == -1.0f) {
                std::cout << "   debug[" << i << "] = " << value << " (not tested)" << std::endl;
            } else if (value >= 100.0f) {
                std::cout << "   debug[" << i << "] = " << value << " ✅" << std::endl;
            } else {
                std::cout << "   debug[" << i << "] = " << value << " (unknown)" << std::endl;
            }
        }

        if (debug_output[11] == 9999.0f) {
            std::cout << "🎉 [DIAGNOSTIC] All buffer tests completed successfully!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ [DIAGNOSTIC] Test failed or incomplete" << std::endl;
            return -3;
        }
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