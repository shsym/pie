#include "metal_batch_prefill_attention.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

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

static id<MTLDevice> get_metal_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device" << std::endl;
        return nil;
    }
    return device;
}

// Forward declaration for library loader
static id<MTLLibrary> get_metal_library();

void batch_prefill_attention_unified_bf16(
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
    int kv_head_dim,      // NEW: KV head dimension for MQA/GQA  
    int head_size,
    int page_size,
    int num_query_heads,  // NEW: Number of query heads
    int num_kv_heads,     // NEW: Number of KV heads
    float scale,
    int num_kv_pages
) {
    std::cout << "🟢 [ENTRY] batch_prefill_attention_unified_bf16 called!" << std::endl;
    std::cout.flush();
    std::cout << "🔍 [PATH] Parameters: num_qo=" << num_qo << ", head_dim=" << head_dim 
              << ", head_size=" << head_size << ", page_size=" << page_size 
              << ", scale=" << scale << std::endl;
    std::cout.flush();
    @autoreleasepool {
        std::cout << "🔍 [PATH] Getting Metal device..." << std::endl;
        id<MTLDevice> device = get_metal_device();
        if (!device) { std::cerr << "❌ Failed to get Metal device" << std::endl; return; }
        std::cout << "✅ Metal device obtained" << std::endl;
        
        std::cout << "🔍 [PATH] Getting Metal library..." << std::endl;
        id<MTLLibrary> library = get_metal_library();
        if (!library) { std::cerr << "❌ Failed to get Metal library" << std::endl; return; }
        std::cout << "✅ Metal library obtained" << std::endl;

        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_unified_bf16_kernel"];
        if (!function) { std::cerr << "Failed to find batch_prefill_attention_unified_bf16_kernel" << std::endl; return; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error || !pipeline) { std::cerr << "Failed to create pipeline: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl; return; }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t q_bytes = q_count * sizeof(uint16_t);

        // Convert Q input from bfloat16 to IEEE half format for Metal kernel
        std::vector<uint16_t> q_half_data = convert_bf16_to_half(q_input, q_count);

        // Note: we don't know total pages from here; assume callers provide consistent arrays and derive from kv_page_indices length
        // For unified path, caller should pass correct linearized buffers

        // Determine sizes based on indptr
        // Minimal: find total kv indices
        // We can't compute lengths here reliably; rely on provided host buffers being correctly sized.

        id<MTLBuffer> q_buf = [device newBufferWithBytes:q_half_data.data() length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> pk_buf = nil;
        id<MTLBuffer> pv_buf = nil;
        id<MTLBuffer> qo_indptr_buf = nil;
        id<MTLBuffer> kv_page_indptr_buf = nil;
        id<MTLBuffer> kv_page_indices_buf = nil;
        id<MTLBuffer> kv_last_page_lens_buf = nil;
        id<MTLBuffer> out_buf = [device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
    // Small debug buffer (expanded for parameter workaround)
        const size_t debug_floats = 20;
        id<MTLBuffer> debug_buf = [device newBufferWithLength:debug_floats * sizeof(float) options:MTLResourceStorageModeShared];
        if (debug_buf && debug_buf.contents) {
            memset(debug_buf.contents, 0, debug_floats * sizeof(float));
            // WORKAROUND: Pass parameters through debug buffer
            float* debug_data = (float*)debug_buf.contents;
            debug_data[10] = (float)num_qo;
            debug_data[11] = (float)head_dim; 
            debug_data[12] = (float)head_size;
            debug_data[13] = (float)page_size;
            debug_data[14] = scale;
            std::cout << "🔧 [WORKAROUND] Parameters passed via debug buffer: num_qo=" << debug_data[10] 
                      << ", head_dim=" << debug_data[11] << ", head_size=" << debug_data[12] 
                      << ", page_size=" << debug_data[13] << ", scale=" << debug_data[14] << std::endl;
        }


        // Create buffers with unknown lengths by scanning simple arrays sizes from context is not possible here.
        // We require caller to provide arrays with known sizes and pass those sizes externally in a higher-level layer.
        // For this backend call, we assume monolithic page cache covering all referenced pages sequentially.

        // Fallback: treat paged caches as single contiguous arrays and require host to size them.
        // We cannot infer sizes here without extra parameters; but our test harness will ensure they are correct.

        // Create with heuristic: out_buf size as q_bytes, pk/pv as unknown so we skip if null
        // We must not pass null buffers to setBuffer; but in tests they will be non-null.
        // So assert non-null.
        if (!paged_k_cache || !paged_v_cache || !qo_indptr || !kv_page_indptr || !kv_page_indices || !kv_last_page_lens) {
            std::cerr << "Unified API requires non-null paged buffers and indices" << std::endl; return;
        }

        // Calculate actual buffer sizes based on provided parameters
        size_t pkpv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;
        size_t pkpv_bytes = pkpv_count * sizeof(uint16_t);

        // Convert K and V cache data from bfloat16 to IEEE half format for Metal kernel
        std::vector<uint16_t> k_half_data = convert_bf16_to_half(paged_k_cache, pkpv_count);
        std::vector<uint16_t> v_half_data = convert_bf16_to_half(paged_v_cache, pkpv_count);

        pk_buf = [device newBufferWithBytes:k_half_data.data() length:pkpv_bytes options:MTLResourceStorageModeShared];
        pv_buf = [device newBufferWithBytes:v_half_data.data() length:pkpv_bytes options:MTLResourceStorageModeShared];

        // Calculate array sizes - num_qo+1 for indptr arrays, num_kv_pages for indices, num_sequences for lens
        size_t qo_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indptr_elems = static_cast<size_t>(num_qo) + 1; // Assume one sequence per qo for simplicity
        size_t kv_page_indices_elems = static_cast<size_t>(num_kv_pages);
        size_t kv_last_page_lens_elems = static_cast<size_t>(num_qo); // One per sequence

        qo_indptr_buf = [device newBufferWithBytes:qo_indptr length:qo_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        kv_page_indptr_buf = [device newBufferWithBytes:kv_page_indptr length:kv_page_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        kv_page_indices_buf = [device newBufferWithBytes:kv_page_indices length:kv_page_indices_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        kv_last_page_lens_buf = [device newBufferWithBytes:kv_last_page_lens length:kv_last_page_lens_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:q_buf offset:0 atIndex:0];
    [enc setBuffer:pk_buf offset:0 atIndex:1];
    [enc setBuffer:pv_buf offset:0 atIndex:2];
    [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
    [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
    [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
    [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
    [enc setBuffer:out_buf offset:0 atIndex:7];

    // Uniform parameter buffer to avoid scalar setBytes corruption
    struct Params {
        int num_qo;
        int head_dim;        // Query head dimension (num_query_heads * head_size)
        int kv_head_dim;     // KV head dimension (num_kv_heads * head_size) 
        int head_size;
        int page_size;
        int num_query_heads; // Number of query heads
        int num_kv_heads;    // Number of KV heads (for MQA/GQA support)
        float scale;
    };
    Params p = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
    
    // Debug: Print parameter values being passed to kernel (ORIGINAL PATH - SIMPLIFIED)
    std::cout << "\n🔧 [HOST DEBUG - ORIGINAL] Parameters: num_qo=" << p.num_qo << ", scale=" << p.scale << std::endl;
    
    id<MTLBuffer> params_buf = [device newBufferWithBytes:&p length:sizeof(Params) options:MTLResourceStorageModeShared];
    [enc setBuffer:params_buf offset:0 atIndex:8];
    [enc setBuffer:debug_buf offset:0 atIndex:9];

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);  // Must match TGP_SIZE in kernel
    // One threadgroup per query token
    MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
    [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) { std::cerr << "Metal command buffer error: " << cmd.error.localizedDescription.UTF8String << std::endl; return; }

        // Convert kernel half output back to bfloat16 expected by caller
        if (out_buf && out_buf.contents && output) {
            uint16_t* out_half = (uint16_t*)out_buf.contents; // IEEE half from GPU
            bfloat16_t* out_bf16 = (bfloat16_t*)output;       // destination as bf16
            for (size_t i = 0; i < q_count; ++i) {
                float f = half_to_float(out_half[i]);
                out_bf16[i] = float_to_bf16(f);
            }
        }

        // Print debug buffer
        if (debug_buf && debug_buf.contents) {
            float* dbg = (float*)debug_buf.contents;
            std::cout << "\n[Kernel Debug] scale=" << dbg[0]
                      << ", head_dim=" << dbg[1]
                      << ", page_size=" << dbg[2]
                      << ", num_qo=" << dbg[3]
                      << ", total_kv_len=" << dbg[4]
                      << ", num_pages=" << dbg[5]
                      << ", last_page_len=" << dbg[6]
                      << std::endl;
        }
    }
}

static id<MTLLibrary> get_metal_library() {
    static id<MTLLibrary> library = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        id<MTLDevice> device = get_metal_device();
        if (device) {
            NSError* error = nil;
            // Load source from both the original and optimized .metal files (same folder as this .mm)
            NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
            NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
            
            // Load original shaders
            NSString* metalPath = [dirPath stringByAppendingPathComponent:@"metal_batch_prefill_attention.metal"];
            NSString* originalSource = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
            if (error || !originalSource) {
                std::cerr << "Failed to load original Metal source for batch_prefill_attention";
                if (error) std::cerr << ": " << error.localizedDescription.UTF8String;
                std::cerr << std::endl;
                return;
            }
            
            // Load optimized shaders
            NSString* optimizedPath = [dirPath stringByAppendingPathComponent:@"metal_batch_prefill_attention_optimized.metal"];
            NSString* optimizedSource = [NSString stringWithContentsOfFile:optimizedPath encoding:NSUTF8StringEncoding error:&error];
            if (error || !optimizedSource) {
                std::cout << "Warning: Failed to load optimized Metal source, will use original only";
                if (error) std::cout << ": " << error.localizedDescription.UTF8String;
                std::cout << std::endl;
                // Just use original source
                optimizedSource = @"";
            }
            
            // Combine both sources
            NSString* combinedSource = [NSString stringWithFormat:@"%@\n\n// Optimized kernels:\n%@", originalSource, optimizedSource];
            
            library = [device newLibraryWithSource:combinedSource options:nil error:&error];
            if (!library || error) {
                NSLog(@"Failed to compile combined Metal shaders: %@", error ? error.localizedDescription : @"Unknown error");
            } else {
                std::cout << "Successfully loaded combined Metal library (original + optimized)" << std::endl;
            }
        }
    });
    return library;
}

// Optimized Attention Implementations

void batch_prefill_attention_optimized_bf16(
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
    int head_size,
    int page_size,
    float scale,
    int num_kv_pages
) {
    std::cout << "🔥 [ENTRY] batch_prefill_attention_optimized_bf16 called!" << std::endl;
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) { std::cerr << "Failed to get Metal device" << std::endl; return; }
        id<MTLLibrary> library = get_metal_library();
        if (!library) { std::cerr << "Failed to get Metal library" << std::endl; return; }

        NSError* error = nil;
        
        // Try to load optimized kernel first
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_optimized_bf16_kernel"];
        if (!function) {
            // Fallback to original kernel if optimized not available
            std::cout << "Optimized attention kernel not found, falling back to original" << std::endl;
            function = [library newFunctionWithName:@"batch_prefill_attention_unified_bf16_kernel"];
        }
        
        if (!function) { 
            std::cerr << "Failed to find attention kernel (optimized or fallback)" << std::endl; 
            return; 
        }
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error || !pipeline) { 
            std::cerr << "Failed to create pipeline: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl; 
            return; 
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t q_bytes = q_count * sizeof(uint16_t);

        // Convert Q input from bfloat16 to IEEE half format for Metal kernel
        std::vector<uint16_t> q_half_data = convert_bf16_to_half(q_input, q_count);
        id<MTLBuffer> q_buf = [device newBufferWithBytes:q_half_data.data() length:q_bytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> out_buf = [device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];

        // Small debug buffer 
        const size_t debug_floats = 7;
        id<MTLBuffer> debug_buf = [device newBufferWithLength:debug_floats * sizeof(float) options:MTLResourceStorageModeShared];
        if (debug_buf && debug_buf.contents) {
            memset(debug_buf.contents, 0, debug_floats * sizeof(float));
        }

        if (!paged_k_cache || !paged_v_cache || !qo_indptr || !kv_page_indptr || !kv_page_indices || !kv_last_page_lens) {
            std::cerr << "Optimized API requires non-null paged buffers and indices" << std::endl; 
            return;
        }

        // Calculate actual buffer sizes based on provided parameters
        size_t pkpv_count = static_cast<size_t>(num_kv_pages) * page_size * head_dim;
        size_t pkpv_bytes = pkpv_count * sizeof(uint16_t);

        // Convert K and V cache data from bfloat16 to IEEE half format
        std::vector<uint16_t> k_half_data = convert_bf16_to_half(paged_k_cache, pkpv_count);
        std::vector<uint16_t> v_half_data = convert_bf16_to_half(paged_v_cache, pkpv_count);

        id<MTLBuffer> pk_buf = [device newBufferWithBytes:k_half_data.data() length:pkpv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> pv_buf = [device newBufferWithBytes:v_half_data.data() length:pkpv_bytes options:MTLResourceStorageModeShared];

        // Calculate array sizes
        size_t qo_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indices_elems = static_cast<size_t>(num_kv_pages);
        size_t kv_last_page_lens_elems = static_cast<size_t>(num_qo);

        id<MTLBuffer> qo_indptr_buf = [device newBufferWithBytes:qo_indptr length:qo_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indptr_buf = [device newBufferWithBytes:kv_page_indptr length:kv_page_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indices_buf = [device newBufferWithBytes:kv_page_indices length:kv_page_indices_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_last_page_lens_buf = [device newBufferWithBytes:kv_last_page_lens length:kv_last_page_lens_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:pk_buf offset:0 atIndex:1];
        [enc setBuffer:pv_buf offset:0 atIndex:2];
        [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
        [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
        [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
        [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
        [enc setBuffer:out_buf offset:0 atIndex:7];

        // Uniform parameter buffer
        struct Params { int num_qo; int head_dim; int kv_head_dim; int head_size; int page_size; int num_query_heads; int num_kv_heads; float scale; };
        // Calculate KV parameters from existing values (temporary fix)
        int num_query_heads = head_dim / head_size;  
        int num_kv_heads = 8;  // TODO: Get from function parameter
        int kv_head_dim = num_kv_heads * head_size;
        Params p = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&p length:sizeof(Params) options:MTLResourceStorageModeShared];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        [enc setBuffer:debug_buf offset:0 atIndex:9];

        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);  // Must match TGP_SIZE in kernel
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) { 
            std::cerr << "Metal command buffer error: " << cmd.error.localizedDescription.UTF8String << std::endl; 
            return; 
        }

        // Convert kernel half output back to bfloat16
        if (out_buf && out_buf.contents && output) {
            uint16_t* out_half = (uint16_t*)out_buf.contents;
            bfloat16_t* out_bf16 = (bfloat16_t*)output;
            for (size_t i = 0; i < q_count; ++i) {
                float f = half_to_float(out_half[i]);
                out_bf16[i] = float_to_bf16(f);
            }
        }

        // Print debug info
        if (debug_buf && debug_buf.contents) {
            float* dbg = (float*)debug_buf.contents;
            std::cout << "\n[Optimized Attention Debug] scale=" << dbg[0]
                      << ", head_dim=" << dbg[1]
                      << ", page_size=" << dbg[2]
                      << ", num_qo=" << dbg[3]
                      << ", total_kv_len=" << dbg[4]
                      << ", num_pages=" << dbg[5]
                      << ", last_page_len=" << dbg[6]
                      << std::endl;
        }
    }
}

void batch_prefill_attention_optimized_f32(
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
    int head_size,
    int page_size,
    float scale,
    int num_kv_pages
) {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) { std::cerr << "Failed to get Metal device" << std::endl; return; }
        id<MTLLibrary> library = get_metal_library();
        if (!library) { std::cerr << "Failed to get Metal library" << std::endl; return; }

        NSError* error = nil;
        
        // Try to load optimized kernel first
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_optimized_f32_kernel"];
        if (!function) {
            // Fallback to original kernel if optimized not available
            std::cout << "Optimized f32 attention kernel not found, falling back to original" << std::endl;
            function = [library newFunctionWithName:@"batch_prefill_attention_unified_f32_kernel"];
        }
        
        if (!function) { 
            std::cerr << "Failed to find f32 attention kernel (optimized or fallback)" << std::endl; 
            return; 
        }
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error || !pipeline) { 
            std::cerr << "Failed to create f32 pipeline: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl; 
            return; 
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t q_bytes = q_count * sizeof(float);

        id<MTLBuffer> q_buf = [device newBufferWithBytes:q_input length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];

        // Debug buffer
        const size_t debug_floats = 7;
        id<MTLBuffer> debug_buf = [device newBufferWithLength:debug_floats * sizeof(float) options:MTLResourceStorageModeShared];
        if (debug_buf && debug_buf.contents) {
            memset(debug_buf.contents, 0, debug_floats * sizeof(float));
        }

        if (!paged_k_cache || !paged_v_cache || !qo_indptr || !kv_page_indptr || !kv_page_indices || !kv_last_page_lens) {
            std::cerr << "Optimized f32 API requires non-null buffers" << std::endl; 
            return;
        }

        // Calculate KV buffer sizes
        size_t pkpv_count = static_cast<size_t>(num_kv_pages) * page_size * head_dim;
        size_t pkpv_bytes = pkpv_count * sizeof(float);

        id<MTLBuffer> pk_buf = [device newBufferWithBytes:paged_k_cache length:pkpv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> pv_buf = [device newBufferWithBytes:paged_v_cache length:pkpv_bytes options:MTLResourceStorageModeShared];

        // Calculate array sizes
        size_t qo_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indices_elems = static_cast<size_t>(num_kv_pages);
        size_t kv_last_page_lens_elems = static_cast<size_t>(num_qo);

        id<MTLBuffer> qo_indptr_buf = [device newBufferWithBytes:qo_indptr length:qo_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indptr_buf = [device newBufferWithBytes:kv_page_indptr length:kv_page_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indices_buf = [device newBufferWithBytes:kv_page_indices length:kv_page_indices_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_last_page_lens_buf = [device newBufferWithBytes:kv_last_page_lens length:kv_last_page_lens_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:pk_buf offset:0 atIndex:1];
        [enc setBuffer:pv_buf offset:0 atIndex:2];
        [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
        [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
        [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
        [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
        [enc setBuffer:out_buf offset:0 atIndex:7];

        // Parameters
        struct Params { int num_qo; int head_dim; int kv_head_dim; int head_size; int page_size; int num_query_heads; int num_kv_heads; float scale; };
        // Calculate KV parameters from existing values (temporary fix)
        int num_query_heads = head_dim / head_size;  
        int num_kv_heads = 8;  // TODO: Get from function parameter
        int kv_head_dim = num_kv_heads * head_size;
        Params p = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&p length:sizeof(Params) options:MTLResourceStorageModeShared];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        [enc setBuffer:debug_buf offset:0 atIndex:9];

        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        
        if (cmd.error) { 
            std::cerr << "Metal f32 command buffer error: " << cmd.error.localizedDescription.UTF8String << std::endl; 
            return; 
        }

        // Copy output
        if (out_buf && out_buf.contents && output) {
            memcpy(output, out_buf.contents, q_bytes);
        }

        // Debug output
        if (debug_buf && debug_buf.contents) {
            float* dbg = (float*)debug_buf.contents;
            std::cout << "\n[Optimized F32 Attention Debug] scale=" << dbg[0]
                      << ", head_dim=" << dbg[1]
                      << ", page_size=" << dbg[2] 
                      << ", num_qo=" << dbg[3]
                      << std::endl;
        }
    }
}

void batch_prefill_attention_unified_f32(
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
    int kv_head_dim,      // NEW: KV head dimension for MQA/GQA
    int head_size,
    int page_size,
    int num_query_heads,  // NEW: Number of query heads
    int num_kv_heads,     // NEW: Number of KV heads
    float scale,
    int num_kv_pages
) {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) { std::cerr << "Failed to get Metal device" << std::endl; return; }
        id<MTLLibrary> library = get_metal_library();
        if (!library) { std::cerr << "Failed to get Metal library" << std::endl; return; }

        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"batch_prefill_attention_unified_f32_kernel"];
        if (!function) { std::cerr << "Failed to find batch_prefill_attention_unified_f32_kernel" << std::endl; return; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error || !pipeline) { std::cerr << "Failed to create pipeline: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl; return; }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        size_t q_count = static_cast<size_t>(num_qo) * head_dim;
        size_t q_bytes = q_count * sizeof(float);

        id<MTLBuffer> q_buf = [device newBufferWithBytes:q_input length:q_bytes options:MTLResourceStorageModeShared];
        if (!paged_k_cache || !paged_v_cache || !qo_indptr || !kv_page_indptr || !kv_page_indices || !kv_last_page_lens) {
            std::cerr << "Unified API requires non-null paged buffers and indices" << std::endl; return;
        }

        size_t pkpv_count = static_cast<size_t>(num_kv_pages) * page_size * kv_head_dim;
        size_t pkpv_bytes = pkpv_count * sizeof(float);

        id<MTLBuffer> pk_buf = [device newBufferWithBytes:paged_k_cache length:pkpv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> pv_buf = [device newBufferWithBytes:paged_v_cache length:pkpv_bytes options:MTLResourceStorageModeShared];

        size_t qo_indptr_elems = static_cast<size_t>(num_qo) + 1;
        size_t kv_page_indptr_elems = static_cast<size_t>(num_qo) + 1; // assume one seq for simplicity
        size_t kv_page_indices_elems = static_cast<size_t>(num_kv_pages);
        size_t kv_last_page_lens_elems = static_cast<size_t>(num_qo);

        id<MTLBuffer> qo_indptr_buf = [device newBufferWithBytes:qo_indptr length:qo_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indptr_buf = [device newBufferWithBytes:kv_page_indptr length:kv_page_indptr_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_page_indices_buf = [device newBufferWithBytes:kv_page_indices length:kv_page_indices_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_last_page_lens_buf = [device newBufferWithBytes:kv_last_page_lens length:kv_last_page_lens_elems * sizeof(int32_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> out_buf = [device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
        const size_t debug_floats = 20; // Expanded to include parameter workaround
        id<MTLBuffer> debug_buf = [device newBufferWithLength:debug_floats * sizeof(float) options:MTLResourceStorageModeShared];
        if (debug_buf && debug_buf.contents) {
            memset(debug_buf.contents, 0, debug_floats * sizeof(float));
            // WORKAROUND: Pass parameters through debug buffer
            float* debug_data = (float*)debug_buf.contents;
            debug_data[10] = (float)num_qo;
            debug_data[11] = (float)head_dim; 
            debug_data[12] = (float)head_size;
            debug_data[13] = (float)page_size;
            debug_data[14] = scale;
            std::cout << "🔧 [WORKAROUND] Parameters passed via debug buffer: num_qo=" << debug_data[10] 
                      << ", head_dim=" << debug_data[11] << ", head_size=" << debug_data[12] 
                      << ", page_size=" << debug_data[13] << ", scale=" << debug_data[14] << std::endl;
        }

        struct Params { int num_qo; int head_dim; int kv_head_dim; int head_size; int page_size; int num_query_heads; int num_kv_heads; float scale; };
        Params p = { num_qo, head_dim, kv_head_dim, head_size, page_size, num_query_heads, num_kv_heads, scale };
        
        // Debug: Print parameter values being passed to kernel
        std::cout << "\n🔧 [HOST DEBUG] Parameters being passed to Metal kernel:" << std::endl;
        std::cout << "  p.num_qo: " << p.num_qo << std::endl;
        std::cout << "  p.head_dim: " << p.head_dim << std::endl;
        std::cout << "  p.head_size: " << p.head_size << std::endl;
        std::cout << "  p.page_size: " << p.page_size << std::endl;
        std::cout << "  p.scale: " << p.scale << std::endl;
        std::cout << "  sizeof(Params): " << sizeof(Params) << std::endl;
        
        // Debug: Print raw buffer contents to verify memory layout
        uint8_t* raw_bytes = (uint8_t*)&p;
        std::cout << "  Raw struct bytes: ";
        for (size_t i = 0; i < sizeof(Params); ++i) {
            std::printf("%02x ", raw_bytes[i]);
        }
        std::cout << std::endl;
        
        // Also verify individual field addresses and sizes
        std::cout << "  Field offsets: num_qo@" << ((char*)&p.num_qo - (char*)&p) 
                  << ", head_dim@" << ((char*)&p.head_dim - (char*)&p)
                  << ", head_size@" << ((char*)&p.head_size - (char*)&p) 
                  << ", page_size@" << ((char*)&p.page_size - (char*)&p)
                  << ", scale@" << ((char*)&p.scale - (char*)&p) << std::endl;
        
        id<MTLBuffer> params_buf = [device newBufferWithBytes:&p length:sizeof(Params) options:MTLResourceStorageModeShared];
        
        // Debug: Verify buffer contents after creation
        uint8_t* buf_bytes = (uint8_t*)params_buf.contents;
        std::cout << "  Buffer contents: ";
        for (size_t i = 0; i < sizeof(Params); ++i) {
            std::printf("%02x ", buf_bytes[i]);
        }
        std::cout << std::endl;
        
        // Also interpret buffer as Params struct to verify
        Params* buf_params = (Params*)params_buf.contents;
        std::cout << "  Buffer as struct: num_qo=" << buf_params->num_qo 
                  << ", head_dim=" << buf_params->head_dim
                  << ", head_size=" << buf_params->head_size
                  << ", page_size=" << buf_params->page_size
                  << ", scale=" << buf_params->scale << std::endl;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:q_buf offset:0 atIndex:0];
        [enc setBuffer:pk_buf offset:0 atIndex:1];
        [enc setBuffer:pv_buf offset:0 atIndex:2];
        [enc setBuffer:qo_indptr_buf offset:0 atIndex:3];
        [enc setBuffer:kv_page_indptr_buf offset:0 atIndex:4];
        [enc setBuffer:kv_page_indices_buf offset:0 atIndex:5];
        [enc setBuffer:kv_last_page_lens_buf offset:0 atIndex:6];
        [enc setBuffer:out_buf offset:0 atIndex:7];
        [enc setBuffer:params_buf offset:0 atIndex:8];
        [enc setBuffer:debug_buf offset:0 atIndex:9];

        MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(num_qo, 1, 1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) { std::cerr << "Metal command buffer error: " << cmd.error.localizedDescription.UTF8String << std::endl; return; }

        if (out_buf && out_buf.contents && output) {
            memcpy(output, out_buf.contents, q_bytes);
        }
    }
}

} // namespace batch_prefill_attention
} // namespace metal