#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_grouped_gemm.hpp"
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace {
    static id<MTLDevice> sDevice = nil;
    static id<MTLCommandQueue> sQueue = nil;
    static id<MTLLibrary> sLibrary = nil;
    static id<MTLComputePipelineState> sPSO_bf16 = nil;
    static id<MTLComputePipelineState> sPSO_f32 = nil;

    struct ParamsHost {
        uint32_t num_groups;
        uint8_t transa;
        uint8_t transb;
        uint8_t pad[2];
        uint32_t has_bias;
    };

    bool ensurePipelines() {
        if (!sDevice) {
            sDevice = MTLCreateSystemDefaultDevice();
            if (!sDevice) { std::cerr << "Metal device not available" << std::endl; return false; }
        }
        if (!sQueue) {
            sQueue = [sDevice newCommandQueue];
        }
        if (!sLibrary) {
            sLibrary = [sDevice newDefaultLibrary];
            if (!sLibrary) {
                // Fallback: compile from .metal source file adjacent to this .mm (no embedded strings)
                NSError* readErr = nil;
                NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
                NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
                NSString* metalFilePath = [dirPath stringByAppendingPathComponent:@"metal_grouped_gemm.metal"];
                NSString* src = [NSString stringWithContentsOfFile:metalFilePath encoding:NSUTF8StringEncoding error:&readErr];
                if (readErr || !src) {
                    std::cerr << "Failed to create default Metal library and read source at: "
                              << (metalFilePath ? metalFilePath.UTF8String : "(null)")
                              << ", error: "
                              << (readErr ? readErr.localizedDescription.UTF8String : "unknown")
                              << std::endl;
                    return false;
                }
                NSError* err = nil;
                MTLCompileOptions* opts = [MTLCompileOptions new];
                sLibrary = [sDevice newLibraryWithSource:src options:opts error:&err];
                if (!sLibrary) {
                    std::cerr << "Failed to compile Metal library from source: "
                              << (err ? err.localizedDescription.UTF8String : "unknown") << std::endl;
                    return false;
                }
            }
        }
        if (!sPSO_bf16) {
            NSError* err = nil;
            id<MTLFunction> fn = [sLibrary newFunctionWithName:@"metal_grouped_gemm_bfloat16"];
            if (!fn) { std::cerr << "Missing kernel metal_grouped_gemm_bfloat16" << std::endl; return false; }
            sPSO_bf16 = [sDevice newComputePipelineStateWithFunction:fn error:&err];
            if (!sPSO_bf16) {
                std::cerr << "Failed to create PSO for bf16 grouped gemm: "
                          << (err ? err.localizedDescription.UTF8String : "unknown") << std::endl;
                return false;
            }
        }
        if (!sPSO_f32) {
            NSError* err = nil;
            id<MTLFunction> fn = [sLibrary newFunctionWithName:@"metal_grouped_gemm_float32"];
            if (!fn) { std::cerr << "Missing kernel metal_grouped_gemm_float32" << std::endl; return false; }
            sPSO_f32 = [sDevice newComputePipelineStateWithFunction:fn error:&err];
            if (!sPSO_f32) {
                std::cerr << "Failed to create PSO for f32 grouped gemm: "
                          << (err ? err.localizedDescription.UTF8String : "unknown") << std::endl;
                return false;
            }
        }
        return true;
    }

    template <typename T>
    inline uint32_t ceil_div_u32(T a, T b) { return static_cast<uint32_t>((a + b - 1) / b); }

    inline id<MTLBuffer> makeBuffer(const void* data, size_t bytes) {
        id<MTLBuffer> buf = [sDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (data && bytes) { memcpy([buf contents], data, bytes); }
        return buf;
    }

    inline void copyFromConcat(const void* concat, size_t elem_size, const std::vector<uint32_t>& offsets_bytes,
                               const std::vector<size_t>& sizes_elems, void** dst_ptrs) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(concat);
        for (size_t g = 0; g < offsets_bytes.size(); ++g) {
            size_t bytes = sizes_elems[g] * elem_size;
            memcpy(dst_ptrs[g], base + offsets_bytes[g], bytes);
        }
    }
}

int metal_grouped_gemm_bfloat16(
    void** A_ptrs,
    void** B_ptrs,
    void** C_ptrs,
    void** bias_ptrs,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    unsigned int num_groups,
    bool transa,
    bool transb
) {
    if (!A_ptrs || !B_ptrs || !C_ptrs || !m_array || !n_array || !k_array || num_groups == 0) {
        std::cerr << "Invalid grouped GEMM parameters" << std::endl; return -2;
    }
    if (!ensurePipelines()) return -5;

    const uint32_t TILE_M = 16, TILE_N = 16;
    const size_t elem = sizeof(uint16_t); // bfloat16 host representation

    std::vector<uint32_t> m_u(num_groups), n_u(num_groups), k_u(num_groups);
    std::vector<uint32_t> A_off(num_groups), B_off(num_groups), C_off(num_groups), Bias_off(num_groups);
    std::vector<size_t> A_sizes(num_groups), B_sizes(num_groups), C_sizes(num_groups), Bias_sizes(num_groups);

    size_t a_cursor = 0, b_cursor = 0, c_cursor = 0, bias_cursor = 0;
    for (unsigned g = 0; g < num_groups; ++g) {
        const uint32_t m = static_cast<uint32_t>(m_array[g]);
        const uint32_t n = static_cast<uint32_t>(n_array[g]);
        const uint32_t k = static_cast<uint32_t>(k_array[g]);
        m_u[g] = m; n_u[g] = n; k_u[g] = k;
        size_t a_e = static_cast<size_t>(transa ? (k * m) : (m * k));
        size_t b_e = static_cast<size_t>(transb ? (n * k) : (k * n));
        size_t c_e = static_cast<size_t>(m) * n;
        size_t bias_e = (bias_ptrs && bias_ptrs[g]) ? static_cast<size_t>(n) : 0;
        A_off[g] = static_cast<uint32_t>(a_cursor);
        B_off[g] = static_cast<uint32_t>(b_cursor);
        C_off[g] = static_cast<uint32_t>(c_cursor);
        Bias_off[g] = static_cast<uint32_t>(bias_cursor);
        A_sizes[g] = a_e; B_sizes[g] = b_e; C_sizes[g] = c_e; Bias_sizes[g] = bias_e;
        a_cursor += a_e * elem;
        b_cursor += b_e * elem;
        c_cursor += c_e * elem;
        bias_cursor += bias_e * elem;
    }

    // Concatenate buffers in shared memory
    std::vector<uint8_t> A_concat(a_cursor), B_concat(b_cursor), C_concat(c_cursor), Bias_concat(bias_cursor);
    for (unsigned g = 0; g < num_groups; ++g) {
        if (A_sizes[g]) memcpy(A_concat.data() + A_off[g], A_ptrs[g], A_sizes[g] * elem);
        if (B_sizes[g]) memcpy(B_concat.data() + B_off[g], B_ptrs[g], B_sizes[g] * elem);
        if (C_sizes[g]) memcpy(C_concat.data() + C_off[g], C_ptrs[g], C_sizes[g] * elem);
        if (Bias_sizes[g]) memcpy(Bias_concat.data() + Bias_off[g], bias_ptrs[g], Bias_sizes[g] * elem);
    }

    id<MTLBuffer> A_buf = makeBuffer(A_concat.data(), A_concat.size());
    id<MTLBuffer> B_buf = makeBuffer(B_concat.data(), B_concat.size());
    id<MTLBuffer> C_buf = makeBuffer(C_concat.data(), C_concat.size());
    id<MTLBuffer> Bias_buf = makeBuffer(Bias_concat.data(), Bias_concat.size());
    id<MTLBuffer> Aoff_buf = makeBuffer(A_off.data(), A_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Boff_buf = makeBuffer(B_off.data(), B_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Coff_buf = makeBuffer(C_off.data(), C_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Biasoff_buf = makeBuffer(Bias_off.data(), Bias_off.size() * sizeof(uint32_t));
    id<MTLBuffer> M_buf = makeBuffer(m_u.data(), m_u.size() * sizeof(uint32_t));
    id<MTLBuffer> N_buf = makeBuffer(n_u.data(), n_u.size() * sizeof(uint32_t));
    id<MTLBuffer> K_buf = makeBuffer(k_u.data(), k_u.size() * sizeof(uint32_t));

    ParamsHost params { num_groups, static_cast<uint8_t>(transa), static_cast<uint8_t>(transb), {0,0}, (bias_ptrs ? 1u : 0u) };
    id<MTLBuffer> P_buf = makeBuffer(&params, sizeof(params));

    id<MTLCommandBuffer> cmd = [sQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:sPSO_bf16];
    [enc setBuffer:A_buf offset:0 atIndex:0];
    [enc setBuffer:B_buf offset:0 atIndex:1];
    [enc setBuffer:C_buf offset:0 atIndex:2];
    [enc setBuffer:Bias_buf offset:0 atIndex:3];
    [enc setBuffer:Aoff_buf offset:0 atIndex:4];
    [enc setBuffer:Boff_buf offset:0 atIndex:5];
    [enc setBuffer:Coff_buf offset:0 atIndex:6];
    [enc setBuffer:Biasoff_buf offset:0 atIndex:7];
    [enc setBuffer:M_buf offset:0 atIndex:8];
    [enc setBuffer:N_buf offset:0 atIndex:9];
    [enc setBuffer:K_buf offset:0 atIndex:10];
    [enc setBuffer:P_buf offset:0 atIndex:11];

    // Threadgroup memory lengths
    const uint32_t TILE_K = 16;
    [enc setThreadgroupMemoryLength:(TILE_M*TILE_K*elem) atIndex:0];
    [enc setThreadgroupMemoryLength:(TILE_K*TILE_N*elem) atIndex:1];

    // Determine max m,n across groups to size grid in x,y; kernel early-outs for bounds
    uint32_t m_max = 0, n_max = 0;
    for (unsigned g = 0; g < num_groups; ++g) { m_max = std::max(m_max, m_u[g]); n_max = std::max(n_max, n_u[g]); }

    MTLSize tpt = MTLSizeMake(16, 16, 1);
    MTLSize tg = MTLSizeMake(ceil_div_u32(n_max, 16u), ceil_div_u32(m_max, 16u), num_groups);
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy results back into provided C_ptrs
    copyFromConcat([C_buf contents], elem, C_off, C_sizes, C_ptrs);
    return 0;
}

int metal_grouped_gemm_float32(
    const float* const* A_ptrs,
    const float* const* B_ptrs,
    float** C_ptrs,
    const float* const* bias_ptrs,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    unsigned int num_groups,
    bool transa,
    bool transb
) {
    if (!A_ptrs || !B_ptrs || !C_ptrs || !m_array || !n_array || !k_array || num_groups == 0) {
        std::cerr << "Invalid grouped GEMM parameters" << std::endl; return -2;
    }
    if (!ensurePipelines()) return -5;

    const uint32_t TILE_M = 16, TILE_N = 16;
    const size_t elem = sizeof(float);

    std::vector<uint32_t> m_u(num_groups), n_u(num_groups), k_u(num_groups);
    std::vector<uint32_t> A_off(num_groups), B_off(num_groups), C_off(num_groups), Bias_off(num_groups);
    std::vector<size_t> A_sizes(num_groups), B_sizes(num_groups), C_sizes(num_groups), Bias_sizes(num_groups);

    size_t a_cursor = 0, b_cursor = 0, c_cursor = 0, bias_cursor = 0;
    for (unsigned g = 0; g < num_groups; ++g) {
        const uint32_t m = static_cast<uint32_t>(m_array[g]);
        const uint32_t n = static_cast<uint32_t>(n_array[g]);
        const uint32_t k = static_cast<uint32_t>(k_array[g]);
        m_u[g] = m; n_u[g] = n; k_u[g] = k;
        size_t a_e = static_cast<size_t>(transa ? (k * m) : (m * k));
        size_t b_e = static_cast<size_t>(transb ? (n * k) : (k * n));
        size_t c_e = static_cast<size_t>(m) * n;
        size_t bias_e = (bias_ptrs && bias_ptrs[g]) ? static_cast<size_t>(n) : 0;
        A_off[g] = static_cast<uint32_t>(a_cursor);
        B_off[g] = static_cast<uint32_t>(b_cursor);
        C_off[g] = static_cast<uint32_t>(c_cursor);
        Bias_off[g] = static_cast<uint32_t>(bias_cursor);
        A_sizes[g] = a_e; B_sizes[g] = b_e; C_sizes[g] = c_e; Bias_sizes[g] = bias_e;
        a_cursor += a_e * elem;
        b_cursor += b_e * elem;
        c_cursor += c_e * elem;
        bias_cursor += bias_e * elem;
    }

    std::vector<uint8_t> A_concat(a_cursor), B_concat(b_cursor), C_concat(c_cursor), Bias_concat(bias_cursor);
    for (unsigned g = 0; g < num_groups; ++g) {
        if (A_sizes[g]) memcpy(A_concat.data() + A_off[g], A_ptrs[g], A_sizes[g] * elem);
        if (B_sizes[g]) memcpy(B_concat.data() + B_off[g], B_ptrs[g], B_sizes[g] * elem);
        if (C_sizes[g]) memcpy(C_concat.data() + C_off[g], C_ptrs[g], C_sizes[g] * elem);
        if (Bias_sizes[g]) memcpy(Bias_concat.data() + Bias_off[g], bias_ptrs[g], Bias_sizes[g] * elem);
    }

    id<MTLBuffer> A_buf = makeBuffer(A_concat.data(), A_concat.size());
    id<MTLBuffer> B_buf = makeBuffer(B_concat.data(), B_concat.size());
    id<MTLBuffer> C_buf = makeBuffer(C_concat.data(), C_concat.size());
    id<MTLBuffer> Bias_buf = makeBuffer(Bias_concat.data(), Bias_concat.size());
    id<MTLBuffer> Aoff_buf = makeBuffer(A_off.data(), A_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Boff_buf = makeBuffer(B_off.data(), B_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Coff_buf = makeBuffer(C_off.data(), C_off.size() * sizeof(uint32_t));
    id<MTLBuffer> Biasoff_buf = makeBuffer(Bias_off.data(), Bias_off.size() * sizeof(uint32_t));
    id<MTLBuffer> M_buf = makeBuffer(m_u.data(), m_u.size() * sizeof(uint32_t));
    id<MTLBuffer> N_buf = makeBuffer(n_u.data(), n_u.size() * sizeof(uint32_t));
    id<MTLBuffer> K_buf = makeBuffer(k_u.data(), k_u.size() * sizeof(uint32_t));

    ParamsHost params { num_groups, static_cast<uint8_t>(transa), static_cast<uint8_t>(transb), {0,0}, (bias_ptrs ? 1u : 0u) };
    id<MTLBuffer> P_buf = makeBuffer(&params, sizeof(params));

    id<MTLCommandBuffer> cmd = [sQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:sPSO_f32];
    [enc setBuffer:A_buf offset:0 atIndex:0];
    [enc setBuffer:B_buf offset:0 atIndex:1];
    [enc setBuffer:C_buf offset:0 atIndex:2];
    [enc setBuffer:Bias_buf offset:0 atIndex:3];
    [enc setBuffer:Aoff_buf offset:0 atIndex:4];
    [enc setBuffer:Boff_buf offset:0 atIndex:5];
    [enc setBuffer:Coff_buf offset:0 atIndex:6];
    [enc setBuffer:Biasoff_buf offset:0 atIndex:7];
    [enc setBuffer:M_buf offset:0 atIndex:8];
    [enc setBuffer:N_buf offset:0 atIndex:9];
    [enc setBuffer:K_buf offset:0 atIndex:10];
    [enc setBuffer:P_buf offset:0 atIndex:11];

    const uint32_t TILE_K = 16;
    [enc setThreadgroupMemoryLength:(TILE_M*TILE_K*sizeof(float)) atIndex:0];
    [enc setThreadgroupMemoryLength:(TILE_K*TILE_N*sizeof(float)) atIndex:1];

    uint32_t m_max = 0, n_max = 0;
    for (unsigned g = 0; g < num_groups; ++g) { m_max = std::max(m_max, m_u[g]); n_max = std::max(n_max, n_u[g]); }

    MTLSize tpt = MTLSizeMake(16, 16, 1);
    MTLSize tg = MTLSizeMake(ceil_div_u32(n_max, 16u), ceil_div_u32(m_max, 16u), num_groups);
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    copyFromConcat([C_buf contents], elem, C_off, C_sizes, reinterpret_cast<void**>(C_ptrs));
    return 0;
}