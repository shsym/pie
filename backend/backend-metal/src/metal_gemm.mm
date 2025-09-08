#include "metal_gemm.hpp"
#include "metal_dtype_conversion.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <iostream>

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLComputePipelineState> g_gemmPipeline = nil;
static id<MTLLibrary> g_library = nil;

// Tile size must match the Metal shader
static const uint32_t TILE_SIZE = 16;

bool initialize_metal_gemm() {
    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "Failed to create Metal device" << std::endl;
            return false;
        }

        // Create command queue
        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return false;
        }

            // Load Metal shader library
            NSError* error = nil;
            NSString* currentPath = [NSString stringWithUTF8String:__FILE__];
            NSString* dirPath = [currentPath stringByDeletingLastPathComponent];
            NSString* metalFilePath = [dirPath stringByAppendingPathComponent:@"metal_gemm.metal"];
            NSString* metalSource = [NSString stringWithContentsOfFile:metalFilePath
                                                              encoding:NSUTF8StringEncoding
                                                                 error:&error];

            if (error || !metalSource) {
                std::cerr << "Failed to load Metal shader source: " <<
                             (error ? error.localizedDescription.UTF8String : "unknown error") << std::endl;
                return false;
            }

        // Compile Metal library
        g_library = [g_device newLibraryWithSource:metalSource options:nil error:&error];
        if (error || !g_library) {
            std::cerr << "Failed to compile Metal library: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        // Get the gemm kernel function - using simple version (swapped version was incorrect)
        id<MTLFunction> gemmFunction = [g_library newFunctionWithName:@"metal_gemm_float32_simple"];
        if (!gemmFunction) {
            std::cerr << "Failed to find metal_gemm_float32 function in library" << std::endl;
            return false;
        }

        // Create compute pipeline state
        g_gemmPipeline = [g_device newComputePipelineStateWithFunction:gemmFunction error:&error];
        if (error || !g_gemmPipeline) {
            std::cerr << "Failed to create compute pipeline: " <<
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        std::cout << "Metal GEMM initialized successfully" << std::endl;
        return true;
    }
}

void cleanup_metal_gemm() {
    g_gemmPipeline = nil;
    g_library = nil;
    g_commandQueue = nil;
    g_device = nil;
}

void metal_gemm_bfloat16(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const bfloat16_t* A,
    const bfloat16_t* B,
    const bfloat16_t* bias,
    bfloat16_t* C,
    int m, int n, int k,
    void* workspace,        // Unused for API compatibility
    size_t workspace_size,  // Unused for API compatibility
    bool transa,
    bool transb
) {
    @autoreleasepool {
        // Use provided device and commandQueue instead of globals
        if (!device || !commandQueue || !g_gemmPipeline) {
            throw std::runtime_error("Metal GEMM not initialized. Call initialize_metal_gemm() first.");
        }

        // Calculate leading dimensions for C++ row-major format
        // lda/ldb/ldc = number of columns per row (row-major stride)
        // For row-major storage: index = row * leading_dim + col
        const uint32_t lda = transa ? m : k;  // A storage: if transposed [k][m] else [m][k]
        const uint32_t ldb = transb ? k : n;  // B storage: if transposed [n][k] else [k][n]
        const uint32_t ldc = n;               // C storage: always [m][n] in row-major

        // Convert bfloat16 input to float32 for testing
        const size_t A_elems = (transa ? k : m) * (transa ? m : k);
        const size_t B_elems = (transb ? n : k) * (transb ? k : n);
        const size_t C_elems = m * n;

        std::vector<float> A_f32(A_elems);
        std::vector<float> B_f32(B_elems);
        std::vector<float> bias_f32(bias ? n : 0);
        std::vector<float> C_f32(C_elems);

        // Convert bfloat16 to float32 using existing conversion
        for (size_t i = 0; i < A_elems; ++i) {
            uint32_t bf16_bits = A[i];  // bfloat16_t is stored as uint16_t
            uint32_t f32_bits = bf16_bits << 16;  // Expand to float32
            A_f32[i] = *reinterpret_cast<float*>(&f32_bits);
        }
        for (size_t i = 0; i < B_elems; ++i) {
            uint32_t bf16_bits = B[i];
            uint32_t f32_bits = bf16_bits << 16;
            B_f32[i] = *reinterpret_cast<float*>(&f32_bits);
        }
        if (bias) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t bf16_bits = bias[i];
                uint32_t f32_bits = bf16_bits << 16;
                bias_f32[i] = *reinterpret_cast<float*>(&f32_bits);
            }
        }

        const size_t A_size = A_elems * sizeof(float);
        const size_t B_size = B_elems * sizeof(float);
        const size_t C_size = C_elems * sizeof(float);
        const size_t bias_size = bias ? n * sizeof(float) : 0;

        // Create Metal buffers using provided device with float32 data
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A_f32.data() length:A_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B_f32.data() length:B_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:C_size options:MTLResourceStorageModeShared];

        // Always create bias buffer (safer than nil binding)
        id<MTLBuffer> bufferBias;
        if (bias) {
            bufferBias = [device newBufferWithBytes:bias_f32.data() length:bias_size options:MTLResourceStorageModeShared];
        } else {
            // Create tiny dummy buffer to avoid nil binding issues
            float dummy = 0.0f;
            bufferBias = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
        }

        // Create parameters struct
        struct GemmParamsHost {
            uint32_t m, n, k;
            uint32_t lda, ldb, ldc;
            uint32_t transa, transb, use_bias;
        } params = {
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            lda, ldb, ldc,
            transa ? 1u : 0u,
            transb ? 1u : 0u,
            bias ? 1u : 0u
        };

        id<MTLBuffer> bufferParams = [device newBufferWithBytes:&params
                                                           length:sizeof(GemmParamsHost)
                                                          options:MTLResourceStorageModeShared];

        // Create command buffer and encoder using provided commandQueue
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set compute pipeline and buffers
        [encoder setComputePipelineState:g_gemmPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferBias offset:0 atIndex:2];  // Can be nil
        [encoder setBuffer:bufferC offset:0 atIndex:3];
        [encoder setBuffer:bufferParams offset:0 atIndex:4];

        // Configure threadgroup sizes for swapped version
        // Swapped version uses: row = gid.y, col = gid.x
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);  // 256 threads per group
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (n + 15) / 16,   // X dimension: columns (n)
            (m + 15) / 16,   // Y dimension: rows (m)
            1
        );

        // No threadgroup memory needed for simple version

        // Dispatch compute kernel
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.error) {
            NSString* errorDesc = [commandBuffer.error localizedDescription];
            throw std::runtime_error("Metal compute failed: " + std::string([errorDesc UTF8String]));
        }

        // Copy float32 result back - temporarily keeping bfloat16 conversion for interface compatibility
        float* C_f32_result = static_cast<float*>([bufferC contents]);

        // Convert float32 result to bfloat16 using centralized conversion
        for (size_t i = 0; i < C_elems; ++i) {
            C[i] = metal::DTypeConverter::f32_to_bf16(C_f32_result[i]);
        }
    }
}

void metal_gemm_float32(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int m, int n, int k,
    void* workspace,        // Unused for API compatibility
    size_t workspace_size,  // Unused for API compatibility
    bool transa,
    bool transb
) {
    @autoreleasepool {
        // Use provided device and commandQueue instead of globals
        if (!device || !commandQueue || !g_gemmPipeline) {
            throw std::runtime_error("Metal GEMM not initialized. Call initialize_metal_gemm() first.");
        }

        // Calculate leading dimensions for C++ row-major format
        const uint32_t lda = transa ? m : k;  // A storage: if transposed [k][m] else [m][k]
        const uint32_t ldb = transb ? k : n;  // B storage: if transposed [n][k] else [k][n]
        const uint32_t ldc = n;               // C storage: always [m][n] in row-major

        // Pure f32 computation - no bfloat16 conversion
        const size_t A_elems = (transa ? k : m) * (transa ? m : k);
        const size_t B_elems = (transb ? n : k) * (transb ? k : n);
        const size_t C_elems = m * n;

        const size_t A_size = A_elems * sizeof(float);
        const size_t B_size = B_elems * sizeof(float);
        const size_t C_size = C_elems * sizeof(float);
        const size_t bias_size = bias ? n * sizeof(float) : 0;

        // Create Metal buffers using provided device with float32 data directly
        id<MTLBuffer> bufferA = [device newBufferWithBytes:A length:A_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:B length:B_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithLength:C_size options:MTLResourceStorageModeShared];

        // Always create bias buffer (safer than nil binding)
        id<MTLBuffer> bufferBias;
        if (bias) {
            bufferBias = [device newBufferWithBytes:bias length:bias_size options:MTLResourceStorageModeShared];
        } else {
            // Create tiny dummy buffer to avoid nil binding issues
            float dummy = 0.0f;
            bufferBias = [device newBufferWithBytes:&dummy length:sizeof(float) options:MTLResourceStorageModeShared];
        }

        // Create parameters struct
        struct GemmParamsHost {
            uint32_t m, n, k;
            uint32_t lda, ldb, ldc;
            uint32_t transa, transb, use_bias;
        } params = {
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(k),
            lda, ldb, ldc,
            transa ? 1u : 0u,
            transb ? 1u : 0u,
            bias ? 1u : 0u
        };

        id<MTLBuffer> bufferParams = [device newBufferWithBytes:&params
                                                           length:sizeof(GemmParamsHost)
                                                          options:MTLResourceStorageModeShared];

        // Create command buffer and encoder using provided commandQueue
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set compute pipeline and buffers
        [encoder setComputePipelineState:g_gemmPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferBias offset:0 atIndex:2];  // Can be dummy buffer
        [encoder setBuffer:bufferC offset:0 atIndex:3];
        [encoder setBuffer:bufferParams offset:0 atIndex:4];

        // Configure threadgroup sizes
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);  // 256 threads per group
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (n + 15) / 16,   // X dimension: columns (n)
            (m + 15) / 16,   // Y dimension: rows (m)
            1
        );

        // Dispatch compute kernel
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Execute and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check for errors
        if (commandBuffer.error) {
            NSString* errorDesc = [commandBuffer.error localizedDescription];
            throw std::runtime_error("Metal compute failed: " + std::string([errorDesc UTF8String]));
        }

        // Copy float32 result directly to output buffer - no conversion needed
        float* C_result = static_cast<float*>([bufferC contents]);
        std::memcpy(C, C_result, C_size);
    }
}