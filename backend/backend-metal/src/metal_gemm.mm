#include "metal_gemm.hpp"
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
        NSString* metalFilePath = @"../backend/backend-metal/src/metal_gemm.metal";
        NSString* metalSource = [NSString stringWithContentsOfFile:metalFilePath 
                                                          encoding:NSUTF8StringEncoding 
                                                             error:&error];
        
        if (error || !metalSource) {
            // Try loading from current directory
            metalFilePath = @"metal_gemm.metal";
            metalSource = [NSString stringWithContentsOfFile:metalFilePath 
                                                    encoding:NSUTF8StringEncoding 
                                                       error:&error];
        }
        
        if (error || !metalSource) {
            std::cerr << "Failed to load Metal shader source: " << 
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        // Compile Metal library
        g_library = [g_device newLibraryWithSource:metalSource options:nil error:&error];
        if (error || !g_library) {
            std::cerr << "Failed to compile Metal library: " << 
                         [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        // Get the gemm kernel function
        id<MTLFunction> gemmFunction = [g_library newFunctionWithName:@"metal_gemm_bfloat16"];
        if (!gemmFunction) {
            std::cerr << "Failed to find metal_gemm_bfloat16 function in library" << std::endl;
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
        if (!g_device || !g_commandQueue || !g_gemmPipeline) {
            throw std::runtime_error("Metal GEMM not initialized. Call initialize_metal_gemm() first.");
        }
        
        // Calculate matrix dimensions and leading dimensions (matches cuBLAS convention)
        const uint32_t lda = transa ? m : k;
        const uint32_t ldb = transb ? k : n;
        const uint32_t ldc = n;
        
        const size_t A_size = (transa ? k : m) * (transa ? m : k) * sizeof(bfloat16_t);
        const size_t B_size = (transb ? n : k) * (transb ? k : n) * sizeof(bfloat16_t);
        const size_t C_size = m * n * sizeof(bfloat16_t);
        const size_t bias_size = bias ? n * sizeof(bfloat16_t) : 0;
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:A length:A_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:B length:B_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:C_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferBias = nil;
        
        if (bias) {
            bufferBias = [g_device newBufferWithBytes:bias length:bias_size options:MTLResourceStorageModeShared];
        }
        
        // Create parameters struct
        struct {
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
        
        id<MTLBuffer> bufferParams = [g_device newBufferWithBytes:&params 
                                                           length:sizeof(params) 
                                                          options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set compute pipeline and buffers
        [encoder setComputePipelineState:g_gemmPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferBias offset:0 atIndex:2];  // Can be nil
        [encoder setBuffer:bufferC offset:0 atIndex:3];
        [encoder setBuffer:bufferParams offset:0 atIndex:4];
        
        // Configure threadgroup sizes
        MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (n + TILE_SIZE - 1) / TILE_SIZE,   // X dimension: columns 
            (m + TILE_SIZE - 1) / TILE_SIZE,   // Y dimension: rows
            1
        );
        
        // Set threadgroup memory for tiles
        NSUInteger tileMemorySize = TILE_SIZE * TILE_SIZE * sizeof(bfloat16_t);
        [encoder setThreadgroupMemoryLength:tileMemorySize atIndex:0];  // tile_A
        [encoder setThreadgroupMemoryLength:tileMemorySize atIndex:1];  // tile_B
        
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
        
        // Copy result back to output buffer
        memcpy(C, [bufferC contents], C_size);
    }
}