#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

// MetalContext implementation
MetalContext& MetalContext::getInstance() {
    static MetalContext instance;
    return instance;
}

bool MetalContext::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Get default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "Metal device creation failed - Metal not supported on this system" << std::endl;
        return false;
    }
    
    // Create command queue
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
        std::cerr << "Metal command queue creation failed" << std::endl;
        device_ = nullptr;
        return false;
    }
    
    initialized_ = true;
    std::cout << "Metal context initialized successfully" << std::endl;
    std::cout << "Device: " << [[device_ name] UTF8String] << std::endl;
    
    return true;
}

void MetalContext::cleanup() {
    if (initialized_) {
        commandQueue_ = nullptr;
        device_ = nullptr;
        initialized_ = false;
        std::cout << "Metal context cleaned up" << std::endl;
    }
}

// MetalProfileScope implementation
MetalProfileScope::MetalProfileScope(MetalProfiler* profiler, const std::string& name, id<MTLCommandBuffer> commandBuffer)
    : profiler_(profiler), name_(name), commandBuffer_(commandBuffer) {
    // Metal profiling would be implemented here
    // For now, we'll do basic timing
}

MetalProfileScope::~MetalProfileScope() {
    // End profiling scope
}

void MetalProfileScope::record(const std::string& checkpoint) {
    // Record profiling checkpoint
    if (profiler_ && profiler_->isEnabled()) {
        // Implementation would record timing checkpoint
    }
}

MetalProfileScope MetalProfileScope::scope(const std::string& name) {
    return MetalProfileScope(profiler_, name_, commandBuffer_);
}

// MetalProfiler implementation
MetalProfiler::MetalProfiler(bool enabled) : enabled_(enabled) {
}

MetalProfileScope MetalProfiler::scope(const std::string& name, id<MTLCommandBuffer> commandBuffer) {
    return MetalProfileScope(this, name, commandBuffer);
}

void MetalProfiler::print_report() {
    if (!enabled_ || timings_.empty()) {
        return;
    }
    
    std::cout << "\n=== Metal Profiling Report ===" << std::endl;
    for (const auto& timing : timings_) {
        std::cout << timing.first << ": " << timing.second << "ms" << std::endl;
    }
    std::cout << "==============================\n" << std::endl;
}

// MetalCast implementations
namespace MetalCast {
    
    void bfloat16_to_float(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const bfloat16_t* input, float* output, size_t count) {
        // For now, do CPU conversion - should be replaced with Metal kernel
        for (size_t i = 0; i < count; i++) {
            // Convert bfloat16 to float (simple bit manipulation)
            uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
            output[i] = *reinterpret_cast<float*>(&bits);
        }
    }
    
    void float_to_bfloat16(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const float* input, bfloat16_t* output, size_t count) {
        // For now, do CPU conversion - should be replaced with Metal kernel  
        for (size_t i = 0; i < count; i++) {
            // Convert float to bfloat16 (truncation)
            uint32_t bits = *reinterpret_cast<const uint32_t*>(&input[i]);
            output[i] = static_cast<bfloat16_t>(bits >> 16);
        }
    }
}

// MetalMemory implementations
namespace MetalMemory {
    
    void copyBuffer(id<MTLCommandQueue> commandQueue,
                   id<MTLBuffer> source, size_t sourceOffset,
                   id<MTLBuffer> destination, size_t destOffset,
                   size_t size) {
        
        METAL_CHECK(commandQueue != nullptr, "Command queue is null");
        METAL_CHECK(source != nullptr, "Source buffer is null");
        METAL_CHECK(destination != nullptr, "Destination buffer is null");
        METAL_CHECK(sourceOffset + size <= [source length], "Source copy exceeds buffer bounds");
        METAL_CHECK(destOffset + size <= [destination length], "Destination copy exceeds buffer bounds");
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        [blitEncoder copyFromBuffer:source 
                       sourceOffset:sourceOffset
                           toBuffer:destination
                  destinationOffset:destOffset
                               size:size];
        
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    void zeroBuffer(id<MTLCommandQueue> commandQueue,
                   id<MTLBuffer> buffer, size_t size) {
        
        METAL_CHECK(commandQueue != nullptr, "Command queue is null");
        METAL_CHECK(buffer != nullptr, "Buffer is null");
        METAL_CHECK(size <= [buffer length], "Zero size exceeds buffer bounds");
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        [blitEncoder fillBuffer:buffer range:NSMakeRange(0, size) value:0];
        
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    // Template specializations for fillBuffer
    template<>
    void fillBuffer<float>(id<MTLCommandQueue> commandQueue,
                          id<MTLBuffer> buffer, const float& value, size_t count) {
        METAL_CHECK(commandQueue != nullptr, "Command queue is null");
        METAL_CHECK(buffer != nullptr, "Buffer is null");
        METAL_CHECK(count * sizeof(float) <= [buffer length], "Fill size exceeds buffer bounds");
        
        // For now, use CPU-based fill - TODO: implement Metal kernel
        float* bufferData = static_cast<float*>([buffer contents]);
        std::fill(bufferData, bufferData + count, value);
    }
    
    template<>
    void fillBuffer<bfloat16_t>(id<MTLCommandQueue> commandQueue,
                               id<MTLBuffer> buffer, const bfloat16_t& value, size_t count) {
        METAL_CHECK(commandQueue != nullptr, "Command queue is null");
        METAL_CHECK(buffer != nullptr, "Buffer is null");
        METAL_CHECK(count * sizeof(bfloat16_t) <= [buffer length], "Fill size exceeds buffer bounds");
        
        // For now, use CPU-based fill - TODO: implement Metal kernel
        bfloat16_t* bufferData = static_cast<bfloat16_t*>([buffer contents]);
        std::fill(bufferData, bufferData + count, value);
    }
}

// MetalComputePipelineManager implementation
MetalComputePipelineManager& MetalComputePipelineManager::getInstance() {
    static MetalComputePipelineManager instance;
    return instance;
}

id<MTLComputePipelineState> MetalComputePipelineManager::getComputePipeline(const std::string& kernelName) {
    auto it = pipelines_.find(kernelName);
    if (it != pipelines_.end()) {
        return it->second;
    }
    
    // Try to create pipeline
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        std::cerr << "Metal context not initialized" << std::endl;
        return nullptr;
    }
    
    if (!defaultLibrary_) {
        defaultLibrary_ = [context.getDevice() newDefaultLibrary];
        if (!defaultLibrary_) {
            std::cerr << "Failed to create default Metal library" << std::endl;
            return nullptr;
        }
    }
    
    NSString* kernelNameStr = [NSString stringWithUTF8String:kernelName.c_str()];
    id<MTLFunction> kernelFunction = [defaultLibrary_ newFunctionWithName:kernelNameStr];
    
    if (!kernelFunction) {
        std::cerr << "Failed to find Metal kernel function: " << kernelName << std::endl;
        return nullptr;
    }
    
    NSError* error = nil;
    id<MTLComputePipelineState> pipelineState = 
        [context.getDevice() newComputePipelineStateWithFunction:kernelFunction error:&error];
    
    if (error) {
        std::cerr << "Failed to create Metal compute pipeline for " << kernelName 
                  << ": " << [[error localizedDescription] UTF8String] << std::endl;
        return nullptr;
    }
    
    pipelines_[kernelName] = pipelineState;
    return pipelineState;
}

bool MetalComputePipelineManager::registerKernelLibrary(const std::string& libraryPath) {
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        return false;
    }
    
    NSString* pathStr = [NSString stringWithUTF8String:libraryPath.c_str()];
    NSError* error = nil;
    
    id<MTLLibrary> library = [context.getDevice() newLibraryWithFile:pathStr error:&error];
    
    if (error) {
        std::cerr << "Failed to load Metal library from " << libraryPath 
                  << ": " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    defaultLibrary_ = library;
    return true;
}

bool MetalComputePipelineManager::registerKernelLibraryFromSource(const std::string& source) {
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        return false;
    }
    
    NSString* sourceStr = [NSString stringWithUTF8String:source.c_str()];
    NSError* error = nil;
    
    id<MTLLibrary> library = [context.getDevice() newLibraryWithSource:sourceStr 
                                                               options:nil 
                                                                 error:&error];
    
    if (error) {
        std::cerr << "Failed to compile Metal library from source: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    defaultLibrary_ = library;
    return true;
}

// MetalDispatch implementations
namespace MetalDispatch {
    
    std::pair<MTLSize, MTLSize> calculateThreadGroups1D(size_t totalThreads, size_t maxThreadsPerGroup) {
        size_t threadsPerGroup = std::min(totalThreads, maxThreadsPerGroup);
        size_t numGroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(numGroups, 1, 1);
        
        return {threadgroupsPerGrid, threadsPerThreadgroup};
    }
    
    std::pair<MTLSize, MTLSize> calculateThreadGroups2D(size_t width, size_t height, size_t maxThreadsPerGroup) {
        // Try to make a square-ish threadgroup
        size_t maxDim = static_cast<size_t>(sqrt(maxThreadsPerGroup));
        
        size_t groupWidth = std::min(width, maxDim);
        size_t groupHeight = std::min(height, maxThreadsPerGroup / groupWidth);
        
        size_t numGroupsX = (width + groupWidth - 1) / groupWidth;
        size_t numGroupsY = (height + groupHeight - 1) / groupHeight;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(groupWidth, groupHeight, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(numGroupsX, numGroupsY, 1);
        
        return {threadgroupsPerGrid, threadsPerThreadgroup};
    }
    
    std::pair<MTLSize, MTLSize> calculateThreadGroups3D(size_t width, size_t height, size_t depth, size_t maxThreadsPerGroup) {
        // Try to make a cubic-ish threadgroup
        size_t maxDim = static_cast<size_t>(cbrt(maxThreadsPerGroup));
        
        size_t groupWidth = std::min(width, maxDim);
        size_t groupHeight = std::min(height, maxDim);
        size_t groupDepth = std::min(depth, maxThreadsPerGroup / (groupWidth * groupHeight));
        
        size_t numGroupsX = (width + groupWidth - 1) / groupWidth;
        size_t numGroupsY = (height + groupHeight - 1) / groupHeight;
        size_t numGroupsZ = (depth + groupDepth - 1) / groupDepth;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(groupWidth, groupHeight, groupDepth);
        MTLSize threadgroupsPerGrid = MTLSizeMake(numGroupsX, numGroupsY, numGroupsZ);
        
        return {threadgroupsPerGrid, threadsPerThreadgroup};
    }
}