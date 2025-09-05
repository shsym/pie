#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <thread>

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

// MetalProfiler singleton implementation
MetalProfiler::MetalProfiler(bool enabled) : enabled_(enabled) {
}

MetalProfiler& MetalProfiler::getInstance() {
    static MetalProfiler instance(true);
    return instance;
}

void MetalProfiler::recordStart(const std::string& name, id<MTLCommandBuffer> commandBuffer) {
    if (!enabled_) return;
    start_times_[name] = std::chrono::high_resolution_clock::now();
}

void MetalProfiler::recordEnd(const std::string& name) {
    if (!enabled_) return;
    auto it = start_times_.find(name);
    if (it != start_times_.end()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second);
        timings_.push_back({name, duration.count() / 1000.0}); // Convert to milliseconds
        start_times_.erase(it);
    }
}

void MetalProfiler::record(const std::string& checkpoint) {
    if (!enabled_) return;
    // For simple checkpoints, just record them with current timestamp
    timings_.push_back({checkpoint, 0.0});
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

// MetalErrorHandling implementations
namespace MetalErrorHandling {

    GPUMemoryInfo getGPUMemoryInfo(id<MTLDevice> device) {
        GPUMemoryInfo info = {};

        if (!device) {
            return info;
        }

        info.current_allocated = [device currentAllocatedSize];
        info.recommended_max = [device recommendedMaxWorkingSetSize];
        info.max_buffer_length = [device maxBufferLength];
        info.has_unified_memory = [device hasUnifiedMemory];

        if (info.recommended_max > 0) {
            info.usage_ratio = static_cast<double>(info.current_allocated) / static_cast<double>(info.recommended_max);
        } else {
            info.usage_ratio = 0.0;
        }

        return info;
    }

    bool isMemoryPressureHigh(id<MTLDevice> device, double threshold) {
        GPUMemoryInfo info = getGPUMemoryInfo(device);
        return info.usage_ratio > threshold;
    }

    bool executeCommandBufferWithRetry(id<MTLCommandBuffer> commandBuffer, const RetryConfig& config) {
        if (!commandBuffer) {
            std::cerr << "MetalErrorHandling: Command buffer is null" << std::endl;
            return false;
        }

        int attempts = 0;
        double current_delay = config.initial_delay_ms;

        while (attempts <= config.max_retries) {
            attempts++;

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            NSError* error = commandBuffer.error;
            if (!error) {
                // Success
                if (attempts > 1) {
                    std::cout << "MetalErrorHandling: Command succeeded after " << attempts << " attempts" << std::endl;
                }
                return true;
            }

            // Log the error
            std::string errorDesc = [error.localizedDescription UTF8String];
            std::cerr << "MetalErrorHandling: Command buffer failed (attempt " << attempts << "/" << (config.max_retries + 1)
                      << "): " << errorDesc << std::endl;

            // Check if this is a recoverable error
            NSInteger errorCode = error.code;
            bool is_recoverable = false;

            // Metal error codes that might be recoverable
            if (errorCode == 14 || // Internal Error
                errorCode == 5 ||  // kIOGPUCommandBufferCallbackErrorInnocentVictim
                errorCode == 3 ||  // kIOGPUCommandBufferCallbackErrorTimeout
                errorCode == 4) {  // kIOGPUCommandBufferCallbackErrorResourceError
                is_recoverable = true;
            }

            if (!is_recoverable || attempts > config.max_retries) {
                std::cerr << "MetalErrorHandling: Command buffer failed after " << attempts << " attempts: " << errorDesc << std::endl;
                return false;
            }

            // Wait before retry (exponential backoff)
            if (config.exponential_backoff && attempts < config.max_retries) {
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(current_delay * 1000)));
                current_delay = std::min(current_delay * config.backoff_multiplier, config.max_delay_ms);
            }

            // Need to recreate command buffer for next attempt
            // This will be handled by the caller
            std::cout << "MetalErrorHandling: Retrying with delay " << current_delay << "ms..." << std::endl;
        }

        return false;
    }

    id<MTLCommandBuffer> createSafeCommandBuffer(id<MTLCommandQueue> commandQueue, id<MTLDevice> device) {
        if (!commandQueue) {
            std::cerr << "MetalErrorHandling: Command queue is null" << std::endl;
            return nullptr;
        }

        // Check memory pressure before creating command buffer
        if (isMemoryPressureHigh(device, 0.85)) {
            GPUMemoryInfo info = getGPUMemoryInfo(device);
            std::cerr << "MetalErrorHandling: High memory pressure detected - "
                      << info.current_allocated / (1024 * 1024) << " MB / "
                      << info.recommended_max / (1024 * 1024) << " MB ("
                      << static_cast<int>(info.usage_ratio * 100) << "%)" << std::endl;

            // Still create the command buffer but log the warning
            // In production, you might want to force garbage collection or defer the operation
        }

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        if (!commandBuffer) {
            std::cerr << "MetalErrorHandling: Failed to create command buffer" << std::endl;
        }

        return commandBuffer;
    }
}

// MetalCast implementations
namespace MetalCast {

    void bfloat16_to_float(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const bfloat16_t* input, float* output, size_t count) {
        // Convert bfloat16 to float32 by extending precision
        for (size_t i = 0; i < count; i++) {
            // bfloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
            // float32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
            // Conversion: shift bfloat16 left by 16 bits to add 16 zeros to mantissa
            uint32_t extended_bits = static_cast<uint32_t>(input[i]) << 16;
            output[i] = *reinterpret_cast<float*>(&extended_bits);
        }
    }

    void float_to_bfloat16(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const float* input, bfloat16_t* output, size_t count) {
        // Convert float32 to bfloat16 by truncating precision
        for (size_t i = 0; i < count; i++) {
            // Extract the float32 bits
            uint32_t float_bits = *reinterpret_cast<const uint32_t*>(&input[i]);

            // Handle special cases (NaN, infinity, zero)
            if ((float_bits & 0x7F800000) == 0x7F800000) {
                // NaN or infinity - preserve as is
                output[i] = static_cast<bfloat16_t>(float_bits >> 16);
            } else if ((float_bits & 0x7FFFFFFF) == 0) {
                // Zero (positive or negative) - preserve as is
                output[i] = static_cast<bfloat16_t>(float_bits >> 16);
            } else {
                // Normal case: truncate mantissa with proper rounding
                // Add rounding bias to bit 15 (the bit that will become the LSB of bfloat16)
                uint32_t rounding_bias = 0x7FFF + ((float_bits >> 16) & 1);  // round-to-even
                uint32_t rounded = float_bits + rounding_bias;
                output[i] = static_cast<bfloat16_t>(rounded >> 16);
            }
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
    NSURL* libraryURL = [NSURL fileURLWithPath:pathStr];
    NSError* error = nil;

    id<MTLLibrary> library = [context.getDevice() newLibraryWithURL:libraryURL error:&error];

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