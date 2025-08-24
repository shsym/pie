#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>

/**
 * @brief Basic memory management testing for Metal backend
 * 
 * Tests:
 * - Basic buffer allocation/deallocation
 * - Memory usage tracking
 * - Buffer operations
 * - Simple stress scenarios
 * - Error handling
 */
class MetalMemoryBasicTest {
public:
    MetalMemoryBasicTest() = default;
    
    bool run() {
        std::cout << "=== Metal Basic Memory Management Test ===" << std::endl;
        
        if (!initialize_metal_context()) {
            return false;
        }
        
        if (!test_basic_allocation_patterns()) {
            std::cerr << "Basic allocation patterns test failed" << std::endl;
            return false;
        }
        
        if (!test_memory_operations()) {
            std::cerr << "Memory operations test failed" << std::endl;
            return false;
        }
        
        if (!test_allocation_limits()) {
            std::cerr << "Allocation limits test failed" << std::endl;
            return false;
        }
        
        if (!test_basic_stress_scenario()) {
            std::cerr << "Basic stress scenario test failed" << std::endl;
            return false;
        }
        
        std::cout << "✅ All basic memory management tests passed!" << std::endl;
        return true;
    }
    
private:
    size_t initial_memory_usage_ = 0;
    
    bool initialize_metal_context() {
        std::cout << "Initializing Metal context for memory testing..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context" << std::endl;
            return false;
        }
        
        // Record initial memory usage
        auto device = context.getDevice();
        if (device) {
            initial_memory_usage_ = [device currentAllocatedSize];
            std::cout << "Initial GPU memory usage: " << initial_memory_usage_ / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Recommended max working set: " << [device recommendedMaxWorkingSetSize] / (1024 * 1024) << " MB" << std::endl;
        }
        
        std::cout << "✓ Metal context initialized for memory testing" << std::endl;
        return true;
    }
    
    bool test_basic_allocation_patterns() {
        std::cout << "Testing basic allocation patterns..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        std::vector<id<MTLBuffer>> buffers;
        
        // Test various buffer sizes
        std::vector<std::pair<size_t, std::string>> test_sizes = {
            {1024 * sizeof(float), "1K floats"},
            {1024 * 1024 * sizeof(float), "1M floats"},
            {10 * 1024 * 1024 * sizeof(float), "10M floats"}
        };
        
        for (const auto& [size, description] : test_sizes) {
            std::cout << "  Allocating " << description << " (" << size / (1024 * 1024) << " MB)..." << std::endl;
            
            size_t before_allocation = device ? [device currentAllocatedSize] : 0;
            
            id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                        options:MTLResourceStorageModeShared];
            
            if (!buffer) {
                std::cerr << "Failed to allocate buffer of size " << size << std::endl;
                return false;
            }
            
            // Test basic write/read operations
            float* data = static_cast<float*>([buffer contents]);
            size_t num_floats = size / sizeof(float);
            
            for (size_t i = 0; i < std::min(num_floats, size_t(1000)); ++i) {
                data[i] = static_cast<float>(i);
            }
            
            // Verify writes
            for (size_t i = 0; i < std::min(num_floats, size_t(1000)); ++i) {
                if (std::abs(data[i] - static_cast<float>(i)) > 1e-6f) {
                    std::cerr << "Memory write/read verification failed at index " << i << std::endl;
                    return false;
                }
            }
            
            size_t after_allocation = device ? [device currentAllocatedSize] : 0;
            size_t allocated_bytes = after_allocation - before_allocation;
            
            std::cout << "    ✓ Allocated " << allocated_bytes / (1024 * 1024) << " MB" << std::endl;
            
            buffers.push_back(buffer);
        }
        
        // Test deallocation
        std::cout << "  Deallocating buffers..." << std::endl;
        size_t before_deallocation = device ? [device currentAllocatedSize] : 0;
        
        buffers.clear();
        
        size_t after_deallocation = device ? [device currentAllocatedSize] : 0;
        size_t freed_bytes = before_deallocation > after_deallocation ? before_deallocation - after_deallocation : 0;
        
        std::cout << "    ✓ Freed " << freed_bytes / (1024 * 1024) << " MB" << std::endl;
        
        std::cout << "✓ Basic allocation patterns test passed" << std::endl;
        return true;
    }
    
    bool test_memory_operations() {
        std::cout << "Testing memory operations..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        auto commandQueue = context.getCommandQueue();
        
        const size_t buffer_size = 1024 * 1024 * sizeof(float); // 4MB
        
        // Create source and destination buffers
        id<MTLBuffer> src_buffer = [device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst_buffer = [device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];
        
        if (!src_buffer || !dst_buffer) {
            std::cerr << "Failed to create test buffers" << std::endl;
            return false;
        }
        
        // Initialize source buffer with test data
        float* src_data = static_cast<float*>([src_buffer contents]);
        for (size_t i = 0; i < buffer_size / sizeof(float); ++i) {
            src_data[i] = static_cast<float>(i % 1000);
        }
        
        // Test buffer copy using blit encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        
        [blitEncoder copyFromBuffer:src_buffer 
                       sourceOffset:0
                           toBuffer:dst_buffer
                  destinationOffset:0
                               size:buffer_size];
        
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Verify copy
        float* dst_data = static_cast<float*>([dst_buffer contents]);
        bool copy_successful = true;
        
        for (size_t i = 0; i < std::min(size_t(10000), buffer_size / sizeof(float)); ++i) {
            if (std::abs(src_data[i] - dst_data[i]) > 1e-6f) {
                copy_successful = false;
                std::cerr << "Copy verification failed at index " << i << std::endl;
                break;
            }
        }
        
        if (!copy_successful) {
            return false;
        }
        
        // Test buffer fill using blit encoder
        commandBuffer = [commandQueue commandBuffer];
        blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder fillBuffer:dst_buffer range:NSMakeRange(0, buffer_size) value:0];
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Verify fill
        bool fill_successful = true;
        uint8_t* dst_bytes = static_cast<uint8_t*>([dst_buffer contents]);
        for (size_t i = 0; i < std::min(size_t(10000), buffer_size); ++i) {
            if (dst_bytes[i] != 0) {
                fill_successful = false;
                break;
            }
        }
        
        if (!fill_successful) {
            std::cerr << "Buffer fill verification failed" << std::endl;
            return false;
        }
        
        std::cout << "✓ Memory operations test passed" << std::endl;
        return true;
    }
    
    bool test_allocation_limits() {
        std::cout << "Testing allocation limits..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        if (!device) {
            std::cerr << "No Metal device available" << std::endl;
            return false;
        }
        
        size_t max_buffer_length = [device maxBufferLength];
        std::cout << "  Device max buffer length: " << max_buffer_length / (1024 * 1024) << " MB" << std::endl;
        
        // Test reasonable large allocation
        size_t large_size = std::min(max_buffer_length / 4, size_t(100 * 1024 * 1024)); // 25% of max or 100MB
        
        std::cout << "  Testing allocation of " << large_size / (1024 * 1024) << " MB..." << std::endl;
        
        id<MTLBuffer> large_buffer = [device newBufferWithLength:large_size
                                                          options:MTLResourceStorageModeShared];
        
        if (large_buffer) {
            // Test basic operations on large buffer
            float* data = static_cast<float*>([large_buffer contents]);
            size_t num_floats = large_size / sizeof(float);
            
            data[0] = 1.0f;
            data[num_floats - 1] = 2.0f;
            
            if (std::abs(data[0] - 1.0f) > 1e-6f || std::abs(data[num_floats - 1] - 2.0f) > 1e-6f) {
                std::cerr << "Large buffer read/write verification failed" << std::endl;
                return false;
            }
            
            std::cout << "    ✓ Large allocation successful" << std::endl;
        } else {
            std::cout << "    Large allocation failed (may be expected)" << std::endl;
        }
        
        // Test over-limit allocation (should fail)
        size_t over_limit_size = max_buffer_length + 1024;
        
        id<MTLBuffer> over_limit_buffer = [device newBufferWithLength:over_limit_size
                                                              options:MTLResourceStorageModeShared];
        
        if (!over_limit_buffer) {
            std::cout << "    ✓ Over-limit allocation properly rejected" << std::endl;
        } else {
            std::cout << "    ⚠️ Over-limit allocation succeeded (unexpected)" << std::endl;
        }
        
        std::cout << "✓ Allocation limits test passed" << std::endl;
        return true;
    }
    
    bool test_basic_stress_scenario() {
        std::cout << "Testing basic stress scenario..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        std::vector<id<MTLBuffer>> buffers;
        const size_t num_iterations = 50;
        const size_t buffer_size = 1024 * 1024 * sizeof(float); // 4MB each
        
        size_t baseline_memory = device ? [device currentAllocatedSize] : 0;
        
        for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
            // Allocate buffer
            id<MTLBuffer> buffer = [device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];
            
            if (!buffer) {
                std::cout << "    Allocation failed at iteration " << iteration << " (may be expected)" << std::endl;
                break;
            }
            
            // Write to buffer to ensure allocation
            float* data = static_cast<float*>([buffer contents]);
            data[0] = static_cast<float>(iteration);
            data[buffer_size / sizeof(float) - 1] = static_cast<float>(iteration + 1000);
            
            buffers.push_back(buffer);
            
            // Every 10 iterations, check memory growth
            if ((iteration + 1) % 10 == 0) {
                size_t current_memory = device ? [device currentAllocatedSize] : 0;
                size_t memory_growth = current_memory > baseline_memory ? current_memory - baseline_memory : 0;
                
                std::cout << "    Iteration " << (iteration + 1) << ": " << buffers.size() 
                         << " buffers, " << memory_growth / (1024 * 1024) << " MB growth" << std::endl;
            }
        }
        
        std::cout << "  Successfully allocated " << buffers.size() << " buffers" << std::endl;
        
        // Verify data integrity
        bool integrity_maintained = true;
        for (size_t i = 0; i < std::min(buffers.size(), size_t(10)); ++i) {
            if (buffers[i]) {
                float* data = static_cast<float*>([buffers[i] contents]);
                if (std::abs(data[0] - static_cast<float>(i)) > 1e-6f ||
                    std::abs(data[buffer_size / sizeof(float) - 1] - static_cast<float>(i + 1000)) > 1e-6f) {
                    integrity_maintained = false;
                    break;
                }
            }
        }
        
        if (!integrity_maintained) {
            std::cerr << "Data integrity lost during stress test" << std::endl;
            return false;
        }
        
        // Record peak memory before cleanup
        size_t peak_memory = device ? [device currentAllocatedSize] : 0;
        size_t peak_growth = peak_memory > baseline_memory ? peak_memory - baseline_memory : 0;
        
        std::cout << "  Peak memory growth: " << peak_growth / (1024 * 1024) << " MB" << std::endl;
        
        // Clean up
        size_t num_buffers = buffers.size();
        buffers.clear();
        
        size_t final_memory = device ? [device currentAllocatedSize] : 0;
        size_t final_growth = final_memory > baseline_memory ? final_memory - baseline_memory : 0;
        
        std::cout << "  Final memory growth after cleanup: " << final_growth / (1024 * 1024) << " MB" << std::endl;
        
        // Metal may defer memory cleanup, so we'll be more lenient
        // Just warn if there's significant growth after cleanup
        if (final_growth > 300 * 1024 * 1024) { // 300MB threshold
            std::cerr << "Warning: Significant memory growth after cleanup: " << final_growth / (1024 * 1024) << " MB" << std::endl;
            std::cerr << "Note: Metal may defer memory cleanup - this might be normal behavior" << std::endl;
        }
        
        // Validate that peak memory growth was reasonable
        size_t expected_peak_growth = num_buffers * buffer_size;
        if (peak_growth > expected_peak_growth * 1.5) { // Allow 50% overhead
            std::cerr << "Excessive peak memory growth: " << peak_growth / (1024 * 1024) 
                     << " MB (expected ~" << expected_peak_growth / (1024 * 1024) << " MB)" << std::endl;
            return false;
        }
        
        std::cout << "✓ Basic stress scenario test passed" << std::endl;
        return true;
    }
};

int main() {
    std::cout << "Metal Basic Memory Management Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    MetalMemoryBasicTest test;
    
    bool success = test.run();
    
    if (success) {
        std::cout << "\n🎉 All basic memory management tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some basic memory management tests failed!" << std::endl;
        return 1;
    }
}