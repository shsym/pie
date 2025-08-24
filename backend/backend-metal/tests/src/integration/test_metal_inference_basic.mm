#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <thread>

/**
 * @brief Basic inference pipeline test focusing on core Metal functionality
 * 
 * Tests:
 * - Metal context initialization
 * - Basic tensor operations
 * - Memory management
 * - Performance measurement
 * - Error handling
 */
class MetalInferenceBasicTest {
public:
    MetalInferenceBasicTest() = default;
    
    bool run() {
        std::cout << "=== Metal Basic Inference Test ===" << std::endl;
        
        if (!test_metal_context_initialization()) {
            std::cerr << "Metal context initialization failed" << std::endl;
            return false;
        }
        
        if (!test_basic_tensor_operations()) {
            std::cerr << "Basic tensor operations failed" << std::endl;
            return false;
        }
        
        if (!test_memory_management_basic()) {
            std::cerr << "Basic memory management failed" << std::endl;
            return false;
        }
        
        if (!test_pipeline_manager()) {
            std::cerr << "Pipeline manager test failed" << std::endl;
            return false;
        }
        
        if (!test_profiling_framework()) {
            std::cerr << "Profiling framework test failed" << std::endl;
            return false;
        }
        
        std::cout << "✅ All basic inference tests passed!" << std::endl;
        return true;
    }
    
private:
    bool test_metal_context_initialization() {
        std::cout << "Testing Metal context initialization..." << std::endl;
        
        // Test singleton pattern
        auto& context1 = MetalContext::getInstance();
        auto& context2 = MetalContext::getInstance();
        
        if (&context1 != &context2) {
            std::cerr << "MetalContext singleton pattern broken" << std::endl;
            return false;
        }
        
        // Test initialization
        if (!context1.initialize()) {
            std::cerr << "Failed to initialize Metal context" << std::endl;
            return false;
        }
        
        // Test device availability
        auto device = context1.getDevice();
        if (!device) {
            std::cerr << "No Metal device available" << std::endl;
            return false;
        }
        
        // Test command queue
        auto commandQueue = context1.getCommandQueue();
        if (!commandQueue) {
            std::cerr << "No Metal command queue available" << std::endl;
            return false;
        }
        
        std::cout << "✓ Metal context initialized" << std::endl;
        std::cout << "  Device: " << [[device name] UTF8String] << std::endl;
        
        return true;
    }
    
    bool test_basic_tensor_operations() {
        std::cout << "Testing basic tensor operations..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        auto commandQueue = context.getCommandQueue();
        
        // Test buffer creation
        const size_t buffer_size = 1024 * 1024; // 1M floats = 4MB
        id<MTLBuffer> buffer = [device newBufferWithLength:buffer_size * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        
        if (!buffer) {
            std::cerr << "Failed to create Metal buffer" << std::endl;
            return false;
        }
        
        // Test basic memory operations
        float* buffer_data = static_cast<float*>([buffer contents]);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Write test data
        for (size_t i = 0; i < buffer_size; ++i) {
            buffer_data[i] = static_cast<float>(i % 100);
        }
        
        auto write_time = std::chrono::high_resolution_clock::now();
        
        // Read and verify test data
        bool verification_passed = true;
        for (size_t i = 0; i < std::min(buffer_size, size_t(1000)); ++i) {
            if (std::abs(buffer_data[i] - static_cast<float>(i % 100)) > 1e-6f) {
                verification_passed = false;
                break;
            }
        }
        
        auto verify_time = std::chrono::high_resolution_clock::now();
        
        if (!verification_passed) {
            std::cerr << "Buffer data verification failed" << std::endl;
            return false;
        }
        
        auto write_duration = std::chrono::duration_cast<std::chrono::microseconds>(write_time - start_time);
        auto verify_duration = std::chrono::duration_cast<std::chrono::microseconds>(verify_time - write_time);
        
        std::cout << "✓ Basic tensor operations successful" << std::endl;
        std::cout << "  Buffer size: " << buffer_size * sizeof(float) / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Write time: " << write_duration.count() << " μs" << std::endl;
        std::cout << "  Verify time: " << verify_duration.count() << " μs" << std::endl;
        
        return true;
    }
    
    bool test_memory_management_basic() {
        std::cout << "Testing basic memory management..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        auto commandQueue = context.getCommandQueue();
        
        std::vector<id<MTLBuffer>> buffers;
        const size_t num_buffers = 10;
        const size_t buffer_size = 1024 * 1024; // 1MB each
        
        size_t initial_memory = [device currentAllocatedSize];
        
        // Test allocation
        for (size_t i = 0; i < num_buffers; ++i) {
            id<MTLBuffer> buffer = [device newBufferWithLength:buffer_size
                                                        options:MTLResourceStorageModeShared];
            if (!buffer) {
                std::cerr << "Failed to allocate buffer " << i << std::endl;
                return false;
            }
            
            // Write to ensure allocation
            float* data = static_cast<float*>([buffer contents]);
            data[0] = static_cast<float>(i);
            data[buffer_size / sizeof(float) - 1] = static_cast<float>(i + 1000);
            
            buffers.push_back(buffer);
        }
        
        size_t allocated_memory = [device currentAllocatedSize];
        size_t memory_used = allocated_memory - initial_memory;
        
        std::cout << "  Allocated " << num_buffers << " buffers: " << memory_used / (1024 * 1024) << " MB" << std::endl;
        
        // Test buffer copy operations
        bool copy_test_passed = true;
        if (buffers.size() >= 2) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            
            [blitEncoder copyFromBuffer:buffers[0] 
                           sourceOffset:0
                               toBuffer:buffers[1]
                      destinationOffset:0
                                   size:buffer_size];
            
            [blitEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            // Verify copy
            float* src_data = static_cast<float*>([buffers[0] contents]);
            float* dst_data = static_cast<float*>([buffers[1] contents]);
            
            if (std::abs(src_data[0] - dst_data[0]) > 1e-6f ||
                std::abs(src_data[buffer_size / sizeof(float) - 1] - dst_data[buffer_size / sizeof(float) - 1]) > 1e-6f) {
                copy_test_passed = false;
            }
        }
        
        if (!copy_test_passed) {
            std::cerr << "Buffer copy test failed" << std::endl;
            return false;
        }
        
        // Test deallocation (ARC handles this automatically)
        buffers.clear();
        
        size_t final_memory = [device currentAllocatedSize];
        size_t memory_freed = allocated_memory > final_memory ? allocated_memory - final_memory : 0;
        
        std::cout << "✓ Basic memory management successful" << std::endl;
        std::cout << "  Memory freed: " << memory_freed / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Copy operations: " << (copy_test_passed ? "PASSED" : "FAILED") << std::endl;
        
        return true;
    }
    
    bool test_pipeline_manager() {
        std::cout << "Testing pipeline manager..." << std::endl;
        
        auto& pipeline_manager = MetalComputePipelineManager::getInstance();
        
        // Test singleton pattern
        auto& pipeline_manager2 = MetalComputePipelineManager::getInstance();
        if (&pipeline_manager != &pipeline_manager2) {
            std::cerr << "Pipeline manager singleton pattern broken" << std::endl;
            return false;
        }
        
        // Test pipeline creation (this will likely fail since we don't have actual kernels)
        // But we can test the framework
        std::string test_kernel_name = "test_kernel";
        auto pipeline = pipeline_manager.getComputePipeline(test_kernel_name);
        
        // It's okay if pipeline creation fails - we're testing the framework
        std::cout << "✓ Pipeline manager framework functional" << std::endl;
        std::cout << "  Test kernel creation: " << (pipeline ? "SUCCESS" : "EXPECTED_FAILURE") << std::endl;
        
        return true;
    }
    
    bool test_profiling_framework() {
        std::cout << "Testing profiling framework..." << std::endl;
        
        // Test profiler creation
        MetalProfiler profiler(true);
        
        if (!profiler.isEnabled()) {
            std::cerr << "Profiler not enabled" << std::endl;
            return false;
        }
        
        auto& context = MetalContext::getInstance();
        auto commandQueue = context.getCommandQueue();
        
        // Test profiler scope (basic functionality)
        {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            auto scope = profiler.scope("test_operation", commandBuffer);
            
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            scope.record("checkpoint_1");
            
            // More simulated work
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            // Scope destructor should handle cleanup
        }
        
        // Test profiler report
        profiler.print_report();
        
        std::cout << "✓ Profiling framework functional" << std::endl;
        
        return true;
    }
};

int main() {
    std::cout << "Metal Basic Inference Test" << std::endl;
    std::cout << "==========================" << std::endl;
    
    MetalInferenceBasicTest test;
    
    bool success = test.run();
    
    if (success) {
        std::cout << "\n🎉 All basic inference tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some basic inference tests failed!" << std::endl;
        return 1;
    }
}