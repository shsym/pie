#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>

/**
 * @brief Memory management and stress testing for Metal backend
 * 
 * Tests:
 * - Buffer allocation/deallocation cycles
 * - Memory fragmentation handling
 * - Concurrent memory operations
 * - Memory leak detection
 * - Large allocation patterns
 * - Buffer reuse and pooling
 * - Memory pressure scenarios
 * - GPU memory limits
 */
class MetalMemoryStressTest {
public:
    MetalMemoryStressTest() = default;
    
    bool run() {
        std::cout << "=== Metal Memory Management & Stress Test ===" << std::endl;
        
        if (!initialize_metal_context()) {
            return false;
        }
        
        if (!test_basic_allocation_patterns()) {
            std::cerr << "Basic allocation patterns test failed" << std::endl;
            return false;
        }
        
        if (!test_memory_fragmentation()) {
            std::cerr << "Memory fragmentation test failed" << std::endl;
            return false;
        }
        
        if (!test_concurrent_memory_operations()) {
            std::cerr << "Concurrent memory operations test failed" << std::endl;
            return false;
        }
        
        if (!test_memory_leak_detection()) {
            std::cerr << "Memory leak detection test failed" << std::endl;
            return false;
        }
        
        if (!test_large_allocation_stress()) {
            std::cerr << "Large allocation stress test failed" << std::endl;
            return false;
        }
        
        if (!test_buffer_reuse_patterns()) {
            std::cerr << "Buffer reuse patterns test failed" << std::endl;
            return false;
        }
        
        if (!test_memory_pressure_scenarios()) {
            std::cerr << "Memory pressure scenarios test failed" << std::endl;
            return false;
        }
        
        if (!test_gpu_memory_limits()) {
            std::cerr << "GPU memory limits test failed" << std::endl;
            return false;
        }
        
        std::cout << "✅ All memory management and stress tests passed!" << std::endl;
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
        
        std::vector<std::unique_ptr<MetalTensor<float>>> tensors;
        
        // Test various tensor sizes
        std::vector<std::pair<size_t, std::string>> test_sizes = {
            {1024, "1K elements"},
            {1024 * 1024, "1M elements"},
            {10 * 1024 * 1024, "10M elements"},
            {100 * 1024 * 1024, "100M elements"}
        };
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        for (const auto& [size, description] : test_sizes) {
            std::cout << "  Allocating " << description << "..." << std::endl;
            
            size_t before_allocation = device ? [device currentAllocatedSize] : 0;
            
            try {
                auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{size});
                
                // Verify allocation
                if (!tensor || !tensor->data()) {
                    std::cerr << "Failed to allocate tensor of size " << size << std::endl;
                    return false;
                }
                
                // Test basic write/read operations
                float* data = tensor->data();
                for (size_t i = 0; i < std::min(size, size_t(1000)); ++i) {
                    data[i] = static_cast<float>(i);
                }
                
                // Verify writes
                for (size_t i = 0; i < std::min(size, size_t(1000)); ++i) {
                    if (std::abs(data[i] - static_cast<float>(i)) > 1e-6f) {
                        std::cerr << "Memory write/read verification failed at index " << i << std::endl;
                        return false;
                    }
                }
                
                size_t after_allocation = device ? [device currentAllocatedSize] : 0;
                size_t allocated_bytes = after_allocation - before_allocation;
                
                std::cout << "    ✓ Allocated " << allocated_bytes / (1024 * 1024) << " MB" << std::endl;
                
                tensors.push_back(std::move(tensor));
                
            } catch (const std::exception& e) {
                std::cerr << "Exception during allocation of " << description << ": " << e.what() << std::endl;
                return false;
            }
        }
        
        // Test deallocation
        std::cout << "  Deallocating tensors..." << std::endl;
        size_t before_deallocation = device ? [device currentAllocatedSize] : 0;
        
        tensors.clear();
        
        size_t after_deallocation = device ? [device currentAllocatedSize] : 0;
        size_t freed_bytes = before_deallocation - after_deallocation;
        
        std::cout << "    ✓ Freed " << freed_bytes / (1024 * 1024) << " MB" << std::endl;
        
        std::cout << "✓ Basic allocation patterns test passed" << std::endl;
        return true;
    }
    
    bool test_memory_fragmentation() {
        std::cout << "Testing memory fragmentation handling..." << std::endl;
        
        std::vector<std::unique_ptr<MetalTensor<float>>> tensors;
        std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_int_distribution<size_t> size_dist(1024, 10 * 1024 * 1024);
        
        // Allocate many tensors of random sizes
        const size_t num_tensors = 50;
        std::cout << "  Allocating " << num_tensors << " tensors of random sizes..." << std::endl;
        
        for (size_t i = 0; i < num_tensors; ++i) {
            size_t tensor_size = size_dist(gen);
            
            try {
                auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{tensor_size});
                tensors.push_back(std::move(tensor));
            } catch (const std::exception& e) {
                std::cout << "    Allocation " << i << " failed (expected for large sizes): " << e.what() << std::endl;
                break;
            }
        }
        
        std::cout << "    Successfully allocated " << tensors.size() << " tensors" << std::endl;
        
        // Randomly deallocate half of the tensors to create fragmentation
        std::vector<size_t> indices(tensors.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        size_t to_remove = tensors.size() / 2;
        std::cout << "  Randomly deallocating " << to_remove << " tensors to create fragmentation..." << std::endl;
        
        for (size_t i = 0; i < to_remove; ++i) {
            tensors[indices[i]].reset();
        }
        
        // Try to allocate new tensors in the fragmented space
        std::cout << "  Allocating new tensors in fragmented space..." << std::endl;
        
        size_t successful_allocations = 0;
        for (size_t i = 0; i < 20; ++i) {
            size_t tensor_size = size_dist(gen) / 4; // Smaller sizes more likely to fit
            
            try {
                auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{tensor_size});
                if (tensor && tensor->data()) {
                    successful_allocations++;
                    
                    // Find an empty slot
                    for (auto& existing : tensors) {
                        if (!existing) {
                            existing = std::move(tensor);
                            break;
                        }
                    }
                }
            } catch (const std::exception&) {
                // Expected for some allocations
            }
        }
        
        std::cout << "    Successfully allocated " << successful_allocations << "/20 tensors in fragmented space" << std::endl;
        
        // Clean up
        tensors.clear();
        
        std::cout << "✓ Memory fragmentation test passed" << std::endl;
        return true;
    }
    
    bool test_concurrent_memory_operations() {
        std::cout << "Testing concurrent memory operations..." << std::endl;
        
        const size_t num_threads = 4;
        const size_t allocations_per_thread = 10;
        std::vector<std::thread> threads;
        std::atomic<bool> success{true};
        std::atomic<size_t> total_allocations{0};
        
        auto allocation_worker = [&](size_t thread_id) {
            std::vector<std::unique_ptr<MetalTensor<float>>> thread_tensors;
            std::random_device rd;
            std::mt19937 gen(42 + thread_id);
            std::uniform_int_distribution<size_t> size_dist(1024, 1024 * 1024);
            
            try {
                for (size_t i = 0; i < allocations_per_thread; ++i) {
                    size_t tensor_size = size_dist(gen);
                    
                    auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{tensor_size});
                    
                    if (tensor && tensor->data()) {
                        // Test basic operations
                        float* data = tensor->data();
                        for (size_t j = 0; j < std::min(tensor_size, size_t(100)); ++j) {
                            data[j] = static_cast<float>(thread_id * 1000 + i * 100 + j);
                        }
                        
                        thread_tensors.push_back(std::move(tensor));
                        total_allocations++;
                    }
                    
                    // Random delay to increase chance of contention
                    std::this_thread::sleep_for(std::chrono::milliseconds(gen() % 5));
                }
                
                // Verify data integrity
                for (size_t i = 0; i < thread_tensors.size(); ++i) {
                    if (thread_tensors[i] && thread_tensors[i]->data()) {
                        float* data = thread_tensors[i]->data();
                        size_t tensor_size = thread_tensors[i]->shape()[0];
                        
                        for (size_t j = 0; j < std::min(tensor_size, size_t(100)); ++j) {
                            float expected = static_cast<float>(thread_id * 1000 + i * 100 + j);
                            if (std::abs(data[j] - expected) > 1e-6f) {
                                std::cerr << "Thread " << thread_id << " data corruption at tensor " << i << " index " << j << std::endl;
                                success = false;
                                return;
                            }
                        }
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Thread " << thread_id << " exception: " << e.what() << std::endl;
                success = false;
            }
        };
        
        // Launch worker threads
        std::cout << "  Launching " << num_threads << " concurrent allocation threads..." << std::endl;
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(allocation_worker, i);
        }
        
        // Wait for completion
        for (auto& thread : threads) {
            thread.join();
        }
        
        if (!success) {
            return false;
        }
        
        std::cout << "    Total successful allocations: " << total_allocations.load() << std::endl;
        std::cout << "✓ Concurrent memory operations test passed" << std::endl;
        return true;
    }
    
    bool test_memory_leak_detection() {
        std::cout << "Testing memory leak detection..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        size_t baseline_memory = device ? [device currentAllocatedSize] : 0;
        std::cout << "  Baseline memory usage: " << baseline_memory / (1024 * 1024) << " MB" << std::endl;
        
        // Perform allocation/deallocation cycles
        const size_t num_cycles = 100;
        const size_t tensors_per_cycle = 10;
        
        for (size_t cycle = 0; cycle < num_cycles; ++cycle) {
            std::vector<std::unique_ptr<MetalTensor<float>>> tensors;
            
            // Allocate tensors
            for (size_t i = 0; i < tensors_per_cycle; ++i) {
                try {
                    auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{1024 * 1024});
                    tensors.push_back(std::move(tensor));
                } catch (const std::exception&) {
                    // May fail due to memory pressure, which is fine
                    break;
                }
            }
            
            // Deallocate all tensors
            tensors.clear();
            
            // Check for memory growth every 10 cycles
            if ((cycle + 1) % 10 == 0) {
                size_t current_memory = device ? [device currentAllocatedSize] : 0;
                size_t memory_growth = current_memory > baseline_memory ? current_memory - baseline_memory : 0;
                
                std::cout << "    Cycle " << (cycle + 1) << ": Current memory " << current_memory / (1024 * 1024) << " MB, growth: " << memory_growth / (1024 * 1024) << " MB" << std::endl;
                
                // Allow some memory growth but detect significant leaks
                if (memory_growth > 100 * 1024 * 1024) { // 100MB threshold
                    std::cerr << "Potential memory leak detected: " << memory_growth / (1024 * 1024) << " MB growth" << std::endl;
                    return false;
                }
            }
        }
        
        size_t final_memory = device ? [device currentAllocatedSize] : 0;
        size_t total_growth = final_memory > baseline_memory ? final_memory - baseline_memory : 0;
        
        std::cout << "  Final memory usage: " << final_memory / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Total memory growth: " << total_growth / (1024 * 1024) << " MB" << std::endl;
        
        if (total_growth > 50 * 1024 * 1024) { // 50MB final threshold
            std::cerr << "Significant memory leak detected: " << total_growth / (1024 * 1024) << " MB total growth" << std::endl;
            return false;
        }
        
        std::cout << "✓ Memory leak detection test passed" << std::endl;
        return true;
    }
    
    bool test_large_allocation_stress() {
        std::cout << "Testing large allocation stress..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        size_t max_buffer_size = device ? [device maxBufferLength] : SIZE_MAX;
        size_t recommended_max = device ? [device recommendedMaxWorkingSetSize] : SIZE_MAX;
        
        std::cout << "  Device max buffer size: " << max_buffer_size / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Recommended max working set: " << recommended_max / (1024 * 1024) << " MB" << std::endl;
        
        // Test progressively larger allocations
        std::vector<size_t> test_sizes = {
            10 * 1024 * 1024,     // 10MB
            100 * 1024 * 1024,    // 100MB
            500 * 1024 * 1024,    // 500MB
            1024 * 1024 * 1024,   // 1GB
        };
        
        // Only test sizes that are reasonable for the device
        if (recommended_max < SIZE_MAX) {
            test_sizes.erase(
                std::remove_if(test_sizes.begin(), test_sizes.end(),
                    [recommended_max](size_t size) { 
                        return size * sizeof(float) > recommended_max; 
                    }),
                test_sizes.end()
            );
        }
        
        for (size_t byte_size : test_sizes) {
            size_t element_count = byte_size / sizeof(float);
            std::cout << "  Testing allocation of " << byte_size / (1024 * 1024) << " MB (" << element_count << " elements)..." << std::endl;
            
            try {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                auto large_tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{element_count});
                
                auto alloc_time = std::chrono::high_resolution_clock::now();
                
                if (!large_tensor || !large_tensor->data()) {
                    std::cout << "    Allocation failed (may be expected for very large sizes)" << std::endl;
                    continue;
                }
                
                // Test basic operations on large tensor
                float* data = large_tensor->data();
                
                // Write to first and last elements
                data[0] = 1.0f;
                data[element_count - 1] = 2.0f;
                
                // Verify writes
                if (std::abs(data[0] - 1.0f) > 1e-6f || std::abs(data[element_count - 1] - 2.0f) > 1e-6f) {
                    std::cerr << "Large tensor write/read verification failed" << std::endl;
                    return false;
                }
                
                auto verify_time = std::chrono::high_resolution_clock::now();
                
                auto alloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_time - start_time);
                auto verify_duration = std::chrono::duration_cast<std::chrono::milliseconds>(verify_time - alloc_time);
                
                std::cout << "    ✓ Allocation successful: " << alloc_duration.count() << "ms allocation, " 
                         << verify_duration.count() << "ms verification" << std::endl;
                
                // Explicit cleanup to measure deallocation time
                auto dealloc_start = std::chrono::high_resolution_clock::now();
                large_tensor.reset();
                auto dealloc_end = std::chrono::high_resolution_clock::now();
                
                auto dealloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_end - dealloc_start);
                std::cout << "    ✓ Deallocation: " << dealloc_duration.count() << "ms" << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "    Allocation failed (may be expected): " << e.what() << std::endl;
            }
        }
        
        std::cout << "✓ Large allocation stress test passed" << std::endl;
        return true;
    }
    
    bool test_buffer_reuse_patterns() {
        std::cout << "Testing buffer reuse patterns..." << std::endl;
        
        // Simulate common buffer reuse patterns in ML workloads
        const size_t buffer_size = 10 * 1024 * 1024; // 10M elements
        const size_t num_iterations = 50;
        
        std::vector<std::chrono::microseconds> allocation_times;
        std::vector<std::chrono::microseconds> deallocation_times;
        
        for (size_t i = 0; i < num_iterations; ++i) {
            auto alloc_start = std::chrono::high_resolution_clock::now();
            
            auto buffer = std::make_unique<MetalTensor<float>>(std::vector<size_t>{buffer_size});
            
            auto alloc_end = std::chrono::high_resolution_clock::now();
            
            if (!buffer || !buffer->data()) {
                std::cerr << "Buffer allocation failed at iteration " << i << std::endl;
                return false;
            }
            
            // Simulate typical usage pattern
            float* data = buffer->data();
            
            // Write pattern
            for (size_t j = 0; j < std::min(buffer_size, size_t(1000)); ++j) {
                data[j] = static_cast<float>(i * 1000 + j);
            }
            
            // Read pattern
            float sum = 0.0f;
            for (size_t j = 0; j < std::min(buffer_size, size_t(1000)); ++j) {
                sum += data[j];
            }
            
            auto dealloc_start = std::chrono::high_resolution_clock::now();
            buffer.reset();
            auto dealloc_end = std::chrono::high_resolution_clock::now();
            
            allocation_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start));
            deallocation_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(dealloc_end - dealloc_start));
        }
        
        // Analyze timing patterns
        auto avg_alloc = std::accumulate(allocation_times.begin(), allocation_times.end(), std::chrono::microseconds(0)) / allocation_times.size();
        auto avg_dealloc = std::accumulate(deallocation_times.begin(), deallocation_times.end(), std::chrono::microseconds(0)) / deallocation_times.size();
        
        auto min_alloc = *std::min_element(allocation_times.begin(), allocation_times.end());
        auto max_alloc = *std::max_element(allocation_times.begin(), allocation_times.end());
        
        std::cout << "  Buffer reuse timing analysis:" << std::endl;
        std::cout << "    Average allocation time: " << avg_alloc.count() << " μs" << std::endl;
        std::cout << "    Average deallocation time: " << avg_dealloc.count() << " μs" << std::endl;
        std::cout << "    Allocation time range: " << min_alloc.count() << " - " << max_alloc.count() << " μs" << std::endl;
        
        // Check for performance degradation
        auto first_10_avg = std::accumulate(allocation_times.begin(), allocation_times.begin() + 10, std::chrono::microseconds(0)) / 10;
        auto last_10_avg = std::accumulate(allocation_times.end() - 10, allocation_times.end(), std::chrono::microseconds(0)) / 10;
        
        double degradation_ratio = static_cast<double>(last_10_avg.count()) / first_10_avg.count();
        
        std::cout << "    Performance degradation ratio: " << degradation_ratio << std::endl;
        
        if (degradation_ratio > 2.0) {
            std::cerr << "Significant performance degradation detected in buffer reuse" << std::endl;
            return false;
        }
        
        std::cout << "✓ Buffer reuse patterns test passed" << std::endl;
        return true;
    }
    
    bool test_memory_pressure_scenarios() {
        std::cout << "Testing memory pressure scenarios..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        size_t recommended_max = device ? [device recommendedMaxWorkingSetSize] : SIZE_MAX;
        if (recommended_max == SIZE_MAX) {
            std::cout << "  Cannot determine device memory limits, skipping pressure test" << std::endl;
            return true;
        }
        
        std::cout << "  Recommended max working set: " << recommended_max / (1024 * 1024) << " MB" << std::endl;
        
        // Try to allocate up to 80% of recommended max
        size_t target_usage = recommended_max * 8 / 10;
        size_t tensor_size = 10 * 1024 * 1024; // 10MB per tensor
        size_t num_tensors = target_usage / (tensor_size * sizeof(float));
        
        std::vector<std::unique_ptr<MetalTensor<float>>> pressure_tensors;
        
        std::cout << "  Attempting to allocate " << num_tensors << " tensors (80% of max working set)..." << std::endl;
        
        size_t successful_allocations = 0;
        for (size_t i = 0; i < num_tensors; ++i) {
            try {
                auto tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{tensor_size});
                
                if (tensor && tensor->data()) {
                    // Write to tensor to ensure it's actually allocated
                    float* data = tensor->data();
                    data[0] = static_cast<float>(i);
                    data[tensor_size - 1] = static_cast<float>(i + 1000);
                    
                    pressure_tensors.push_back(std::move(tensor));
                    successful_allocations++;
                } else {
                    break;
                }
            } catch (const std::exception& e) {
                std::cout << "    Allocation failed at tensor " << i << ": " << e.what() << std::endl;
                break;
            }
        }
        
        size_t allocated_bytes = successful_allocations * tensor_size * sizeof(float);
        std::cout << "    Successfully allocated " << successful_allocations << " tensors (" 
                 << allocated_bytes / (1024 * 1024) << " MB)" << std::endl;
        
        // Test operations under memory pressure
        std::cout << "  Testing operations under memory pressure..." << std::endl;
        
        // Try to allocate one more tensor (should fail gracefully)
        bool additional_allocation_failed = false;
        try {
            auto extra_tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{tensor_size * 2});
            if (!extra_tensor || !extra_tensor->data()) {
                additional_allocation_failed = true;
            }
        } catch (const std::exception&) {
            additional_allocation_failed = true;
        }
        
        if (additional_allocation_failed) {
            std::cout << "    ✓ Additional allocation properly failed under pressure" << std::endl;
        } else {
            std::cout << "    Additional allocation succeeded (device has more memory than expected)" << std::endl;
        }
        
        // Verify existing tensors are still valid
        bool integrity_maintained = true;
        for (size_t i = 0; i < std::min(successful_allocations, size_t(10)); ++i) {
            if (pressure_tensors[i] && pressure_tensors[i]->data()) {
                float* data = pressure_tensors[i]->data();
                if (std::abs(data[0] - static_cast<float>(i)) > 1e-6f ||
                    std::abs(data[tensor_size - 1] - static_cast<float>(i + 1000)) > 1e-6f) {
                    integrity_maintained = false;
                    break;
                }
            }
        }
        
        if (!integrity_maintained) {
            std::cerr << "Data integrity lost under memory pressure" << std::endl;
            return false;
        }
        
        std::cout << "    ✓ Data integrity maintained under pressure" << std::endl;
        
        // Clean up gradually
        std::cout << "  Releasing memory pressure..." << std::endl;
        pressure_tensors.clear();
        
        std::cout << "✓ Memory pressure scenarios test passed" << std::endl;
        return true;
    }
    
    bool test_gpu_memory_limits() {
        std::cout << "Testing GPU memory limits..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        auto device = context.getDevice();
        
        if (!device) {
            std::cerr << "No Metal device available" << std::endl;
            return false;
        }
        
        // Query device capabilities
        size_t max_buffer_length = [device maxBufferLength];
        size_t max_threadgroup_memory = [device maxThreadgroupMemoryLength];
        size_t recommended_max_working_set = [device recommendedMaxWorkingSetSize];
        bool supports_unified_memory = [device hasUnifiedMemory];
        
        std::cout << "  Device memory capabilities:" << std::endl;
        std::cout << "    Max buffer length: " << max_buffer_length / (1024 * 1024) << " MB" << std::endl;
        std::cout << "    Max threadgroup memory: " << max_threadgroup_memory / 1024 << " KB" << std::endl;
        std::cout << "    Recommended max working set: " << recommended_max_working_set / (1024 * 1024) << " MB" << std::endl;
        std::cout << "    Unified memory: " << (supports_unified_memory ? "Yes" : "No") << std::endl;
        
        // Test buffer size limits
        std::cout << "  Testing buffer size limits..." << std::endl;
        
        // Try to allocate maximum size buffer (this may fail)
        size_t max_elements = max_buffer_length / sizeof(float);
        
        bool max_allocation_succeeded = false;
        try {
            auto max_tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{max_elements});
            if (max_tensor && max_tensor->data()) {
                max_allocation_succeeded = true;
                std::cout << "    ✓ Maximum buffer allocation succeeded" << std::endl;
                
                // Test basic operations on max buffer
                float* data = max_tensor->data();
                data[0] = 1.0f;
                data[max_elements - 1] = 2.0f;
                
                if (std::abs(data[0] - 1.0f) < 1e-6f && std::abs(data[max_elements - 1] - 2.0f) < 1e-6f) {
                    std::cout << "    ✓ Maximum buffer read/write operations successful" << std::endl;
                } else {
                    std::cerr << "Maximum buffer read/write verification failed" << std::endl;
                    return false;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "    Maximum buffer allocation failed (may be expected): " << e.what() << std::endl;
        }
        
        // Test allocation beyond limits (should fail gracefully)
        std::cout << "  Testing over-limit allocation handling..." << std::endl;
        
        bool over_limit_failed = false;
        try {
            // Try to allocate 150% of max buffer size
            size_t over_limit_elements = (max_elements * 3) / 2;
            auto over_limit_tensor = std::make_unique<MetalTensor<float>>(std::vector<size_t>{over_limit_elements});
            
            if (!over_limit_tensor || !over_limit_tensor->data()) {
                over_limit_failed = true;
            }
        } catch (const std::exception&) {
            over_limit_failed = true;
        }
        
        if (over_limit_failed) {
            std::cout << "    ✓ Over-limit allocation properly rejected" << std::endl;
        } else {
            std::cout << "    ⚠️ Over-limit allocation succeeded (unexpected)" << std::endl;
        }
        
        // Test current memory usage tracking
        size_t current_allocated = [device currentAllocatedSize];
        std::cout << "  Current GPU memory usage: " << current_allocated / (1024 * 1024) << " MB" << std::endl;
        
        std::cout << "✓ GPU memory limits test passed" << std::endl;
        return true;
    }
};

int main() {
    std::cout << "Metal Memory Management & Stress Test" << std::endl;
    std::cout << "======================================" << std::endl;
    
    MetalMemoryStressTest test;
    
    bool success = test.run();
    
    if (success) {
        std::cout << "\n🎉 All memory management and stress tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some memory management and stress tests failed!" << std::endl;
        return 1;
    }
}