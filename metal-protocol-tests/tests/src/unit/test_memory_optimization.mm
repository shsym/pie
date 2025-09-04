#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <iostream>
#import <vector>
#import <cassert>

// Include the headers for our memory optimization system
#import "metal_common.hpp"
#import "metal_tensor.hpp"
#import "metal_buffer.hpp"

class MemoryOptimizationTests {
public:
    static void run_all_tests() {
        std::cout << "=== Running Memory Optimization Tests ===" << std::endl;
        
        test_metal_tensor_view();
        test_persistent_memory_pool();
        test_zero_copy_buffer_mapping();
        test_cross_layer_memory_reuse();
        test_memory_pool_validation();
        test_performance_comparison();
        
        std::cout << "=== All Memory Optimization Tests Passed ===" << std::endl;
    }

private:
    static void test_metal_tensor_view() {
        std::cout << "Testing MetalTensorView zero-copy functionality..." << std::endl;
        
        // Create a large buffer
        const size_t buffer_size = 1024 * 1024; // 1MB
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
        
        id<MTLBuffer> large_buffer = [context.getDevice() newBufferWithLength:buffer_size
                                                                       options:MTLResourceStorageModeShared];
        assert(large_buffer != nullptr);
        
        // Fill buffer with test data
        float* buffer_data = static_cast<float*>([large_buffer contents]);
        for (size_t i = 0; i < buffer_size / sizeof(float); ++i) {
            buffer_data[i] = static_cast<float>(i);
        }
        
        // Create tensor views at different offsets
        size_t offset1 = 0;
        size_t offset2 = 1000 * sizeof(float);
        size_t offset3 = 2000 * sizeof(float);
        
        MetalTensorView<float> view1(large_buffer, {100, 10}, offset1);
        MetalTensorView<float> view2(large_buffer, {50, 20}, offset2);
        MetalTensorView<float> view3(large_buffer, {25, 40}, offset3);
        
        // Verify views reference correct memory locations
        assert(view1.data()[0] == 0.0f);
        assert(view1.data()[1] == 1.0f);
        assert(view2.data()[0] == 1000.0f);
        assert(view3.data()[0] == 2000.0f);
        
        // Verify views are marked as views
        assert(view1.isView() == true);
        assert(view2.isView() == true);
        assert(view3.isView() == true);
        
        // Verify shapes are correct
        assert(view1.shape()[0] == 100 && view1.shape()[1] == 10);
        assert(view2.shape()[0] == 50 && view2.shape()[1] == 20);
        assert(view3.shape()[0] == 25 && view3.shape()[1] == 40);
        
        std::cout << "  ✅ MetalTensorView zero-copy functionality working correctly" << std::endl;
    }
    
    static void test_persistent_memory_pool() {
        std::cout << "Testing PersistentMemoryPool allocation and management..." << std::endl;
        
        const size_t pool_size = 10 * 1024 * 1024; // 10MB
        PersistentMemoryPool pool(pool_size);
        
        // Initialize the pool
        assert(pool.initialize() == true);
        assert(pool.total_size() == pool_size);
        assert(pool.allocated_size() == 0);
        assert(pool.available_size() == pool_size);
        
        // Allocate some persistent regions
        auto* region1 = pool.allocate_persistent(1024, "test_region_1");
        auto* region2 = pool.allocate_persistent(2048, "test_region_2");
        auto* region3 = pool.allocate_persistent(4096, "test_region_3");
        
        assert(region1 != nullptr);
        assert(region2 != nullptr);
        assert(region3 != nullptr);
        
        // Verify region properties
        assert(region1->size >= 1024);  // May be aligned
        assert(region1->in_use == true);
        assert(region1->is_persistent == true);
        assert(region1->name == "test_region_1");
        
        // Allocate some temporary regions
        auto* temp1 = pool.allocate_temporary(512, "temp_1");
        auto* temp2 = pool.allocate_temporary(1024, "temp_2");
        
        assert(temp1 != nullptr);
        assert(temp2 != nullptr);
        assert(temp1->is_persistent == false);
        assert(temp2->is_persistent == false);
        
        // Test statistics
        size_t persistent_size = pool.persistent_allocated();
        size_t temporary_size = pool.temporary_allocated();
        size_t total_allocated = pool.allocated_size();
        
        assert(persistent_size > 0);
        assert(temporary_size > 0);
        assert(total_allocated == persistent_size + temporary_size);
        
        // Test temporary reset
        pool.reset_temporary();
        assert(pool.temporary_allocated() == 0);
        assert(pool.persistent_allocated() == persistent_size); // Should remain unchanged
        
        // Test full reset
        pool.reset_all();
        assert(pool.allocated_size() == 0);
        assert(pool.persistent_allocated() == 0);
        assert(pool.temporary_allocated() == 0);
        
        std::cout << "  ✅ PersistentMemoryPool allocation and management working correctly" << std::endl;
    }
    
    static void test_zero_copy_buffer_mapping() {
        std::cout << "Testing zero-copy buffer mapping..." << std::endl;
        
        // Initialize Metal context
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
        
        // Create test configuration
        L4maConfig config = {
            .type = "llama",
            .num_layers = 4,
            .num_query_heads = 8,
            .num_key_value_heads = 8,
            .head_size = 64,
            .hidden_size = 512,
            .intermediate_size = 1376,
            .vocab_size = 32000,
            .use_qkv_bias = false,
            .rms_norm_eps = 1e-6f,
            .rope_factor = 1.0f,
            .rope_high_frequency_factor = 1.0f,
            .rope_low_frequency_factor = 1.0f,
            .rope_theta = 10000.0f
        };
        
        const size_t buffer_size = 50 * 1024 * 1024; // 50MB
        MetalL4maBuffer<bfloat16_t> buffer(config, 16, 50, buffer_size);
        
        // Create memory pool
        PersistentMemoryPool pool(100 * 1024 * 1024); // 100MB
        assert(pool.initialize() == true);
        
        // Test host memory mapping
        const size_t host_memory_size = 4096;
        std::vector<uint8_t> host_data(host_memory_size, 42); // Fill with test pattern
        
        buffer.mapHostMemory(host_data.data(), host_memory_size);
        
        // Prepare test data for zero-copy planning
        std::vector<int32_t> input_ids = {1, 2, 3, 4, 5};
        std::vector<int32_t> position_ids = {0, 1, 2, 3, 4};
        std::vector<int32_t> qo_indptr = {0, 5}; // Single batch
        
        // Test zero-copy planning
        auto command_buffer = [context.getCommandQueue() commandBuffer];
        buffer.planWithMapping(
            command_buffer,
            pool,
            input_ids.data(), input_ids.size(),
            position_ids.data(), position_ids.size(),
            nullptr, 0, // kv_page_indices
            nullptr, 0, // kv_page_indptr  
            nullptr, 0, // kv_last_page_lens
            qo_indptr.data(), qo_indptr.size(),
            nullptr, 0, // custom_mask
            nullptr, 0, // mask_indptr
            nullptr, 0, // kv_batch_indices
            nullptr, 0, // kv_positions
            nullptr, 0  // output_indices_src
        );
        
        // Verify that tensors are properly mapped
        assert(buffer.input_ids.size() == input_ids.size());
        assert(buffer.position_ids.size() == position_ids.size());
        assert(buffer.qo_indptr.size() == qo_indptr.size());
        
        // Verify data integrity
        const int32_t* mapped_input_ids = buffer.input_ids.data();
        const int32_t* mapped_position_ids = buffer.position_ids.data();
        
        for (size_t i = 0; i < input_ids.size(); ++i) {
            assert(mapped_input_ids[i] == input_ids[i]);
            assert(mapped_position_ids[i] == position_ids[i]);
        }
        
        std::cout << "  ✅ Zero-copy buffer mapping working correctly" << std::endl;
    }
    
    static void test_cross_layer_memory_reuse() {
        std::cout << "Testing cross-layer memory reuse..." << std::endl;
        
        // Initialize Metal context
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
        
        // Create test configuration
        L4maConfig config = {
            .type = "llama",
            .num_layers = 3,
            .num_query_heads = 4,
            .num_key_value_heads = 4,
            .head_size = 32,
            .hidden_size = 128,
            .intermediate_size = 256,
            .vocab_size = 1000,
            .use_qkv_bias = false,
            .rms_norm_eps = 1e-6f,
            .rope_factor = 1.0f,
            .rope_high_frequency_factor = 1.0f,
            .rope_low_frequency_factor = 1.0f,
            .rope_theta = 10000.0f
        };
        
        const size_t buffer_size = 10 * 1024 * 1024; // 10MB
        MetalL4maBuffer<float> buffer(config, 16, 50, buffer_size);
        
        // Create memory pool
        PersistentMemoryPool pool(50 * 1024 * 1024); // 50MB
        assert(pool.initialize() == true);
        
        // Set buffer properties needed for workspace initialization
        buffer.num_tokens = 10;
        buffer.batch_size = 1;
        
        // Initialize persistent workspaces
        assert(buffer.initializePersistentWorkspaces(pool) == true);
        
        // Verify workspaces were created
        for (size_t layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
            auto* workspace = buffer.getLayerWorkspace(layer_idx);
            assert(workspace != nullptr);
            
            // Verify workspace tensor shapes
            assert(workspace->hidden_states.shape()[0] == buffer.num_tokens);
            assert(workspace->hidden_states.shape()[1] == config.hidden_size);
            assert(workspace->attention_output.shape()[0] == buffer.num_tokens);
            assert(workspace->attention_output.shape()[1] == config.hidden_size);
            assert(workspace->mlp_output.shape()[0] == buffer.num_tokens);
            assert(workspace->mlp_output.shape()[1] == config.hidden_size);
        }
        
        // Test memory pool statistics
        size_t allocated_before_temp = pool.allocated_size();
        
        // Allocate some temporary regions
        auto* temp1 = pool.allocate_temporary(1024, "temp_computation");
        auto* temp2 = pool.allocate_temporary(2048, "temp_intermediate");
        
        assert(temp1 != nullptr);
        assert(temp2 != nullptr);
        
        size_t allocated_after_temp = pool.allocated_size();
        assert(allocated_after_temp > allocated_before_temp);
        
        // Reset temporary regions (simulating end of layer)
        buffer.resetTemporaryRegions();
        
        size_t allocated_after_reset = pool.allocated_size();
        assert(allocated_after_reset == allocated_before_temp); // Temporary memory freed
        
        std::cout << "  ✅ Cross-layer memory reuse working correctly" << std::endl;
    }
    
    static void test_memory_pool_validation() {
        std::cout << "Testing memory pool validation and error handling..." << std::endl;
        
        const size_t pool_size = 1024; // Small pool to test limits
        PersistentMemoryPool pool(pool_size);
        assert(pool.initialize() == true);
        
        // Test allocation that should succeed
        auto* region1 = pool.allocate_persistent(256, "small_region");
        assert(region1 != nullptr);
        
        // Test allocation that should fail (too large)
        auto* region2 = pool.allocate_persistent(2048, "too_large");
        assert(region2 == nullptr); // Should fail
        
        // Test layout validation
        assert(pool.validate_layout() == true);
        
        // Create tensor view with valid region
        try {
            auto tensor_view = pool.create_tensor_view<float>(region1, {16, 4}); // 64 floats = 256 bytes
            assert(tensor_view.isView() == true);
        } catch (...) {
            assert(false && "Valid tensor view creation should not throw");
        }
        
        // Test tensor view with invalid size
        try {
            auto tensor_view = pool.create_tensor_view<float>(region1, {100, 100}); // Too large
            assert(false && "Oversized tensor view should throw");
        } catch (const std::runtime_error&) {
            // Expected
        }
        
        // Test statistics consistency
        size_t allocated = pool.allocated_size();
        size_t persistent = pool.persistent_allocated();
        size_t temporary = pool.temporary_allocated();
        assert(allocated == persistent + temporary);
        
        std::cout << "  ✅ Memory pool validation and error handling working correctly" << std::endl;
    }
    
    static void test_performance_comparison() {
        std::cout << "Testing performance comparison between traditional and zero-copy approaches..." << std::endl;
        
        // Initialize Metal context
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
        
        const size_t data_size = 1024 * 1024; // 1MB of floats
        std::vector<float> test_data(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            test_data[i] = static_cast<float>(i);
        }
        
        // Traditional approach: Create tensor with copy
        auto start_traditional = std::chrono::high_resolution_clock::now();
        {
            MetalTensor<float> traditional_tensor(test_data.data(), {data_size});
            // Simulate some access
            volatile float sum = 0;
            for (size_t i = 0; i < 1000; ++i) {
                sum += traditional_tensor.data()[i];
            }
        }
        auto end_traditional = std::chrono::high_resolution_clock::now();
        
        // Zero-copy approach: Create view
        auto start_zerocopy = std::chrono::high_resolution_clock::now();
        {
            // Create buffer and view
            id<MTLBuffer> buffer = [context.getDevice() newBufferWithBytesNoCopy:test_data.data()
                                                                           length:data_size * sizeof(float)
                                                                          options:MTLResourceStorageModeShared
                                                                      deallocator:nil];
            MetalTensorView<float> view(buffer, {data_size}, 0);
            
            // Simulate some access
            volatile float sum = 0;
            for (size_t i = 0; i < 1000; ++i) {
                sum += view.data()[i];
            }
        }
        auto end_zerocopy = std::chrono::high_resolution_clock::now();
        
        auto traditional_time = std::chrono::duration_cast<std::chrono::microseconds>(end_traditional - start_traditional).count();
        auto zerocopy_time = std::chrono::duration_cast<std::chrono::microseconds>(end_zerocopy - start_zerocopy).count();
        
        std::cout << "  Traditional approach: " << traditional_time << " μs" << std::endl;
        std::cout << "  Zero-copy approach: " << zerocopy_time << " μs" << std::endl;
        std::cout << "  Speedup: " << (double)traditional_time / zerocopy_time << "x" << std::endl;
        
        // Zero-copy should be faster (no memory copy)
        assert(zerocopy_time <= traditional_time);
        
        std::cout << "  ✅ Performance comparison shows zero-copy benefits" << std::endl;
    }
};

int main() {
    try {
        MemoryOptimizationTests::run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}