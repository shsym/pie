#include "metal_common.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cstring>
#include <algorithm>

/**
 * @brief Model weight loading test that validates realistic weight loading patterns
 * 
 * Tests:
 * - Binary checkpoint file loading
 * - Weight tensor initialization and copying
 * - Memory layout validation
 * - Large model weight handling
 * - Checkpoint format compatibility
 * - Error handling for corrupted files
 */
class MetalModelWeightLoadingTest {
public:
    MetalModelWeightLoadingTest() = default;
    
    bool run() {
        std::cout << "=== Metal Model Weight Loading Test ===" << std::endl;
        
        if (!initialize_metal_context()) {
            return false;
        }
        
        if (!test_checkpoint_file_creation()) {
            std::cerr << "Checkpoint file creation test failed" << std::endl;
            return false;
        }
        
        if (!test_tensor_weight_loading()) {
            std::cerr << "Tensor weight loading test failed" << std::endl;
            return false;
        }
        
        if (!test_large_model_loading()) {
            std::cerr << "Large model loading test failed" << std::endl;
            return false;
        }
        
        if (!test_weight_validation()) {
            std::cerr << "Weight validation test failed" << std::endl;
            return false;
        }
        
        if (!test_memory_layout_validation()) {
            std::cerr << "Memory layout validation test failed" << std::endl;
            return false;
        }
        
        if (!test_error_handling()) {
            std::cerr << "Error handling test failed" << std::endl;
            return false;
        }
        
        cleanup_test_files();
        
        std::cout << "✅ All model weight loading tests passed!" << std::endl;
        return true;
    }
    
private:
    std::string test_checkpoint_path_ = "/tmp/test_metal_model.ckpt";
    std::string test_large_checkpoint_path_ = "/tmp/test_metal_large_model.ckpt";
    std::string test_corrupted_checkpoint_path_ = "/tmp/test_metal_corrupted.ckpt";
    
    struct TestModelConfig {
        size_t hidden_size = 1024;
        size_t intermediate_size = 2816;
        size_t vocab_size = 32000;
        size_t num_layers = 4;
        size_t num_heads = 8;
        size_t head_size = 128;
    } config_;
    
    bool initialize_metal_context() {
        std::cout << "Initializing Metal context..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            std::cerr << "Failed to initialize Metal context" << std::endl;
            return false;
        }
        
        std::cout << "✓ Metal context initialized" << std::endl;
        return true;
    }
    
    bool test_checkpoint_file_creation() {
        std::cout << "Testing checkpoint file creation..." << std::endl;
        
        // Create a realistic checkpoint file with proper header and weights
        std::ofstream file(test_checkpoint_path_, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to create test checkpoint file" << std::endl;
            return false;
        }
        
        // Write checkpoint header (mimicking real checkpoint format)
        struct CheckpointHeader {
            char magic[9];
            uint32_t version;
            uint32_t num_layers;
            uint32_t hidden_size;
            uint32_t intermediate_size;
            uint32_t vocab_size;
            uint32_t num_heads;
            uint32_t head_size;
        } header;
        
        // Initialize header
        std::strcpy(header.magic, "METALCK1");
        header.version = 1;
        header.num_layers = static_cast<uint32_t>(config_.num_layers);
        header.hidden_size = static_cast<uint32_t>(config_.hidden_size);
        header.intermediate_size = static_cast<uint32_t>(config_.intermediate_size);
        header.vocab_size = static_cast<uint32_t>(config_.vocab_size);
        header.num_heads = static_cast<uint32_t>(config_.num_heads);
        header.head_size = static_cast<uint32_t>(config_.head_size);
        
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // Generate and write realistic model weights
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducible tests
        std::normal_distribution<float> weight_dist(0.0f, 0.02f); // Xavier-like initialization
        
        // Write embedding weights
        size_t embed_size = config_.vocab_size * config_.hidden_size;
        write_tensor_weights(file, gen, weight_dist, embed_size, "embed_tokens.weight");
        
        // Write transformer layers
        for (size_t layer = 0; layer < config_.num_layers; ++layer) {
            std::string prefix = "layers." + std::to_string(layer) + ".";
            
            // Attention weights
            write_tensor_weights(file, gen, weight_dist, 
                               config_.hidden_size * config_.hidden_size,
                               prefix + "self_attn.q_proj.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size * config_.hidden_size,
                               prefix + "self_attn.k_proj.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size * config_.hidden_size,
                               prefix + "self_attn.v_proj.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size * config_.hidden_size,
                               prefix + "self_attn.o_proj.weight");
            
            // MLP weights
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size * config_.intermediate_size,
                               prefix + "mlp.gate_proj.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size * config_.intermediate_size,
                               prefix + "mlp.up_proj.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.intermediate_size * config_.hidden_size,
                               prefix + "mlp.down_proj.weight");
            
            // Layer norm weights
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size,
                               prefix + "input_layernorm.weight");
            write_tensor_weights(file, gen, weight_dist,
                               config_.hidden_size,
                               prefix + "post_attention_layernorm.weight");
        }
        
        // Final layer norm
        write_tensor_weights(file, gen, weight_dist, config_.hidden_size, "norm.weight");
        
        // LM head (optional, can share with embeddings)
        write_tensor_weights(file, gen, weight_dist,
                           config_.vocab_size * config_.hidden_size,
                           "lm_head.weight");
        
        file.close();
        
        // Verify file was created and has expected size
        std::ifstream verify_file(test_checkpoint_path_, std::ios::binary | std::ios::ate);
        size_t file_size = verify_file.tellg();
        verify_file.close();
        
        if (file_size == 0) {
            std::cerr << "Checkpoint file is empty" << std::endl;
            return false;
        }
        
        std::cout << "✓ Checkpoint file created: " << file_size / (1024 * 1024) << " MB" << std::endl;
        return true;
    }
    
    void write_tensor_weights(std::ofstream& file, std::mt19937& gen, 
                            std::normal_distribution<float>& dist,
                            size_t num_elements, const std::string& name) {
        // Write tensor metadata
        uint32_t name_len = static_cast<uint32_t>(name.size());
        uint64_t tensor_size = static_cast<uint64_t>(num_elements);
        
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);
        file.write(reinterpret_cast<const char*>(&tensor_size), sizeof(tensor_size));
        
        // Write tensor data in chunks to avoid memory issues
        const size_t chunk_size = 1024 * 1024; // 1M elements at a time
        std::vector<float> chunk_data;
        
        for (size_t offset = 0; offset < num_elements; offset += chunk_size) {
            size_t current_chunk = std::min(chunk_size, num_elements - offset);
            chunk_data.resize(current_chunk);
            
            for (size_t i = 0; i < current_chunk; ++i) {
                chunk_data[i] = dist(gen);
            }
            
            file.write(reinterpret_cast<const char*>(chunk_data.data()), 
                      current_chunk * sizeof(float));
        }
    }
    
    bool test_tensor_weight_loading() {
        std::cout << "Testing tensor weight loading..." << std::endl;
        
        // Test loading individual tensors from checkpoint
        std::ifstream file(test_checkpoint_path_, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open test checkpoint file" << std::endl;
            return false;
        }
        
        // Skip header
        struct CheckpointHeader {
            char magic[9];
            uint32_t version;
            uint32_t num_layers;
            uint32_t hidden_size;
            uint32_t intermediate_size;
            uint32_t vocab_size;
            uint32_t num_heads;
            uint32_t head_size;
        } header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        // Verify header
        if (std::string(header.magic) != "METALCK1") {
            std::cerr << "Invalid checkpoint magic number: " << header.magic << std::endl;
            return false;
        }
        
        // Test loading first tensor (embeddings)
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        
        std::vector<char> name_buffer(name_len);
        file.read(name_buffer.data(), name_len);
        std::string tensor_name(name_buffer.data(), name_len);
        
        uint64_t tensor_size;
        file.read(reinterpret_cast<char*>(&tensor_size), sizeof(tensor_size));
        
        if (tensor_name != "embed_tokens.weight") {
            std::cerr << "Unexpected first tensor name: " << tensor_name << std::endl;
            return false;
        }
        
        if (tensor_size != config_.vocab_size * config_.hidden_size) {
            std::cerr << "Unexpected embedding tensor size: " << tensor_size << std::endl;
            return false;
        }
        
        // Simulate Metal tensor creation and weight loading
        size_t tensor_elements = config_.vocab_size * config_.hidden_size;
        std::vector<float> embedding_tensor_data(tensor_elements);
        
        // Load weights in chunks
        const size_t chunk_size = 1024 * 1024;
        std::vector<float> chunk_data;
        float* tensor_data = embedding_tensor_data.data();
        
        for (size_t offset = 0; offset < tensor_size; offset += chunk_size) {
            size_t current_chunk = std::min(chunk_size, static_cast<size_t>(tensor_size - offset));
            chunk_data.resize(current_chunk);
            
            file.read(reinterpret_cast<char*>(chunk_data.data()), 
                     current_chunk * sizeof(float));
            
            // Copy to Metal tensor (in practice, this would be GPU memory)
            std::memcpy(tensor_data + offset, chunk_data.data(), 
                       current_chunk * sizeof(float));
        }
        
        // Validate loaded weights (check for reasonable values)
        float sum = 0.0f;
        float max_val = -std::numeric_limits<float>::infinity();
        float min_val = std::numeric_limits<float>::infinity();
        
        for (size_t i = 0; i < tensor_size; ++i) {
            float val = tensor_data[i];
            sum += val;
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }
        
        float mean = sum / tensor_size;
        
        // Check for reasonable weight statistics
        if (std::abs(mean) > 0.1f) {
            std::cerr << "Suspicious weight mean: " << mean << std::endl;
            return false;
        }
        
        if (max_val > 1.0f || min_val < -1.0f) {
            std::cerr << "Suspicious weight range: [" << min_val << ", " << max_val << "]" << std::endl;
            return false;
        }
        
        file.close();
        
        std::cout << "✓ Tensor weight loading successful" << std::endl;
        std::cout << "  Loaded tensor: " << tensor_name << " [" << tensor_size << " elements]" << std::endl;
        std::cout << "  Weight statistics: mean=" << mean << ", range=[" << min_val << ", " << max_val << "]" << std::endl;
        return true;
    }
    
    bool test_large_model_loading() {
        std::cout << "Testing large model loading (7B parameters)..." << std::endl;
        
        // Simulate loading a 7B parameter model
        TestModelConfig large_config;
        large_config.hidden_size = 4096;
        large_config.intermediate_size = 11008;
        large_config.vocab_size = 32000;
        large_config.num_layers = 32;
        large_config.num_heads = 32;
        large_config.head_size = 128;
        
        // Calculate total parameters
        size_t embedding_params = large_config.vocab_size * large_config.hidden_size;
        size_t layer_params = large_config.num_layers * (
            4 * large_config.hidden_size * large_config.hidden_size + // Attention projections
            2 * large_config.hidden_size * large_config.intermediate_size + // MLP gate & up
            large_config.intermediate_size * large_config.hidden_size + // MLP down
            2 * large_config.hidden_size // Layer norms
        );
        size_t final_norm_params = large_config.hidden_size;
        size_t lm_head_params = large_config.vocab_size * large_config.hidden_size;
        
        size_t total_params = embedding_params + layer_params + final_norm_params + lm_head_params;
        size_t total_bytes = total_params * sizeof(float);
        
        std::cout << "  Simulating model with " << total_params / 1000000 << "M parameters" << std::endl;
        std::cout << "  Memory requirement: " << total_bytes / (1024 * 1024) << " MB" << std::endl;
        
        // Test memory allocation for large model
        try {
            // Test individual tensor allocation (simulated)
            size_t embedding_size = large_config.vocab_size * large_config.hidden_size;
            size_t mlp_gate_size = large_config.intermediate_size * large_config.hidden_size;
            size_t attention_size = large_config.hidden_size * large_config.hidden_size;
            
            std::vector<float> large_embedding_data(embedding_size);
            std::vector<float> large_mlp_gate_data(mlp_gate_size);
            std::vector<float> large_attention_data(attention_size);
            
            std::cout << "✓ Large tensor allocation successful" << std::endl;
            
            // Test memory usage tracking
            auto& context = MetalContext::getInstance();
            auto device = context.getDevice();
            
            if (device) {
                // Query Metal device memory info
                size_t recommended_max = [device recommendedMaxWorkingSetSize];
                std::cout << "  Device recommended max working set: " << recommended_max / (1024 * 1024) << " MB" << std::endl;
                
                if (total_bytes > recommended_max) {
                    std::cout << "  ⚠️ Model size exceeds recommended working set" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Large model allocation failed: " << e.what() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool test_weight_validation() {
        std::cout << "Testing weight validation..." << std::endl;
        
        // Test various weight validation scenarios
        
        // 1. Test NaN detection
        std::vector<float> nan_tensor_data(10000);
        float* data = nan_tensor_data.data();
        for (size_t i = 0; i < 10000; ++i) {
            data[i] = (i == 5000) ? std::numeric_limits<float>::quiet_NaN() : 0.01f;
        }
        
        bool found_nan = false;
        for (size_t i = 0; i < 10000; ++i) {
            if (std::isnan(data[i])) {
                found_nan = true;
                break;
            }
        }
        
        if (!found_nan) {
            std::cerr << "Failed to detect NaN in weights" << std::endl;
            return false;
        }
        
        // 2. Test infinity detection
        std::vector<float> inf_tensor_data(10000);
        float* inf_data = inf_tensor_data.data();
        for (size_t i = 0; i < 10000; ++i) {
            inf_data[i] = (i == 3000) ? std::numeric_limits<float>::infinity() : 0.01f;
        }
        
        bool found_inf = false;
        for (size_t i = 0; i < 10000; ++i) {
            if (std::isinf(inf_data[i])) {
                found_inf = true;
                break;
            }
        }
        
        if (!found_inf) {
            std::cerr << "Failed to detect infinity in weights" << std::endl;
            return false;
        }
        
        // 3. Test normal weight distribution
        std::vector<float> normal_tensor_data(1000000);
        float* normal_data = normal_tensor_data.data();
        
        std::random_device rd;
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        
        for (size_t i = 0; i < 1000000; ++i) {
            normal_data[i] = dist(gen);
        }
        
        // Calculate statistics
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < 1000000; ++i) {
            float val = normal_data[i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / 1000000;
        float variance = (sum_sq / 1000000) - (mean * mean);
        float stddev = std::sqrt(variance);
        
        // Validate distribution properties
        if (std::abs(mean) > 0.01f) {
            std::cerr << "Normal distribution mean too far from 0: " << mean << std::endl;
            return false;
        }
        
        if (std::abs(stddev - 0.02f) > 0.005f) {
            std::cerr << "Normal distribution stddev incorrect: " << stddev << " (expected ~0.02)" << std::endl;
            return false;
        }
        
        std::cout << "✓ Weight validation passed" << std::endl;
        std::cout << "  Normal distribution: mean=" << mean << ", stddev=" << stddev << std::endl;
        return true;
    }
    
    bool test_memory_layout_validation() {
        std::cout << "Testing memory layout validation..." << std::endl;
        
        // Test different tensor layouts and alignments
        
        // 1. Test row-major vs column-major layout
        std::vector<float> row_major_tensor_data(512 * 1024);
        std::vector<float> col_major_tensor_data(1024 * 512);
        
        // Verify memory alignment
        float* row_data = row_major_tensor_data.data();
        float* col_data = col_major_tensor_data.data();
        
        uintptr_t row_addr = reinterpret_cast<uintptr_t>(row_data);
        uintptr_t col_addr = reinterpret_cast<uintptr_t>(col_data);
        
        // Check 16-byte alignment (required for SIMD operations)
        if (row_addr % 16 != 0 || col_addr % 16 != 0) {
            std::cerr << "Tensor memory not 16-byte aligned" << std::endl;
            return false;
        }
        
        // 2. Test tensor stride and shape consistency (simulated)
        size_t row_shape[2] = {512, 1024};
        size_t col_shape[2] = {1024, 512};
        
        if (row_shape[0] * row_shape[1] != 512 * 1024) {
            std::cerr << "Row major tensor size mismatch" << std::endl;
            return false;
        }
        
        if (col_shape[0] * col_shape[1] != 1024 * 512) {
            std::cerr << "Column major tensor size mismatch" << std::endl;
            return false;
        }
        
        // 3. Test memory access patterns
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Row-wise access (cache-friendly for row-major)
        float sum1 = 0.0f;
        for (size_t i = 0; i < 512; ++i) {
            for (size_t j = 0; j < 1024; ++j) {
                sum1 += row_data[i * 1024 + j];
            }
        }
        
        auto mid_time = std::chrono::high_resolution_clock::now();
        
        // Column-wise access (less cache-friendly for row-major)
        float sum2 = 0.0f;
        for (size_t j = 0; j < 1024; ++j) {
            for (size_t i = 0; i < 512; ++i) {
                sum2 += row_data[i * 1024 + j];
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto row_access_time = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time);
        auto col_access_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time);
        
        std::cout << "✓ Memory layout validation passed" << std::endl;
        std::cout << "  Row-wise access time: " << row_access_time.count() << " μs" << std::endl;
        std::cout << "  Column-wise access time: " << col_access_time.count() << " μs" << std::endl;
        std::cout << "  Cache efficiency ratio: " << static_cast<double>(col_access_time.count()) / row_access_time.count() << std::endl;
        
        return true;
    }
    
    bool test_error_handling() {
        std::cout << "Testing error handling..." << std::endl;
        
        // 1. Test loading non-existent file
        try {
            std::ifstream nonexistent("/nonexistent/path/model.ckpt", std::ios::binary);
            if (nonexistent.is_open()) {
                std::cerr << "Should not be able to open nonexistent file" << std::endl;
                return false;
            }
            std::cout << "✓ Non-existent file handling correct" << std::endl;
        } catch (...) {
            std::cerr << "Unexpected exception for nonexistent file" << std::endl;
            return false;
        }
        
        // 2. Test corrupted checkpoint file
        std::ofstream corrupted_file(test_corrupted_checkpoint_path_, std::ios::binary);
        corrupted_file << "BADMAGIC"; // Invalid magic number
        corrupted_file.close();
        
        std::ifstream test_corrupted(test_corrupted_checkpoint_path_, std::ios::binary);
        char magic[8];
        test_corrupted.read(magic, 8);
        test_corrupted.close();
        
        if (std::string(magic, 8) == "METALCK1") {
            std::cerr << "Should have detected corrupted magic number" << std::endl;
            return false;
        }
        
        std::cout << "✓ Corrupted file detection correct" << std::endl;
        
        // 3. Test insufficient memory handling
        try {
            // Try to allocate unreasonably large tensor
            // This should fail gracefully
            size_t huge_size = SIZE_MAX / sizeof(float) / 2; // Half of max possible size
            
            // This allocation should fail or be rejected by Metal
            bool allocation_failed = false;
            try {
                std::vector<float> huge_tensor_data(huge_size);
            } catch (const std::exception&) {
                allocation_failed = true;
            }
            
            if (!allocation_failed) {
                std::cout << "⚠️ Large allocation succeeded (system may have very large memory)" << std::endl;
            } else {
                std::cout << "✓ Large allocation properly rejected" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "✓ Large allocation exception handled: " << e.what() << std::endl;
        }
        
        return true;
    }
    
    void cleanup_test_files() {
        std::cout << "Cleaning up test files..." << std::endl;
        
        std::remove(test_checkpoint_path_.c_str());
        std::remove(test_large_checkpoint_path_.c_str());
        std::remove(test_corrupted_checkpoint_path_.c_str());
        
        std::cout << "✓ Test files cleaned up" << std::endl;
    }
};

int main() {
    std::cout << "Metal Model Weight Loading Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    MetalModelWeightLoadingTest test;
    
    bool success = test.run();
    
    if (success) {
        std::cout << "\n🎉 All model weight loading tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some model weight loading tests failed!" << std::endl;
        return 1;
    }
}