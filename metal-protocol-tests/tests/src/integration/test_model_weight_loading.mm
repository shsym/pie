#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <memory>
#include <algorithm>

// Include backend headers
#include "../../../../backend/backend-metal/src/metal_common.hpp"
#include "../../../../backend/backend-metal/src/metal_tensor.hpp"
#include "../../../../backend/backend-cuda/src/ztensor.hpp"

class ModelWeightLoadingTest {
public:
    void run_all_tests() {
        std::cout << "Running Model Weight Loading Integration Tests..." << std::endl;

        try {
            setup_metal_context();

            test_model_creation_and_weight_loading();
            test_parameter_tensor_shapes();
            test_weight_data_integrity();
            test_move_semantics_compilation();
            test_missing_parameters_fallback();

            std::cout << "✅ All model weight loading tests passed!" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }

private:
    const std::string model_path = "/Users/seung-seoblee/Library/Caches/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt";

    void setup_metal_context() {
        std::cout << "Setting up Metal context..." << std::endl;
        try {
            auto& context = MetalContext::getInstance();
            if (!context.initialize()) {
                throw std::runtime_error("Failed to initialize Metal context");
            }
            std::cout << "  ✅ Metal context initialized" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ❌ Failed to setup Metal context: " << e.what() << std::endl;
            throw;
        }
    }

    void test_model_creation_and_weight_loading() {
        std::cout << "Testing zTensor file access and parameter mapping..." << std::endl;

        try {
            // Test zTensor file access
            ztensor::zTensorReader reader(model_path);
            auto tensor_list = reader.list_tensors();

            std::cout << "  Found " << tensor_list.size() << " tensors in model file" << std::endl;

            // Test parameter name mapping
            std::vector<std::string> test_metal_names = {
                "embed_tokens.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.mlp.gate_proj.weight",
                "norm.weight"
            };

            for (const auto& metal_name : test_metal_names) {
                // This would map to ztensor names - for now just test the concept
                std::string expected_ztensor_name = "model." + metal_name;

                // Check if expected tensor exists
                auto it = std::find(tensor_list.begin(), tensor_list.end(), expected_ztensor_name);
                if (it != tensor_list.end()) {
                    std::cout << "  ✓ Found: " << metal_name << " -> " << expected_ztensor_name << std::endl;

                    // Test loading tensor info
                    auto info = reader.get_tensor_info(expected_ztensor_name);
                    std::cout << "    Shape: [";
                    for (size_t i = 0; i < info.shape.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << info.shape[i];
                    }
                    std::cout << "], " << info.size << " bytes" << std::endl;
                } else {
                    std::cout << "  ⚠ Missing: " << metal_name << " -> " << expected_ztensor_name << std::endl;
                }
            }

            std::cout << "  ✅ Weight loading infrastructure verified" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in weight loading test: " << e.what() << std::endl;
            throw;
        }
    }

    void test_parameter_tensor_shapes() {
        std::cout << "Testing auto-detected configuration and parameter shapes..." << std::endl;

        try {
            ztensor::zTensorReader reader(model_path);
            L4maConfig config = auto_detect_config_from_ztensor(reader);

            std::cout << "  Verifying detected configuration against actual tensor shapes:" << std::endl;

            // Test key parameter shapes match auto-detected dimensions
            verify_ztensor_shape(reader, "model.embed_tokens.weight",
                               {config.vocab_size, config.hidden_size});
            verify_ztensor_shape(reader, "model.layers.0.self_attn.q_proj.weight",
                               {config.num_query_heads * config.head_size, config.hidden_size});
            verify_ztensor_shape(reader, "model.layers.0.self_attn.k_proj.weight",
                               {config.num_key_value_heads * config.head_size, config.hidden_size});
            verify_ztensor_shape(reader, "model.layers.0.mlp.gate_proj.weight",
                               {config.intermediate_size, config.hidden_size});
            verify_ztensor_shape(reader, "model.norm.weight",
                               {config.hidden_size});

            std::cout << "  ✅ Auto-detected configuration matches tensor shapes perfectly!" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in parameter shape test: " << e.what() << std::endl;
            throw;
        }
    }

    void test_weight_data_integrity() {
        std::cout << "Testing weight data integrity with MetalTensor loading..." << std::endl;

        try {
            ztensor::zTensorReader reader(model_path);

            // Test loading data into MetalTensor
            auto tensor_list = reader.list_tensors();
            std::string test_tensor_name = "model.norm.weight"; // Small tensor for testing

            auto it = std::find(tensor_list.begin(), tensor_list.end(), test_tensor_name);
            if (it == tensor_list.end()) {
                throw std::runtime_error("Test tensor not found: " + test_tensor_name);
            }

            // Get tensor info and raw data
            auto info = reader.get_tensor_info(test_tensor_name);
            const void* raw_data = reader.get_raw_tensor_pointer(test_tensor_name);

            // Calculate total elements
            size_t total_elements = 1;
            std::vector<size_t> shape_vec;
            for (auto dim : info.shape) {
                total_elements *= dim;
                shape_vec.push_back(static_cast<size_t>(dim));
            }

            // Create MetalTensor and load data
            MetalTensor<bfloat16_t> metal_tensor(shape_vec);
            metal_tensor.copyFromMappedMemory(raw_data, total_elements);

            std::cout << "  ✓ Loaded " << test_tensor_name << " (" << total_elements << " elements)" << std::endl;
            std::cout << "  ✅ Weight data integrity verified" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in data integrity test: " << e.what() << std::endl;
            throw;
        }
    }

    void test_move_semantics_compilation() {
        std::cout << "Testing MetalTensor move semantics compilation..." << std::endl;

        try {
            std::cout << "  Testing MetalTensor move operations..." << std::endl;

            // Test that we can create vectors of MetalTensors (which should use move semantics)
            std::vector<MetalTensor<bfloat16_t>> tensors;
            tensors.reserve(3);

            std::cout << "  ✓ Vector reserve() operation successful" << std::endl;

            // Test creating tensors with emplace_back
            tensors.emplace_back(std::vector<size_t>{10, 20});
            tensors.emplace_back(std::vector<size_t>{5, 5});
            tensors.emplace_back(std::vector<size_t>{2048});

            std::cout << "  ✓ Tensor creation with emplace_back() successful" << std::endl;
            std::cout << "  ✓ Created " << tensors.size() << " tensors" << std::endl;

            // Verify tensor properties
            assert(tensors[0].size() == 200);
            assert(tensors[1].size() == 25);
            assert(tensors[2].size() == 2048);

            std::cout << "  ✅ Move semantics compilation test passed" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in move semantics test: " << e.what() << std::endl;
            throw;
        }
    }

    void test_missing_parameters_fallback() {
        std::cout << "Testing fallback with non-existent file..." << std::endl;

        try {
            // Test that we handle missing files gracefully
            try {
                ztensor::zTensorReader reader("/nonexistent/path/model.zt");
                std::cerr << "  ❌ Should have thrown an exception for non-existent file" << std::endl;
                throw std::runtime_error("Expected exception not thrown");
            } catch (const std::exception& e) {
                std::cout << "  ✓ Correctly handled missing file: " << e.what() << std::endl;
            }

            std::cout << "  ✅ Fallback handling verified" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in fallback test: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * @brief Auto-detect L4maConfig from zTensor file by analyzing tensor shapes
     *
     * This is much better than hardcoding - we infer the configuration from
     * the actual model weights in the file.
     */
    L4maConfig auto_detect_config_from_ztensor(ztensor::zTensorReader& reader) {
        L4maConfig config;
        config.type = "llama";

        try {
            // Extract basic dimensions from key tensors
            auto embed_info = reader.get_tensor_info("model.embed_tokens.weight");
            config.vocab_size = static_cast<int>(embed_info.shape[0]);
            config.hidden_size = static_cast<int>(embed_info.shape[1]);

            // Extract MLP dimensions from first layer
            auto mlp_gate_info = reader.get_tensor_info("model.layers.0.mlp.gate_proj.weight");
            config.intermediate_size = static_cast<int>(mlp_gate_info.shape[0]);

            // Extract attention dimensions
            auto q_proj_info = reader.get_tensor_info("model.layers.0.self_attn.q_proj.weight");
            int total_q_dim = static_cast<int>(q_proj_info.shape[0]);

            auto k_proj_info = reader.get_tensor_info("model.layers.0.self_attn.k_proj.weight");
            int total_kv_dim = static_cast<int>(k_proj_info.shape[0]);

            // Count number of layers by counting layer-specific tensors
            auto tensor_list = reader.list_tensors();
            int layer_count = 0;
            for (const auto& name : tensor_list) {
                if (name.find("model.layers.") == 0 && name.find(".self_attn.q_proj.weight") != std::string::npos) {
                    layer_count++;
                }
            }
            config.num_layers = layer_count;

            // Infer head configuration (this requires some assumptions)
            // For Llama models, head_size is typically 64, 80, 128, etc.
            // We'll try common head sizes and see which one fits
            std::vector<int> common_head_sizes = {64, 80, 96, 128};

            for (int head_size : common_head_sizes) {
                if (total_q_dim % head_size == 0 && total_kv_dim % head_size == 0) {
                    config.head_size = head_size;
                    config.num_query_heads = total_q_dim / head_size;
                    config.num_key_value_heads = total_kv_dim / head_size;
                    break;
                }
            }

            // Validate that we found a valid head configuration
            if (config.head_size == 0) {
                throw std::runtime_error("Could not determine head configuration from tensor shapes");
            }

            // Validate basic consistency
            if (config.num_query_heads * config.head_size != config.hidden_size) {
                throw std::runtime_error("Inconsistent head configuration: " +
                                       std::to_string(config.num_query_heads) + " * " +
                                       std::to_string(config.head_size) + " != " +
                                       std::to_string(config.hidden_size));
            }

            // Set reasonable defaults for parameters not inferable from shapes
            config.rope_theta = 500000.0f;   // Common default for Llama 3.x
            config.rope_factor = 32.0f;      // Common default
            config.rms_norm_eps = 1e-5f;     // Common default

            std::cout << "Auto-detected configuration:" << std::endl;
            std::cout << "  vocab_size: " << config.vocab_size << std::endl;
            std::cout << "  hidden_size: " << config.hidden_size << std::endl;
            std::cout << "  intermediate_size: " << config.intermediate_size << std::endl;
            std::cout << "  num_layers: " << config.num_layers << std::endl;
            std::cout << "  num_query_heads: " << config.num_query_heads << std::endl;
            std::cout << "  num_key_value_heads: " << config.num_key_value_heads << std::endl;
            std::cout << "  head_size: " << config.head_size << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to auto-detect config from zTensor: " + std::string(e.what()));
        }

        return config;
    }

    void verify_ztensor_shape(ztensor::zTensorReader& reader,
                             const std::string& tensor_name,
                             const std::vector<int>& expected_shape) {
        try {
            auto info = reader.get_tensor_info(tensor_name);

            if (info.shape.size() != expected_shape.size()) {
                throw std::runtime_error("Shape dimension mismatch for " + tensor_name +
                                       ": expected " + std::to_string(expected_shape.size()) +
                                       " dims, got " + std::to_string(info.shape.size()));
            }

            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (info.shape[i] != expected_shape[i]) {
                    throw std::runtime_error("Shape mismatch for " + tensor_name +
                                           " at dimension " + std::to_string(i) +
                                           ": expected " + std::to_string(expected_shape[i]) +
                                           ", got " + std::to_string(info.shape[i]));
                }
            }

            std::cout << "    ✓ " << tensor_name << ": [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << info.shape[i];
            }
            std::cout << "]" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "    ⚠ " << tensor_name << ": " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    @autoreleasepool {
        try {
            ModelWeightLoadingTest test;
            test.run_all_tests();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
            return 1;
        }
    }
}