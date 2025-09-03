#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

// Include backend headers
#include "../../../../backend/backend-metal/src/metal_common.hpp"
#include "../../../../backend/backend-metal/src/metal_tensor.hpp"

// Include the CUDA backend's ztensor implementation
#include "../../../../backend/backend-cuda/src/ztensor.hpp"

class MetalTensorLoadingTest {
public:
    void run_all_tests() {
        std::cout << "Running MetalTensor Loading Tests...\n";

        try {
            setup_metal_context();

            test_basic_tensor_creation();
            test_load_from_ztensor_same_type();
            test_load_from_ztensor_type_conversion();
            test_tensor_data_validation();

            std::cout << "✅ All MetalTensor loading tests passed!\n";

        } catch (const std::exception& e) {
            std::cerr << "❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }

private:
    const std::string model_path = "/Users/seung-seoblee/Library/Caches/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt";

    void setup_metal_context() {
        std::cout << "Setting up Metal context...\n";
        try {
            auto& context = MetalContext::getInstance();
            if (!context.initialize()) {
                throw std::runtime_error("Failed to initialize Metal context");
            }
            std::cout << "  ✅ Metal context initialized\n";
        } catch (const std::exception& e) {
            std::cerr << "  ❌ Failed to setup Metal context: " << e.what() << std::endl;
            throw;
        }
    }

    void test_basic_tensor_creation() {
        std::cout << "Testing basic MetalTensor creation...\n";

        // Test float tensor
        MetalTensor<float> float_tensor({10, 20});
        assert(float_tensor.shape().size() == 2);
        assert(float_tensor.shape()[0] == 10);
        assert(float_tensor.shape()[1] == 20);
        assert(float_tensor.size() == 200);

        // Test half tensor (for bfloat16 compatibility)
        MetalTensor<uint16_t> half_tensor({5, 5});
        assert(half_tensor.size() == 25);

        std::cout << "  ✅ Basic tensor creation test passed\n";
    }

    void test_load_from_ztensor_same_type() {
        std::cout << "Testing loading from zTensor (same type)...\n";

        try {
            ztensor::zTensorReader reader(model_path);
            auto tensor_list = reader.list_tensors();

            // Find a small tensor to test with (layer norm weights are small)
            std::string test_tensor_name;
            for (const auto& name : tensor_list) {
                if (name.find("layernorm.weight") != std::string::npos) {
                    test_tensor_name = name;
                    break;
                }
            }

            if (test_tensor_name.empty()) {
                throw std::runtime_error("Could not find a suitable test tensor");
            }

            // Get tensor info and raw data pointer
            auto info = reader.get_tensor_info(test_tensor_name);
            const void* raw_data = reader.get_raw_tensor_pointer(test_tensor_name);

            std::cout << "  Testing with tensor: " << test_tensor_name << "\n";
            std::cout << "    Shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << info.shape[i];
            }
            std::cout << "], size: " << info.size << " bytes\n";

            // Calculate total elements
            size_t total_elements = 1;
            std::vector<size_t> shape_vec;
            for (auto dim : info.shape) {
                total_elements *= dim;
                shape_vec.push_back(static_cast<size_t>(dim));
            }

            // Create MetalTensor with same shape
            // Note: We'll use uint16_t to represent bfloat16 data
            MetalTensor<uint16_t> metal_tensor(shape_vec);

            // Load data using copyFromMappedMemory
            metal_tensor.copyFromMappedMemory(raw_data, total_elements);

            std::cout << "  ✅ Successfully loaded tensor data to Metal buffer\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in same-type loading: " << e.what() << std::endl;
            throw;
        }
    }

    void test_load_from_ztensor_type_conversion() {
        std::cout << "Testing loading from zTensor (with type conversion)...\n";

        try {
            ztensor::zTensorReader reader(model_path);
            auto tensor_list = reader.list_tensors();

            // Find a small tensor to test with
            std::string test_tensor_name;
            for (const auto& name : tensor_list) {
                if (name.find("layernorm.weight") != std::string::npos) {
                    test_tensor_name = name;
                    break;
                }
            }

            if (test_tensor_name.empty()) {
                throw std::runtime_error("Could not find a suitable test tensor");
            }

            // Get tensor info and raw data pointer
            auto info = reader.get_tensor_info(test_tensor_name);
            const uint16_t* raw_data = static_cast<const uint16_t*>(reader.get_raw_tensor_pointer(test_tensor_name));

            // Calculate total elements
            size_t total_elements = 1;
            std::vector<size_t> shape_vec;
            for (auto dim : info.shape) {
                total_elements *= dim;
                shape_vec.push_back(static_cast<size_t>(dim));
            }

            // Create MetalTensor with float32 type (conversion from bfloat16)
            MetalTensor<float> metal_tensor(shape_vec);

            // Load data using copyFromMappedMemory with type conversion
            metal_tensor.copyFromMappedMemory<uint16_t>(raw_data, total_elements);

            std::cout << "  ✅ Successfully loaded tensor data with type conversion\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in type conversion loading: " << e.what() << std::endl;
            throw;
        }
    }

    void test_tensor_data_validation() {
        std::cout << "Testing tensor data validation...\n";

        try {
            // Test with some dummy data to verify the copying works
            std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            std::vector<size_t> shape = {2, 3};

            MetalTensor<float> tensor(shape);
            tensor.copyFromMappedMemory(test_data.data(), test_data.size());

            // Copy data back from Metal buffer to verify
            std::vector<float> result_data(test_data.size());
            tensor.copyToHost(result_data.data());

            // Validate the data matches
            for (size_t i = 0; i < test_data.size(); ++i) {
                if (std::abs(test_data[i] - result_data[i]) > 1e-6f) {
                    throw std::runtime_error("Data mismatch at index " + std::to_string(i) +
                                           ": expected " + std::to_string(test_data[i]) +
                                           ", got " + std::to_string(result_data[i]));
                }
            }

            std::cout << "  ✅ Tensor data validation test passed\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in data validation: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    @autoreleasepool {
        try {
            MetalTensorLoadingTest test;
            test.run_all_tests();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
            return 1;
        }
    }
}