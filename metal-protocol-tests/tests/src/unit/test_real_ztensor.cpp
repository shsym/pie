#include <iostream>
#include <vector>
#include <string>
#include <cassert>

// Include the CUDA backend's ztensor implementation
#include "../../../../backend/backend-cuda/src/ztensor.hpp"

class RealZTensorTest {
public:
    void run_all_tests() {
        std::cout << "Running Real zTensor Tests with llama-3.2-1b-instruct.zt...\n";

        test_real_model_loading();
        test_tensor_listing();
        test_tensor_info_extraction();

        std::cout << "✅ All real zTensor tests passed!\n";
    }

private:
    const std::string model_path = "/Users/seung-seoblee/Library/Caches/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt";

    void test_real_model_loading() {
        std::cout << "Testing real model loading...\n";

        try {
            ztensor::zTensorReader reader(model_path);
            std::cout << "  ✅ Successfully loaded real zTensor file\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in real model loading: " << e.what() << std::endl;
            throw;
        }
    }

    void test_tensor_listing() {
        std::cout << "Testing tensor listing...\n";

        try {
            ztensor::zTensorReader reader(model_path);
            auto tensor_list = reader.list_tensors();

            std::cout << "  Found " << tensor_list.size() << " tensors:\n";
            for (const auto& tensor_name : tensor_list) {
                std::cout << "    - " << tensor_name << "\n";
            }

            // Basic sanity checks for Llama model
            assert(tensor_list.size() > 0);
            std::cout << "  ✅ Tensor listing test passed\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in tensor listing: " << e.what() << std::endl;
            throw;
        }
    }

    void test_tensor_info_extraction() {
        std::cout << "Testing tensor info extraction...\n";

        try {
            ztensor::zTensorReader reader(model_path);
            auto tensor_list = reader.list_tensors();

            // Test info extraction for first few tensors
            int tensors_to_test = std::min(5, (int)tensor_list.size());
            for (int i = 0; i < tensors_to_test; ++i) {
                const auto& tensor_name = tensor_list[i];
                auto info = reader.get_tensor_info(tensor_name);

                std::cout << "  Tensor: " << tensor_name << "\n";
                std::cout << "    Shape: [";
                for (size_t j = 0; j < info.shape.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << info.shape[j];
                }
                std::cout << "]\n";
                std::cout << "    Data type: " << info.dtype << "\n";
                std::cout << "    Offset: " << info.offset << "\n";
                std::cout << "    Size: " << info.size << " bytes\n";
                std::cout << "    Encoding: " << info.encoding << "\n";
                std::cout << "    Layout: " << info.layout << "\n";

                // Basic sanity checks
                assert(info.size > 0);
                assert(!info.shape.empty());
                assert(!info.dtype.empty());
            }

            std::cout << "  ✅ Tensor info extraction test passed\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in tensor info extraction: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        RealZTensorTest test;
        test.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}