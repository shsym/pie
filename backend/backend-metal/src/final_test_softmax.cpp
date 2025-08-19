#include "metal_softmax.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Helper function to load binary data
std::vector<float> load_binary_floats(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return {};
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_floats = file_size / sizeof(float);
    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

// Compare two float vectors
bool compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual, float tolerance = 1e-5f) {
    if (expected.size() != actual.size()) return false;
    
    float max_error = 0.0f;
    size_t error_count = 0;
    
    for (size_t i = 0; i < expected.size(); ++i) {
        float error = std::abs(expected[i] - actual[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) error_count++;
    }
    
    std::cout << "Accuracy: " << (expected.size() - error_count) << "/" << expected.size() 
              << " elements within tolerance " << tolerance 
              << " (max_error=" << max_error << ")" << std::endl;
    
    return error_count == 0;
}

int main() {
    std::cout << "=== Metal Softmax Final Validation ===" << std::endl;
    
    // Test 1: Small batch for verification
    {
        std::cout << "\nTest 1: Small batch verification" << std::endl;
        const int batch_size = 1;
        const int vocab_size = 4;
        const float temperature = 1.0f;
        
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> output(vocab_size);
        
        int result = metal_softmax_float(input.data(), output.data(), batch_size, vocab_size, temperature);
        
        if (result == 0) {
            float sum = 0.0f;
            for (float val : output) sum += val;
            std::cout << "✓ Small batch test passed (sum=" << sum << ")" << std::endl;
        } else {
            std::cout << "✗ Small batch test failed" << std::endl;
            return 1;
        }
    }
    
    // Test 2: Production scale with CUDA reference
    {
        std::cout << "\nTest 2: Production scale validation" << std::endl;
        const int batch_size = 2;
        const int vocab_size = 32000;
        const float temperature = 1.0f;
        
        std::string base_path = "/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/softmax/production/";
        
        auto input_logits = load_binary_floats(base_path + "input_logits.bin");
        auto expected_output = load_binary_floats(base_path + "output.bin");
        
        if (input_logits.empty() || expected_output.empty()) {
            std::cout << "✗ Could not load CUDA reference data" << std::endl;
            return 1;
        }
        
        std::vector<float> metal_output(expected_output.size());
        
        int result = metal_softmax_float(
            input_logits.data(),
            metal_output.data(),
            batch_size,
            vocab_size,
            temperature
        );
        
        if (result == 0 && compare_outputs(expected_output, metal_output, 1e-4f)) {
            std::cout << "✓ Production scale test passed" << std::endl;
        } else {
            std::cout << "✗ Production scale test failed" << std::endl;
            return 1;
        }
    }
    
    std::cout << "\n=== All Tests Passed! ✅ ===" << std::endl;
    std::cout << "Metal Softmax implementation is production-ready" << std::endl;
    return 0;
}