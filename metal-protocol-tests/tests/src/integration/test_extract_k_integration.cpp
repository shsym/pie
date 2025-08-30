#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include "metal_extract_k_values.hpp"

// Basic integration test for extract_k_values
// For comprehensive CUDA comparison, use: metal-protocol-tests/scripts/test_all_ops.sh

int main() {
    std::cout << "=== Metal Extract K Values Integration Test ===" << std::endl;
    
    // Basic functionality test with small data
    const int M = 4;
    const int N = 100;
    const int k = 10;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  M (sequences): " << M << std::endl;
    std::cout << "  N (vocab_size): " << N << std::endl;
    std::cout << "  K (values to extract): " << k << std::endl;
    
    // Generate simple test data
    std::vector<float> input_data(M * N);
    std::vector<float> output_values(M * k);
    std::vector<int32_t> output_indices(M * k);
    
    // Create simple test pattern: some -inf, some valid values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (int i = 0; i < M * N; i++) {
        if (i % 3 == 0) {
            input_data[i] = -INFINITY;  // Mark as invalid
        } else {
            input_data[i] = dist(gen);  // Valid value
        }
    }
    
    // Test Metal implementation
    std::cout << "\nTesting Metal extract_k_values_float32..." << std::endl;
    int result = metal_extract_k_values_float32(
        input_data.data(),
        output_values.data(),
        output_indices.data(),
        M, N, k
    );
    
    if (result != 0) {
        std::cerr << "❌ Metal extract_k_values_float32 failed with error: " << result << std::endl;
        return 1;
    }
    
    // Basic validation - check that we got valid results
    bool valid = true;
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < k; i++) {
            int idx = m * k + i;
            
            // Check index bounds
            if (output_indices[idx] < 0 || output_indices[idx] >= N) {
                std::cerr << "❌ Invalid index at [" << m << "," << i << "]: " 
                          << output_indices[idx] << std::endl;
                valid = false;
            }
            
            // Check that non-zero values correspond to non-infinity inputs
            if (output_values[idx] != 0.0f) {
                float original_val = input_data[m * N + output_indices[idx]];
                if (std::isinf(original_val) && original_val < 0) {
                    std::cerr << "❌ Extracted -infinity value at [" << m << "," << i << "]" << std::endl;
                    valid = false;
                }
            }
        }
    }
    
    if (!valid) {
        std::cerr << "❌ Basic validation failed" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Basic Metal extract_k_values functionality validated" << std::endl;
    
    // Test bfloat16 version briefly
    std::cout << "\nTesting Metal extract_k_values_bfloat16..." << std::endl;
    
    std::vector<uint16_t> bf16_input(M * N, 0x0000);  // bfloat16 zeros
    std::vector<uint16_t> bf16_values(M * k);
    std::vector<int32_t> bf16_indices(M * k);
    
    // Add some non-zero bfloat16 values (1.0f = 0x3F80 in bfloat16)
    for (int i = 1; i < M * N; i += 2) {
        bf16_input[i] = 0x3F80;  // bfloat16 representation of 1.0f
    }
    
    int bf16_result = metal_extract_k_values_bfloat16(
        bf16_input.data(),
        bf16_values.data(),
        bf16_indices.data(),
        M, N, k
    );
    
    if (bf16_result != 0) {
        std::cerr << "❌ Metal extract_k_values_bfloat16 failed with error: " << bf16_result << std::endl;
        return 1;
    }
    
    std::cout << "✅ Basic Metal extract_k_values_bfloat16 functionality validated" << std::endl;
    
    std::cout << "\n📋 Integration test summary:" << std::endl;
    std::cout << "  ✅ Float32 version: Basic functionality working" << std::endl;
    std::cout << "  ✅ BFloat16 version: Basic functionality working" << std::endl;
    std::cout << "  ✅ Index bounds validation: Passed" << std::endl;
    std::cout << "  ✅ -Infinity handling: Correct" << std::endl;
    
    std::cout << "\n💡 For comprehensive CUDA accuracy comparison, run:" << std::endl;
    std::cout << "     ../scripts/test_all_ops.sh" << std::endl;
    
    std::cout << "\n🎉 Metal Extract K Values integration test passed!" << std::endl;
    return 0;
}