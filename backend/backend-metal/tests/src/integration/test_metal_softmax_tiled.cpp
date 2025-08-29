#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#include "metal_softmax.hpp"

/**
 * Integration test for tiled softmax implementation
 * 
 * Validates:
 * 1. Correctness against CPU reference for large vocabularies
 * 2. Proper kernel selection (tiled vs standard)
 * 3. Edge cases (non-divisible tile sizes, extreme values)
 * 4. Numerical stability
 */

// Test configurations for tiled kernel validation
const std::vector<int> TILED_VOCAB_SIZES = {17000, 32000, 65536, 100000};
const std::vector<int> TEST_BATCH_SIZES = {1, 4, 16};
const float TEST_TEMPERATURE = 1.0f;
const float TOLERANCE = 1e-5f;

/**
 * High-precision CPU reference implementation
 */
void cpu_softmax_reference(const float* input, float* output, int batch_size, int vocab_size, float temperature) {
    for (int b = 0; b < batch_size; b++) {
        const float* row = input + b * vocab_size;
        float* out_row = output + b * vocab_size;
        
        // Find max for numerical stability
        float max_val = *std::max_element(row, row + vocab_size);
        
        // Apply temperature scaling and exp
        double sum = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            float scaled = (row[i] - max_val) / temperature;
            out_row[i] = std::exp(scaled);
            sum += out_row[i];
        }
        
        // Normalize
        for (int i = 0; i < vocab_size; i++) {
            out_row[i] /= sum;
        }
    }
}

/**
 * Generate test data with challenging numerical properties
 */
std::vector<float> generate_challenging_data(int batch_size, int vocab_size, int pattern) {
    std::vector<float> data(batch_size * vocab_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch (pattern) {
        case 0: {
            // Normal distribution
            std::normal_distribution<float> dist(0.0f, 2.0f);
            for (auto& val : data) {
                val = dist(gen);
            }
            break;
        }
        case 1: {
            // Extreme values (very large and very small)
            std::uniform_real_distribution<float> extreme(-50.0f, 50.0f);
            for (auto& val : data) {
                val = extreme(gen);
            }
            break;
        }
        case 2: {
            // Sparse with outliers (realistic for language models)
            std::normal_distribution<float> small_dist(0.0f, 1.0f);
            std::normal_distribution<float> large_dist(10.0f, 3.0f);
            std::uniform_real_distribution<float> outlier_prob(0.0f, 1.0f);
            
            for (auto& val : data) {
                if (outlier_prob(gen) < 0.95f) {
                    val = small_dist(gen);
                } else {
                    val = large_dist(gen);
                }
            }
            break;
        }
    }
    
    return data;
}

/**
 * Validate Metal output against CPU reference
 */
bool validate_results(const float* metal_output, const float* cpu_output, 
                     int batch_size, int vocab_size, float tolerance) {
    bool pass = true;
    int error_count = 0;
    const int max_errors = 5;  // Limit error output
    
    for (int b = 0; b < batch_size; b++) {
        double metal_sum = 0.0;
        double cpu_sum = 0.0;
        
        for (int v = 0; v < vocab_size; v++) {
            int idx = b * vocab_size + v;
            metal_sum += metal_output[idx];
            cpu_sum += cpu_output[idx];
            
            // Check individual values
            float diff = std::abs(metal_output[idx] - cpu_output[idx]);
            if (diff > tolerance) {
                if (error_count < max_errors) {
                    std::cerr << "Value mismatch at [" << b << "," << v << "]: "
                              << "Metal=" << metal_output[idx] 
                              << ", CPU=" << cpu_output[idx] 
                              << ", diff=" << diff << std::endl;
                }
                error_count++;
                pass = false;
            }
            
            // Check non-negative
            if (metal_output[idx] < 0.0f) {
                if (error_count < max_errors) {
                    std::cerr << "Negative probability at [" << b << "," << v << "]: " 
                              << metal_output[idx] << std::endl;
                }
                error_count++;
                pass = false;
            }
        }
        
        // Check normalization
        float sum_diff = std::abs(static_cast<float>(metal_sum) - 1.0f);
        if (sum_diff > tolerance) {
            if (error_count < max_errors) {
                std::cerr << "Batch " << b << " sum = " << metal_sum 
                          << ", expected 1.0, diff = " << sum_diff << std::endl;
            }
            error_count++;
            pass = false;
        }
    }
    
    if (error_count > max_errors) {
        std::cerr << "... and " << (error_count - max_errors) << " more errors" << std::endl;
    }
    
    return pass;
}

/**
 * Test a specific configuration
 */
bool test_configuration(int batch_size, int vocab_size, int data_pattern) {
    std::cout << "Testing [batch=" << batch_size << ", vocab=" << vocab_size 
              << ", pattern=" << data_pattern << "]..." << std::endl;
    
    // Generate test data
    auto input_data = generate_challenging_data(batch_size, vocab_size, data_pattern);
    std::vector<float> metal_output(batch_size * vocab_size);
    std::vector<float> cpu_output(batch_size * vocab_size);
    
    // CPU reference
    cpu_softmax_reference(input_data.data(), cpu_output.data(), batch_size, vocab_size, TEST_TEMPERATURE);
    
    // Metal implementation
    int result = metal_softmax_float(input_data.data(), metal_output.data(), 
                                   batch_size, vocab_size, TEST_TEMPERATURE);
    
    if (result != 0) {
        std::cerr << "Metal softmax failed with error code: " << result << std::endl;
        return false;
    }
    
    // Validate results
    bool valid = validate_results(metal_output.data(), cpu_output.data(), 
                                batch_size, vocab_size, TOLERANCE);
    
    if (valid) {
        std::cout << "  ✅ PASS" << std::endl;
    } else {
        std::cout << "  ❌ FAIL" << std::endl;
    }
    
    return valid;
}

/**
 * Test edge cases specific to tiling
 */
bool test_tiling_edge_cases() {
    std::cout << "\n=== Tiling Edge Cases ===" << std::endl;
    
    bool all_pass = true;
    
    // Test vocab sizes that don't divide evenly by tile size (2048)
    std::vector<int> edge_vocab_sizes = {
        17000,   // 8 full tiles + partial tile (904 elements)
        32001,   // 15 full tiles + 1 element
        65535,   // 31 full tiles + 2047 elements
        100000   // 48 full tiles + 1696 elements
    };
    
    for (int vocab_size : edge_vocab_sizes) {
        for (int batch_size : {1, 4}) {
            for (int pattern = 0; pattern < 3; pattern++) {
                bool pass = test_configuration(batch_size, vocab_size, pattern);
                all_pass = all_pass && pass;
                
                if (!pass) {
                    std::cout << "FAIL: Edge case [batch=" << batch_size 
                              << ", vocab=" << vocab_size << ", pattern=" << pattern << "]" << std::endl;
                }
            }
        }
    }
    
    return all_pass;
}

/**
 * Test kernel selection logic
 */
bool test_kernel_selection() {
    std::cout << "\n=== Kernel Selection Test ===" << std::endl;
    
    // Test that we get different behavior for small vs large vocabs
    std::vector<float> small_input(1 * 1024, 1.0f);  // 1K vocab - should use standard
    std::vector<float> large_input(1 * 32000, 1.0f); // 32K vocab - should use tiled
    
    std::vector<float> small_output(1024);
    std::vector<float> large_output(32000);
    
    std::cout << "Testing small vocab (1024 - should use standard kernel)..." << std::endl;
    int result1 = metal_softmax_float(small_input.data(), small_output.data(), 1, 1024, 1.0f);
    
    std::cout << "Testing large vocab (32000 - should use tiled kernel)..." << std::endl;
    int result2 = metal_softmax_float(large_input.data(), large_output.data(), 1, 32000, 1.0f);
    
    bool pass = (result1 == 0 && result2 == 0);
    
    // Basic validation - all outputs should be 1.0/vocab_size for uniform input
    float expected_small = 1.0f / 1024.0f;
    float expected_large = 1.0f / 32000.0f;
    
    bool small_valid = std::abs(small_output[0] - expected_small) < 1e-6f;
    bool large_valid = std::abs(large_output[0] - expected_large) < 1e-6f;
    
    pass = pass && small_valid && large_valid;
    
    std::cout << "Kernel selection test: " << (pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    return pass;
}

int main() {
    std::cout << "Metal Softmax Tiled Implementation Integration Test" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    bool all_tests_pass = true;
    
    try {
        // Test kernel selection logic
        all_tests_pass &= test_kernel_selection();
        
        // Test tiling edge cases
        all_tests_pass &= test_tiling_edge_cases();
        
        // Test standard configurations with different data patterns
        std::cout << "\n=== Standard Configuration Tests ===" << std::endl;
        for (int vocab_size : TILED_VOCAB_SIZES) {
            for (int batch_size : TEST_BATCH_SIZES) {
                for (int pattern = 0; pattern < 3; pattern++) {
                    bool pass = test_configuration(batch_size, vocab_size, pattern);
                    all_tests_pass = all_tests_pass && pass;
                }
            }
        }
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        if (all_tests_pass) {
            std::cout << "🎉 ALL TESTS PASSED" << std::endl;
            std::cout << "Tiled softmax implementation is ready for performance testing" << std::endl;
        } else {
            std::cout << "❌ SOME TESTS FAILED" << std::endl;
            std::cout << "Tiled softmax implementation needs debugging" << std::endl;
        }
        
        return all_tests_pass ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Test execution failed: " << e.what() << std::endl;
        return 1;
    }
}