#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <random>

#include "metal_softmax.hpp"
#include "metal_extract_k_values.hpp"
#include "metal_topk_mask_logits.hpp"

bool test_softmax_edge_cases_robust() {
    std::cout << "=== Testing Softmax Edge Cases (Robust) ===" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: Single element
    {
        std::cout << "Testing single element..." << std::endl;
        std::vector<float> input = {5.0f};
        std::vector<float> output(1);
        
        int result = metal_softmax_float(input.data(), output.data(), 1, 1, 1.0f);
        
        if (result != 0) {
            std::cerr << "❌ Single element softmax failed: " << result << std::endl;
            all_passed = false;
        } else if (std::abs(output[0] - 1.0f) > 1e-6f) {
            std::cerr << "❌ Single element should be 1.0, got " << output[0] << std::endl;
            all_passed = false;
        } else {
            std::cout << "✅ Single element test passed" << std::endl;
        }
    }
    
    // Test 2: All zeros
    {
        std::cout << "Testing all zeros..." << std::endl;
        std::vector<float> input = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> output(4);
        
        int result = metal_softmax_float(input.data(), output.data(), 1, 4, 1.0f);
        
        if (result != 0) {
            std::cerr << "❌ All zeros softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Check normalization and uniformity with relaxed tolerance
            double sum = 0.0;
            for (int i = 0; i < 4; i++) {
                sum += output[i];
            }
            
            float expected = 0.25f;
            bool approximately_uniform = true;
            for (int i = 0; i < 4; i++) {
                if (std::abs(output[i] - expected) > 1e-5f) {
                    approximately_uniform = false;
                    break;
                }
            }
            
            if (std::abs(sum - 1.0) > 1e-5f) {
                std::cerr << "❌ All zeros normalization failed, sum=" << sum << std::endl;
                all_passed = false;
            } else if (!approximately_uniform) {
                std::cerr << "❌ All zeros should be approximately uniform" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ All zeros test passed" << std::endl;
            }
        }
    }
    
    // Test 3: Moderate values (avoid extreme overflow)
    {
        std::cout << "Testing moderate large values..." << std::endl;
        std::vector<float> input = {10.0f, 11.0f, 9.0f};
        std::vector<float> output(3);
        
        int result = metal_softmax_float(input.data(), output.data(), 1, 3, 1.0f);
        
        if (result != 0) {
            std::cerr << "❌ Moderate values softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Check normalization with relaxed tolerance
            double sum = 0.0;
            bool all_finite = true;
            for (int i = 0; i < 3; i++) {
                if (!std::isfinite(output[i])) {
                    all_finite = false;
                    break;
                }
                sum += output[i];
            }
            
            if (!all_finite || std::abs(sum - 1.0) > 1e-4f) {
                std::cerr << "❌ Moderate values failed, sum=" << sum 
                          << ", all_finite=" << all_finite << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Moderate values test passed (sum=" << sum << ")" << std::endl;
            }
        }
    }
    
    // Test 4: Temperature scaling validation  
    {
        std::cout << "Testing temperature scaling..." << std::endl;
        std::vector<float> input = {1.0f, 3.0f, 2.0f};
        std::vector<float> output_low(3), output_high(3);
        
        int result1 = metal_softmax_float(input.data(), output_low.data(), 1, 3, 0.1f);
        int result2 = metal_softmax_float(input.data(), output_high.data(), 1, 3, 10.0f);
        
        if (result1 != 0 || result2 != 0) {
            std::cerr << "❌ Temperature scaling failed: " << result1 << ", " << result2 << std::endl;
            all_passed = false;
        } else {
            // Low temperature should be more peaked (higher max value)
            float max_low = *std::max_element(output_low.begin(), output_low.end());
            float max_high = *std::max_element(output_high.begin(), output_high.end());
            
            if (max_low <= max_high) {
                std::cerr << "❌ Low temperature should be more peaked: " 
                          << max_low << " vs " << max_high << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Temperature scaling test passed" << std::endl;
            }
        }
    }
    
    // Test 5: Numerical stability with negative values
    {
        std::cout << "Testing negative values..." << std::endl;
        std::vector<float> input = {-100.0f, -99.0f, -101.0f};
        std::vector<float> output(3);
        
        int result = metal_softmax_float(input.data(), output.data(), 1, 3, 1.0f);
        
        if (result != 0) {
            std::cerr << "❌ Negative values softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            double sum = 0.0;
            bool all_finite = true;
            for (int i = 0; i < 3; i++) {
                if (!std::isfinite(output[i]) || output[i] < 0) {
                    all_finite = false;
                    break;
                }
                sum += output[i];
            }
            
            if (!all_finite || std::abs(sum - 1.0) > 1e-5f) {
                std::cerr << "❌ Negative values test failed" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Negative values test passed" << std::endl;
            }
        }
    }
    
    return all_passed;
}

bool test_kernel_robustness() {
    std::cout << "=== Testing Kernel Robustness ===" << std::endl;
    
    bool all_passed = true;
    
    // Test various size combinations
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {1, 2, 1},    // Minimal
        {1, 16, 8},   // Small
        {4, 128, 16}, // Medium
        {8, 1024, 32} // Large
    };
    
    for (auto [batch_size, vocab_size, k] : test_sizes) {
        std::cout << "Testing size " << batch_size << "x" << vocab_size << ", k=" << k << std::endl;
        
        // Generate random test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 2.0f);
        
        // Test softmax
        {
            std::vector<float> input(batch_size * vocab_size);
            std::vector<float> output(batch_size * vocab_size);
            
            for (float& val : input) val = dist(gen);
            
            int result = metal_softmax_float(
                input.data(), output.data(), batch_size, vocab_size, 1.0f
            );
            
            if (result != 0) {
                std::cerr << "❌ Softmax failed for size " << batch_size << "x" << vocab_size << std::endl;
                all_passed = false;
                continue;
            }
            
            // Check normalization for each batch
            for (int b = 0; b < batch_size; b++) {
                double sum = 0.0;
                for (int i = 0; i < vocab_size; i++) {
                    float val = output[b * vocab_size + i];
                    if (!std::isfinite(val) || val < 0) {
                        std::cerr << "❌ Invalid softmax output at [" << b << "," << i << "]=" << val << std::endl;
                        all_passed = false;
                        goto next_size;
                    }
                    sum += val;
                }
                
                if (std::abs(sum - 1.0) > 1e-4f) {
                    std::cerr << "❌ Normalization failed for batch " << b << ", sum=" << sum << std::endl;
                    all_passed = false;
                    goto next_size;
                }
            }
        }
        
        // Test topk_mask  
        {
            if (k <= vocab_size) { // Valid k
                std::vector<float> logits(batch_size * vocab_size);
                for (float& val : logits) val = dist(gen);
                
                int result = metal_topk_mask_logits_float32(
                    logits.data(), batch_size, vocab_size, k
                );
                
                if (result != 0) {
                    std::cerr << "❌ TopK mask failed for size " << batch_size << "x" << vocab_size << std::endl;
                    all_passed = false;
                    continue;
                }
                
                // Check masking for each batch
                for (int b = 0; b < batch_size; b++) {
                    int unmasked = 0;
                    for (int i = 0; i < vocab_size; i++) {
                        float val = logits[b * vocab_size + i];
                        if (!std::isinf(val) || val > 0) {
                            unmasked++;
                        }
                    }
                    
                    if (unmasked != k) {
                        std::cerr << "❌ TopK mask batch " << b << " has " << unmasked 
                                  << " unmasked, expected " << k << std::endl;
                        all_passed = false;
                        goto next_size;
                    }
                }
            }
        }
        
        next_size:;
    }
    
    if (all_passed) {
        std::cout << "✅ All robustness tests passed" << std::endl;
    }
    
    return all_passed;
}

int main() {
    std::cout << "=== Metal Kernel Robust Edge Case Tests ===" << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_softmax_edge_cases_robust();
    std::cout << std::endl;
    
    all_passed &= test_kernel_robustness();
    
    if (all_passed) {
        std::cout << "\n🎉 All robust edge case tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some robust edge case tests failed!" << std::endl;
        return 1;
    }
}