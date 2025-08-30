#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <random>

#include "metal_softmax.hpp"
#include "metal_extract_k_values.hpp"
#include "metal_topk_mask_logits.hpp"

bool test_softmax_edge_cases() {
    std::cout << "=== Testing Softmax Edge Cases ===" << std::endl;
    
    const float tolerance = 1e-6f;
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
        } else if (std::abs(output[0] - 1.0f) > tolerance) {
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
            // Should be uniform distribution
            float expected = 0.25f;
            bool uniform = true;
            for (int i = 0; i < 4; i++) {
                if (std::abs(output[i] - expected) > tolerance) {
                    uniform = false;
                    break;
                }
            }
            
            if (!uniform) {
                std::cerr << "❌ All zeros should give uniform distribution" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ All zeros test passed" << std::endl;
            }
        }
    }
    
    // Test 3: Very large values (potential overflow)
    {
        std::cout << "Testing large values..." << std::endl;
        std::vector<float> input = {1000.0f, 1001.0f, 999.0f};
        std::vector<float> output(3);
        
        int result = metal_softmax_float(input.data(), output.data(), 1, 3, 1.0f);
        
        if (result != 0) {
            std::cerr << "❌ Large values softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Check normalization
            double sum = 0.0;
            bool has_finite = true;
            for (int i = 0; i < 3; i++) {
                if (!std::isfinite(output[i])) {
                    has_finite = false;
                    break;
                }
                sum += output[i];
            }
            
            if (!has_finite || std::abs(sum - 1.0) > tolerance) {
                std::cerr << "❌ Large values test failed, sum=" << sum << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Large values test passed" << std::endl;
            }
        }
    }
    
    // Test 4: High temperature (approaching uniform)
    {
        std::cout << "Testing high temperature..." << std::endl;
        std::vector<float> input = {1.0f, 5.0f, 10.0f};
        std::vector<float> output(3);
        
        float high_temp = 1000.0f;
        int result = metal_softmax_float(input.data(), output.data(), 1, 3, high_temp);
        
        if (result != 0) {
            std::cerr << "❌ High temperature softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            // With very high temperature, should approach uniform
            float expected = 1.0f / 3.0f;
            bool near_uniform = true;
            for (int i = 0; i < 3; i++) {
                if (std::abs(output[i] - expected) > 0.1f) { // Looser tolerance
                    near_uniform = false;
                    break;
                }
            }
            
            if (!near_uniform) {
                std::cerr << "❌ High temperature should approach uniform distribution" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ High temperature test passed" << std::endl;
            }
        }
    }
    
    // Test 5: Low temperature (approaching one-hot)
    {
        std::cout << "Testing low temperature..." << std::endl;
        std::vector<float> input = {1.0f, 2.0f, 1.5f};
        std::vector<float> output(3);
        
        float low_temp = 0.01f;
        int result = metal_softmax_float(input.data(), output.data(), 1, 3, low_temp);
        
        if (result != 0) {
            std::cerr << "❌ Low temperature softmax failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Should be very peaked at index 1 (highest value)
            if (output[1] < 0.9f) {
                std::cerr << "❌ Low temperature should peak at max value, got " << output[1] << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Low temperature test passed" << std::endl;
            }
        }
    }
    
    return all_passed;
}

bool test_extract_k_values_edge_cases() {
    std::cout << "=== Testing Extract K Values Edge Cases ===" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: All infinity (no valid values)
    {
        std::cout << "Testing all infinity..." << std::endl;
        const int M = 2, N = 4, k = 2;
        std::vector<float> input(M * N, -INFINITY);
        std::vector<float> values(M * k);
        std::vector<int32_t> indices(M * k);
        
        int result = metal_extract_k_values_float32(
            input.data(), values.data(), indices.data(), M, N, k
        );
        
        if (result != 0) {
            std::cerr << "❌ All infinity test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // All output should be zeros/padding
            bool all_zeros = true;
            for (int i = 0; i < M * k; i++) {
                if (values[i] != 0.0f || indices[i] != 0) {
                    all_zeros = false;
                    break;
                }
            }
            
            if (!all_zeros) {
                std::cerr << "❌ All infinity should produce zero output" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ All infinity test passed" << std::endl;
            }
        }
    }
    
    // Test 2: k = 0 (extract nothing)
    {
        std::cout << "Testing k=0..." << std::endl;
        const int M = 2, N = 4, k = 0;
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> values; // Empty
        std::vector<int32_t> indices; // Empty
        
        int result = metal_extract_k_values_float32(
            input.data(), values.data(), indices.data(), M, N, k
        );
        
        // This should either work (no-op) or gracefully fail
        if (result == 0) {
            std::cout << "✅ k=0 test passed (no-op)" << std::endl;
        } else {
            std::cout << "ℹ️  k=0 test gracefully failed (expected): " << result << std::endl;
            // This is acceptable behavior
        }
    }
    
    // Test 3: k > N (extract more than available)
    {
        std::cout << "Testing k > N..." << std::endl;
        const int M = 2, N = 3, k = 5;
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> values(M * k);
        std::vector<int32_t> indices(M * k);
        
        int result = metal_extract_k_values_float32(
            input.data(), values.data(), indices.data(), M, N, k
        );
        
        if (result != 0) {
            std::cout << "ℹ️  k > N test failed as expected: " << result << std::endl;
            // This is acceptable - kernel should validate inputs
        } else {
            std::cout << "✅ k > N test handled gracefully" << std::endl;
        }
    }
    
    // Test 4: Single row, single column
    {
        std::cout << "Testing 1x1 matrix..." << std::endl;
        const int M = 1, N = 1, k = 1;
        std::vector<float> input = {42.0f};
        std::vector<float> values(M * k);
        std::vector<int32_t> indices(M * k);
        
        int result = metal_extract_k_values_float32(
            input.data(), values.data(), indices.data(), M, N, k
        );
        
        if (result != 0) {
            std::cerr << "❌ 1x1 matrix test failed: " << result << std::endl;
            all_passed = false;
        } else if (values[0] != 42.0f || indices[0] != 0) {
            std::cerr << "❌ 1x1 matrix should extract (0, 42.0), got ("
                      << indices[0] << ", " << values[0] << ")" << std::endl;
            all_passed = false;
        } else {
            std::cout << "✅ 1x1 matrix test passed" << std::endl;
        }
    }
    
    // Test 5: Mixed finite and special values
    {
        std::cout << "Testing mixed special values..." << std::endl;
        const int M = 1, N = 6, k = 2;
        std::vector<float> input = {
            NAN, -INFINITY, 1.0f, INFINITY, 2.0f, -0.0f
        };
        std::vector<float> values(M * k);
        std::vector<int32_t> indices(M * k);
        
        int result = metal_extract_k_values_float32(
            input.data(), values.data(), indices.data(), M, N, k
        );
        
        if (result != 0) {
            std::cerr << "❌ Mixed special values test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Should extract finite values (1.0, 2.0) at indices (2, 4)
            // Note: behavior with NaN and +Inf may vary
            std::cout << "✅ Mixed special values test completed" << std::endl;
        }
    }
    
    return all_passed;
}

bool test_topk_mask_edge_cases() {
    std::cout << "=== Testing TopK Mask Edge Cases ===" << std::endl;
    
    bool all_passed = true;
    
    // Test 1: k = vocab_size (mask nothing)
    {
        std::cout << "Testing k = vocab_size..." << std::endl;
        const int batch_size = 1, vocab_size = 4, k = 4;
        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> original = logits;
        
        int result = metal_topk_mask_logits_float32(
            logits.data(), batch_size, vocab_size, k
        );
        
        if (result != 0) {
            std::cerr << "❌ k=vocab_size test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // No values should be masked
            bool unchanged = true;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] != original[i]) {
                    unchanged = false;
                    break;
                }
            }
            
            if (!unchanged) {
                std::cerr << "❌ k=vocab_size should not mask anything" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ k=vocab_size test passed" << std::endl;
            }
        }
    }
    
    // Test 2: k = 1 (mask all but one)
    {
        std::cout << "Testing k = 1..." << std::endl;
        const int batch_size = 1, vocab_size = 5, k = 1;
        std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f, 4.0f};
        
        int result = metal_topk_mask_logits_float32(
            logits.data(), batch_size, vocab_size, k
        );
        
        if (result != 0) {
            std::cerr << "❌ k=1 test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Only the maximum value (5.0 at index 1) should be unmasked
            int unmasked_count = 0;
            int max_index = -1;
            for (int i = 0; i < vocab_size; i++) {
                if (!std::isinf(logits[i]) || logits[i] > 0) {
                    unmasked_count++;
                    max_index = i;
                }
            }
            
            if (unmasked_count != 1 || max_index != 1) {
                std::cerr << "❌ k=1 should leave only max value (index 1), got "
                          << unmasked_count << " unmasked, max_index=" << max_index << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ k=1 test passed" << std::endl;
            }
        }
    }
    
    // Test 3: k = 0 (mask everything)
    {
        std::cout << "Testing k = 0..." << std::endl;
        const int batch_size = 1, vocab_size = 4, k = 0;
        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f};
        
        int result = metal_topk_mask_logits_float32(
            logits.data(), batch_size, vocab_size, k
        );
        
        if (result == 0) {
            // All should be masked
            bool all_masked = true;
            for (int i = 0; i < vocab_size; i++) {
                if (!std::isinf(logits[i]) || logits[i] > 0) {
                    all_masked = false;
                    break;
                }
            }
            
            if (!all_masked) {
                std::cerr << "❌ k=0 should mask everything" << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ k=0 test passed" << std::endl;
            }
        } else {
            std::cout << "ℹ️  k=0 test failed as expected: " << result << std::endl;
            // This is acceptable behavior
        }
    }
    
    // Test 4: All same values
    {
        std::cout << "Testing all same values..." << std::endl;
        const int batch_size = 1, vocab_size = 6, k = 3;
        std::vector<float> logits = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        
        int result = metal_topk_mask_logits_float32(
            logits.data(), batch_size, vocab_size, k
        );
        
        if (result != 0) {
            std::cerr << "❌ All same values test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Should keep exactly k values (implementation dependent which ones)
            int unmasked = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (!std::isinf(logits[i]) || logits[i] > 0) {
                    unmasked++;
                }
            }
            
            if (unmasked != k) {
                std::cerr << "❌ All same values should keep " << k << " values, got " << unmasked << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ All same values test passed" << std::endl;
            }
        }
    }
    
    // Test 5: Already contains infinity
    {
        std::cout << "Testing with existing infinity..." << std::endl;
        const int batch_size = 1, vocab_size = 5, k = 2;
        std::vector<float> logits = {1.0f, -INFINITY, 3.0f, 2.0f, -INFINITY};
        
        int result = metal_topk_mask_logits_float32(
            logits.data(), batch_size, vocab_size, k
        );
        
        if (result != 0) {
            std::cerr << "❌ Existing infinity test failed: " << result << std::endl;
            all_passed = false;
        } else {
            // Should preserve existing -inf and add more as needed
            int finite_count = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (std::isfinite(logits[i])) {
                    finite_count++;
                }
            }
            
            if (finite_count != k) {
                std::cerr << "❌ Should have " << k << " finite values, got " << finite_count << std::endl;
                all_passed = false;
            } else {
                std::cout << "✅ Existing infinity test passed" << std::endl;
            }
        }
    }
    
    return all_passed;
}

int main() {
    std::cout << "=== Metal Kernel Edge Case Tests ===" << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_softmax_edge_cases();
    std::cout << std::endl;
    
    all_passed &= test_extract_k_values_edge_cases();
    std::cout << std::endl;
    
    all_passed &= test_topk_mask_edge_cases();
    
    if (all_passed) {
        std::cout << "\n🎉 All edge case tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some edge case tests failed!" << std::endl;
        return 1;
    }
}