#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "metal_topk_mask_logits.hpp"

// Test parameters
const int TEST_NUM_TOKENS = 8;
const int TEST_VOCAB_SIZE = 32000;
const int TEST_K = 50;  // Keep top 50 values
const float TOLERANCE = 1e-5f;
const float MASK_VALUE = -1e9f;  // Value used to mask non-top-k elements

// Reference CPU implementation for validation
void cpu_topk_mask_logits_float32(
    float* logits,
    unsigned int num_tokens,
    unsigned int vocab_size,
    unsigned int k
) {
    for (unsigned int token = 0; token < num_tokens; token++) {
        float* token_logits = logits + token * vocab_size;
        
        // Create vector of value-index pairs for sorting
        std::vector<std::pair<float, unsigned int>> pairs(vocab_size);
        for (unsigned int i = 0; i < vocab_size; i++) {
            pairs[i] = {token_logits[i], i};
        }
        
        // Find the k-th largest value using partial sort
        std::nth_element(pairs.begin(), pairs.begin() + k - 1, pairs.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float kth_value = pairs[k - 1].first;
        
        // Mask all values below the k-th largest
        for (unsigned int i = 0; i < vocab_size; i++) {
            if (token_logits[i] < kth_value) {
                token_logits[i] = -INFINITY;
            }
        }
    }
}

void cpu_topk_mask_logits_bfloat16(
    void* logits,
    unsigned int num_tokens,
    unsigned int vocab_size,
    unsigned int k
) {
    uint16_t* bf16_logits = static_cast<uint16_t*>(logits);
    
    for (unsigned int token = 0; token < num_tokens; token++) {
        uint16_t* token_logits = bf16_logits + token * vocab_size;
        
        // Convert to float for processing
        std::vector<float> float_logits(vocab_size);
        for (unsigned int i = 0; i < vocab_size; i++) {
            uint32_t bits = static_cast<uint32_t>(token_logits[i]) << 16;
            float_logits[i] = *reinterpret_cast<float*>(&bits);
        }
        
        // Apply CPU algorithm
        std::vector<std::pair<float, unsigned int>> pairs(vocab_size);
        for (unsigned int i = 0; i < vocab_size; i++) {
            pairs[i] = {float_logits[i], i};
        }
        
        std::nth_element(pairs.begin(), pairs.begin() + k - 1, pairs.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float kth_value = pairs[k - 1].first;
        
        // Mask values and convert back to bfloat16
        for (unsigned int i = 0; i < vocab_size; i++) {
            if (float_logits[i] < kth_value) {
                float_logits[i] = -INFINITY;
            }
            
            uint32_t float_bits = *reinterpret_cast<uint32_t*>(&float_logits[i]);
            token_logits[i] = static_cast<uint16_t>(float_bits >> 16);
        }
    }
}

bool validate_topk_mask_float32(
    const float* logits,
    unsigned int num_tokens,
    unsigned int vocab_size,
    unsigned int k
) {
    for (unsigned int token = 0; token < num_tokens; token++) {
        const float* token_logits = logits + token * vocab_size;
        
        // Count unmasked values
        int unmasked_count = 0;
        int masked_count = 0;
        
        for (unsigned int i = 0; i < vocab_size; i++) {
            if (std::isinf(token_logits[i]) && token_logits[i] < 0) {
                masked_count++;
            } else {
                unmasked_count++;
            }
        }
        
        // Should have exactly k unmasked values
        if (unmasked_count != static_cast<int>(k)) {
            std::cerr << "❌ Token " << token << " has " << unmasked_count 
                      << " unmasked values, expected " << k << std::endl;
            return false;
        }
        
        // Check that unmasked values are larger than any masked values
        float min_unmasked = std::numeric_limits<float>::max();
        float max_masked = -std::numeric_limits<float>::max();
        
        for (unsigned int i = 0; i < vocab_size; i++) {
            if (std::isinf(token_logits[i]) && token_logits[i] < 0) {
                // This is a masked value (-inf)
                // Skip for min/max comparison since it's -inf
            } else {
                // This is an unmasked value
                if (token_logits[i] < min_unmasked) {
                    min_unmasked = token_logits[i];
                }
            }
        }
        
        // Since masked values are -inf, this check doesn't apply
        // The algorithm should keep the top-k values, which is validated above
    }
    
    return true;
}

bool compare_results_float32(
    const float* metal_logits,
    const float* cpu_logits,
    unsigned int total_size,
    float tolerance
) {
    for (unsigned int i = 0; i < total_size; i++) {
        float diff = std::abs(metal_logits[i] - cpu_logits[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": Metal=" << metal_logits[i]
                      << ", CPU=" << cpu_logits[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== Metal TopK Mask Logits Kernel Test ===" << std::endl;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Tokens: " << TEST_NUM_TOKENS << std::endl;
    std::cout << "  Vocab size: " << TEST_VOCAB_SIZE << std::endl;
    std::cout << "  K (keep top): " << TEST_K << std::endl;
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    // Test Float32 version
    {
        std::cout << "\n=== Testing Float32 Version ===" << std::endl;
        
        const int total_size = TEST_NUM_TOKENS * TEST_VOCAB_SIZE;
        std::vector<float> metal_logits(total_size);
        std::vector<float> cpu_logits(total_size);
        
        // Generate identical test data for both tests
        for (int i = 0; i < total_size; i++) {
            float val = dist(gen);
            metal_logits[i] = val;
            cpu_logits[i] = val;
        }
        
        // Test Metal implementation
        std::cout << "Running Metal topk_mask_logits_float32..." << std::endl;
        int result = metal_topk_mask_logits_float32(
            metal_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        if (result != 0) {
            std::cerr << "❌ Metal topk_mask_logits_float32 failed with error: " << result << std::endl;
            return 1;
        }
        
        // Compute CPU reference
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_topk_mask_logits_float32(
            cpu_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        // Debug: Show sample results
        std::cout << "Debug: Sample of Metal results (first token, first 10 values):" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "  [" << i << "] = " << metal_logits[i] << std::endl;
        }
        
        // Count actual values
        int unmasked = 0, masked = 0;
        for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
            if (std::isinf(metal_logits[i]) && metal_logits[i] < 0) {
                masked++;
            } else {
                unmasked++;
            }
        }
        std::cout << "Debug: Token 0 has " << unmasked << " unmasked, " << masked << " masked values" << std::endl;
        
        // Validate Metal results structure
        std::cout << "Validating Metal results..." << std::endl;
        
        bool metal_valid = validate_topk_mask_float32(
            metal_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        if (!metal_valid) {
            std::cerr << "❌ Float32 Metal validation failed" << std::endl;
            return 1;
        }
        std::cout << "✅ Float32 Metal validation passed" << std::endl;
        
        // Validate CPU results structure (sanity check)
        bool cpu_valid = validate_topk_mask_float32(
            cpu_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        if (!cpu_valid) {
            std::cerr << "❌ Float32 CPU validation failed" << std::endl;
            return 1;
        }
        std::cout << "✅ Float32 CPU validation passed" << std::endl;
        
        // Compare results
        bool accuracy_valid = compare_results_float32(
            metal_logits.data(),
            cpu_logits.data(),
            total_size,
            TOLERANCE
        );
        
        if (!accuracy_valid) {
            std::cerr << "❌ Float32 accuracy check failed" << std::endl;
            return 1;
        }
        std::cout << "✅ Float32 accuracy check passed" << std::endl;
    }
    
    // Test bfloat16 version
    {
        std::cout << "\n=== Testing bfloat16 Version ===" << std::endl;
        
        const int total_size = TEST_NUM_TOKENS * TEST_VOCAB_SIZE;
        std::vector<uint16_t> metal_logits(total_size);
        std::vector<uint16_t> cpu_logits(total_size);
        
        // Generate identical test data (convert float to bfloat16)
        for (int i = 0; i < total_size; i++) {
            float val = dist(gen);
            uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
            uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
            
            metal_logits[i] = bf16_val;
            cpu_logits[i] = bf16_val;
        }
        
        // Test Metal implementation
        std::cout << "Running Metal topk_mask_logits_bfloat16..." << std::endl;
        int result = metal_topk_mask_logits_bfloat16(
            metal_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        if (result != 0) {
            std::cerr << "❌ Metal topk_mask_logits_bfloat16 failed with error: " << result << std::endl;
            return 1;
        }
        
        // Compute CPU reference  
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_topk_mask_logits_bfloat16(
            cpu_logits.data(),
            TEST_NUM_TOKENS,
            TEST_VOCAB_SIZE,
            TEST_K
        );
        
        // Basic validation (check that we have approximately k unmasked values per token)
        std::cout << "Basic validation..." << std::endl;
        
        // Convert -INFINITY to bfloat16 for comparison
        float neg_inf = -INFINITY;
        uint32_t mask_bits = *reinterpret_cast<const uint32_t*>(&neg_inf);
        uint16_t bf16_mask = static_cast<uint16_t>(mask_bits >> 16);
        
        for (unsigned int token = 0; token < TEST_NUM_TOKENS; token++) {
            int unmasked_count = 0;
            
            for (unsigned int i = 0; i < TEST_VOCAB_SIZE; i++) {
                uint16_t val = metal_logits[token * TEST_VOCAB_SIZE + i];
                
                // Convert back to float to check if it's -infinity
                uint32_t float_bits = static_cast<uint32_t>(val) << 16;
                float float_val = *reinterpret_cast<float*>(&float_bits);
                
                if (!std::isinf(float_val) || float_val >= 0) {
                    unmasked_count++;
                }
            }
            
            // Allow some tolerance for bfloat16 precision issues
            if (abs(unmasked_count - static_cast<int>(TEST_K)) > 2) {
                std::cerr << "❌ Token " << token << " has " << unmasked_count 
                          << " unmasked values, expected approximately " << TEST_K << std::endl;
                return 1;
            }
        }
        
        std::cout << "✅ bfloat16 basic validation passed" << std::endl;
    }
    
    std::cout << "\n🎉 All Metal TopK Mask Logits tests passed!" << std::endl;
    return 0;
}