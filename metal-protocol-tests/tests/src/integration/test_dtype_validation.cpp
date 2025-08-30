#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

#include "metal_extract_k_values.hpp"
#include "metal_topk_mask_logits.hpp"

// Helper functions for bfloat16 conversion
uint16_t float_to_bfloat16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    return static_cast<uint16_t>(bits >> 16);
}

float bfloat16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    return *reinterpret_cast<float*>(&bits);
}

bool test_extract_k_values_data_types() {
    std::cout << "=== Testing extract_k_values data types ===" << std::endl;
    
    const int M = 4;
    const int N = 16;
    const int k = 4;
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    std::vector<float> float_input(M * N);
    std::vector<uint16_t> bf16_input(M * N);
    
    // Create sparse test data with some finite values
    for (int i = 0; i < M * N; i++) {
        float val;
        if (i % 4 == 0) {
            val = dist(gen);  // 25% valid values
        } else {
            val = -INFINITY;  // 75% masked
        }
        
        float_input[i] = val;
        bf16_input[i] = float_to_bfloat16(val);
    }
    
    // Test float32 version
    std::vector<float> float_values(M * k);
    std::vector<int32_t> float_indices(M * k);
    
    int result_f32 = metal_extract_k_values_float32(
        float_input.data(),
        float_values.data(),
        float_indices.data(),
        M, N, k
    );
    
    if (result_f32 != 0) {
        std::cerr << "❌ Float32 extract_k_values failed: " << result_f32 << std::endl;
        return false;
    }
    
    // Test bfloat16 version
    std::vector<uint16_t> bf16_values(M * k);
    std::vector<int32_t> bf16_indices(M * k);
    
    int result_bf16 = metal_extract_k_values_bfloat16(
        bf16_input.data(),
        bf16_values.data(),
        bf16_indices.data(),
        M, N, k
    );
    
    if (result_bf16 != 0) {
        std::cerr << "❌ bfloat16 extract_k_values failed: " << result_bf16 << std::endl;
        return false;
    }
    
    // Compare results (indices should match)
    bool indices_match = true;
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < k; j++) {
            int idx = m * k + j;
            if (float_indices[idx] != bf16_indices[idx]) {
                std::cerr << "Index mismatch at [" << m << "," << j << "]: "
                          << "float32=" << float_indices[idx] 
                          << ", bfloat16=" << bf16_indices[idx] << std::endl;
                indices_match = false;
            }
        }
    }
    
    if (!indices_match) {
        std::cerr << "❌ Indices don't match between data types" << std::endl;
        return false;
    }
    
    // Compare values (allowing for bfloat16 precision loss)
    bool values_compatible = true;
    const float bf16_tolerance = 1e-2f; // bfloat16 has ~7 bits mantissa
    
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < k; j++) {
            int idx = m * k + j;
            float f32_val = float_values[idx];
            float bf16_val = bfloat16_to_float(bf16_values[idx]);
            
            // Skip zero padding slots
            if (f32_val == 0.0f && bf16_val == 0.0f) continue;
            
            float diff = std::abs(f32_val - bf16_val);
            float rel_diff = diff / std::max(std::abs(f32_val), 1e-8f);
            
            if (rel_diff > bf16_tolerance) {
                std::cerr << "Value precision mismatch at [" << m << "," << j << "]: "
                          << "float32=" << f32_val << ", bfloat16=" << bf16_val
                          << ", rel_diff=" << rel_diff << std::endl;
                values_compatible = false;
            }
        }
    }
    
    if (!values_compatible) {
        std::cerr << "❌ Values don't match within bfloat16 precision" << std::endl;
        return false;
    }
    
    std::cout << "✅ extract_k_values data type validation passed" << std::endl;
    return true;
}

bool test_topk_mask_data_types() {
    std::cout << "=== Testing topk_mask_logits data types ===" << std::endl;
    
    const int batch_size = 4;
    const int vocab_size = 32;
    const int k = 8;
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 5.0f);
    
    std::vector<float> float_logits(batch_size * vocab_size);
    std::vector<uint16_t> bf16_logits(batch_size * vocab_size);
    
    for (int i = 0; i < batch_size * vocab_size; i++) {
        float val = dist(gen);
        float_logits[i] = val;
        bf16_logits[i] = float_to_bfloat16(val);
    }
    
    // Copy for modification
    std::vector<float> float_test = float_logits;
    std::vector<uint16_t> bf16_test = bf16_logits;
    
    // Test float32 version
    int result_f32 = metal_topk_mask_logits_float32(
        float_test.data(),
        batch_size,
        vocab_size,
        k
    );
    
    if (result_f32 != 0) {
        std::cerr << "❌ Float32 topk_mask_logits failed: " << result_f32 << std::endl;
        return false;
    }
    
    // Test bfloat16 version
    int result_bf16 = metal_topk_mask_logits_bfloat16(
        bf16_test.data(),
        batch_size,
        vocab_size,
        k
    );
    
    if (result_bf16 != 0) {
        std::cerr << "❌ bfloat16 topk_mask_logits failed: " << result_bf16 << std::endl;
        return false;
    }
    
    // Compare masking patterns
    bool mask_patterns_match = true;
    for (int b = 0; b < batch_size; b++) {
        int f32_masked = 0, bf16_masked = 0;
        
        for (int i = 0; i < vocab_size; i++) {
            int idx = b * vocab_size + i;
            
            bool f32_mask = std::isinf(float_test[idx]) && float_test[idx] < 0;
            bool bf16_mask = (bf16_test[idx] == 0xFF80); // -infinity in bfloat16
            
            if (f32_mask) f32_masked++;
            if (bf16_mask) bf16_masked++;
        }
        
        // Both should have exactly k unmasked values
        int f32_unmasked = vocab_size - f32_masked;
        int bf16_unmasked = vocab_size - bf16_masked;
        
        if (f32_unmasked != k || bf16_unmasked != k) {
            std::cerr << "Masking count mismatch in batch " << b << ": "
                      << "float32 unmasked=" << f32_unmasked 
                      << ", bfloat16 unmasked=" << bf16_unmasked 
                      << ", expected=" << k << std::endl;
            mask_patterns_match = false;
        }
    }
    
    if (!mask_patterns_match) {
        std::cerr << "❌ Masking patterns don't match between data types" << std::endl;
        return false;
    }
    
    std::cout << "✅ topk_mask_logits data type validation passed" << std::endl;
    return true;
}

bool test_data_type_conversion_precision() {
    std::cout << "=== Testing bfloat16 conversion precision ===" << std::endl;
    
    // Test known values
    struct TestCase {
        float input;
        uint16_t expected_bf16;
        const char* description;
    };
    
    std::vector<TestCase> test_cases = {
        {0.0f, 0x0000, "zero"},
        {1.0f, 0x3F80, "one"},
        {-1.0f, 0xBF80, "negative one"},
        {-INFINITY, 0xFF80, "negative infinity"},
        {INFINITY, 0x7F80, "positive infinity"}
    };
    
    bool conversion_accurate = true;
    for (const auto& tc : test_cases) {
        uint16_t converted = float_to_bfloat16(tc.input);
        if (converted != tc.expected_bf16) {
            std::cerr << "Conversion error for " << tc.description << ": "
                      << "input=" << tc.input << ", got=0x" << std::hex << converted
                      << ", expected=0x" << tc.expected_bf16 << std::dec << std::endl;
            conversion_accurate = false;
        }
        
        // Test round-trip conversion (except for special values)
        if (std::isfinite(tc.input)) {
            float round_trip = bfloat16_to_float(converted);
            float error = std::abs(round_trip - tc.input) / std::max(std::abs(tc.input), 1e-8f);
            if (error > 1e-2f) { // Allow for bfloat16 precision
                std::cerr << "Round-trip error for " << tc.description << ": "
                          << "original=" << tc.input << ", round_trip=" << round_trip 
                          << ", error=" << error << std::endl;
                conversion_accurate = false;
            }
        }
    }
    
    if (!conversion_accurate) {
        std::cerr << "❌ Data type conversion precision test failed" << std::endl;
        return false;
    }
    
    std::cout << "✅ Data type conversion precision test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Metal Data Type Validation Tests ===" << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_data_type_conversion_precision();
    all_passed &= test_extract_k_values_data_types();
    all_passed &= test_topk_mask_data_types();
    
    if (all_passed) {
        std::cout << "\n🎉 All data type validation tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some data type validation tests failed!" << std::endl;
        return 1;
    }
}