#include "metal_softmax.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

/**
 * CPU reference implementation of softmax for verification
 * Implements numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 */
int metal_softmax_reference(
    const float* input,
    float* output,
    int batch_size,
    int vocab_size,
    float temperature
) {
    if (!input || !output || batch_size <= 0 || vocab_size <= 0 || temperature <= 0.0f) {
        std::cerr << "Invalid parameters for metal_softmax_reference" << std::endl;
        return -1;
    }
    
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const float* seq_input = input + batch_idx * vocab_size;
        float* seq_output = output + batch_idx * vocab_size;
        
        // Phase 1: Find maximum value for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab_size; ++i) {
            float val = seq_input[i] / temperature;  // Apply temperature scaling
            max_val = std::max(max_val, val);
        }
        
        // Phase 2: Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            float val = seq_input[i] / temperature - max_val;
            float exp_val = std::exp(val);
            seq_output[i] = exp_val;
            sum += exp_val;
        }
        
        // Phase 3: Normalize by sum to get probabilities
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab_size; ++i) {
            seq_output[i] *= inv_sum;
        }
    }
    
    return 0;
}

/**
 * Validation function to compare Metal and reference implementations
 * Returns 0 if implementations match within tolerance, -1 otherwise
 */
int validate_metal_softmax_accuracy(
    const float* input,
    const float* metal_output,
    int batch_size,
    int vocab_size,
    float temperature,
    float tolerance = 1e-5f
) {
    std::vector<float> reference_output(batch_size * vocab_size);
    
    // Compute reference result
    int ref_result = metal_softmax_reference(
        input,
        reference_output.data(),
        batch_size,
        vocab_size,
        temperature
    );
    
    if (ref_result != 0) {
        std::cerr << "Reference softmax computation failed" << std::endl;
        return -1;
    }
    
    // Compare outputs element by element
    const size_t total_elements = static_cast<size_t>(batch_size) * vocab_size;
    float max_error = 0.0f;
    size_t error_count = 0;
    
    for (size_t i = 0; i < total_elements; ++i) {
        float error = std::abs(metal_output[i] - reference_output[i]);
        max_error = std::max(max_error, error);
        
        if (error > tolerance) {
            error_count++;
            if (error_count <= 5) {  // Show first few errors
                std::cerr << "Softmax mismatch at index " << i 
                         << ": Metal=" << metal_output[i] 
                         << ", Reference=" << reference_output[i]
                         << ", Error=" << error << std::endl;
            }
        }
    }
    
    if (error_count > 0) {
        std::cerr << "Metal softmax validation failed: " 
                 << error_count << "/" << total_elements << " elements exceed tolerance " << tolerance
                 << ", max_error=" << max_error << std::endl;
        return -1;
    }
    
    std::cout << "Metal softmax validation passed: max_error=" << max_error 
             << " (tolerance=" << tolerance << ")" << std::endl;
    return 0;
}

/**
 * Comprehensive test for softmax correctness
 */
void test_metal_softmax() {
    std::cout << "Running Metal softmax correctness tests..." << std::endl;
    
    // Test case 1: Small batch, small vocab
    {
        const int batch_size = 2;
        const int vocab_size = 4;
        const float temperature = 1.0f;
        
        std::vector<float> input = {
            1.0f, 2.0f, 3.0f, 4.0f,  // batch 0
            0.5f, 1.5f, 2.5f, 3.5f   // batch 1
        };
        
        std::vector<float> metal_output(batch_size * vocab_size);
        
        int result = metal_softmax_float(
            input.data(),
            metal_output.data(),
            batch_size,
            vocab_size,
            temperature
        );
        
        if (result != 0) {
            std::cerr << "Test 1: Metal softmax failed" << std::endl;
            return;
        }
        
        // Validate against reference
        if (validate_metal_softmax_accuracy(
            input.data(), metal_output.data(),
            batch_size, vocab_size, temperature
        ) != 0) {
            std::cerr << "Test 1: Validation failed" << std::endl;
            return;
        }
        
        // Check that each row sums to 1.0
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; ++i) {
                sum += metal_output[batch_idx * vocab_size + i];
            }
            if (std::abs(sum - 1.0f) > 1e-5f) {
                std::cerr << "Test 1: Batch " << batch_idx << " sum=" << sum << " (expected 1.0)" << std::endl;
                return;
            }
        }
        
        std::cout << "Test 1 passed: Small batch, small vocab" << std::endl;
    }
    
    // Test case 2: Temperature scaling
    {
        const int batch_size = 1;
        const int vocab_size = 3;
        const float temperature = 0.5f;  // Lower temperature = sharper distribution
        
        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        std::vector<float> metal_output(batch_size * vocab_size);
        
        int result = metal_softmax_float(
            input.data(),
            metal_output.data(),
            batch_size,
            vocab_size,
            temperature
        );
        
        if (result != 0 || validate_metal_softmax_accuracy(
            input.data(), metal_output.data(),
            batch_size, vocab_size, temperature
        ) != 0) {
            std::cerr << "Test 2: Temperature scaling failed" << std::endl;
            return;
        }
        
        std::cout << "Test 2 passed: Temperature scaling" << std::endl;
    }
    
    std::cout << "All Metal softmax tests passed!" << std::endl;
}