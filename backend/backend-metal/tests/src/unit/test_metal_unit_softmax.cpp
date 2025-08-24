#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "metal_softmax.hpp"

// Precise CPU reference implementation
void cpu_softmax_float(const float* input, float* output, int batch_size, int vocab_size, float temperature) {
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

int main() {
    const int batch_size = 2;
    const int vocab_size = 8;
    const float temperature = 1.0f;

    // Test data with known patterns
    std::vector<float> input_data = {
        // Batch 0: simple ascending pattern
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        // Batch 1: mixed pattern with some large values
        0.0f, 10.0f, -5.0f, 3.0f, 1.0f, -2.0f, 8.0f, 0.5f
    };

    std::vector<float> metal_output(batch_size * vocab_size);
    std::vector<float> cpu_output(batch_size * vocab_size);

    // Test Metal implementation
    int result = metal_softmax_float(
        input_data.data(),
        metal_output.data(),
        batch_size,
        vocab_size,
        temperature
    );

    if (result != 0) {
        std::cerr << "FAIL: Metal softmax returned error: " << result << std::endl;
        return 1;
    }

    // Compute CPU reference
    cpu_softmax_float(
        input_data.data(),
        cpu_output.data(),
        batch_size,
        vocab_size,
        temperature
    );

    // Validate results
    const float tolerance = 1e-6f;
    bool pass = true;
    
    for (int b = 0; b < batch_size; b++) {
        // Check normalization (probabilities sum to 1)
        double sum = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            int idx = b * vocab_size + i;
            sum += metal_output[idx];
            
            // Check individual values match CPU
            if (std::abs(metal_output[idx] - cpu_output[idx]) > tolerance) {
                std::cerr << "Batch " << b << ", position " << i << ": got " 
                          << metal_output[idx] << ", expected " << cpu_output[idx] << std::endl;
                pass = false;
            }
            
            // Check values are positive
            if (metal_output[idx] < 0.0f) {
                std::cerr << "Batch " << b << ", position " << i << ": negative probability " 
                          << metal_output[idx] << std::endl;
                pass = false;
            }
        }
        
        // Check normalization
        if (std::abs(sum - 1.0) > tolerance) {
            std::cerr << "Batch " << b << ": sum = " << sum << ", expected 1.0" << std::endl;
            pass = false;
        }
    }

    if (!pass) {
        std::cerr << "FAIL: test_metal_unit_softmax" << std::endl;
        return 1;
    }

    std::cout << "PASS: test_metal_unit_softmax" << std::endl;
    return 0;
}