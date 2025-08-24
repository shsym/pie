#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "metal_topk_mask_logits.hpp"

// CPU reference implementation
void cpu_topk_mask_logits_float(float* logits, int batch_size, int vocab_size, int k) {
    for (int b = 0; b < batch_size; b++) {
        float* row = logits + b * vocab_size;
        
        // Create value-index pairs
        std::vector<std::pair<float, int>> values;
        for (int i = 0; i < vocab_size; i++) {
            values.emplace_back(row[i], i);
        }
        
        // Sort by value descending
        std::sort(values.begin(), values.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Keep top-k, mask the rest
        for (int i = k; i < vocab_size; i++) {
            row[values[i].second] = -INFINITY;
        }
    }
}

int main() {
    const int batch_size = 2;
    const int vocab_size = 10;
    const int k = 3;

    // Test data
    std::vector<float> input_data = {
        // Batch 0: clear top-3 should be indices 7,5,8
        1.0f, 2.0f, 0.5f, 3.0f, 1.5f, 8.0f, 2.5f, 9.0f, 7.0f, 0.1f,
        // Batch 1: clear top-3 should be indices 2,6,9  
        -1.0f, 0.0f, 15.0f, 2.0f, 1.0f, -5.0f, 12.0f, 3.0f, 4.0f, 10.0f
    };

    std::vector<float> metal_output = input_data;
    std::vector<float> cpu_output = input_data;

    // Test Metal implementation
    int result = metal_topk_mask_logits_float32(
        metal_output.data(),
        batch_size,
        vocab_size,
        k
    );

    if (result != 0) {
        std::cerr << "FAIL: Metal topk_mask returned error: " << result << std::endl;
        return 1;
    }

    // Compute CPU reference
    cpu_topk_mask_logits_float(
        cpu_output.data(),
        batch_size,
        vocab_size,
        k
    );

    // Validate results
    bool pass = true;
    
    for (int b = 0; b < batch_size; b++) {
        int unmasked_count = 0;
        int masked_count = 0;
        
        for (int i = 0; i < vocab_size; i++) {
            int idx = b * vocab_size + i;
            
            bool metal_masked = std::isinf(metal_output[idx]) && metal_output[idx] < 0;
            bool cpu_masked = std::isinf(cpu_output[idx]) && cpu_output[idx] < 0;
            
            if (metal_masked != cpu_masked) {
                std::cerr << "Batch " << b << ", position " << i << ": Metal masked=" 
                          << metal_masked << ", CPU masked=" << cpu_masked << std::endl;
                pass = false;
            }
            
            if (metal_masked) {
                masked_count++;
            } else {
                unmasked_count++;
                // Values should match for unmasked positions
                if (std::abs(metal_output[idx] - input_data[idx]) > 1e-6f) {
                    std::cerr << "Batch " << b << ", position " << i 
                              << ": unmasked value changed from " << input_data[idx] 
                              << " to " << metal_output[idx] << std::endl;
                    pass = false;
                }
            }
        }
        
        // Check counts
        if (unmasked_count != k) {
            std::cerr << "Batch " << b << ": expected " << k 
                      << " unmasked values, got " << unmasked_count << std::endl;
            pass = false;
        }
        
        if (masked_count != (vocab_size - k)) {
            std::cerr << "Batch " << b << ": expected " << (vocab_size - k) 
                      << " masked values, got " << masked_count << std::endl;
            pass = false;
        }
    }

    if (!pass) {
        std::cerr << "FAIL: test_metal_unit_topk_mask" << std::endl;
        return 1;
    }

    std::cout << "PASS: test_metal_unit_topk_mask" << std::endl;
    return 0;
}