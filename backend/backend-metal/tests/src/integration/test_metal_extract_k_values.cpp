#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include "metal_extract_k_values.hpp"

// Test parameters
const int TEST_M = 8;      // Number of sequences
const int TEST_N = 32000;  // Vocab size
const int TEST_K = 50;     // Top-K values to extract

struct ValueIndex {
    float value;
    int index;
    
    bool operator>(const ValueIndex& other) const {
        return value > other.value;
    }
};

// Reference CPU implementation for validation
// Extract first k non-zero/non-infinity values from sparse matrix
void cpu_extract_k_values_float32(
    const float* A, float* V, int32_t* I, 
    unsigned int M, unsigned int N, unsigned int k
) {
    for (unsigned int m = 0; m < M; m++) {
        const float* row = A + m * N;
        unsigned int found = 0;
        
        // Find first k valid values
        for (unsigned int n = 0; n < N && found < k; n++) {
            float val = row[n];
            // Extract non-zero, non-infinity values
            if (val != 0.0f && val != -INFINITY && !std::isinf(val)) {
                V[m * k + found] = val;
                I[m * k + found] = static_cast<int32_t>(n);
                found++;
            }
        }
        
        // Zero remaining slots
        for (unsigned int i = found; i < k; i++) {
            V[m * k + i] = 0.0f;
            I[m * k + i] = 0;
        }
    }
}

void cpu_extract_k_values_bfloat16(
    const void* A, void* V, int32_t* I,
    unsigned int M, unsigned int N, unsigned int k
) {
    const uint16_t* input = static_cast<const uint16_t*>(A);
    uint16_t* output_values = static_cast<uint16_t*>(V);
    
    for (unsigned int m = 0; m < M; m++) {
        const uint16_t* row = input + m * N;
        unsigned int found = 0;
        
        // Find first k valid values
        for (unsigned int n = 0; n < N && found < k; n++) {
            uint16_t bf16_val = row[n];
            
            // Convert bfloat16 to float for comparison
            uint32_t float_bits = static_cast<uint32_t>(bf16_val) << 16;
            float val = *reinterpret_cast<float*>(&float_bits);
            
            // Extract non-zero values  
            if (val != 0.0f && !std::isinf(val)) {
                output_values[m * k + found] = bf16_val;
                I[m * k + found] = static_cast<int32_t>(n);
                found++;
            }
        }
        
        // Zero remaining slots
        for (unsigned int i = found; i < k; i++) {
            output_values[m * k + i] = 0;  // bfloat16 zero
            I[m * k + i] = 0;
        }
    }
}

bool compare_results_float32(
    const float* metal_values, const int32_t* metal_indices,
    const float* cpu_values, const int32_t* cpu_indices,
    unsigned int M, unsigned int k, float tolerance = 1e-5f
) {
    bool success = true;
    
    for (unsigned int m = 0; m < M; m++) {
        for (unsigned int i = 0; i < k; i++) {
            int idx = m * k + i;
            
            // Check values
            if (std::abs(metal_values[idx] - cpu_values[idx]) > tolerance) {
                std::cerr << "Value mismatch at [" << m << "," << i << "]: "
                          << metal_values[idx] << " vs " << cpu_values[idx] << std::endl;
                success = false;
            }
            
            // Check indices
            if (metal_indices[idx] != cpu_indices[idx]) {
                std::cerr << "Index mismatch at [" << m << "," << i << "]: "
                          << metal_indices[idx] << " vs " << cpu_indices[idx] << std::endl;
                success = false;
            }
        }
    }
    
    return success;
}

int main() {
    std::cout << "=== Metal Extract K Values Kernel Test ===" << std::endl;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  M (sequences): " << TEST_M << std::endl;
    std::cout << "  N (vocab_size): " << TEST_N << std::endl;
    std::cout << "  K (top values): " << TEST_K << std::endl;
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    // Test Float32 version
    {
        std::cout << "\n=== Testing Float32 Version ===" << std::endl;
        
        std::vector<float> input_data(TEST_M * TEST_N);
        std::vector<float> metal_values(TEST_M * TEST_K);
        std::vector<int32_t> metal_indices(TEST_M * TEST_K);
        std::vector<float> cpu_values(TEST_M * TEST_K);
        std::vector<int32_t> cpu_indices(TEST_M * TEST_K);
        
        // Generate sparse test data with zeros and some infinity values
        std::uniform_real_distribution<float> sparse_prob(0.0f, 1.0f);
        
        for (int i = 0; i < TEST_M * TEST_N; i++) {
            float prob = sparse_prob(gen);
            if (prob < 0.7f) {
                input_data[i] = 0.0f;  // 70% zeros (sparse)
            } else if (prob < 0.8f) {
                input_data[i] = -INFINITY;  // 10% -infinity
            } else {
                input_data[i] = dist(gen);  // 20% valid values
            }
        }
        
        // Test Metal implementation
        std::cout << "Running Metal extract_k_values_float32..." << std::endl;
        int result = metal_extract_k_values_float32(
            input_data.data(),
            metal_values.data(),
            metal_indices.data(),
            TEST_M, TEST_N, TEST_K
        );
        
        if (result != 0) {
            std::cerr << "❌ Metal extract_k_values_float32 failed with error: " << result << std::endl;
            return 1;
        }
        
        // Compute CPU reference
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_extract_k_values_float32(
            input_data.data(),
            cpu_values.data(),
            cpu_indices.data(),
            TEST_M, TEST_N, TEST_K
        );
        
        // Validate results
        std::cout << "Validating results..." << std::endl;
        
        // Check that we only extracted valid values (no zeros, no infinities)
        bool validity_check = true;
        for (unsigned int m = 0; m < TEST_M; m++) {
            for (unsigned int i = 0; i < TEST_K; i++) {
                int idx = m * TEST_K + i;
                float val = metal_values[idx];
                
                // Skip checking trailing zeros (unfilled slots)
                if (val == 0.0f) continue;
                
                if (std::isinf(val)) {
                    std::cerr << "❌ Found infinity at [" << m << "," << i << "]" << std::endl;
                    validity_check = false;
                }
            }
        }
        
        if (!validity_check) {
            std::cerr << "❌ Float32 validity check failed" << std::endl;
            return 1;
        }
        std::cout << "✅ Float32 validity check passed" << std::endl;
        
        // Compare against CPU reference
        bool accuracy_valid = compare_results_float32(
            metal_values.data(), metal_indices.data(),
            cpu_values.data(), cpu_indices.data(),
            TEST_M, TEST_K
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
        
        std::vector<uint16_t> input_data(TEST_M * TEST_N);
        std::vector<uint16_t> metal_values(TEST_M * TEST_K);
        std::vector<int32_t> metal_indices(TEST_M * TEST_K);
        std::vector<uint16_t> cpu_values(TEST_M * TEST_K);
        std::vector<int32_t> cpu_indices(TEST_M * TEST_K);
        
        // Generate sparse test data (convert float to bfloat16)
        std::uniform_real_distribution<float> sparse_prob(0.0f, 1.0f);
        
        for (int i = 0; i < TEST_M * TEST_N; i++) {
            float prob = sparse_prob(gen);
            float val;
            
            if (prob < 0.8f) {
                val = 0.0f;  // 80% zeros (more sparse for bfloat16 test)
            } else {
                val = dist(gen);  // 20% valid values
            }
            
            uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
            input_data[i] = static_cast<uint16_t>(bits >> 16);
        }
        
        // Test Metal implementation
        std::cout << "Running Metal extract_k_values_bfloat16..." << std::endl;
        int result = metal_extract_k_values_bfloat16(
            input_data.data(),
            metal_values.data(),
            metal_indices.data(),
            TEST_M, TEST_N, TEST_K
        );
        
        if (result != 0) {
            std::cerr << "❌ Metal extract_k_values_bfloat16 failed with error: " << result << std::endl;
            return 1;
        }
        
        // Compute CPU reference
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_extract_k_values_bfloat16(
            input_data.data(),
            cpu_values.data(),
            cpu_indices.data(),
            TEST_M, TEST_N, TEST_K
        );
        
        // Basic validation (just check that we got results)
        std::cout << "Basic validation..." << std::endl;
        
        // Check that indices are in valid range
        bool indices_valid = true;
        for (unsigned int m = 0; m < TEST_M; m++) {
            for (unsigned int i = 0; i < TEST_K; i++) {
                int idx = m * TEST_K + i;
                if (metal_indices[idx] < 0 || metal_indices[idx] >= static_cast<int>(TEST_N)) {
                    std::cerr << "❌ Invalid index at [" << m << "," << i << "]: " << metal_indices[idx] << std::endl;
                    indices_valid = false;
                }
            }
        }
        
        if (!indices_valid) {
            std::cerr << "❌ bfloat16 index validation failed" << std::endl;
            return 1;
        }
        std::cout << "✅ bfloat16 index validation passed" << std::endl;
    }
    
    std::cout << "\n🎉 All Metal Extract K Values tests passed!" << std::endl;
    return 0;
}