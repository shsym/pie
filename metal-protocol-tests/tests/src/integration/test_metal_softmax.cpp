#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include "metal_softmax.hpp"

// Test parameters
const int TEST_BATCH_SIZE = 4;
const int TEST_VOCAB_SIZE = 32000;  // Typical vocab size
const float TEST_TEMPERATURE = 1.0f;
const float TOLERANCE = 1e-5f;

// Reference CPU implementation for validation
void cpu_softmax(const float* input, float* output, int batch_size, int vocab_size, float temperature) {
    for (int b = 0; b < batch_size; b++) {
        const float* input_row = input + b * vocab_size;
        float* output_row = output + b * vocab_size;
        
        // Apply temperature scaling and find max for numerical stability
        float max_val = input_row[0] / temperature;
        for (int i = 1; i < vocab_size; i++) {
            float val = input_row[i] / temperature;
            if (val > max_val) max_val = val;
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float val = std::exp(input_row[i] / temperature - max_val);
            output_row[i] = val;
            sum += val;
        }
        
        // Normalize
        for (int i = 0; i < vocab_size; i++) {
            output_row[i] /= sum;
        }
    }
}

bool compare_arrays(const float* a, const float* b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] 
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== Metal Softmax Kernel Test ===" << std::endl;
    
    // Initialize test data
    const int total_size = TEST_BATCH_SIZE * TEST_VOCAB_SIZE;
    std::vector<float> input_data(total_size);
    std::vector<float> metal_output(total_size);
    std::vector<float> cpu_output(total_size);
    
    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 2.0f);  // Normal distribution for logits
    
    for (int i = 0; i < total_size; i++) {
        input_data[i] = dist(gen);
    }
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Batch size: " << TEST_BATCH_SIZE << std::endl;
    std::cout << "  Vocab size: " << TEST_VOCAB_SIZE << std::endl;
    std::cout << "  Temperature: " << TEST_TEMPERATURE << std::endl;
    std::cout << "  Total elements: " << total_size << std::endl;
    
    // Test Metal implementation
    std::cout << "\nRunning Metal softmax..." << std::endl;
    int metal_result = metal_softmax_float(
        input_data.data(),
        metal_output.data(),
        TEST_BATCH_SIZE,
        TEST_VOCAB_SIZE,
        TEST_TEMPERATURE
    );
    
    if (metal_result != 0) {
        std::cerr << "❌ Metal softmax failed with error code: " << metal_result << std::endl;
        return 1;
    }
    
    // Compute reference CPU implementation
    std::cout << "Computing CPU reference..." << std::endl;
    cpu_softmax(input_data.data(), cpu_output.data(), TEST_BATCH_SIZE, TEST_VOCAB_SIZE, TEST_TEMPERATURE);
    
    // Validate results
    std::cout << "\nValidating results..." << std::endl;
    
    // Check if outputs sum to 1 (basic softmax property)
    bool sums_valid = true;
    for (int b = 0; b < TEST_BATCH_SIZE; b++) {
        float sum = 0.0f;
        for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
            sum += metal_output[b * TEST_VOCAB_SIZE + i];
        }
        
        if (std::abs(sum - 1.0f) > TOLERANCE) {
            std::cerr << "❌ Batch " << b << " sum: " << sum << " (should be 1.0)" << std::endl;
            sums_valid = false;
        }
    }
    
    if (!sums_valid) {
        std::cerr << "❌ Softmax normalization check failed" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Softmax normalization check passed" << std::endl;
    
    // Compare against CPU reference
    bool accuracy_valid = compare_arrays(metal_output.data(), cpu_output.data(), total_size, TOLERANCE);
    
    if (!accuracy_valid) {
        std::cerr << "❌ Metal vs CPU accuracy check failed" << std::endl;
        
        // Show first few mismatches for debugging
        std::cout << "\nFirst 10 values comparison:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < std::min(10, total_size); i++) {
            std::cout << "  [" << i << "] Metal: " << metal_output[i] 
                      << ", CPU: " << cpu_output[i]
                      << ", Diff: " << std::abs(metal_output[i] - cpu_output[i]) << std::endl;
        }
        return 1;
    }
    
    std::cout << "✅ Metal vs CPU accuracy check passed" << std::endl;
    
    // Performance test
    std::cout << "\nRunning performance test..." << std::endl;
    const int PERF_ITERATIONS = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        metal_softmax_float(
            input_data.data(),
            metal_output.data(),
            TEST_BATCH_SIZE,
            TEST_VOCAB_SIZE,
            TEST_TEMPERATURE
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = static_cast<double>(duration.count()) / PERF_ITERATIONS;
    double throughput = (static_cast<double>(TEST_BATCH_SIZE * TEST_VOCAB_SIZE) / avg_time_us) * 1000000.0;
    
    std::cout << "Performance results:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avg_time_us << " μs" << std::endl;
    std::cout << "  Throughput: " << std::scientific << std::setprecision(2) << throughput << " elements/sec" << std::endl;
    
    std::cout << "\n🎉 All Metal Softmax tests passed!" << std::endl;
    return 0;
}