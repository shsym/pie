#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <algorithm>

// Metal reference implementation of the Metal RMSNorm kernel logic
// This allows testing the algorithm on Linux without actual Metal execution
// Implements the exact same logic that runs in metal_rmsnorm.metal

using bfloat16_t = uint16_t;

// Helper to convert float to bfloat16 
uint16_t float_to_bfloat16(float f) {
    union { float f; uint32_t i; } u = {f};
    return (u.i + 0x7fff + ((u.i >> 16) & 1)) >> 16;
}

// Helper to convert bfloat16 to float
float bfloat16_to_float(uint16_t bf) {
    union { float f; uint32_t i; } u;
    u.i = static_cast<uint32_t>(bf) << 16;
    return u.f;
}

// CUDA-compatible rsqrt approximation to match FlashInfer's rsqrt.approx.ftz.f32
// This approximates the CUDA hardware instruction behavior
float cuda_rsqrt_approx(float x) {
    // Handle edge cases like CUDA's ftz (flush-to-zero) behavior
    if (x <= 0.0f || !std::isfinite(x)) {
        return 1.0f / std::sqrt(x);  // fallback for edge cases
    }
    
    // Use IEEE 754 bit manipulation for rsqrt approximation (similar to CUDA hardware)
    union { float f; uint32_t i; } u;
    u.f = x;
    
    // Magic number approximation for rsqrt (based on Quake III algorithm, similar to hardware)
    u.i = 0x5f3759df - (u.i >> 1);
    float y = u.f;
    
    // One Newton-Raphson iteration for better accuracy (approximates CUDA precision)
    y = y * (1.5f - 0.5f * x * y * y);
    
    return y;
}

// Metal reference implementation matching FlashInfer's exact algorithm
// RMSNorm formula: output = input * rsqrt(mean(input^2) + eps) * (weight_bias + weight)
// where weight_bias = 0.0f for standard RMSNorm
void metal_rmsnorm_reference(
    const bfloat16_t* input,
    const bfloat16_t* weight,
    bfloat16_t* output,
    int num_tokens,
    int hidden_size,
    float eps
) {
    const float weight_bias = 0.0f;  // FlashInfer uses weight_bias = 0.0f for RMSNorm
    
    // Process each token independently (matches FlashInfer's batch processing)
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const bfloat16_t* input_token = input + token_idx * hidden_size;
        bfloat16_t* output_token = output + token_idx * hidden_size;
        
        // Phase 1: Compute sum of squares (mimic FlashInfer's accumulation)
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = bfloat16_to_float(input_token[i]);
            sum_sq += val * val;
        }
        
        // Phase 2: Compute RMS normalization factor (matches FlashInfer line 81)
        // Use CUDA-compatible rsqrt approximation to match FlashInfer's rsqrt.approx.ftz.f32
        float rms_rcp = cuda_rsqrt_approx(sum_sq / float(hidden_size) + eps);
        
        // Phase 3: Apply normalization and weight scaling (matches FlashInfer line 95)
        for (int i = 0; i < hidden_size; ++i) {
            float input_val = bfloat16_to_float(input_token[i]);
            float weight_val = bfloat16_to_float(weight[i]);
            // FlashInfer formula: input * rms_rcp * (weight_bias + weight)
            float result = input_val * rms_rcp * (weight_bias + weight_val);
            output_token[i] = float_to_bfloat16(result);
        }
    }
}

// Load binary data from test artifact
template<typename T>
std::vector<T> load_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<T> buffer(size / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filepath);
    }
    
    return buffer;
}

// Compare two arrays with strict tolerance: relative OR absolute threshold
bool compare_arrays_strict(const std::vector<bfloat16_t>& a, const std::vector<bfloat16_t>& b, 
                          float relative_tolerance, float absolute_tolerance) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t num_errors = 0;
    const size_t max_show_errors = 20;
    
    for (size_t i = 0; i < a.size(); ++i) {
        float val_a = bfloat16_to_float(a[i]);
        float val_b = bfloat16_to_float(b[i]);
        float abs_error = std::abs(val_a - val_b);
        float rel_error = (std::abs(val_b) > 1e-8f) ? abs_error / std::abs(val_b) : abs_error;
        
        // Error if either relative OR absolute threshold is exceeded  
        bool is_error = (abs_error > absolute_tolerance) || (rel_error > relative_tolerance);
        
        if (is_error) {
            num_errors++;
            if (num_errors <= max_show_errors) {
                std::cerr << "Error at index " << i << ": " << val_a << " vs " << val_b 
                         << " (abs_diff: " << abs_error << ", rel_diff: " << (rel_error*100) << "%)" << std::endl;
            }
        }
        
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    std::cout << "Max absolute error: " << max_abs_error << std::endl;
    std::cout << "Max relative error: " << (max_rel_error*100) << "%" << std::endl;
    std::cout << "Absolute tolerance: " << absolute_tolerance << std::endl;
    std::cout << "Relative tolerance: " << (relative_tolerance*100) << "%" << std::endl;
    std::cout << "Errors above strict threshold: " << num_errors 
              << "/" << a.size() << " (" << (100.0 * num_errors / a.size()) << "%)" << std::endl;
    
    return num_errors == 0;
}

// Legacy comparison function for debugging
bool compare_arrays(const std::vector<bfloat16_t>& a, const std::vector<bfloat16_t>& b, float tolerance = 1e-3f) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    float max_error = 0.0f;
    size_t num_errors = 0;
    const size_t max_show_errors = 10;
    
    for (size_t i = 0; i < a.size(); ++i) {
        float val_a = bfloat16_to_float(a[i]);
        float val_b = bfloat16_to_float(b[i]);
        float error = std::abs(val_a - val_b);
        
        if (error > tolerance) {
            num_errors++;
            if (num_errors <= max_show_errors) {  // Show first few errors
                std::cerr << "Error at index " << i << ": " << val_a << " vs " << val_b 
                         << " (diff: " << error << ")" << std::endl;
            }
        }
        
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error: " << max_error << ", Errors above tolerance: " << num_errors 
              << "/" << a.size() << " (" << (100.0 * num_errors / a.size()) << "%)" << std::endl;
    
    return num_errors == 0;
}

// Test Metal reference implementation against CUDA reference data
bool test_metal_vs_cuda_artifacts() {
    const std::string artifacts_dir = "../../metal-protocol-tests/tests/artifacts/rms_norm/test1/";
    
    try {
        // Load CUDA reference data
        auto input_data = load_binary_file<bfloat16_t>(artifacts_dir + "input.bin");
        auto weight_data = load_binary_file<bfloat16_t>(artifacts_dir + "weight.bin");  
        auto output_expected = load_binary_file<bfloat16_t>(artifacts_dir + "output.bin");
        
        // Test configuration from meta.json: num_tokens=128, hidden_size=4096, eps=1e-05
        const int num_tokens = 128;
        const int hidden_size = 4096;
        const float eps = 1e-05f;
        
        std::cout << "Testing Metal reference RMSNorm with:" << std::endl;
        std::cout << "  num_tokens: " << num_tokens << std::endl;
        std::cout << "  hidden_size: " << hidden_size << std::endl;
        std::cout << "  eps: " << eps << std::endl;
        
        // Verify data sizes match expected dimensions
        const size_t expected_input_size = num_tokens * hidden_size;    // [128, 4096]
        const size_t expected_weight_size = hidden_size;                // [4096]
        const size_t expected_output_size = num_tokens * hidden_size;   // [128, 4096]
        
        assert(input_data.size() == expected_input_size);
        assert(weight_data.size() == expected_weight_size);
        assert(output_expected.size() == expected_output_size);
        
        std::cout << "Input sizes verified: input=" << input_data.size() 
                  << ", weight=" << weight_data.size() 
                  << ", output=" << output_expected.size() << std::endl;
        
        // Allocate output buffer for Metal computation
        std::vector<bfloat16_t> output_metal(expected_output_size, 0);
        
        // Run Metal reference RMSNorm computation
        metal_rmsnorm_reference(
            input_data.data(),
            weight_data.data(),
            output_metal.data(),
            num_tokens,
            hidden_size,
            eps
        );
        
        std::cout << "Metal reference RMSNorm computation completed" << std::endl;
        
        // Compare results with relaxed threshold: 1% relative difference OR 0.01 absolute difference  
        float relative_tolerance = 0.01f;  // 1%
        float absolute_tolerance = 0.01f;   // 0.01 absolute
        bool match = compare_arrays_strict(output_metal, output_expected, relative_tolerance, absolute_tolerance);
        
        if (match) {
            std::cout << "✅ Metal reference RMSNorm matches CUDA reference!" << std::endl;
        } else {
            std::cout << "❌ Metal reference RMSNorm does not match CUDA reference" << std::endl;
            
            // Show a few sample values for debugging
            std::cout << "\nSample comparison (first 10 values):" << std::endl;
            for (size_t i = 0; i < std::min(size_t(10), output_metal.size()); ++i) {
                float metal_val = bfloat16_to_float(output_metal[i]);
                float cuda_val = bfloat16_to_float(output_expected[i]);
                std::cout << "  [" << i << "] Metal: " << metal_val << ", CUDA: " << cuda_val 
                         << ", diff: " << std::abs(metal_val - cuda_val) << std::endl;
            }
        }
        
        return match;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load test artifacts: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Metal Reference RMSNorm Test vs CUDA Reference ===" << std::endl;
    
    bool success = test_metal_vs_cuda_artifacts();
    
    if (success) {
        std::cout << "All tests passed! ✅" << std::endl;
        return 0;
    } else {
        std::cout << "Tests failed! ❌" << std::endl;
        return 1;
    }
}