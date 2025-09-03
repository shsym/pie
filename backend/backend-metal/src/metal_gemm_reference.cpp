#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include "workspace_utils.hpp"

// Metal reference implementation of the Metal GEMM kernel logic
// This allows testing the algorithm on Linux without actual Metal execution
// Implements the exact same logic that runs in metal_gemm.metal

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

// Metal reference implementation matching Metal kernel logic
void metal_gemm_bfloat16_reference(
    const bfloat16_t* A,
    const bfloat16_t* B,
    const bfloat16_t* bias,
    bfloat16_t* C,
    int m, int n, int k,
    bool transa, bool transb, bool use_bias
) {
    // Calculate leading dimensions (matches cuBLAS convention)
    const int lda = transa ? m : k;
    const int ldb = transb ? k : n;
    const int ldc = n;
    
    // Initialize output to zero
    for (int i = 0; i < m * n; ++i) {
        C[i] = float_to_bfloat16(0.0f);
    }
    
    // Main GEMM computation: C = A × B (with transposes)
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;  // Use float accumulation like Metal kernel
            
            for (int ki = 0; ki < k; ++ki) {
                float a_val, b_val;
                
                // Load from A with transpose handling
                if (transa) {
                    // A is transposed: access A[ki][row] stored as A[ki * lda + row]
                    a_val = bfloat16_to_float(A[ki * lda + row]);
                } else {
                    // A is not transposed: access A[row][ki] stored as A[row * lda + ki]
                    a_val = bfloat16_to_float(A[row * lda + ki]);
                }
                
                // Load from B with transpose handling
                if (transb) {
                    // B is transposed: access B[col][ki] stored as B[col * ldb + ki]
                    b_val = bfloat16_to_float(B[col * ldb + ki]);
                } else {
                    // B is not transposed: access B[ki][col] stored as B[ki * ldb + col]
                    b_val = bfloat16_to_float(B[ki * ldb + col]);
                }
                
                sum += a_val * b_val;
            }
            
            // Add bias if provided (matches cuBLAS behavior: beta = 1.0f when bias is present)
            if (use_bias && bias) {
                sum += bfloat16_to_float(bias[col]);
            }
            
            // Store result: C[row][col] stored as C[row * ldc + col]
            C[row * ldc + col] = float_to_bfloat16(sum);
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

// Compare two arrays with tolerance for numerical differences
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
    auto artifacts_base = workspace_utils::get_cuda_artifacts_dir();
    const std::string artifacts_dir = (artifacts_base / "gemm" / "test1").string() + "/";
    
    try {
        // Load CUDA reference data (bfloat16 stored as uint16_t)
        auto A_data = load_binary_file<bfloat16_t>(artifacts_dir + "A.bin");
        auto B_data = load_binary_file<bfloat16_t>(artifacts_dir + "B.bin");  
        auto C_expected = load_binary_file<bfloat16_t>(artifacts_dir + "C.bin");
        
        // Test configuration (matches test1 case: m=32, n=128, k=64, transb=true)
        const int m = 32, n = 128, k = 64;
        const bool transa = false, transb = true;
        const bool use_bias = false;
        
        std::cout << "Testing Metal reference GEMM with dimensions: " << m << "x" << n << "x" << k << std::endl;
        std::cout << "transa: " << transa << ", transb: " << transb << ", use_bias: " << use_bias << std::endl;
        
        // Verify data sizes match expected dimensions
        const size_t expected_A_size = m * k;  // [32, 64]
        const size_t expected_B_size = n * k;  // [128, 64] (before transpose)
        const size_t expected_C_size = m * n;  // [32, 128]
        
        assert(A_data.size() == expected_A_size);
        assert(B_data.size() == expected_B_size);
        assert(C_expected.size() == expected_C_size);
        
        std::cout << "Input sizes verified: A=" << A_data.size() << ", B=" << B_data.size() 
                  << ", C=" << C_expected.size() << std::endl;
        
        // Allocate output buffer for Metal computation
        std::vector<bfloat16_t> C_metal(expected_C_size, 0);
        
        // Run Metal reference GEMM computation
        metal_gemm_bfloat16_reference(
            A_data.data(),
            B_data.data(),
            nullptr,  // no bias
            C_metal.data(),
            m, n, k,
            transa, transb, use_bias
        );
        
        std::cout << "Metal reference GEMM computation completed" << std::endl;
        
        // Compare results
        bool match = compare_arrays(C_metal, C_expected, 1e-3f);  // bfloat16 has limited precision
        
        if (match) {
            std::cout << "✅ Metal reference GEMM matches CUDA reference!" << std::endl;
        } else {
            std::cout << "❌ Metal reference GEMM does not match CUDA reference" << std::endl;
            
            // Show a few sample values for debugging
            std::cout << "\nSample comparison (first 10 values):" << std::endl;
            for (size_t i = 0; i < std::min(size_t(10), C_metal.size()); ++i) {
                float metal_val = bfloat16_to_float(C_metal[i]);
                float cuda_val = bfloat16_to_float(C_expected[i]);
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
    std::cout << "=== Metal Reference GEMM Test vs CUDA Reference ===" << std::endl;
    
    bool success = test_metal_vs_cuda_artifacts();
    
    if (success) {
        std::cout << "All tests passed! ✅" << std::endl;
        return 0;
    } else {
        std::cout << "Tests failed! ❌" << std::endl;
        return 1;
    }
}