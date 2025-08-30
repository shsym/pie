#include <cassert>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include "metal_extract_k_values.hpp"

// Test CUDA-style unit test with precise expected outputs
int main() {
    using T = float;

    const int M = 3;
    const int N = 8;
    const int k = 3;

    // Create test data exactly like CUDA test
    std::vector<T> h_A(M * N, -INFINITY);
    
    // Row 0: finite at cols 1,3,6,7 -> expect first 3 -> (1,3,6)
    h_A[0 * N + 1] = 0.1f;
    h_A[0 * N + 3] = 0.3f;
    h_A[0 * N + 6] = 0.6f;
    h_A[0 * N + 7] = 0.7f;
    
    // Row 1: finite at cols 0,4 -> expect (0,4, then padding)
    h_A[1 * N + 0] = 1.0f;
    h_A[1 * N + 4] = 1.4f;
    
    // Row 2: finite at cols 2,5,6 -> expect (2,5,6)
    h_A[2 * N + 2] = 2.2f;
    h_A[2 * N + 5] = 2.5f;
    h_A[2 * N + 6] = 2.6f;

    std::vector<T> h_V(M * k, 0);
    std::vector<int32_t> h_I(M * k, -1);

    // Call Metal kernel
    int result = metal_extract_k_values_float32(
        h_A.data(),
        h_V.data(),
        h_I.data(),
        M, N, k
    );

    if (result != 0) {
        std::cerr << "FAIL: Metal kernel returned error: " << result << std::endl;
        return 1;
    }

    // Validation function matching CUDA test
    auto check_row = [&](int row, std::vector<int> exp_cols, std::vector<T> exp_vals) {
        for (int j = 0; j < (int)exp_cols.size(); ++j) {
            int idx = row * k + j;
            if (h_I[idx] != exp_cols[j] || std::fabs(h_V[idx] - exp_vals[j]) > 1e-6f) {
                std::cerr << "Row " << row << ": j=" << j << " got (" << h_I[idx] << ", " << h_V[idx]
                          << ") expected (" << exp_cols[j] << ", " << exp_vals[j] << ")\n";
                return false;
            }
        }
        return true;
    };

    if (!check_row(0, {1,3,6}, {0.1f,0.3f,0.6f})) return 1;
    if (!check_row(1, {0,4}, {1.0f,1.4f})) return 1; // Note: only 2 finite values in row 1
    if (!check_row(2, {2,5,6}, {2.2f,2.5f,2.6f})) return 1;

    std::cout << "PASS: test_metal_unit_extract_k_values" << std::endl;
    return 0;
}