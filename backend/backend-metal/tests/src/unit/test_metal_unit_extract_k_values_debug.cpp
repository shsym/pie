#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include "metal_extract_k_values.hpp"

int main() {
    std::cout << "Starting extract_k_values debug test..." << std::endl;
    
    using T = float;
    const int M = 3;
    const int N = 8; 
    const int k = 3;

    std::vector<T> h_A(M * N, -INFINITY);
    
    // Row 0: finite at cols 1,3,6,7
    h_A[0 * N + 1] = 0.1f;
    h_A[0 * N + 3] = 0.3f;
    h_A[0 * N + 6] = 0.6f;
    h_A[0 * N + 7] = 0.7f;
    
    // Row 1: finite at cols 0,4  
    h_A[1 * N + 0] = 1.0f;
    h_A[1 * N + 4] = 1.4f;
    
    // Row 2: finite at cols 2,5,6
    h_A[2 * N + 2] = 2.2f;
    h_A[2 * N + 5] = 2.5f;
    h_A[2 * N + 6] = 2.6f;
    
    std::cout << "Test data initialized" << std::endl;

    std::vector<T> h_V(M * k, 999.0f); // Initialize to recognizable values
    std::vector<int32_t> h_I(M * k, -999);

    std::cout << "Calling Metal kernel..." << std::endl;
    int result = metal_extract_k_values_float32(
        h_A.data(),
        h_V.data(), 
        h_I.data(),
        M, N, k
    );

    std::cout << "Metal kernel returned: " << result << std::endl;
    
    if (result != 0) {
        std::cout << "FAIL: Metal kernel error" << std::endl;
        return 1;
    }

    // Print results
    for (int m = 0; m < M; m++) {
        std::cout << "Row " << m << ": ";
        for (int j = 0; j < k; j++) {
            int idx = m * k + j;
            std::cout << "(" << h_I[idx] << "," << h_V[idx] << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "DEBUG test completed" << std::endl;
    return 0;
}