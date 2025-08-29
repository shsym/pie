#include <iostream>
#include <vector>
#include <cmath>
#include "metal_extract_k_values.hpp"

int main() {
    const int M = 2, N = 5, k = 2;
    
    // Create test matrix: [2,5] with -INFINITY and some values
    std::vector<float> A = {
        -INFINITY, 0.5f, -INFINITY, -1.2f, -INFINITY,  // Row 0: values at [1]=0.5, [3]=-1.2
        -INFINITY, -INFINITY, 0.8f, -INFINITY, 2.1f    // Row 1: values at [2]=0.8, [4]=2.1
    };
    
    std::vector<float> V(M * k, 0.0f);
    std::vector<int32_t> I(M * k, 0);
    
    std::cout << "Input matrix A (" << M << "x" << N << "):" << std::endl;
    for (int m = 0; m < M; ++m) {
        std::cout << "Row " << m << ": ";
        for (int n = 0; n < N; ++n) {
            float val = A[m * N + n];
            if (val == -INFINITY) {
                std::cout << "-INF ";
            } else {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }
    
    int result = metal_extract_k_values_float32(A.data(), V.data(), I.data(), M, N, k);
    
    if (result != 0) {
        std::cerr << "Metal extract k values failed with code: " << result << std::endl;
        return 1;
    }
    
    std::cout << "\nResults:" << std::endl;
    for (int m = 0; m < M; ++m) {
        std::cout << "Row " << m << " - Values: ";
        for (int i = 0; i < k; ++i) {
            std::cout << V[m * k + i] << " ";
        }
        std::cout << ", Indices: ";
        for (int i = 0; i < k; ++i) {
            std::cout << I[m * k + i] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}