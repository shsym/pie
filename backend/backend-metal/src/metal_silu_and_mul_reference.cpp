#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include "workspace_utils.hpp"
#include <random>

// Metal reference implementation of the Metal SiLU and Mul kernel logic
// This allows testing the algorithm on Linux without actual Metal execution
// Implements the exact same logic that runs in metal_silu_and_mul.metal

using bfloat16_t = uint16_t;

// Helper to convert float to bfloat16 (matches CUDA backend conversion)
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

// SiLU activation function (matches CUDA implementation)
float silu_activation(float x) {
    return x / (1.0f + std::exp(-x));
}

// Metal reference implementation matching the metal_silu_and_mul.metal kernel
void metal_silu_and_mul_reference_bfloat16(
    const bfloat16_t* gate,
    const bfloat16_t* up,
    bfloat16_t* output,
    uint32_t num_tokens,
    uint32_t intermediate_size
) {
    for (uint32_t token_idx = 0; token_idx < num_tokens; token_idx++) {
        for (uint32_t dim_idx = 0; dim_idx < intermediate_size; dim_idx++) {
            uint32_t idx = token_idx * intermediate_size + dim_idx;

            float gate_val = bfloat16_to_float(gate[idx]);
            float up_val = bfloat16_to_float(up[idx]);

            // Apply SiLU activation to gate, then multiply by up
            float result = silu_activation(gate_val) * up_val;

            output[idx] = float_to_bfloat16(result);
        }
    }
}

// Load binary file into vector
template<typename T>
std::vector<T> load_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = size / sizeof(T);
    std::vector<T> data(num_elements);

    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Compare two vectors with tolerance
bool compare_vectors_bfloat16(const std::vector<bfloat16_t>& a, const std::vector<bfloat16_t>& b, float tolerance = 1e-3f) {
    if (a.size() != b.size()) {
        std::cerr << "Vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    size_t diff_count = 0;

    for (size_t i = 0; i < a.size(); i++) {
        float val_a = bfloat16_to_float(a[i]);
        float val_b = bfloat16_to_float(b[i]);

        float abs_diff = std::abs(val_a - val_b);
        float rel_diff = std::abs(val_a) > 1e-8f ? abs_diff / std::abs(val_a) : abs_diff;

        max_diff = std::max(max_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        // Use relative tolerance for larger values, absolute for smaller ones
        float effective_tolerance = std::max(tolerance, std::abs(val_a) * tolerance);
        if (abs_diff > effective_tolerance) {
            diff_count++;
            if (diff_count <= 10) {  // Show first 10 mismatches
                std::cout << "Mismatch at " << i << ": " << val_a << " vs " << val_b
                         << " (diff: " << abs_diff << ", rel: " << rel_diff << ")" << std::endl;
            }
        }
    }

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "Total mismatches: " << diff_count << " / " << a.size() << std::endl;

    return diff_count == 0;
}

int main() {
    std::cout << "=== Metal SiLU and Mul Reference Implementation Test ===" << std::endl;

    // Test against CUDA artifacts from metal-protocol-tests
    auto artifacts_base = workspace_utils::get_cuda_artifacts_dir();
    const std::string artifacts_dir = (artifacts_base / "silu_and_mul" / "production").string() + "/";

    std::cout << "Loading CUDA artifacts from: " << artifacts_dir << std::endl;

    // Load CUDA artifacts
    auto gate_data = load_binary_file<bfloat16_t>(artifacts_dir + "gate.bin");
    auto up_data = load_binary_file<bfloat16_t>(artifacts_dir + "up.bin");
    auto expected_output = load_binary_file<bfloat16_t>(artifacts_dir + "output.bin");

    if (gate_data.empty() || up_data.empty() || expected_output.empty()) {
        std::cerr << "Failed to load artifact files. Make sure to run CUDA tests first:" << std::endl;
        return 1;
    }

    std::cout << "Loaded gate: " << gate_data.size() << " elements" << std::endl;
    std::cout << "Loaded up: " << up_data.size() << " elements" << std::endl;
    std::cout << "Loaded expected output: " << expected_output.size() << " elements" << std::endl;

    // Expected dimensions from meta.json: 128 tokens × 11008 intermediate_size
    const uint32_t num_tokens = 128;
    const uint32_t intermediate_size = 11008;
    const uint32_t expected_size = num_tokens * intermediate_size;

    if (gate_data.size() != expected_size || up_data.size() != expected_size || expected_output.size() != expected_size) {
        std::cerr << "Size mismatch. Expected " << expected_size << " elements." << std::endl;
        return 1;
    }

    // Run Metal reference implementation
    std::vector<bfloat16_t> metal_output(expected_size);

    std::cout << "Running Metal reference implementation..." << std::endl;
    metal_silu_and_mul_reference_bfloat16(
        gate_data.data(),
        up_data.data(),
        metal_output.data(),
        num_tokens,
        intermediate_size
    );

    // Compare results
    std::cout << "Comparing Metal reference vs CUDA golden reference..." << std::endl;
    bool success = compare_vectors_bfloat16(metal_output, expected_output, 1e-3f);

    if (success) {
        std::cout << "✅ SUCCESS: Metal SiLU and Mul reference matches CUDA implementation!" << std::endl;
        std::cout << "The Metal kernel logic is mathematically correct." << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAILURE: Metal SiLU and Mul reference does not match CUDA implementation." << std::endl;
        std::cout << "Check the Metal kernel implementation." << std::endl;
        return 1;
    }
}