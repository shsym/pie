#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <algorithm>

// Metal reference implementation of the Metal Embedding kernel logic  
// This allows testing the algorithm on Linux without actual Metal execution
// Implements the exact same logic that runs in metal_embedding.metal

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
void metal_embedding_lookup_reference(
    const bfloat16_t* embedding_matrix,
    size_t vocab_size,
    const int32_t* indices,
    size_t num_tokens,
    bfloat16_t* output,
    int hidden_size
) {
    // Process each token lookup
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const int32_t vocab_idx = indices[token_idx];
        
        // Bounds checking (matches Metal kernel behavior)
        if (vocab_idx < 0 || vocab_idx >= static_cast<int32_t>(vocab_size)) {
            // Invalid index - zero out the output
            for (int i = 0; i < hidden_size; ++i) {
                output[token_idx * hidden_size + i] = float_to_bfloat16(0.0f);
            }
            continue;
        }
        
        // Copy embedding vector: output[token_idx] = embedding_matrix[vocab_idx]
        const bfloat16_t* source_embedding = embedding_matrix + vocab_idx * hidden_size;
        bfloat16_t* dest_embedding = output + token_idx * hidden_size;
        
        for (int i = 0; i < hidden_size; ++i) {
            dest_embedding[i] = source_embedding[i];
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
bool compare_arrays(const std::vector<bfloat16_t>& a, const std::vector<bfloat16_t>& b, float tolerance = 1e-6f) {
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
    const std::string artifacts_dir = "../../metal-protocol-tests/tests/artifacts/embedding_lookup_forward/test1/";
    
    try {
        // Load CUDA reference data
        auto embedding_data = load_binary_file<bfloat16_t>(artifacts_dir + "embedding.bin");
        auto indices_data = load_binary_file<int32_t>(artifacts_dir + "indices.bin");  
        auto output_expected = load_binary_file<bfloat16_t>(artifacts_dir + "output.bin");
        
        // Test configuration from meta.json: vocab_size=32000, hidden_size=4096, num_tokens=128
        const size_t vocab_size = 32000;
        const int hidden_size = 4096;
        const size_t num_tokens = 128;
        
        std::cout << "Testing Metal reference Embedding with:" << std::endl;
        std::cout << "  vocab_size: " << vocab_size << std::endl;
        std::cout << "  hidden_size: " << hidden_size << std::endl;
        std::cout << "  num_tokens: " << num_tokens << std::endl;
        
        // Verify data sizes match expected dimensions
        const size_t expected_embedding_size = vocab_size * hidden_size;  // [32000, 4096]
        const size_t expected_indices_size = num_tokens;                   // [128]
        const size_t expected_output_size = num_tokens * hidden_size;      // [128, 4096]
        
        assert(embedding_data.size() == expected_embedding_size);
        assert(indices_data.size() == expected_indices_size);
        assert(output_expected.size() == expected_output_size);
        
        std::cout << "Input sizes verified: embedding=" << embedding_data.size() 
                  << ", indices=" << indices_data.size() 
                  << ", output=" << output_expected.size() << std::endl;
        
        // Allocate output buffer for Metal computation
        std::vector<bfloat16_t> output_metal(expected_output_size, 0);
        
        // Run Metal reference embedding lookup
        metal_embedding_lookup_reference(
            embedding_data.data(),
            vocab_size,
            indices_data.data(),
            num_tokens,
            output_metal.data(),
            hidden_size
        );
        
        std::cout << "Metal reference embedding lookup completed" << std::endl;
        
        // Compare results (embedding lookup should be exact - no numerical errors)
        bool match = compare_arrays(output_metal, output_expected, 1e-8f);
        
        if (match) {
            std::cout << "✅ Metal reference Embedding matches CUDA reference!" << std::endl;
        } else {
            std::cout << "❌ Metal reference Embedding does not match CUDA reference" << std::endl;
            
            // Show a few sample indices and their lookups for debugging
            std::cout << "\nSample comparison (first 5 tokens, first 5 dimensions each):" << std::endl;
            for (size_t token = 0; token < std::min(size_t(5), num_tokens); ++token) {
                int32_t vocab_idx = indices_data[token];
                std::cout << "  Token " << token << " (vocab_idx=" << vocab_idx << "):" << std::endl;
                
                for (size_t dim = 0; dim < std::min(size_t(5), size_t(hidden_size)); ++dim) {
                    size_t output_idx = token * hidden_size + dim;
                    float metal_val = bfloat16_to_float(output_metal[output_idx]);
                    float cuda_val = bfloat16_to_float(output_expected[output_idx]);
                    std::cout << "    [" << dim << "] Metal: " << metal_val << ", CUDA: " << cuda_val 
                             << ", diff: " << std::abs(metal_val - cuda_val) << std::endl;
                }
            }
        }
        
        return match;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load test artifacts: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Metal Reference Embedding Test vs CUDA Reference ===" << std::endl;
    
    bool success = test_metal_vs_cuda_artifacts();
    
    if (success) {
        std::cout << "All tests passed! ✅" << std::endl;
        return 0;
    } else {
        std::cout << "Tests failed! ❌" << std::endl;
        return 1;
    }
}