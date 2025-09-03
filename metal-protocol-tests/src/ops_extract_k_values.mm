// Per-op Metal wrapper for Extract K Values
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <cmath>
#include <fstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_extract_k_values.hpp"
#include "dtype_utils.hpp"
#include "workspace_utils.hpp"

namespace ops {

void run_extract_k_values_metal(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed) {
    const int M = cfg.M;
    const int N = cfg.N;
    const int k = cfg.k;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("extract_k_values", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for extract_k_values/" << case_id 
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Extract K Values: M=" << M << ", N=" << N << ", k=" << k 
              << ", dtype=" << dtype_info.dtype_str << std::endl;

    const size_t A_size = static_cast<size_t>(M) * N;
    const size_t V_size = static_cast<size_t>(M) * k;
    const size_t I_size = static_cast<size_t>(M) * k;

    std::vector<bfloat16_t> h_A_bf16(A_size);
    std::vector<bfloat16_t> h_V_bf16(V_size);
    std::vector<int32_t> h_I(I_size);

    // Load input data from CUDA reference artifacts to ensure exact match
    auto cuda_artifacts_base = workspace_utils::get_cuda_artifacts_dir();
    std::string cuda_case_dir = (cuda_artifacts_base / "extract_k_values" / case_id).string();
    std::string input_path = cuda_case_dir + "/A.bin";
    
    std::ifstream cuda_file(input_path, std::ios::binary);
    if (cuda_file.is_open()) {
        if (dtype_info.dtype == DType::FP32) {
            // Read fp32 data and convert to bf16
            std::vector<float> cuda_fp32(A_size);
            cuda_file.read(reinterpret_cast<char*>(cuda_fp32.data()), A_size * sizeof(float));
            for (size_t i = 0; i < A_size; ++i) {
                h_A_bf16[i] = float_to_bf16(cuda_fp32[i]);
            }
        } else if (dtype_info.dtype == DType::FP16) {
            // Read fp16 data and convert to bf16
            std::vector<uint16_t> cuda_fp16(A_size);
            cuda_file.read(reinterpret_cast<char*>(cuda_fp16.data()), A_size * sizeof(uint16_t));
            for (size_t i = 0; i < A_size; ++i) {
                h_A_bf16[i] = float_to_bf16(half_to_float(cuda_fp16[i]));
            }
        } else {
            // Read bf16 data directly
            cuda_file.read(reinterpret_cast<char*>(h_A_bf16.data()), A_size * sizeof(bfloat16_t));
        }
        cuda_file.close();
        std::cout << "Loaded input matrix from CUDA reference: " << input_path << std::endl;
    } else {
        std::cerr << "Failed to load CUDA reference input, falling back to generated data" << std::endl;
        // Fallback to generated data
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Initialize all to negative infinity
        for (auto& v : h_A_bf16) v = float_to_bf16(-INFINITY);

        // For each row, place k values at deterministic positions
        for (int m = 0; m < M; ++m) {
            for (int j = 0; j < k; ++j) {
                int col = (m * 131 + j * 17) % N; // same hash as CUDA
                h_A_bf16[static_cast<size_t>(m) * N + col] = float_to_bf16(dist(rng));
            }
        }
    }

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Convert bf16 input to fp32 for native fp32 kernel
        std::vector<float> h_A_fp32(A_size), h_V_fp32(V_size);
        
        for (size_t i = 0; i < A_size; ++i) {
            h_A_fp32[i] = bf16_to_float(h_A_bf16[i]);
        }
        
        
        result = metal_extract_k_values_float32(
            h_A_fp32.data(), h_V_fp32.data(), h_I.data(),
            M, N, k
        );
        
        
        // Convert fp32 results back to bf16 for consistent artifact writing
        for (size_t i = 0; i < A_size; ++i) {
            h_A_bf16[i] = float_to_bf16(h_A_fp32[i]);
        }
        for (size_t i = 0; i < V_size; ++i) {
            h_V_bf16[i] = float_to_bf16(h_V_fp32[i]);
        }
    }
    else if (dtype_info.dtype == DType::FP16) {
        // Route fp16 through fp32 kernel due to lack of native Metal support
        std::vector<float> h_A_fp32_conv(A_size), h_V_fp32_conv(V_size);
        
        for (size_t i = 0; i < A_size; ++i) {
            h_A_fp32_conv[i] = bf16_to_float(h_A_bf16[i]);
        }
        
        result = metal_extract_k_values_float32(
            h_A_fp32_conv.data(), h_V_fp32_conv.data(), h_I.data(),
            M, N, k
        );
        
        // Convert fp32 results back to bf16
        for (size_t i = 0; i < V_size; ++i) {
            h_V_bf16[i] = float_to_bf16(h_V_fp32_conv[i]);
        }
        std::cout << "Note: Using fp32 kernel for fp16 request (no native Metal fp16 support)" << std::endl;
    }
    else {
        // Route bf16 through fp32 kernel due to lack of native Metal support  
        std::vector<float> h_A_fp32_conv(A_size), h_V_fp32_conv(V_size);
        
        for (size_t i = 0; i < A_size; ++i) {
            h_A_fp32_conv[i] = bf16_to_float(h_A_bf16[i]);
        }
        
        
        result = metal_extract_k_values_float32(
            h_A_fp32_conv.data(), h_V_fp32_conv.data(), h_I.data(),
            M, N, k
        );
        
        // Convert fp32 results back to bf16
        for (size_t i = 0; i < V_size; ++i) {
            h_V_bf16[i] = float_to_bf16(h_V_fp32_conv[i]);
        }
        std::cout << "Note: Using fp32 kernel for bf16 request (no native Metal bf16 support)" << std::endl;
    }

    if (result != 0) {
        throw std::runtime_error("Metal extract k values execution failed");
    }

    // Write artifacts in requested dtype
    if (artifacts::op_enabled("extract_k_values")) {
        auto dir = artifacts::ensure_dir_for_case("extract_k_values", case_id + "_metal");

        // Write artifacts in the requested dtype to match CUDA reference
        if (dtype_info.dtype == DType::FP32) {
            // Convert to fp32 for artifact writing
            std::vector<float> A_fp32(A_size), V_fp32(V_size);
            for (size_t i = 0; i < A_size; ++i) {
                A_fp32[i] = bf16_to_float(h_A_bf16[i]);
            }
            for (size_t i = 0; i < V_size; ++i) {
                V_fp32[i] = bf16_to_float(h_V_bf16[i]);
            }
            artifacts::write_vector_bin(dir, "A", A_fp32);
            artifacts::write_vector_bin(dir, "V", V_fp32);
            artifacts::write_vector_bin(dir, "I", h_I);
        }
        else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> A_fp16(A_size), V_fp16(V_size);
            for (size_t i = 0; i < A_size; ++i) {
                A_fp16[i] = float_to_half(bf16_to_float(h_A_bf16[i]));
            }
            for (size_t i = 0; i < V_size; ++i) {
                V_fp16[i] = float_to_half(bf16_to_float(h_V_bf16[i]));
            }
            artifacts::write_vector_bin(dir, "A", A_fp16);
            artifacts::write_vector_bin(dir, "V", V_fp16);
            artifacts::write_vector_bin(dir, "I", h_I);
        }
        else {
            // Write as bf16
            artifacts::write_vector_bin(dir, "A", h_A_bf16);
            artifacts::write_vector_bin(dir, "V", h_V_bf16);
            artifacts::write_vector_bin(dir, "I", h_I);
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"extract_k_values\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"M\": " << M << ", \"N\": " << N << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"A\": \"" << dtype_info.dtype_str << "\", \"V\": \"" << dtype_info.dtype_str << "\", \"I\": \"s32\"},\n"
             << "\"shape_map\": {\"A\": [" << M << ", " << N << "], \"V\": [" << M << ", " << k << "], \"I\": [" << M << ", " << k << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Extract K Values completed successfully" << std::endl;
}

} // namespace ops
