// Per-op Metal wrapper for Extract K Values
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <cmath>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_extract_k_values.hpp"
#include "dtype_utils.hpp"

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

    // Generate test data in bf16 (consistent host type)
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t A_size = static_cast<size_t>(M) * N;
    const size_t V_size = static_cast<size_t>(M) * k;
    const size_t I_size = static_cast<size_t>(M) * k;

    std::vector<bfloat16_t> h_A_bf16(A_size);
    std::vector<bfloat16_t> h_V_bf16(V_size);
    std::vector<int32_t> h_I(I_size);

    // Initialize all to negative infinity
    for (auto& v : h_A_bf16) v = float_to_bf16(-INFINITY);

    // For each row, place k values at deterministic positions
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < k; ++j) {
            int col = (m * 131 + j * 17) % N; // same hash as CUDA
            h_A_bf16[static_cast<size_t>(m) * N + col] = float_to_bf16(dist(rng));
        }
    }

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use native fp32 kernel with fp32 data
        std::vector<float> h_A_fp32(A_size), h_V_fp32(V_size);
        
        // Initialize all to negative infinity
        for (auto& v : h_A_fp32) v = -INFINITY;
        
        // For each row, place k values at deterministic positions
        for (int m = 0; m < M; ++m) {
            for (int j = 0; j < k; ++j) {
                int col = (m * 131 + j * 17) % N; // same hash as CUDA
                h_A_fp32[static_cast<size_t>(m) * N + col] = dist(rng);
            }
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
        // TODO: Use fp16 kernel when available (metal_extract_k_values_float16)
        // For now, use bf16 kernel
        result = metal_extract_k_values_bfloat16(
            h_A_bf16.data(), h_V_bf16.data(), h_I.data(),
            M, N, k
        );
        std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // Default bf16 path
        result = metal_extract_k_values_bfloat16(
            h_A_bf16.data(), h_V_bf16.data(), h_I.data(),
            M, N, k
        );
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
