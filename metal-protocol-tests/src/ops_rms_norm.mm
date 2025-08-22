// Per-op Metal wrapper for RMS Normalization
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_rmsnorm.hpp"
#include "dtype_utils.hpp"

namespace ops {

void run_rms_norm_metal(const std::string& case_id, const RMSNormConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.eps;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("rms_norm", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for rms_norm/" << case_id 
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal RMS Norm: tokens=" << num_tokens << ", hidden=" << hidden_size
              << ", eps=" << eps << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Generate test data in bf16 (consistent host type)
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t input_size = static_cast<size_t>(num_tokens) * hidden_size;
    const size_t weight_size = hidden_size;

    // Input tensor [num_tokens, hidden_size]
    std::vector<bfloat16_t> h_input_bf16(input_size);
    for (auto& v : h_input_bf16) v = float_to_bf16(dist(rng));

    // Weight tensor [hidden_size]
    std::vector<bfloat16_t> h_weight_bf16(weight_size);
    for (auto& v : h_weight_bf16) v = float_to_bf16(dist(rng));

    // Output tensor [num_tokens, hidden_size]
    std::vector<bfloat16_t> h_output_bf16(input_size);

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use native fp32 kernel with fp32 data
        std::vector<float> h_input_fp32(input_size), h_weight_fp32(weight_size), h_output_fp32(input_size);
        
        // Generate test data in fp32
        for (auto& v : h_input_fp32) v = dist(rng);
        for (auto& v : h_weight_fp32) v = dist(rng);
        
        result = metal_rmsnorm_float32(
            h_input_fp32.data(), h_weight_fp32.data(), h_output_fp32.data(),
            num_tokens, hidden_size, eps
        );
        
        // Convert fp32 results back to bf16 for consistent artifact writing
        for (size_t i = 0; i < input_size; ++i) {
            h_input_bf16[i] = float_to_bf16(h_input_fp32[i]);
            h_output_bf16[i] = float_to_bf16(h_output_fp32[i]);
        }
        for (size_t i = 0; i < weight_size; ++i) {
            h_weight_bf16[i] = float_to_bf16(h_weight_fp32[i]);
        }
    }
    else if (dtype_info.dtype == DType::FP16) {
        // TODO: Use fp16 kernel when available (metal_rmsnorm_float16)
        // For now, use bf16 kernel
        result = metal_rmsnorm_bfloat16(
            h_input_bf16.data(), h_weight_bf16.data(), h_output_bf16.data(),
            num_tokens, hidden_size, eps
        );
        std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // Default bf16 path
        result = metal_rmsnorm_bfloat16(
            h_input_bf16.data(), h_weight_bf16.data(), h_output_bf16.data(),
            num_tokens, hidden_size, eps
        );
    }

    if (result != 0) {
        throw std::runtime_error("Metal RMS norm execution failed with code: " + std::to_string(result));
    }

    // Write artifacts in requested dtype
    if (artifacts::op_enabled("rms_norm")) {
        auto dir = artifacts::ensure_dir_for_case("rms_norm", case_id + "_metal");

        // Write artifacts in the requested dtype to match CUDA reference
        if (dtype_info.dtype == DType::FP32) {
            // Convert to fp32 for artifact writing
            std::vector<float> input_fp32(input_size), weight_fp32(weight_size), output_fp32(input_size);
            for (size_t i = 0; i < input_size; ++i) {
                input_fp32[i] = bf16_to_float(h_input_bf16[i]);
                output_fp32[i] = bf16_to_float(h_output_bf16[i]);
            }
            for (size_t i = 0; i < weight_size; ++i) {
                weight_fp32[i] = bf16_to_float(h_weight_bf16[i]);
            }
            artifacts::write_vector_bin(dir, "input", input_fp32);
            artifacts::write_vector_bin(dir, "weight", weight_fp32);
            artifacts::write_vector_bin(dir, "output", output_fp32);
        }
        else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> input_fp16(input_size), weight_fp16(weight_size), output_fp16(input_size);
            for (size_t i = 0; i < input_size; ++i) {
                input_fp16[i] = float_to_half(bf16_to_float(h_input_bf16[i]));
                output_fp16[i] = float_to_half(bf16_to_float(h_output_bf16[i]));
            }
            for (size_t i = 0; i < weight_size; ++i) {
                weight_fp16[i] = float_to_half(bf16_to_float(h_weight_bf16[i]));
            }
            artifacts::write_vector_bin(dir, "input", input_fp16);
            artifacts::write_vector_bin(dir, "weight", weight_fp16);
            artifacts::write_vector_bin(dir, "output", output_fp16);
        }
        else {
            // Write as bf16
            artifacts::write_vector_bin(dir, "input", h_input_bf16);
            artifacts::write_vector_bin(dir, "weight", h_weight_bf16);
            artifacts::write_vector_bin(dir, "output", h_output_bf16);
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rms_norm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"hidden_size\": " << hidden_size
             << ", \"eps\": " << eps << "},\n"
             << "\"dtype_map\": {\"input\": \"" << dtype_info.dtype_str << "\", \"weight\": \"" << dtype_info.dtype_str << "\", \"output\": \"" << dtype_info.dtype_str << "\"},\n"
             << "\"shape_map\": {\"input\": [" << num_tokens << ", " << hidden_size
             << "], \"weight\": [" << hidden_size
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal RMS Norm completed successfully" << std::endl;
}

} // namespace ops
