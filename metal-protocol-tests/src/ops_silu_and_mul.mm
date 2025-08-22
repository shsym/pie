// Per-op Metal wrapper for SiLU and Mul
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_silu_and_mul.hpp"
#include "dtype_utils.hpp"

namespace ops {

void run_silu_and_mul_metal(const std::string& case_id, const SiLUAndMulConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int intermediate_size = cfg.intermediate_size;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("silu_and_mul", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for silu_and_mul/" << case_id 
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal SiLU and Mul: tokens=" << num_tokens << ", intermediate=" << intermediate_size 
              << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Generate test data in bf16 (consistent host type)
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t tensor_size = static_cast<size_t>(num_tokens) * intermediate_size;
    std::vector<bfloat16_t> h_gate_bf16(tensor_size);
    std::vector<bfloat16_t> h_up_bf16(tensor_size);
    std::vector<bfloat16_t> h_output_bf16(tensor_size);

    for (auto& v : h_gate_bf16) v = float_to_bf16(dist(rng));
    for (auto& v : h_up_bf16) v = float_to_bf16(dist(rng));

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use native fp32 kernel with fp32 data
        std::vector<float> h_gate_fp32(tensor_size), h_up_fp32(tensor_size), h_output_fp32(tensor_size);
        
        // Generate test data in fp32
        for (auto& v : h_gate_fp32) v = dist(rng);
        for (auto& v : h_up_fp32) v = dist(rng);
        
        result = metal_silu_and_mul_float32(
            h_gate_fp32.data(), h_up_fp32.data(), h_output_fp32.data(),
            num_tokens, intermediate_size
        );
        
        // Convert fp32 results back to bf16 for consistent artifact writing
        for (size_t i = 0; i < tensor_size; ++i) {
            h_gate_bf16[i] = float_to_bf16(h_gate_fp32[i]);
            h_up_bf16[i] = float_to_bf16(h_up_fp32[i]);
            h_output_bf16[i] = float_to_bf16(h_output_fp32[i]);
        }
    }
    else if (dtype_info.dtype == DType::FP16) {
        // TODO: Use fp16 kernel when available (metal_silu_and_mul_float16)
        // For now, use bf16 kernel
        result = metal_silu_and_mul_bfloat16(
            h_gate_bf16.data(), h_up_bf16.data(), h_output_bf16.data(),
            num_tokens, intermediate_size
        );
        std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // Default bf16 path
        result = metal_silu_and_mul_bfloat16(
            h_gate_bf16.data(), h_up_bf16.data(), h_output_bf16.data(),
            num_tokens, intermediate_size
        );
    }

    if (result != 0) {
        throw std::runtime_error("Metal SiLU and multiply execution failed");
    }

    // Write artifacts in requested dtype
    if (artifacts::op_enabled("silu_and_mul")) {
        auto dir = artifacts::ensure_dir_for_case("silu_and_mul", case_id + "_metal");

        // Write artifacts in the requested dtype to match CUDA reference
        if (dtype_info.dtype == DType::FP32) {
            // Convert to fp32 for artifact writing
            std::vector<float> gate_fp32(tensor_size), up_fp32(tensor_size), output_fp32(tensor_size);
            for (size_t i = 0; i < tensor_size; ++i) {
                gate_fp32[i] = bf16_to_float(h_gate_bf16[i]);
                up_fp32[i] = bf16_to_float(h_up_bf16[i]);
                output_fp32[i] = bf16_to_float(h_output_bf16[i]);
            }
            artifacts::write_vector_bin(dir, "gate", gate_fp32);
            artifacts::write_vector_bin(dir, "up", up_fp32);
            artifacts::write_vector_bin(dir, "output", output_fp32);
        }
        else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> gate_fp16(tensor_size), up_fp16(tensor_size), output_fp16(tensor_size);
            for (size_t i = 0; i < tensor_size; ++i) {
                gate_fp16[i] = float_to_half(bf16_to_float(h_gate_bf16[i]));
                up_fp16[i] = float_to_half(bf16_to_float(h_up_bf16[i]));
                output_fp16[i] = float_to_half(bf16_to_float(h_output_bf16[i]));
            }
            artifacts::write_vector_bin(dir, "gate", gate_fp16);
            artifacts::write_vector_bin(dir, "up", up_fp16);
            artifacts::write_vector_bin(dir, "output", output_fp16);
        }
        else {
            // Write as bf16
            artifacts::write_vector_bin(dir, "gate", h_gate_bf16);
            artifacts::write_vector_bin(dir, "up", h_up_bf16);
            artifacts::write_vector_bin(dir, "output", h_output_bf16);
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"silu_and_mul\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens << ", \"intermediate_size\": " << intermediate_size << "},\n"
             << "\"dtype_map\": {\"gate\": \"" << dtype_info.dtype_str << "\", \"up\": \"" << dtype_info.dtype_str << "\", \"output\": \"" << dtype_info.dtype_str << "\"},\n"
             << "\"shape_map\": {\"gate\": [" << num_tokens << ", " << intermediate_size << "], \"up\": [" << num_tokens << ", " << intermediate_size << "], \"output\": [" << num_tokens << ", " << intermediate_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal SiLU and Mul completed successfully" << std::endl;
}

} // namespace ops
