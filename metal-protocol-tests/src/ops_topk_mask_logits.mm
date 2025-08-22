// Separated compilation unit for Top-K Mask Logits Metal harness

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_topk_mask_logits.hpp"
#include "dtype_utils.hpp"
#include "metal_helpers.hpp"

namespace ops {

void run_topk_mask_logits_metal(const std::string& case_id, const TopKMaskConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int vocab_size = cfg.vocab_size;
    const int k = cfg.k;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("topk_mask_logits", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for topk_mask_logits/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Top-K Mask Logits: tokens=" << num_tokens << ", vocab=" << vocab_size
              << ", k=" << k << ", dtype=" << dtype_info.dtype_str << std::endl;

    const size_t logits_size = static_cast<size_t>(num_tokens) * vocab_size;
    std::vector<bfloat16_t> h_input_bf16(logits_size);
    std::vector<bfloat16_t> h_output_bf16(logits_size);

    // Load input data from CUDA reference (always load as float for simplicity)
    auto cuda_input = load_cuda_tensor<float>("topk_mask_logits", case_id, "input_logits");
    if (cuda_input.empty() || cuda_input.size() != logits_size) {
        std::cerr << "Warning: Could not load CUDA input data, size mismatch or file missing" << std::endl;
        return;
    }
    std::cout << "Loaded " << cuda_input.size() << " input values from CUDA reference" << std::endl;
    
    // Convert to bf16 for consistent host representation (will be converted back as needed)
    for (size_t i = 0; i < logits_size; ++i) {
        h_input_bf16[i] = float_to_bf16(cuda_input[i]);
    }

    h_output_bf16 = h_input_bf16;  // Start with copy since operation is in-place

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use fp32 kernel with native fp32 data (no precision loss)
        std::vector<float> h_logits_fp32 = cuda_input;  // Use original fp32 data directly

        result = metal_topk_mask_logits_float32(
            h_logits_fp32.data(), num_tokens, vocab_size, k
        );

        // Convert to bf16 for artifact writing consistency
        for (size_t i = 0; i < logits_size; ++i) {
            h_output_bf16[i] = float_to_bf16(h_logits_fp32[i]);
        }
    }
    else if (dtype_info.dtype == DType::FP16) {
        // TODO: Use fp16 kernel when available (metal_topk_mask_logits_float16)
        // For now, convert bf16 -> fp32 -> fp32 kernel -> bf16
        std::vector<float> h_logits_fp32(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            h_logits_fp32[i] = bf16_to_float(h_input_bf16[i]);
        }

        result = metal_topk_mask_logits_float32(
            h_logits_fp32.data(), num_tokens, vocab_size, k
        );

        // Convert back to bf16 for artifact writing consistency
        for (size_t i = 0; i < logits_size; ++i) {
            h_output_bf16[i] = float_to_bf16(h_logits_fp32[i]);
        }

        std::cout << "Note: Using fp32 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // BF16 path - use native bfloat16 kernel
        result = metal_topk_mask_logits_bfloat16(
            h_input_bf16.data(), num_tokens, vocab_size, k
        );

        // Output is already in bf16 format
        h_output_bf16 = h_input_bf16;  // bfloat16 kernel modifies in-place
    }

    if (result != 0) {
        throw std::runtime_error("Metal Top-K mask execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA - write in requested dtype
    if (artifacts::op_enabled("topk_mask_logits")) {
        auto dir = artifacts::ensure_dir_for_case("topk_mask_logits", case_id + "_metal");

        // Write artifacts in the requested dtype to match CUDA reference
        if (dtype_info.dtype == DType::FP32) {
            // For fp32, write original input and converted output
            std::vector<float> output_fp32(logits_size);
            for (size_t i = 0; i < logits_size; ++i) {
                output_fp32[i] = bf16_to_float(h_output_bf16[i]);
            }
            artifacts::write_vector_bin(dir, "input_logits", cuda_input);  // Use original fp32 input
            artifacts::write_vector_bin(dir, "masked_logits", output_fp32);
        }
        else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> input_fp16(logits_size), output_fp16(logits_size);
            for (size_t i = 0; i < logits_size; ++i) {
                input_fp16[i] = float_to_half(bf16_to_float(h_input_bf16[i]));
                output_fp16[i] = float_to_half(bf16_to_float(h_output_bf16[i]));
            }
            artifacts::write_vector_bin(dir, "input_logits", input_fp16);
            artifacts::write_vector_bin(dir, "masked_logits", output_fp16);
        }
        else {
            // Write as bf16
            artifacts::write_vector_bin(dir, "input_logits", h_input_bf16);
            artifacts::write_vector_bin(dir, "masked_logits", h_output_bf16);
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"topk_mask_logits\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"vocab_size\": " << vocab_size
             << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"" << dtype_info.dtype_str
             << "\", \"masked_logits\": \"" << dtype_info.dtype_str << "\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << num_tokens << ", " << vocab_size
             << "], \"masked_logits\": [" << num_tokens << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Top-K Mask Logits completed successfully" << std::endl;
}

} // namespace ops
