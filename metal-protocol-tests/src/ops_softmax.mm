// Per-op Metal wrapper for Softmax
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_softmax.hpp"
#include "dtype_utils.hpp"
#include "metal_helpers.hpp"

namespace ops {

void run_softmax_metal(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed) {
    const int batch_size = cfg.batch_size;
    const int vocab_size = cfg.vocab_size;
    const float temperature = cfg.temperature;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("softmax", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for softmax/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Softmax: batch_size=" << batch_size << ", vocab_size=" << vocab_size
              << ", temperature=" << temperature << ", dtype=" << dtype_info.dtype_str << std::endl;
    std::cout << "  Expected total elements: " << batch_size * vocab_size << std::endl;

    const size_t logits_size = static_cast<size_t>(batch_size) * vocab_size;
    std::vector<bfloat16_t> h_input_bf16(logits_size);
    std::vector<bfloat16_t> h_output_bf16(logits_size, float_to_bf16(0.0f));

    // Generate test data in bf16 (consistent host type)
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (auto& v : h_input_bf16) v = float_to_bf16(dist(rng));

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use fp32 kernel (existing)
        std::vector<float> h_input_fp32(logits_size), h_output_fp32(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            h_input_fp32[i] = bf16_to_float(h_input_bf16[i]);
        }

        result = metal_softmax_float(
            h_input_fp32.data(), h_output_fp32.data(),
            batch_size, vocab_size, temperature
        );

        // Convert back to bf16 for consistent artifact storage
        for (size_t i = 0; i < logits_size; ++i) {
            h_output_bf16[i] = float_to_bf16(h_output_fp32[i]);
        }
    }
    else if (dtype_info.dtype == DType::FP16) {
        // TODO: Use fp16 kernel when available (metal_softmax_float16)
        // For now, convert bf16 -> fp32 -> fp32 kernel -> bf16
        std::vector<float> h_input_fp32(logits_size), h_output_fp32(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            h_input_fp32[i] = bf16_to_float(h_input_bf16[i]);
        }

        result = metal_softmax_float(
            h_input_fp32.data(), h_output_fp32.data(),
            batch_size, vocab_size, temperature
        );

        // Convert back to bf16 for consistent artifact storage
        for (size_t i = 0; i < logits_size; ++i) {
            h_output_bf16[i] = float_to_bf16(h_output_fp32[i]);
        }

        std::cout << "Note: Using fp32 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // Default bf16 path - convert to fp32 for existing kernel
        std::vector<float> h_input_fp32(logits_size), h_output_fp32(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            h_input_fp32[i] = bf16_to_float(h_input_bf16[i]);
        }

        result = metal_softmax_float(
            h_input_fp32.data(), h_output_fp32.data(),
            batch_size, vocab_size, temperature
        );

        // Convert back to bf16
        for (size_t i = 0; i < logits_size; ++i) {
            h_output_bf16[i] = float_to_bf16(h_output_fp32[i]);
        }
    }

    if (result != 0) {
        std::cerr << "Metal softmax failed with error: " << result << std::endl;
        return;
    }

    // Always write meta.json and input_logits.bin/output.bin in the Metal artifacts directory
    auto dir = artifacts::ensure_dir_for_case("softmax", case_id + "_metal");

    // Write artifacts in the requested dtype to match CUDA reference
    if (dtype_info.dtype == DType::FP32) {
        // Convert to fp32 for artifact writing
        std::vector<float> input_fp32(logits_size), output_fp32(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            input_fp32[i] = bf16_to_float(h_input_bf16[i]);
            output_fp32[i] = bf16_to_float(h_output_bf16[i]);
        }
        artifacts::write_vector_bin(dir, "input_logits", input_fp32);
        artifacts::write_vector_bin(dir, "output", output_fp32);
    }
    else if (dtype_info.dtype == DType::FP16) {
        // Convert to fp16 for artifact writing
        std::vector<uint16_t> input_fp16(logits_size), output_fp16(logits_size);
        for (size_t i = 0; i < logits_size; ++i) {
            input_fp16[i] = float_to_half(bf16_to_float(h_input_bf16[i]));
            output_fp16[i] = float_to_half(bf16_to_float(h_output_bf16[i]));
        }
        artifacts::write_vector_bin(dir, "input_logits", input_fp16);
        artifacts::write_vector_bin(dir, "output", output_fp16);
    }
    else {
        // Write as bf16
        artifacts::write_vector_bin(dir, "input_logits", h_input_bf16);
        artifacts::write_vector_bin(dir, "output", h_output_bf16);
    }

    std::ostringstream meta;
    meta << "\"version\": \"1\",\n"
         << "\"op\": \"softmax\",\n"
         << "\"backend\": \"metal\",\n"
         << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
         << "\"config\": {\"batch_size\": " << batch_size
         << ", \"vocab_size\": " << vocab_size
         << ", \"temperature\": " << temperature << "},\n"
         << "\"dtype_map\": {\"input_logits\": \"" << dtype_info.dtype_str
         << "\", \"output\": \"" << dtype_info.dtype_str << "\"},\n"
         << "\"shape_map\": {\"input_logits\": [" << batch_size << ", " << vocab_size
         << "], \"output\": [" << batch_size << ", " << vocab_size << "]}";
    artifacts::write_meta_json(dir, meta.str());

    // Debug: Check if softmax outputs sum to 1
    std::vector<float> debug_output_fp32(logits_size);
    for (size_t i = 0; i < logits_size; ++i) {
        debug_output_fp32[i] = bf16_to_float(h_output_bf16[i]);
    }

    float total_sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        float batch_sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            batch_sum += debug_output_fp32[b * vocab_size + v];
        }
        total_sum += batch_sum;
        std::cout << "  Batch " << b << " sum: " << batch_sum << std::endl;
    }

    std::cout << "Metal Softmax completed successfully" << std::endl;
}

} // namespace ops
