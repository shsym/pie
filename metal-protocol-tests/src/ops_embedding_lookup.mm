// Per-op Metal wrapper for Embedding Lookup
#import <Foundation/Foundation.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_embedding.hpp"
#include "dtype_utils.hpp"

namespace ops {

void run_embedding_lookup_metal(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const int vocab_size = cfg.vocab_size;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("embedding_lookup_forward", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for embedding_lookup_forward/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Embedding: tokens=" << num_tokens << ", hidden=" << hidden_size
              << ", vocab=" << vocab_size << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Generate test data; use full precision when targeting fp32
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t embedding_size = static_cast<size_t>(vocab_size) * hidden_size;
    const size_t output_size = static_cast<size_t>(num_tokens) * hidden_size;
    std::vector<int32_t> h_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) h_indices[i] = static_cast<int32_t>(i % vocab_size);

    // Allocate dtype-specific host buffers
    std::vector<bfloat16_t> h_embedding_bf16;
    std::vector<bfloat16_t> h_output_bf16;
    std::vector<float> h_embedding_f32;
    std::vector<float> h_output_f32;
    if (dtype_info.dtype == DType::FP32) {
        h_embedding_f32.resize(embedding_size);
        h_output_f32.resize(output_size);
        for (auto& v : h_embedding_f32) v = dist(rng);
    } else {
        h_embedding_bf16.resize(embedding_size);
        h_output_bf16.resize(output_size);
        for (auto& v : h_embedding_bf16) v = float_to_bf16(dist(rng));
    }

    // Initialize Metal embedding
    if (!initialize_metal_embedding()) {
        throw std::runtime_error("Failed to initialize Metal embedding");
    }

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Native float32 kernel path
        metal_embedding_lookup_float32(
            nil /*device*/, nil /*queue*/,
            h_embedding_f32.data(), static_cast<size_t>(vocab_size),
            h_indices.data(), static_cast<size_t>(num_tokens),
            h_output_f32.data(), hidden_size
        );
    }
    else if (dtype_info.dtype == DType::FP16) {
        // TODO: Use fp16 kernel when available (metal_embedding_lookup_float16)
        // For now, use bf16 kernel
        metal_embedding_lookup_bfloat16(
            nil /*device*/, nil /*queue*/,
            h_embedding_bf16.data(), static_cast<size_t>(vocab_size),
            h_indices.data(), static_cast<size_t>(num_tokens),
            h_output_bf16.data(), hidden_size
        );
        std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
    }
    else {
        // Default bf16 path
        metal_embedding_lookup_bfloat16(
            nil /*device*/, nil /*queue*/,
            h_embedding_bf16.data(), static_cast<size_t>(vocab_size),
            h_indices.data(), static_cast<size_t>(num_tokens),
            h_output_bf16.data(), hidden_size
        );
    }

    // Write artifacts in requested dtype
    if (artifacts::op_enabled("embedding_lookup_forward")) {
        auto dir = artifacts::ensure_dir_for_case("embedding_lookup_forward", case_id + "_metal");

        // Write artifacts in the requested dtype to match CUDA reference
        if (dtype_info.dtype == DType::FP32) {
            // Write native fp32 artifacts
            artifacts::write_vector_bin(dir, "embedding", h_embedding_f32);
            artifacts::write_vector_bin(dir, "indices", h_indices);
            artifacts::write_vector_bin(dir, "output", h_output_f32);
        }
        else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> embedding_fp16(embedding_size), output_fp16(output_size);
            for (size_t i = 0; i < embedding_size; ++i) {
                embedding_fp16[i] = float_to_half(bf16_to_float(h_embedding_bf16[i]));
            }
            for (size_t i = 0; i < output_size; ++i) {
                output_fp16[i] = float_to_half(bf16_to_float(h_output_bf16[i]));
            }
            artifacts::write_vector_bin(dir, "embedding", embedding_fp16);
            artifacts::write_vector_bin(dir, "indices", h_indices);
            artifacts::write_vector_bin(dir, "output", output_fp16);
        }
        else {
            // Write as bf16
            artifacts::write_vector_bin(dir, "embedding", h_embedding_bf16);
            artifacts::write_vector_bin(dir, "indices", h_indices);
            artifacts::write_vector_bin(dir, "output", h_output_bf16);
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"embedding_lookup_forward\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"hidden_size\": " << hidden_size << ", \"vocab_size\": " << vocab_size << ", \"num_tokens\": " << num_tokens << "},\n"
             << "\"dtype_map\": {\"embedding\": \"" << dtype_info.dtype_str << "\", \"indices\": \"s32\", \"output\": \"" << dtype_info.dtype_str << "\"},\n"
             << "\"shape_map\": {\"embedding\": [" << vocab_size << ", " << hidden_size << "], \"indices\": [" << num_tokens << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Embedding completed successfully" << std::endl;
}

} // namespace ops
