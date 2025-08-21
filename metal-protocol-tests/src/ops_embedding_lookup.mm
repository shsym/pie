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

namespace ops {

void run_embedding_lookup_metal(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;
    using I = int32_t;

    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const int vocab_size = cfg.vocab_size;

    std::cout << "Running Metal Embedding: tokens=" << num_tokens << ", hidden=" << hidden_size << ", vocab=" << vocab_size << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_embedding(static_cast<size_t>(vocab_size) * hidden_size);
    for (auto& v : h_embedding) v = float_to_bf16(dist(rng));

    std::vector<I> h_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        h_indices[i] = static_cast<I>(i % vocab_size);
    }

    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    // Call Metal embedding implementation
    if (!initialize_metal_embedding()) {
        throw std::runtime_error("Failed to initialize Metal embedding");
    }
    metal_embedding_lookup_bfloat16(
        nil /*device*/, nil /*queue*/,
        h_embedding.data(), static_cast<size_t>(vocab_size),
        h_indices.data(), static_cast<size_t>(num_tokens),
        h_output.data(), hidden_size
    );

    // Write artifacts
    if (artifacts::op_enabled("embedding_lookup_forward")) {
        auto dir = artifacts::ensure_dir_for_case("embedding_lookup_forward", case_id + "_metal");

        artifacts::write_host_bin(dir, "embedding", h_embedding.data(), h_embedding.size());
        artifacts::write_host_bin(dir, "indices", h_indices.data(), h_indices.size());
        artifacts::write_host_bin(dir, "output", h_output.data(), h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"embedding_lookup_forward\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"hidden_size\": " << hidden_size << ", \"vocab_size\": " << vocab_size << ", \"num_tokens\": " << num_tokens << "},\n"
             << "\"dtype_map\": {\"embedding\": \"bf16\", \"indices\": \"s32\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"embedding\": [" << vocab_size << ", " << hidden_size << "], \"indices\": [" << num_tokens << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Embedding completed successfully" << std::endl;
}

} // namespace ops
