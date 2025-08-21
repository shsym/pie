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

namespace ops {

void run_topk_mask_logits_metal(const std::string& case_id, const TopKMaskConfig& cfg, uint64_t seed) {
    using T = float;  // Use float for logits (common for sampling operations)

    const int num_tokens = cfg.num_tokens;
    const int vocab_size = cfg.vocab_size;
    const int k = cfg.k;

    std::cout << "Running Metal Top-K Mask Logits: tokens=" << num_tokens << ", vocab=" << vocab_size
              << ", k=" << k << std::endl;

    // Try to load CUDA reference input logits for apples-to-apples comparison
    std::vector<T> h_logits(static_cast<size_t>(num_tokens) * vocab_size);
    std::vector<T> h_original_logits;
    {
        std::filesystem::path cuda_base_dir;
        if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
            cuda_base_dir = std::filesystem::path(envp);
        } else {
            std::filesystem::path this_file(__FILE__);
            cuda_base_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
        }
        auto cuda_case_dir = cuda_base_dir / "topk_mask_logits" / case_id;
        auto p = cuda_case_dir / "input_logits.bin";
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            std::ifstream ifs(p, std::ios::binary);
            ifs.seekg(0, std::ios::end);
            size_t bytes = static_cast<size_t>(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            h_original_logits.resize(bytes / sizeof(float));
            if (bytes) ifs.read(reinterpret_cast<char*>(h_original_logits.data()), bytes);
            h_logits = h_original_logits;
        } else {
            // Generate same test data as CUDA version expectations
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
            for (auto& v : h_logits) v = dist(rng);
            h_original_logits = h_logits;
        }
    }

    // Call Metal Top-K mask implementation (in-place operation)
    int result = metal_topk_mask_logits_float32(
        h_logits.data(), num_tokens, vocab_size, k
    );

    if (result != 0) {
        throw std::runtime_error("Metal Top-K mask execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("topk_mask_logits")) {
        auto dir = artifacts::ensure_dir_for_case("topk_mask_logits", case_id + "_metal");

        artifacts::write_host_bin(dir, "input_logits", h_original_logits.data(), h_original_logits.size());
        // Match CUDA naming: masked logits after operation
        artifacts::write_host_bin(dir, "masked_logits", h_logits.data(), h_logits.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"topk_mask_logits\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"vocab_size\": " << vocab_size
             << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"masked_logits\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << num_tokens << ", " << vocab_size
             << "], \"masked_logits\": [" << num_tokens << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Top-K Mask Logits completed successfully" << std::endl;
}

} // namespace ops
