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

namespace ops {

void run_rms_norm_metal(const std::string& case_id, const RMSNormConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // bfloat16 on Metal host side

    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.eps;

    std::cout << "Running Metal RMS Norm: tokens=" << num_tokens << ", hidden=" << hidden_size
              << ", eps=" << eps << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Input tensor [num_tokens, hidden_size]
    std::vector<T> h_input(static_cast<size_t>(num_tokens) * hidden_size);
    for (auto& v : h_input) v = float_to_bf16(dist(rng));

    // Weight tensor [hidden_size]
    std::vector<T> h_weight(hidden_size);
    for (auto& v : h_weight) v = float_to_bf16(dist(rng));

    // Output tensor [num_tokens, hidden_size]
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    // Call Metal RMS norm implementation
    int result = metal_rmsnorm_bfloat16(
        h_input.data(), h_weight.data(), h_output.data(),
        num_tokens, hidden_size, eps
    );

    if (result != 0) {
        throw std::runtime_error("Metal RMS norm execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("rms_norm")) {
        auto dir = artifacts::ensure_dir_for_case("rms_norm", case_id + "_metal");

        artifacts::write_host_bin(dir, "input", h_input.data(), h_input.size());
        artifacts::write_host_bin(dir, "weight", h_weight.data(), h_weight.size());
        artifacts::write_host_bin(dir, "output", h_output.data(), h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rms_norm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"hidden_size\": " << hidden_size
             << ", \"eps\": " << eps << "},\n"
             << "\"dtype_map\": {\"input\": \"bf16\", \"weight\": \"bf16\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"input\": [" << num_tokens << ", " << hidden_size
             << "], \"weight\": [" << hidden_size
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal RMS Norm completed successfully" << std::endl;
}

} // namespace ops
