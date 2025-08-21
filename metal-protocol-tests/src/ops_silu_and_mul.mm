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

namespace ops {

void run_silu_and_mul_metal(const std::string& case_id, const SiLUAndMulConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;

    const int num_tokens = cfg.num_tokens;
    const int intermediate_size = cfg.intermediate_size;

    std::cout << "Running Metal SiLU and Mul: tokens=" << num_tokens << ", intermediate=" << intermediate_size << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_gate(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_up(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * intermediate_size, 0);

    for (auto& v : h_gate) v = float_to_bf16(dist(rng));
    for (auto& v : h_up) v = float_to_bf16(dist(rng));

    // Call Metal SiLU and multiply implementation
    int result = metal_silu_and_mul_bfloat16(
        h_gate.data(), h_up.data(), h_output.data(),
        num_tokens, intermediate_size
    );

    if (result != 0) {
        throw std::runtime_error("Metal SiLU and multiply execution failed");
    }

    // Write artifacts
    if (artifacts::op_enabled("silu_and_mul")) {
        auto dir = artifacts::ensure_dir_for_case("silu_and_mul", case_id + "_metal");

        artifacts::write_host_bin(dir, "gate", h_gate.data(), h_gate.size());
        artifacts::write_host_bin(dir, "up", h_up.data(), h_up.size());
        artifacts::write_host_bin(dir, "output", h_output.data(), h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"silu_and_mul\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens << ", \"intermediate_size\": " << intermediate_size << "},\n"
             << "\"dtype_map\": {\"gate\": \"bf16\", \"up\": \"bf16\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"gate\": [" << num_tokens << ", " << intermediate_size << "], \"up\": [" << num_tokens << ", " << intermediate_size << "], \"output\": [" << num_tokens << ", " << intermediate_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal SiLU and Mul completed successfully" << std::endl;
}

} // namespace ops
