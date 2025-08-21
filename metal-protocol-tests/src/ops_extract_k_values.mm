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

namespace ops {

void run_extract_k_values_metal(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;

    const int M = cfg.M;
    const int N = cfg.N;
    const int k = cfg.k;

    std::cout << "Running Metal Extract K Values: M=" << M << ", N=" << N << ", k=" << k << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_A(static_cast<size_t>(M) * N);
    // Initialize all to negative infinity
    for (auto& v : h_A) v = float_to_bf16(-INFINITY);

    // For each row, place k values at deterministic positions
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < k; ++j) {
            int col = (m * 131 + j * 17) % N; // same hash as CUDA
            h_A[static_cast<size_t>(m) * N + col] = float_to_bf16(dist(rng));
        }
    }

    std::vector<T> h_V(static_cast<size_t>(M) * k, 0);
    std::vector<int32_t> h_I(static_cast<size_t>(M) * k, 0);

    // Call Metal extract k values implementation
    int result = metal_extract_k_values_bfloat16(
        h_A.data(), h_V.data(), h_I.data(),
        M, N, k
    );

    if (result != 0) {
        throw std::runtime_error("Metal extract k values execution failed");
    }

    // Write artifacts
    if (artifacts::op_enabled("extract_k_values")) {
        auto dir = artifacts::ensure_dir_for_case("extract_k_values", case_id + "_metal");

        artifacts::write_host_bin(dir, "A", h_A.data(), h_A.size());
        artifacts::write_host_bin(dir, "V", h_V.data(), h_V.size());
        artifacts::write_host_bin(dir, "I", h_I.data(), h_I.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"extract_k_values\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"M\": " << M << ", \"N\": " << N << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"A\": \"bf16\", \"V\": \"bf16\", \"I\": \"s32\"},\n"
             << "\"shape_map\": {\"A\": [" << M << ", " << N << "], \"V\": [" << M << ", " << k << "], \"I\": [" << M << ", " << k << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Extract K Values completed successfully" << std::endl;
}

} // namespace ops
