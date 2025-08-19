// Metal GPU implementations that call into backend/backend-metal kernels
// This allows metal-protocol-tests to test actual Metal GPU execution

#ifdef METAL_SUPPORT_ENABLED

#include "ops.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <random>
#include <cmath>

// Include Metal implementations from backend-metal
#include "metal_gemm.hpp"
#include "metal_embedding.hpp"
#include "metal_silu_and_mul.hpp"
#include "metal_extract_k_values.hpp"
#include "metal_softmax.hpp"
#include "metal_rmsnorm.hpp"

#include "artifacts.hpp"

namespace ops {

// Simple helpers to convert between float32 and bfloat16_t (uint16_t)
static inline bfloat16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest by adding 0x8000 before truncation
    return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
}

static inline float bf16_to_float(bfloat16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Metal implementation of GEMM operation
void run_gemm_metal(const std::string& case_id, const GemmConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // Metal host-side bfloat16

    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;

    std::cout << "Running Metal GEMM: m=" << m << ", n=" << n << ", k=" << k << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_A(static_cast<size_t>(m) * k);
    std::vector<T> h_B(static_cast<size_t>(k) * n);
    std::vector<T> h_C(static_cast<size_t>(m) * n, 0);

    for (auto& v : h_A) v = float_to_bf16(dist(rng));
    // Initialize B with layout matching transb flag and CUDA artifact convention
    // If transb=true, B is stored as [n, k] row-major (so op uses B^T)
    // If transb=false, B is stored as [k, n] row-major
    if (cfg.transb) {
        // rows = n, cols = k, index = row * k + col
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < k; ++col) {
                h_B[static_cast<size_t>(row) * k + col] = float_to_bf16(dist(rng));
            }
        }
    } else {
        // rows = k, cols = n, index = row * n + col
        for (int row = 0; row < k; ++row) {
            for (int col = 0; col < n; ++col) {
                h_B[static_cast<size_t>(row) * n + col] = float_to_bf16(dist(rng));
            }
        }
    }

    // Call Metal GEMM implementation
    // Initialize Metal GEMM once
    if (!initialize_metal_gemm()) {
        throw std::runtime_error("Failed to initialize Metal GEMM");
    }
    // Note: current backend-metal API ignores device/queue parameters and uses internal globals
    metal_gemm_bfloat16(
        nil /*device*/, nil /*queue*/,
        h_A.data(), h_B.data(), nullptr /*bias*/, h_C.data(),
        m, n, k,
        nullptr /*workspace*/, 0 /*workspace_size*/,
        cfg.transa, cfg.transb
    );

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("gemm")) {
        auto dir = artifacts::ensure_dir_for_case("gemm", case_id + "_metal");

        artifacts::write_host_bin(dir, "A", h_A.data(), h_A.size());
        artifacts::write_host_bin(dir, "B", h_B.data(), h_B.size());
        artifacts::write_host_bin(dir, "C", h_C.data(), h_C.size());

    std::ostringstream meta;
    const int A_dim0 = cfg.transa ? k : m;
    const int A_dim1 = cfg.transa ? m : k;
    const int B_dim0 = cfg.transb ? n : k;
    const int B_dim1 = cfg.transb ? k : n;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"gemm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (cfg.transa ? "true" : "false")
             << ", \"transb\": " << (cfg.transb ? "true" : "false")
             << ", \"use_bias\": " << (cfg.use_bias ? "true" : "false") << "},\n"
         << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"C\": \"bf16\"},\n"
         << "\"shape_map\": {\"A\": [" << A_dim0 << ", " << A_dim1 << "], \"B\": [" << B_dim0 << ", " << B_dim1 << "], \"C\": [" << m << ", " << n << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal GEMM completed successfully" << std::endl;
}

// Metal implementation of embedding lookup
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

// Metal implementation of SiLU and multiply
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

// Metal implementation of extract k values
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

// Metal implementation of Softmax operation
void run_softmax_metal(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed) {
    using T = float;  // Use float to match CUDA FlashInfer OnlineSoftmax

    const int batch_size = cfg.batch_size;
    const int vocab_size = cfg.vocab_size;
    const float temperature = cfg.temperature;

    std::cout << "Running Metal Softmax: batch_size=" << batch_size << ", vocab_size=" << vocab_size
              << ", temperature=" << temperature << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Reasonable logit range

    const size_t logits_size = static_cast<size_t>(batch_size) * vocab_size;
    std::vector<T> h_input_logits(logits_size);
    std::vector<T> h_output(logits_size, 0);

    for (auto& v : h_input_logits) v = dist(rng);

    // Call Metal softmax implementation
    int result = metal_softmax_float(
        h_input_logits.data(),
        h_output.data(),
        batch_size,
        vocab_size,
        temperature
    );

    if (result != 0) {
        std::cerr << "Metal softmax failed with error: " << result << std::endl;
        return;
    }

    // Generate artifacts for comparison with CUDA
    if (artifacts::op_enabled("softmax")) {
        auto dir = artifacts::ensure_dir_for_case("softmax", case_id + "_metal");

        artifacts::write_vector_bin(dir, "input_logits", h_input_logits);
        artifacts::write_vector_bin(dir, "output", h_output);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"softmax\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_metal") << ",\n"
             << "\"config\": {\"batch_size\": " << batch_size
             << ", \"vocab_size\": " << vocab_size
             << ", \"temperature\": " << temperature << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"output\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << batch_size << ", " << vocab_size
             << "], \"output\": [" << batch_size << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Softmax completed successfully" << std::endl;
}

// Metal implementation of RMS Normalization
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

#endif // METAL_SUPPORT_ENABLED