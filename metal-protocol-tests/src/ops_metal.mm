// Metal GPU implementations that call into backend/backend-metal kernels
// This allows metal-protocol-tests to test actual Metal GPU execution

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_gemm.hpp"
#include "metal_embedding.hpp"
#include "metal_silu_and_mul.hpp"
#include "metal_extract_k_values.hpp"
#include "metal_softmax.hpp"
#include "metal_rmsnorm.hpp"
#include "metal_rope.hpp"
#include "metal_topk_mask_logits.hpp"
#include "metal_grouped_gemm.hpp"
#include "metal_batch_prefill_attention.hpp"


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

// Minimal IEEE fp16 conversion helpers
static inline uint16_t float_to_half(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    if (f == 0.0f) return u.i >> 16; // preserve sign of zero
    if (!std::isfinite(f)) {
        if (std::isnan(f)) return 0x7e00; // qNaN
        return (u.i >> 16) | 0x7c00; // inf with sign
    }
    uint32_t sign = (u.i >> 16) & 0x8000;
    int32_t exp = ((u.i >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = (u.i >> 13) & 0x3ff;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
}
static inline float half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u) >> 10;
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = sign;
        } else {
            int shift = 0;
            while ((h_sig & 0x0400u) == 0) { h_sig <<= 1; ++shift; }
            h_sig &= 0x03FFu;
            uint32_t exp = 127 - 15 - shift;
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        }
    } else if (h_exp == 0x1Fu) {
        uint32_t exp = 0xFFu;
        uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
        f = sign | (exp << 23) | mant;
    } else {
        uint32_t exp = static_cast<uint32_t>(h_exp) - 15 + 127;
        uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
        f = sign | (exp << 23) | mant;
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

template<typename T>
void print_vec_stats(const std::string& name, const std::vector<T>& vec) {
    if (vec.empty()) {
        std::cout << "📊 " << name << " is empty." << std::endl;
        return;
    }
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum_val = 0.0;
    size_t non_zero_count = 0;
    for (const auto& val_bf16 : vec) {
        float val = bf16_to_float(val_bf16);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
        if (std::abs(val) > 1e-9f) non_zero_count++;
    }
    std::cout << "📊 " << name << " range: [" << min_val << ", " << max_val
              << "], avg=" << sum_val / vec.size()
              << ", non_zero=" << non_zero_count << "/" << vec.size()
              << " (" << (100.0 * non_zero_count / vec.size()) << "%)" << std::endl;
}

template<typename T>
void print_vec_stats(const std::string& name, const std::vector<T>& vec, size_t n) {
    if (vec.empty()) {
        std::cout << "📊 " << name << " is empty." << std::endl;
        return;
    }
    size_t count = std::min(n, vec.size());
    if (count == 0) {
        print_vec_stats(name, vec);
        return;
    }
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum_val = 0.0;
    size_t non_zero_count = 0;
    for (size_t i = 0; i < count; ++i) {
        float val = bf16_to_float(vec[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
        if (std::abs(val) > 1e-9f) non_zero_count++;
    }
    std::cout << "📊 " << name << " range (first " << count << "): [" << min_val << ", " << max_val
              << "], avg=" << sum_val / count
              << ", non_zero=" << non_zero_count << "/" << count
              << ", total_size=" << vec.size() << std::endl;
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

// Metal implementation of RoPE (Rotary Position Embedding)
void run_rope_metal(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // bfloat16 on Metal host side

    const int num_tokens = cfg.num_tokens;
    const int num_q_heads = cfg.num_query_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;
    const float rope_theta = cfg.rope_theta;
    const float rope_factor = cfg.rope_factor;

    std::cout << "Running Metal RoPE: tokens=" << num_tokens
              << ", q_heads=" << num_q_heads
              << ", kv_heads=" << num_kv_heads
              << ", head_size=" << head_size
              << ", theta=" << rope_theta
              << ", factor=" << rope_factor << std::endl;

    // Generate same test data as CUDA version would use
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t q_elems = static_cast<size_t>(num_tokens) * num_q_heads * head_size;
    const size_t k_elems = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;

    // Query and Key tensors [num_tokens, num_heads, head_size]
    std::vector<T> h_q(q_elems);
    std::vector<T> h_k(k_elems);
    for (auto& v : h_q) v = float_to_bf16(dist(rng));
    for (auto& v : h_k) v = float_to_bf16(dist(rng));

    // Save originals for artifact q_input/k_input
    std::vector<T> h_q_input = h_q;
    std::vector<T> h_k_input = h_k;

    // Position IDs [num_tokens] - sequential positions starting from 0
    std::vector<int32_t> h_position_ids(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        h_position_ids[i] = i;
    }

    // Select kernel precision based on CUDA reference dtype (meta.json)
    bool use_fp16_kernel = false; // default to fp32
    try {
        std::filesystem::path cuda_base_dir;
        if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
            cuda_base_dir = std::filesystem::path(envp);
        } else {
            std::filesystem::path this_file = __FILE__;
            auto tests_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
            cuda_base_dir = tests_dir;
        }
        std::filesystem::path meta_path = cuda_base_dir / "rope" / case_id / "meta.json";
        if (std::filesystem::exists(meta_path)) {
            std::ifstream fin(meta_path);
            std::string meta((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            if (meta.find("\"bf16\"") != std::string::npos) use_fp16_kernel = true;
            if (meta.find("\"float32\"") != std::string::npos) use_fp16_kernel = false;
            if (meta.find("\"fp32\"") != std::string::npos) use_fp16_kernel = false;
            if (meta.find("\"fp16\"") != std::string::npos) use_fp16_kernel = true; // be generous
        }
    } catch (...) { /* ignore */ }

    if (use_fp16_kernel) {
        // CUDA ref is bf16: use fp16 kernels through host conversion bf16 -> fp32 -> fp16
        std::vector<uint16_t> hq_fp16(q_elems), hk_fp16(k_elems);
        for (size_t i = 0; i < q_elems; ++i) {
            float v = bf16_to_float(h_q[i]);
            hq_fp16[i] = float_to_half(v);
        }
        for (size_t i = 0; i < k_elems; ++i) {
            float v = bf16_to_float(h_k[i]);
            hk_fp16[i] = float_to_half(v);
        }

        int res_q = metal_rope_float16(
            hq_fp16.data(), h_position_ids.data(),
            num_tokens, num_q_heads, head_size,
            rope_theta, rope_factor
        );
        if (res_q != 0) {
            throw std::runtime_error("Metal RoPE(Q fp16) failed with code: " + std::to_string(res_q));
        }
        int res_k = metal_rope_float16(
            hk_fp16.data(), h_position_ids.data(),
            num_tokens, num_kv_heads, head_size,
            rope_theta, rope_factor
        );
        if (res_k != 0) {
            throw std::runtime_error("Metal RoPE(K fp16) failed with code: " + std::to_string(res_k));
        }
        for (size_t i = 0; i < q_elems; ++i) h_q[i] = float_to_bf16(half_to_float(hq_fp16[i]));
        for (size_t i = 0; i < k_elems; ++i) h_k[i] = float_to_bf16(half_to_float(hk_fp16[i]));
    } else {
        // CUDA ref is fp32: use fp32 kernels
        std::vector<float> fq(q_elems), fk(k_elems);
        for (size_t i = 0; i < q_elems; ++i) fq[i] = bf16_to_float(h_q[i]);
        for (size_t i = 0; i < k_elems; ++i) fk[i] = bf16_to_float(h_k[i]);
        int res_q = metal_rope_float32(
            fq.data(), h_position_ids.data(),
            num_tokens, num_q_heads, head_size,
            rope_theta, rope_factor
        );
        if (res_q != 0) {
            throw std::runtime_error("Metal RoPE(Q fp32) failed with code: " + std::to_string(res_q));
        }
        int res_k = metal_rope_float32(
            fk.data(), h_position_ids.data(),
            num_tokens, num_kv_heads, head_size,
            rope_theta, rope_factor
        );
        if (res_k != 0) {
            throw std::runtime_error("Metal RoPE(K fp32) failed with code: " + std::to_string(res_k));
        }
        for (size_t i = 0; i < q_elems; ++i) h_q[i] = float_to_bf16(fq[i]);
        for (size_t i = 0; i < k_elems; ++i) h_k[i] = float_to_bf16(fk[i]);
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("rope")) {
        auto dir = artifacts::ensure_dir_for_case("rope", case_id + "_metal");

        // Inputs (pre-RoPE)
        artifacts::write_host_bin(dir, "q_input", h_q_input.data(), h_q_input.size());
        artifacts::write_host_bin(dir, "k_input", h_k_input.data(), h_k_input.size());
        artifacts::write_host_bin(dir, "pos_ids", h_position_ids.data(), h_position_ids.size());

        // Outputs (post-RoPE)
        artifacts::write_host_bin(dir, "q_output", h_q.data(), h_q.size());
        artifacts::write_host_bin(dir, "k_output", h_k.data(), h_k.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rope\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_query_heads\": " << num_q_heads
             << ", \"num_kv_heads\": " << num_kv_heads
             << ", \"head_size\": " << head_size
             << ", \"rope_theta\": " << rope_theta
             << ", \"rope_factor\": " << rope_factor
             << ", \"rope_low_frequency_factor\": " << cfg.rope_low_frequency_factor
             << ", \"rope_high_frequency_factor\": " << cfg.rope_high_frequency_factor
             << ", \"max_position_embeddings\": " << cfg.max_position_embeddings << "},\n"
             << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"pos_ids\": \"s32\", \"q_output\": \"bf16\", \"k_output\": \"bf16\"},\n"
             << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_q_heads * head_size)
             << "], \"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
             << "], \"pos_ids\": [" << num_tokens
             << "], \"q_output\": [" << num_tokens << ", " << (num_q_heads * head_size)
             << "], \"k_output\": [" << num_tokens << ", " << (num_kv_heads * head_size) << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal RoPE completed successfully" << std::endl;
}

// Metal implementation of Top-K Mask Logits
void run_topk_mask_logits_metal(const std::string& case_id, const TopKMaskConfig& cfg, uint64_t seed) {
    using T = float;  // Use float for logits (common for sampling operations)

    const int num_tokens = cfg.num_tokens;
    const int vocab_size = cfg.vocab_size;
    const int k = cfg.k;

    std::cout << "Running Metal Top-K Mask Logits: tokens=" << num_tokens << ", vocab=" << vocab_size
              << ", k=" << k << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Reasonable logit range

    // Input logits [num_tokens, vocab_size]
    std::vector<T> h_logits(static_cast<size_t>(num_tokens) * vocab_size);
    for (auto& v : h_logits) v = dist(rng);

    // Make a copy for comparison (since operation is in-place)
    std::vector<T> h_original_logits = h_logits;

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
        artifacts::write_host_bin(dir, "output_logits", h_logits.data(), h_logits.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"topk_mask_logits\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"vocab_size\": " << vocab_size
             << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"output_logits\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << num_tokens << ", " << vocab_size
             << "], \"output_logits\": [" << num_tokens << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Top-K Mask Logits completed successfully" << std::endl;
}

// Metal implementation of Grouped GEMM
void run_grouped_gemm_metal(const std::string& case_id, const GroupedGemmConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // bfloat16 on Metal host side

    const int num_groups = cfg.num_groups;
    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    std::cout << "Running Metal Grouped GEMM: groups=" << num_groups << ", m=" << m << ", n=" << n
              << ", k=" << k << ", transa=" << transa << ", transb=" << transb
              << ", use_bias=" << use_bias << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create arrays to hold matrix data for each group
    std::vector<std::vector<T>> A_matrices(num_groups);
    std::vector<std::vector<T>> B_matrices(num_groups);
    std::vector<std::vector<T>> C_matrices(num_groups);
    std::vector<std::vector<T>> bias_matrices(num_groups);

    std::vector<void*> A_ptrs(num_groups);
    std::vector<void*> B_ptrs(num_groups);
    std::vector<void*> C_ptrs(num_groups);
    std::vector<void*> bias_ptrs(num_groups);

    std::vector<int> m_array(num_groups, m);
    std::vector<int> n_array(num_groups, n);
    std::vector<int> k_array(num_groups, k);

    // Initialize matrices for each group
    for (int group = 0; group < num_groups; ++group) {
        // A matrix size depends on transpose flag
        size_t A_size = static_cast<size_t>(transa ? k * m : m * k);
        A_matrices[group].resize(A_size);
        for (auto& v : A_matrices[group]) v = float_to_bf16(dist(rng));
        A_ptrs[group] = A_matrices[group].data();

        // B matrix size depends on transpose flag
        size_t B_size = static_cast<size_t>(transb ? n * k : k * n);
        B_matrices[group].resize(B_size);
        for (auto& v : B_matrices[group]) v = float_to_bf16(dist(rng));
        B_ptrs[group] = B_matrices[group].data();

        // C matrix (output)
        size_t C_size = static_cast<size_t>(m) * n;
        C_matrices[group].resize(C_size, 0);
        C_ptrs[group] = C_matrices[group].data();

        // Bias vector (optional)
        if (use_bias) {
            bias_matrices[group].resize(n);
            for (auto& v : bias_matrices[group]) v = float_to_bf16(dist(rng));
            bias_ptrs[group] = bias_matrices[group].data();
        } else {
            bias_ptrs[group] = nullptr;
        }
    }

    // Call Metal Grouped GEMM implementation
    int result = metal_grouped_gemm_bfloat16(
        A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
        use_bias ? bias_ptrs.data() : nullptr,
        m_array.data(), n_array.data(), k_array.data(),
        num_groups, transa, transb
    );

    if (result != 0) {
        throw std::runtime_error("Metal Grouped GEMM execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("grouped_gemm")) {
        auto dir = artifacts::ensure_dir_for_case("grouped_gemm", case_id + "_metal");

        // Save each group's matrices
        for (int group = 0; group < num_groups; ++group) {
            std::string group_suffix = "_group" + std::to_string(group);

            artifacts::write_host_bin(dir, "A" + group_suffix, A_matrices[group].data(), A_matrices[group].size());
            artifacts::write_host_bin(dir, "B" + group_suffix, B_matrices[group].data(), B_matrices[group].size());
            artifacts::write_host_bin(dir, "C" + group_suffix, C_matrices[group].data(), C_matrices[group].size());

            if (use_bias) {
                artifacts::write_host_bin(dir, "bias" + group_suffix, bias_matrices[group].data(), bias_matrices[group].size());
            }
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"grouped_gemm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_groups\": " << num_groups
             << ", \"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (transa ? "true" : "false")
             << ", \"transb\": " << (transb ? "true" : "false")
             << ", \"use_bias\": " << (use_bias ? "true" : "false") << "},\n"
             << "\"dtype_map\": {";

        for (int group = 0; group < num_groups; ++group) {
            if (group > 0) meta << ", ";
            std::string suffix = "_group" + std::to_string(group);
            meta << "\"A" << suffix << "\": \"bf16\", \"B" << suffix << "\": \"bf16\", \"C" << suffix << "\": \"bf16\"";
            if (use_bias) {
                meta << ", \"bias" << suffix << "\": \"bf16\"";
            }
        }

        meta << "},\n\"shape_map\": {";

        for (int group = 0; group < num_groups; ++group) {
            if (group > 0) meta << ", ";
            std::string suffix = "_group" + std::to_string(group);

            int A_dim0 = transa ? k : m;
            int A_dim1 = transa ? m : k;
            int B_dim0 = transb ? n : k;
            int B_dim1 = transb ? k : n;

            meta << "\"A" << suffix << "\": [" << A_dim0 << ", " << A_dim1 << "], "
                 << "\"B" << suffix << "\": [" << B_dim0 << ", " << B_dim1 << "], "
                 << "\"C" << suffix << "\": [" << m << ", " << n << "]";

            if (use_bias) {
                meta << ", \"bias" << suffix << "\": [" << n << "]";
            }
        }

        meta << "}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Grouped GEMM completed successfully" << std::endl;
}

// Metal implementation of Batch Prefill Attention operation
void run_batch_prefill_attention_metal(const std::string& case_id, const BatchPrefillAttentionConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // Metal host-side bfloat16

    const int num_tokens = cfg.num_tokens;         // qo tokens
    const int num_query_heads = cfg.num_query_heads;
    const int head_size = cfg.head_size;
    const int head_dim = num_query_heads * head_size;
    const int page_size = cfg.page_size;

    std::cout << "\n🚀 Running Metal Batch Prefill Attention: tokens=" << num_tokens
              << ", query_heads=" << num_query_heads
              << ", head_size=" << head_size
              << ", head_dim=" << head_dim
              << ", page_size=" << page_size << std::endl;

    // Try to load CUDA reference inputs if available to ensure apples-to-apples comparison
    // Resolve CUDA artifacts base directory:
    // 1) PIE_CUDA_ARTIFACTS_DIR env var, or
    // 2) Absolute path derived from this source file: <repo>/metal-protocol-tests/tests/artifacts
    std::filesystem::path cuda_base_dir;
    if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
        cuda_base_dir = std::filesystem::path(envp);
    } else {
        // __FILE__ points to .../metal-protocol-tests/src/ops_metal.mm
        std::filesystem::path this_file(__FILE__);
        auto project_root = this_file.parent_path().parent_path(); // .../metal-protocol-tests
        cuda_base_dir = project_root / "tests" / "artifacts";
    }
    std::filesystem::path cuda_case_dir = cuda_base_dir / "batch_prefill_attention" / case_id;

    auto file_exists = [](const std::filesystem::path& p) -> bool {
        std::error_code ec; return std::filesystem::exists(p, ec);
    };
    auto read_bytes = [](const std::filesystem::path& p) -> std::vector<uint8_t> {
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs.is_open()) return {};
        ifs.seekg(0, std::ios::end);
        std::streamsize size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(static_cast<size_t>(std::max<int64_t>(0, size)));
        if (size > 0) ifs.read(reinterpret_cast<char*>(buf.data()), size);
        return buf;
    };
    auto read_vec_s32 = [&](const std::filesystem::path& p) -> std::vector<int32_t> {
        std::vector<uint8_t> bytes = read_bytes(p);
        size_t n = bytes.size() / sizeof(int32_t);
        std::vector<int32_t> v(n);
        if (n) memcpy(v.data(), bytes.data(), n * sizeof(int32_t));
        return v;
    };
    auto read_vec_bf16 = [&](const std::filesystem::path& p, size_t expected = 0) -> std::vector<T> {
        std::vector<uint8_t> bytes = read_bytes(p);
        size_t n = bytes.size() / sizeof(T);
        if (expected && n != expected) {
            std::cerr << "Warning: " << p << " element count mismatch; expected " << expected << ", got " << n << std::endl;
        }
        std::vector<T> v(n);
        if (n) memcpy(v.data(), bytes.data(), n * sizeof(T));
        return v;
    };

    // Allocate containers
    std::vector<T> q_input(static_cast<size_t>(num_tokens) * head_dim);
    std::vector<T> paged_k_cache;
    std::vector<T> paged_v_cache;
    std::vector<T> output(static_cast<size_t>(num_tokens) * head_dim);
    std::vector<int32_t> qo_indptr{0, num_tokens};
    std::vector<int32_t> kv_page_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;

    // Try to load CUDA reference inputs if available, otherwise generate test data
    bool use_cuda_artifacts = false;
    if (file_exists(cuda_case_dir)) {
        auto q_input_p = cuda_case_dir / "q_input.bin";
        auto pkv_p = cuda_case_dir / "paged_k_cache.bin";
        auto pvv_p = cuda_case_dir / "paged_v_cache.bin";
        auto qo_indptr_p = cuda_case_dir / "qo_indptr.bin";
        auto kv_page_indptr_p = cuda_case_dir / "kv_page_indptr.bin";
        auto kv_page_indices_p = cuda_case_dir / "kv_page_indices.bin";
        auto kv_last_page_lens_p = cuda_case_dir / "kv_last_page_lens.bin";

        // Check if all required CUDA reference files exist
        std::vector<std::filesystem::path> required_files = {
            q_input_p, pkv_p, pvv_p, qo_indptr_p, kv_page_indptr_p, kv_page_indices_p, kv_last_page_lens_p
        };

        bool all_files_exist = true;
        for (const auto& file : required_files) {
            if (!file_exists(file)) {
                all_files_exist = false;
                break;
            }
        }

        if (all_files_exist) {
            // Load CUDA reference inputs
            qo_indptr = read_vec_s32(qo_indptr_p);
            kv_page_indptr = read_vec_s32(kv_page_indptr_p);
            kv_page_indices = read_vec_s32(kv_page_indices_p);
            kv_last_page_lens = read_vec_s32(kv_last_page_lens_p);
            q_input = read_vec_bf16(q_input_p, static_cast<size_t>(num_tokens) * head_dim);
            paged_k_cache = read_vec_bf16(pkv_p);
            paged_v_cache = read_vec_bf16(pvv_p);
            use_cuda_artifacts = true;
            std::cout << "✅ Loaded CUDA reference inputs from: " << cuda_case_dir << std::endl;
        }
    }

    if (!use_cuda_artifacts) {
        // Generate synthetic test data for FlashInfer interface testing
        std::cout << "🔧 Generating synthetic test data for FlashInfer interface testing" << std::endl;

        // Set up paging structure to match CUDA reference expectations
        // Calculate number of pages needed for kv_len tokens
        const int num_pages = (cfg.kv_len + page_size - 1) / page_size;  // Round up
        qo_indptr = {0, num_tokens};
        kv_page_indptr = {0, num_pages};
        kv_page_indices.resize(num_pages);
        for (int i = 0; i < num_pages; ++i) {
            kv_page_indices[i] = i;
        }
        // Last page might be partially filled
        const int last_page_len = cfg.kv_len - (num_pages - 1) * page_size;
        kv_last_page_lens = {last_page_len};

        // Generate random test data using seed
        std::mt19937_64 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Generate q_input
        q_input.resize(static_cast<size_t>(num_tokens) * head_dim);
        for (size_t i = 0; i < q_input.size(); ++i) {
            float f = dist(gen);
            q_input[i] = float_to_bf16(f);
        }

        // Generate paged K and V caches - each key should have head_dim elements (all heads concatenated)
        size_t cache_size = static_cast<size_t>(num_pages) * page_size * head_dim;
        paged_k_cache.resize(cache_size);
        paged_v_cache.resize(cache_size);
        for (auto& val : paged_k_cache) {
            val = float_to_bf16(dist(gen));
        }
        for (auto& val : paged_v_cache) {
            val = float_to_bf16(dist(gen));
        }
    }
    std::fill(output.begin(), output.end(), static_cast<T>(0));

    print_vec_stats("q_input", q_input, 16);
    print_vec_stats("paged_k_cache", paged_k_cache, 16);
    print_vec_stats("paged_v_cache", paged_v_cache, 16);

    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    std::cout << "\n🔍 DEBUG: Host-side parameter values:" << std::endl;
    std::cout << "  head_size: " << head_size << std::endl;
    std::cout << "  sqrt(head_size): " << sqrtf(static_cast<float>(head_size)) << std::endl;
    std::cout << "  scale = 1/sqrt(head_size): " << scale << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;
    std::cout << "  page_size: " << page_size << std::endl;
    std::cout << "  kv_page_indices.size(): " << kv_page_indices.size() << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        // Call new unified Metal implementation
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input.data(),
            paged_k_cache.data(),
            paged_v_cache.data(),
            qo_indptr.data(),
            kv_page_indptr.data(),
            kv_page_indices.data(),
            kv_last_page_lens.data(),
            output.data(),
            num_tokens, head_dim, head_size, page_size, scale,
            static_cast<int>(kv_page_indices.size())  // Pass actual number of pages
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "\n⏱️  Metal kernel execution time: " << elapsed.count() << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Metal kernel execution failed: " << e.what() << std::endl;
        // Optionally re-throw or handle the error as needed
        throw;
    }


    print_vec_stats("output", output);

    // Derive kv_len from paging arrays to reflect CUDA metadata
    int num_pages_actual = static_cast<int>(kv_page_indices.size());
    int kv_len_actual = 0;
    if (!kv_last_page_lens.empty()) {
        kv_len_actual = (num_pages_actual > 0)
            ? (num_pages_actual - 1) * page_size + kv_last_page_lens[0]
            : 0;
    }

    // Write artifacts to match CUDA test_unified tensors
    if (artifacts::op_enabled("batch_prefill_attention")) {
        auto dir = artifacts::ensure_dir_for_case("batch_prefill_attention", case_id + "_metal");

        artifacts::write_host_bin(dir, "q_input", q_input.data(), q_input.size());
        artifacts::write_host_bin(dir, "k_input", paged_k_cache.data(), paged_k_cache.size());
        artifacts::write_host_bin(dir, "v_input", paged_v_cache.data(), paged_v_cache.size());
        artifacts::write_host_bin(dir, "paged_k_cache", paged_k_cache.data(), paged_k_cache.size());
        artifacts::write_host_bin(dir, "paged_v_cache", paged_v_cache.data(), paged_v_cache.size());
        artifacts::write_host_bin(dir, "output", output.data(), output.size());
        artifacts::write_host_bin(dir, "qo_indptr", qo_indptr.data(), qo_indptr.size());
        artifacts::write_host_bin(dir, "kv_page_indptr", kv_page_indptr.data(), kv_page_indptr.size());
        artifacts::write_host_bin(dir, "kv_page_indices", kv_page_indices.data(), kv_page_indices.size());
        artifacts::write_host_bin(dir, "kv_last_page_lens", kv_last_page_lens.data(), kv_last_page_lens.size());

        std::ostringstream meta;
    meta << "\"version\": \"1\",\n"
             << "\"op\": \"batch_prefill_attention\",\n"
             << "\"case_id\": \"" << case_id << "\",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_query_heads\": " << num_query_heads
             << ", \"num_kv_heads\": " << cfg.num_kv_heads
             << ", \"head_size\": " << head_size
         << ", \"kv_len\": " << (kv_len_actual > 0 ? kv_len_actual : cfg.kv_len)
             << ", \"page_size\": " << page_size
         << ", \"batch_size\": 1, \"num_pages\": " << num_pages_actual << "},\n"
             << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"v_input\": \"bf16\", \"paged_k_cache\": \"bf16\", \"paged_v_cache\": \"bf16\", \"output\": \"bf16\", \"qo_indptr\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_page_indices\": \"s32\", \"kv_last_page_lens\": \"s32\"},\n"
             << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << head_dim << "], "
         << "\"k_input\": [" << (num_pages_actual * page_size) << ", " << head_dim << "], "
         << "\"v_input\": [" << (num_pages_actual * page_size) << ", " << head_dim << "], "
         << "\"paged_k_cache\": [" << num_pages_actual << ", " << page_size << ", " << head_dim << "], "
         << "\"paged_v_cache\": [" << num_pages_actual << ", " << page_size << ", " << head_dim << "], "
             << "\"output\": [" << num_tokens << ", " << head_dim << "], "
         << "\"qo_indptr\": [2], \"kv_page_indptr\": [2], \"kv_page_indices\": [" << num_pages_actual << "], \"kv_last_page_lens\": [1]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "\n✅ Metal Batch Prefill Attention completed successfully" << std::endl;
}

} // namespace ops