// Metal GPU implementations that call into backend/backend-metal kernels
// This allows metal-protocol-tests to test actual Metal GPU execution

#ifdef METAL_SUPPORT_ENABLED

#include "ops.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <random>

// Include Metal implementations from backend-metal
#include "metal_gemm.hpp"
#include "metal_embedding.hpp"
#include "metal_silu_and_mul.hpp"
#include "metal_extract_k_values.hpp"
#include "metal_softmax.hpp"

#include "artifacts.hpp"

namespace ops {

// Metal implementation of GEMM operation
void run_gemm_metal(const std::string& case_id, const GemmConfig& cfg, uint64_t seed) {
    using T = __nv_bfloat16;  // Use same data type as CUDA for comparison
    
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
    
    for (auto& v : h_A) v = static_cast<T>(dist(rng));
    for (auto& v : h_B) v = static_cast<T>(dist(rng));
    
    // Call Metal GEMM implementation
    int result = metal_gemm_bfloat16(
        h_A.data(), h_B.data(), nullptr, h_C.data(),
        m, n, k, k, n, n,  // lda, ldb, ldc
        cfg.transa, cfg.transb, cfg.use_bias
    );
    
    if (result != 0) {
        throw std::runtime_error("Metal GEMM execution failed");
    }
    
    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("gemm")) {
        auto dir = artifacts::ensure_dir_for_case("gemm", case_id + "_metal");
        
        artifacts::write_host_bin(dir, "A", h_A.data(), h_A.size());
        artifacts::write_host_bin(dir, "B", h_B.data(), h_B.size());
        artifacts::write_host_bin(dir, "C", h_C.data(), h_C.size());
        
        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"gemm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (cfg.transa ? "true" : "false")
             << ", \"transb\": " << (cfg.transb ? "true" : "false")
             << ", \"use_bias\": " << (cfg.use_bias ? "true" : "false") << "},\n"
             << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"C\": \"bf16\"},\n"
             << "\"shape_map\": {\"A\": [" << m << ", " << k << "], \"B\": [" << k << ", " << n << "], \"C\": [" << m << ", " << n << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }
    
    std::cout << "Metal GEMM completed successfully" << std::endl;
}

// Metal implementation of embedding lookup
void run_embedding_lookup_metal(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed) {
    using T = __nv_bfloat16;
    using I = int32_t;
    
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const int vocab_size = cfg.vocab_size;
    
    std::cout << "Running Metal Embedding: tokens=" << num_tokens << ", hidden=" << hidden_size << ", vocab=" << vocab_size << std::endl;
    
    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<T> h_embedding(static_cast<size_t>(vocab_size) * hidden_size);
    for (auto& v : h_embedding) v = static_cast<T>(dist(rng));
    
    std::vector<I> h_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        h_indices[i] = static_cast<I>(i % vocab_size);
    }
    
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);
    
    // Call Metal embedding implementation
    int result = metal_embedding_lookup_bfloat16(
        h_embedding.data(), h_indices.data(), h_output.data(),
        num_tokens, hidden_size, vocab_size
    );
    
    if (result != 0) {
        throw std::runtime_error("Metal embedding lookup execution failed");
    }
    
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
    using T = __nv_bfloat16;
    
    const int num_tokens = cfg.num_tokens;
    const int intermediate_size = cfg.intermediate_size;
    
    std::cout << "Running Metal SiLU and Mul: tokens=" << num_tokens << ", intermediate=" << intermediate_size << std::endl;
    
    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<T> h_gate(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_up(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * intermediate_size, 0);
    
    for (auto& v : h_gate) v = static_cast<T>(dist(rng));
    for (auto& v : h_up) v = static_cast<T>(dist(rng));
    
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
    using T = __nv_bfloat16;
    
    const int M = cfg.M;
    const int N = cfg.N;
    const int k = cfg.k;
    
    std::cout << "Running Metal Extract K Values: M=" << M << ", N=" << N << ", k=" << k << std::endl;
    
    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<T> h_A(static_cast<size_t>(M) * N);
    // Initialize all to negative infinity
    for (auto& v : h_A) v = static_cast<T>(-INFINITY);
    
    // For each row, place k values at deterministic positions
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < k; ++j) {
            int col = (m * 131 + j * 17) % N; // same hash as CUDA
            h_A[static_cast<size_t>(m) * N + col] = static_cast<T>(dist(rng));
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

} // namespace ops

#endif // METAL_SUPPORT_ENABLED