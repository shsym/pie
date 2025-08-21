// Per-op Metal wrapper for RoPE
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <filesystem>
#include <fstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_rope.hpp"

namespace ops {

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

} // namespace ops
