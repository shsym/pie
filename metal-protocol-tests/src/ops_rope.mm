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
#include "dtype_utils.hpp"

namespace ops {

void run_rope_metal(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // bfloat16 on Metal host side

    const int num_tokens = cfg.num_tokens;
    const int num_q_heads = cfg.num_query_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;
    const float rope_theta = cfg.rope_theta;
    const float rope_factor = cfg.rope_factor;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("rope", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for rope/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal RoPE: tokens=" << num_tokens
              << ", q_heads=" << num_q_heads
              << ", kv_heads=" << num_kv_heads
              << ", head_size=" << head_size
              << ", theta=" << rope_theta
              << ", factor=" << rope_factor
              << ", dtype=" << dtype_info.dtype_str << std::endl;

    const size_t q_elems = static_cast<size_t>(num_tokens) * num_q_heads * head_size;
    const size_t k_elems = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;

    // Query and Key tensors [num_tokens, num_heads, head_size]
    std::vector<T> h_q(q_elems);
    std::vector<T> h_k(k_elems);
    std::vector<T> h_q_input(q_elems);
    std::vector<T> h_k_input(k_elems);

    // Load input data from CUDA reference instead of generating random data
    if (dtype_info.dtype == DType::BF16) {
        // Load bf16 data directly
        auto cuda_q = ops::load_cuda_tensor<uint16_t>("rope", case_id, "q_input");
        auto cuda_k = ops::load_cuda_tensor<uint16_t>("rope", case_id, "k_input");

        if (!cuda_q.empty() && cuda_q.size() == q_elems &&
            !cuda_k.empty() && cuda_k.size() == k_elems) {
            std::cout << "Loaded " << cuda_q.size() << " Q values and " << cuda_k.size() << " K values from CUDA reference" << std::endl;

            // Convert bf16 raw data to bfloat16_t
            for (size_t i = 0; i < q_elems; ++i) {
                h_q[i] = bfloat16_t{cuda_q[i]};
                h_q_input[i] = h_q[i];
            }
            for (size_t i = 0; i < k_elems; ++i) {
                h_k[i] = bfloat16_t{cuda_k[i]};
                h_k_input[i] = h_k[i];
            }
        } else {
            std::cerr << "Warning: Could not load CUDA input data for RoPE, size mismatch or file missing" << std::endl;
            return;
        }
    } else {
        // For fp32/fp16, load and convert
        auto cuda_q = ops::load_cuda_tensor<float>("rope", case_id, "q_input");
        auto cuda_k = ops::load_cuda_tensor<float>("rope", case_id, "k_input");

        if (!cuda_q.empty() && cuda_q.size() == q_elems &&
            !cuda_k.empty() && cuda_k.size() == k_elems) {
            std::cout << "Loaded " << cuda_q.size() << " Q values and " << cuda_k.size() << " K values from CUDA reference" << std::endl;

            for (size_t i = 0; i < q_elems; ++i) {
                h_q[i] = float_to_bf16(cuda_q[i]);
                h_q_input[i] = h_q[i];
            }
            for (size_t i = 0; i < k_elems; ++i) {
                h_k[i] = float_to_bf16(cuda_k[i]);
                h_k_input[i] = h_k[i];
            }
        } else {
            std::cerr << "Warning: Could not load CUDA input data for RoPE, size mismatch or file missing" << std::endl;
            return;
        }
    }

    // Load position IDs from CUDA reference
    auto cuda_pos_ids = ops::load_cuda_tensor<int32_t>("rope", case_id, "pos_ids");
    std::vector<int32_t> h_position_ids(num_tokens);
    if (!cuda_pos_ids.empty() && cuda_pos_ids.size() == static_cast<size_t>(num_tokens)) {
        std::cout << "Loaded " << cuda_pos_ids.size() << " position IDs from CUDA reference" << std::endl;
        h_position_ids = cuda_pos_ids;
    } else {
        std::cerr << "Warning: Could not load CUDA position IDs, using sequential fallback" << std::endl;
        for (int i = 0; i < num_tokens; ++i) {
            h_position_ids[i] = i;
        }
    }

    // Route based on requested dtype
    if (dtype_info.dtype == DType::FP16) {
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
    } else if (dtype_info.dtype == DType::FP32) {
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
        // Keep fq/fk in scope for artifact writing by copying to persistent buffers
        // Note: we prefer writing native fp32 artifacts to match CUDA meta
        // We'll re-construct these buffers again below if needed
    } else {
        // Default bf16 path: need to call bf16 kernel
        std::cout << "Using bf16 kernel path" << std::endl;
        std::cout << "Calling metal_rope_float32 with: tokens=" << num_tokens
                  << ", heads=" << num_q_heads << ", head_size=" << head_size << std::endl;
        // TODO: Call metal_rope_bfloat16 when available
        // For now, convert to fp32, apply fp32 kernel, convert back
        std::vector<float> fq(q_elems), fk(k_elems);
        for (size_t i = 0; i < q_elems; ++i) fq[i] = bf16_to_float(h_q[i]);
        for (size_t i = 0; i < k_elems; ++i) fk[i] = bf16_to_float(h_k[i]);

        int res_q = metal_rope_float32(
            fq.data(), h_position_ids.data(),
            num_tokens, num_q_heads, head_size,
            rope_theta, rope_factor
        );
        std::cout << "metal_rope_float32 Q returned: " << res_q << std::endl;
        if (res_q != 0) {
            throw std::runtime_error("Metal RoPE(Q fp32 fallback) failed with code: " + std::to_string(res_q));
        }
        int res_k = metal_rope_float32(
            fk.data(), h_position_ids.data(),
            num_tokens, num_kv_heads, head_size,
            rope_theta, rope_factor
        );
        if (res_k != 0) {
            throw std::runtime_error("Metal RoPE(K fp32 fallback) failed with code: " + std::to_string(res_k));
        }

        for (size_t i = 0; i < q_elems; ++i) h_q[i] = float_to_bf16(fq[i]);
        for (size_t i = 0; i < k_elems; ++i) h_k[i] = float_to_bf16(fk[i]);

        // Debug: Print first few values after Metal kernel
        std::cout << "After Metal kernel (bf16->fp32): First 10 Q values:" << std::endl;
        for (int i = 0; i < 10 && i < static_cast<int>(fq.size()); ++i) {
            std::cout << "  Q[" << i << "] = " << fq[i] << std::endl;
        }
    }

    // Write artifacts for comparison with CUDA in the requested dtype
    if (artifacts::op_enabled("rope")) {
        auto dir = artifacts::ensure_dir_for_case("rope", case_id + "_metal");

        if (dtype_info.dtype == DType::FP32) {
            // Convert to fp32 for artifact writing from bf16 host buffers
            std::vector<float> q_in_fp32(h_q_input.size()), k_in_fp32(h_k_input.size());
            std::vector<float> q_out_fp32(h_q.size()), k_out_fp32(h_k.size());
            for (size_t i = 0; i < h_q_input.size(); ++i) q_in_fp32[i] = bf16_to_float(h_q_input[i]);
            for (size_t i = 0; i < h_k_input.size(); ++i) k_in_fp32[i] = bf16_to_float(h_k_input[i]);
            for (size_t i = 0; i < h_q.size(); ++i) q_out_fp32[i] = bf16_to_float(h_q[i]);
            for (size_t i = 0; i < h_k.size(); ++i) k_out_fp32[i] = bf16_to_float(h_k[i]);
            artifacts::write_vector_bin(dir, "q_input", q_in_fp32);
            artifacts::write_vector_bin(dir, "k_input", k_in_fp32);
            artifacts::write_vector_bin(dir, "pos_ids", h_position_ids);
            artifacts::write_vector_bin(dir, "q_output", q_out_fp32);
            artifacts::write_vector_bin(dir, "k_output", k_out_fp32);
        } else if (dtype_info.dtype == DType::FP16) {
            // Convert to fp16 for artifact writing
            std::vector<uint16_t> q_in_fp16(h_q_input.size()), k_in_fp16(h_k_input.size());
            std::vector<uint16_t> q_out_fp16(h_q.size()), k_out_fp16(h_k.size());
            for (size_t i = 0; i < h_q_input.size(); ++i) q_in_fp16[i] = float_to_half(bf16_to_float(h_q_input[i]));
            for (size_t i = 0; i < h_k_input.size(); ++i) k_in_fp16[i] = float_to_half(bf16_to_float(h_k_input[i]));
            for (size_t i = 0; i < h_q.size(); ++i) q_out_fp16[i] = float_to_half(bf16_to_float(h_q[i]));
            for (size_t i = 0; i < h_k.size(); ++i) k_out_fp16[i] = float_to_half(bf16_to_float(h_k[i]));
            artifacts::write_vector_bin(dir, "q_input", q_in_fp16);
            artifacts::write_vector_bin(dir, "k_input", k_in_fp16);
            artifacts::write_vector_bin(dir, "pos_ids", h_position_ids);
            artifacts::write_vector_bin(dir, "q_output", q_out_fp16);
            artifacts::write_vector_bin(dir, "k_output", k_out_fp16);
        } else {
            // Write as bf16
            artifacts::write_host_bin(dir, "q_input", h_q_input.data(), h_q_input.size());
            artifacts::write_host_bin(dir, "k_input", h_k_input.data(), h_k_input.size());
            artifacts::write_host_bin(dir, "pos_ids", h_position_ids.data(), h_position_ids.size());
            artifacts::write_host_bin(dir, "q_output", h_q.data(), h_q.size());
            artifacts::write_host_bin(dir, "k_output", h_k.data(), h_k.size());
        }

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
         << "\"dtype_map\": {\"q_input\": \"" << dtype_info.dtype_str
         << "\", \"k_input\": \"" << dtype_info.dtype_str
         << "\", \"pos_ids\": \"s32\", \"q_output\": \"" << dtype_info.dtype_str
         << "\", \"k_output\": \"" << dtype_info.dtype_str << "\"},\n"
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
