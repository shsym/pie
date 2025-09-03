// Per-op Metal wrapper for Grouped GEMM
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
#include "metal_grouped_gemm.hpp"
#include "dtype_utils.hpp"
#include "workspace_utils.hpp"

namespace ops {

void run_grouped_gemm_metal(const std::string& case_id, const GroupedGemmConfig& cfg, uint64_t seed) {
    const int num_groups = cfg.num_groups;
    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("grouped_gemm", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for grouped_gemm/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Grouped GEMM: groups=" << num_groups << ", m=" << m << ", n=" << n
              << ", k=" << k << ", transa=" << transa << ", transb=" << transb
              << ", use_bias=" << use_bias << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Create arrays to hold matrix data for each group
    std::vector<std::vector<bfloat16_t>> A_matrices(num_groups);
    std::vector<std::vector<bfloat16_t>> B_matrices(num_groups);
    std::vector<std::vector<bfloat16_t>> C_matrices(num_groups);
    std::vector<std::vector<bfloat16_t>> bias_matrices(num_groups);

    std::vector<void*> A_ptrs(num_groups);
    std::vector<void*> B_ptrs(num_groups);
    std::vector<void*> C_ptrs(num_groups);
    std::vector<void*> bias_ptrs(num_groups);

    std::vector<int> m_array(num_groups, m);
    std::vector<int> n_array(num_groups, n);
    std::vector<int> k_array(num_groups, k);

    // Attempt to load CUDA reference inputs for apples-to-apples comparison
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

    bool loaded_cuda_inputs = false;
    auto cuda_base_dir = workspace_utils::get_cuda_artifacts_dir();
    if (cuda_base_dir.empty()) {
        std::cerr << "Error: Could not find workspace root or CUDA artifacts directory" << std::endl;
        return;
    }
    auto cuda_case_dir = cuda_base_dir / "grouped_gemm" / case_id;

    const size_t A_elems_per_group = static_cast<size_t>(transa ? (k * m) : (m * k));
    const size_t B_elems_per_group = static_cast<size_t>(transb ? (n * k) : (k * n));
    const size_t C_elems_per_group = static_cast<size_t>(m) * n;

    if (file_exists(cuda_case_dir / "A.bin") && file_exists(cuda_case_dir / "B.bin")) {
        // Read flattened bf16 tensors and split into groups
        std::vector<uint8_t> A_bytes = read_bytes(cuda_case_dir / "A.bin");
        std::vector<uint8_t> B_bytes = read_bytes(cuda_case_dir / "B.bin");
        size_t A_elems = A_bytes.size() / sizeof(bfloat16_t);
        size_t B_elems = B_bytes.size() / sizeof(bfloat16_t);
        if (A_elems == static_cast<size_t>(num_groups) * A_elems_per_group &&
            B_elems == static_cast<size_t>(num_groups) * B_elems_per_group) {
            const bfloat16_t* A_flat = reinterpret_cast<const bfloat16_t*>(A_bytes.data());
            const bfloat16_t* B_flat = reinterpret_cast<const bfloat16_t*>(B_bytes.data());
            for (int g = 0; g < num_groups; ++g) {
                A_matrices[g].assign(A_flat + g * A_elems_per_group, A_flat + (g + 1) * A_elems_per_group);
                B_matrices[g].assign(B_flat + g * B_elems_per_group, B_flat + (g + 1) * B_elems_per_group);
                A_ptrs[g] = A_matrices[g].data();
                B_ptrs[g] = B_matrices[g].data();
            }
            loaded_cuda_inputs = true;
            std::cout << "✅ Loaded CUDA reference A/B for grouped_gemm from: " << cuda_case_dir << std::endl;
        } else {
            std::cerr << "Warning: CUDA A/B sizes do not match expected group sizes. Falling back to random inputs." << std::endl;
        }
        // Optional bias
        if (use_bias && file_exists(cuda_case_dir / "bias.bin")) {
            std::vector<uint8_t> bias_bytes = read_bytes(cuda_case_dir / "bias.bin");
            size_t bias_elems = bias_bytes.size() / sizeof(bfloat16_t);
            if (bias_elems == static_cast<size_t>(num_groups) * static_cast<size_t>(n)) {
                const bfloat16_t* Bias_flat = reinterpret_cast<const bfloat16_t*>(bias_bytes.data());
                for (int g = 0; g < num_groups; ++g) {
                    bias_matrices[g].assign(Bias_flat + g * static_cast<size_t>(n), Bias_flat + (g + 1) * static_cast<size_t>(n));
                    bias_ptrs[g] = bias_matrices[g].data();
                }
            }
        }
    }

    if (!loaded_cuda_inputs) {
        // Generate same test data as CUDA version when reference isn't available
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int group = 0; group < num_groups; ++group) {
            // A matrix
            A_matrices[group].resize(A_elems_per_group);
            for (auto& v : A_matrices[group]) v = float_to_bf16(dist(rng));
            A_ptrs[group] = A_matrices[group].data();
            // B matrix
            B_matrices[group].resize(B_elems_per_group);
            for (auto& v : B_matrices[group]) v = float_to_bf16(dist(rng));
            B_ptrs[group] = B_matrices[group].data();
            // Bias
            if (use_bias) {
                bias_matrices[group].resize(n);
                for (auto& v : bias_matrices[group]) v = float_to_bf16(dist(rng));
                bias_ptrs[group] = bias_matrices[group].data();
            } else {
                bias_ptrs[group] = nullptr;
            }
        }
    } else if (!use_bias) {
        // Ensure bias_ptrs are null when not used
        std::fill(bias_ptrs.begin(), bias_ptrs.end(), nullptr);
    }

    // Allocate outputs C
    for (int group = 0; group < num_groups; ++group) {
        C_matrices[group].resize(C_elems_per_group, 0);
        C_ptrs[group] = C_matrices[group].data();
    }

    // Route to appropriate kernel based on detected dtype
    int result = 0;
    // When running fp32, preserve fp32 host-side copies for artifact writing
    std::vector<std::vector<float>> A_matrices_fp32_preserve;
    std::vector<std::vector<float>> B_matrices_fp32_preserve;
    std::vector<std::vector<float>> C_matrices_fp32_preserve;
    std::vector<std::vector<float>> bias_matrices_fp32_preserve;
    if (dtype_info.dtype == DType::FP32) {
        A_matrices_fp32_preserve.resize(num_groups);
        B_matrices_fp32_preserve.resize(num_groups);
        C_matrices_fp32_preserve.resize(num_groups);
        if (use_bias) bias_matrices_fp32_preserve.resize(num_groups);
    }
    if (dtype_info.dtype == DType::FP32) {
        // Use native fp32 kernel with fp32 data
        std::vector<std::vector<float>> A_matrices_fp32(num_groups);
        std::vector<std::vector<float>> B_matrices_fp32(num_groups);
        std::vector<std::vector<float>> C_matrices_fp32(num_groups);
        std::vector<std::vector<float>> bias_matrices_fp32(num_groups);
        std::vector<const float*> A_ptrs_fp32(num_groups);
        std::vector<const float*> B_ptrs_fp32(num_groups);
        std::vector<float*> C_ptrs_fp32(num_groups);
        std::vector<const float*> bias_ptrs_fp32(num_groups);

        // Convert bf16 data to fp32 for computation
        for (int group = 0; group < num_groups; ++group) {
            A_matrices_fp32[group].resize(A_matrices[group].size());
            B_matrices_fp32[group].resize(B_matrices[group].size());
            C_matrices_fp32[group].resize(C_matrices[group].size());

            for (size_t i = 0; i < A_matrices[group].size(); ++i) {
                A_matrices_fp32[group][i] = bf16_to_float(A_matrices[group][i]);
            }
            for (size_t i = 0; i < B_matrices[group].size(); ++i) {
                B_matrices_fp32[group][i] = bf16_to_float(B_matrices[group][i]);
            }

            A_ptrs_fp32[group] = A_matrices_fp32[group].data();
            B_ptrs_fp32[group] = B_matrices_fp32[group].data();
            C_ptrs_fp32[group] = C_matrices_fp32[group].data();

            if (use_bias && !bias_matrices[group].empty()) {
                bias_matrices_fp32[group].resize(bias_matrices[group].size());
                for (size_t i = 0; i < bias_matrices[group].size(); ++i) {
                    bias_matrices_fp32[group][i] = bf16_to_float(bias_matrices[group][i]);
                }
                bias_ptrs_fp32[group] = bias_matrices_fp32[group].data();
            } else {
                bias_ptrs_fp32[group] = nullptr;
            }
        }

        result = metal_grouped_gemm_float32(
            A_ptrs_fp32.data(), B_ptrs_fp32.data(), C_ptrs_fp32.data(),
            use_bias ? const_cast<const float* const*>(bias_ptrs_fp32.data()) : nullptr,
            m_array.data(), n_array.data(), k_array.data(),
            num_groups, transa, transb
        );

        // Convert fp32 results back to bf16 for comparator compatibility when needed,
        // but also preserve fp32 copies for artifact writing.
        for (int group = 0; group < num_groups; ++group) {
            // Preserve fp32 results
            C_matrices_fp32_preserve[group] = C_matrices_fp32[group];
            for (size_t i = 0; i < C_matrices[group].size(); ++i) {
                C_matrices[group][i] = float_to_bf16(C_matrices_fp32[group][i]);
            }
        }
        // Preserve fp32 inputs/bias for artifact writing
        for (int group = 0; group < num_groups; ++group) {
            A_matrices_fp32_preserve[group] = A_matrices_fp32[group];
            B_matrices_fp32_preserve[group] = B_matrices_fp32[group];
            if (use_bias && !bias_matrices_fp32[group].empty()) {
                bias_matrices_fp32_preserve[group] = bias_matrices_fp32[group];
            }
        }
    }
    else {
        // Default bf16 path (includes fp16 fallback)
        result = metal_grouped_gemm_bfloat16(
            A_ptrs.data(), B_ptrs.data(), C_ptrs.data(),
            use_bias ? bias_ptrs.data() : nullptr,
            m_array.data(), n_array.data(), k_array.data(),
            num_groups, transa, transb
        );

        if (dtype_info.dtype == DType::FP16) {
            std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
        }
    }

    if (result != 0) {
        throw std::runtime_error("Metal Grouped GEMM execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("grouped_gemm")) {
        auto dir = artifacts::ensure_dir_for_case("grouped_gemm", case_id + "_metal");
        if (dtype_info.dtype == DType::FP32) {
            // Save each group's matrices in fp32
            for (int group = 0; group < num_groups; ++group) {
                std::string group_suffix = "_group" + std::to_string(group);
                artifacts::write_vector_bin(dir, "A" + group_suffix, A_matrices_fp32_preserve[group]);
                artifacts::write_vector_bin(dir, "B" + group_suffix, B_matrices_fp32_preserve[group]);
                artifacts::write_vector_bin(dir, "C" + group_suffix, C_matrices_fp32_preserve[group]);
                if (use_bias && !bias_matrices_fp32_preserve.empty() && !bias_matrices_fp32_preserve[group].empty()) {
                    artifacts::write_vector_bin(dir, "bias" + group_suffix, bias_matrices_fp32_preserve[group]);
                }
            }
            // Additionally write flattened A/B/C in fp32
            size_t A_total = 0, B_total = 0, C_total = 0;
            for (int g = 0; g < num_groups; ++g) {
                A_total += A_matrices_fp32_preserve[g].size();
                B_total += B_matrices_fp32_preserve[g].size();
                C_total += C_matrices_fp32_preserve[g].size();
            }
            std::vector<float> A_flat; A_flat.reserve(A_total);
            std::vector<float> B_flat; B_flat.reserve(B_total);
            std::vector<float> C_flat; C_flat.reserve(C_total);
            for (int g = 0; g < num_groups; ++g) {
                A_flat.insert(A_flat.end(), A_matrices_fp32_preserve[g].begin(), A_matrices_fp32_preserve[g].end());
                B_flat.insert(B_flat.end(), B_matrices_fp32_preserve[g].begin(), B_matrices_fp32_preserve[g].end());
                C_flat.insert(C_flat.end(), C_matrices_fp32_preserve[g].begin(), C_matrices_fp32_preserve[g].end());
            }
            artifacts::write_vector_bin(dir, "A", A_flat);
            artifacts::write_vector_bin(dir, "B", B_flat);
            artifacts::write_vector_bin(dir, "C", C_flat);
        } else {
            // Save each group's matrices in bf16
            for (int group = 0; group < num_groups; ++group) {
                std::string group_suffix = "_group" + std::to_string(group);
                artifacts::write_host_bin(dir, "A" + group_suffix, A_matrices[group].data(), A_matrices[group].size());
                artifacts::write_host_bin(dir, "B" + group_suffix, B_matrices[group].data(), B_matrices[group].size());
                artifacts::write_host_bin(dir, "C" + group_suffix, C_matrices[group].data(), C_matrices[group].size());
                if (use_bias) {
                    artifacts::write_host_bin(dir, "bias" + group_suffix, bias_matrices[group].data(), bias_matrices[group].size());
                }
            }
            // Additionally, write flattened A/B/C in bf16
            size_t A_total = 0, B_total = 0, C_total = 0;
            for (int g = 0; g < num_groups; ++g) {
                A_total += A_matrices[g].size();
                B_total += B_matrices[g].size();
                C_total += C_matrices[g].size();
            }
            std::vector<bfloat16_t> A_flat; A_flat.reserve(A_total);
            std::vector<bfloat16_t> B_flat; B_flat.reserve(B_total);
            std::vector<bfloat16_t> C_flat; C_flat.reserve(C_total);
            for (int g = 0; g < num_groups; ++g) {
                A_flat.insert(A_flat.end(), A_matrices[g].begin(), A_matrices[g].end());
                B_flat.insert(B_flat.end(), B_matrices[g].begin(), B_matrices[g].end());
                C_flat.insert(C_flat.end(), C_matrices[g].begin(), C_matrices[g].end());
            }
            artifacts::write_host_bin(dir, "A", A_flat.data(), A_flat.size());
            artifacts::write_host_bin(dir, "B", B_flat.data(), B_flat.size());
            artifacts::write_host_bin(dir, "C", C_flat.data(), C_flat.size());
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
            meta << "\"A" << suffix << "\": \"" << dtype_info.dtype_str << "\", \"B" << suffix << "\": \"" << dtype_info.dtype_str << "\", \"C" << suffix << "\": \"" << dtype_info.dtype_str << "\"";
            if (use_bias) {
                meta << ", \"bias" << suffix << "\": \"" << dtype_info.dtype_str << "\"";
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

} // namespace ops
