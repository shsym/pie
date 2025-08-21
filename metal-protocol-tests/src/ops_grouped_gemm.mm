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

namespace ops {

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
    std::filesystem::path cuda_base_dir;
    if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
        cuda_base_dir = std::filesystem::path(envp);
    } else {
        std::filesystem::path this_file(__FILE__);
        cuda_base_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
    }
    auto cuda_case_dir = cuda_base_dir / "grouped_gemm" / case_id;

    const size_t A_elems_per_group = static_cast<size_t>(transa ? (k * m) : (m * k));
    const size_t B_elems_per_group = static_cast<size_t>(transb ? (n * k) : (k * n));
    const size_t C_elems_per_group = static_cast<size_t>(m) * n;

    if (file_exists(cuda_case_dir / "A.bin") && file_exists(cuda_case_dir / "B.bin")) {
        // Read flattened bf16 tensors and split into groups
        std::vector<uint8_t> A_bytes = read_bytes(cuda_case_dir / "A.bin");
        std::vector<uint8_t> B_bytes = read_bytes(cuda_case_dir / "B.bin");
        size_t A_elems = A_bytes.size() / sizeof(T);
        size_t B_elems = B_bytes.size() / sizeof(T);
        if (A_elems == static_cast<size_t>(num_groups) * A_elems_per_group &&
            B_elems == static_cast<size_t>(num_groups) * B_elems_per_group) {
            const T* A_flat = reinterpret_cast<const T*>(A_bytes.data());
            const T* B_flat = reinterpret_cast<const T*>(B_bytes.data());
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
            size_t bias_elems = bias_bytes.size() / sizeof(T);
            if (bias_elems == static_cast<size_t>(num_groups) * static_cast<size_t>(n)) {
                const T* Bias_flat = reinterpret_cast<const T*>(bias_bytes.data());
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

        // Additionally, write flattened A/B/C to match CUDA artifact naming if present
        {
            // Flatten by concatenating groups in order
            size_t A_total = 0, B_total = 0, C_total = 0;
            for (int g = 0; g < num_groups; ++g) {
                A_total += A_matrices[g].size();
                B_total += B_matrices[g].size();
                C_total += C_matrices[g].size();
            }
            std::vector<T> A_flat; A_flat.reserve(A_total);
            std::vector<T> B_flat; B_flat.reserve(B_total);
            std::vector<T> C_flat; C_flat.reserve(C_total);
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

} // namespace ops
