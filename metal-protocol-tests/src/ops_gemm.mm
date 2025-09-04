// Per-op Metal wrapper for GEMM
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <filesystem>
#include <fstream>
#import <Metal/Metal.h>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_gemm.hpp"
#include "dtype_utils.hpp"
#include "workspace_utils.hpp"

namespace ops {

void run_gemm_metal(const std::string& case_id, const GemmConfig& cfg, uint64_t seed) {
    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("gemm", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for gemm/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal GEMM: m=" << m << ", n=" << n << ", k=" << k
              << ", transa=" << transa << ", transb=" << transb
              << ", use_bias=" << use_bias << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Calculate tensor dimensions
    const size_t A_elems = static_cast<size_t>(transa ? (k * m) : (m * k));
    const size_t B_elems = static_cast<size_t>(transb ? (n * k) : (k * n));
    const size_t C_elems = static_cast<size_t>(m) * n;

    // Use different storage based on detected dtype to avoid unnecessary quantization
    std::vector<bfloat16_t> A_data_bf16, B_data_bf16, C_data_bf16, bias_data_bf16;
    std::vector<float> A_data_f32, B_data_f32, C_data_f32, bias_data_f32;

    bool use_f32_storage = (dtype_info.dtype == DType::FP32);

    if (use_f32_storage) {
        A_data_f32.resize(A_elems);
        B_data_f32.resize(B_elems);
        C_data_f32.resize(C_elems);
    } else {
        A_data_bf16.resize(A_elems);
        B_data_bf16.resize(B_elems);
        C_data_bf16.resize(C_elems);
    }

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
    if (!cuda_base_dir.empty()) {
        auto cuda_case_dir = cuda_base_dir / "gemm" / case_id;
        if (file_exists(cuda_case_dir / "A.bin") && file_exists(cuda_case_dir / "B.bin")) {
            std::vector<uint8_t> A_bytes = read_bytes(cuda_case_dir / "A.bin");
            std::vector<uint8_t> B_bytes = read_bytes(cuda_case_dir / "B.bin");

            // Check expected file sizes based on detected dtype
            size_t expected_A_size, expected_B_size, expected_bias_size;
            if (dtype_info.dtype == DType::FP32) {
                expected_A_size = A_elems * sizeof(float);
                expected_B_size = B_elems * sizeof(float);
                expected_bias_size = n * sizeof(float);
            } else if (dtype_info.dtype == DType::FP16) {
                expected_A_size = A_elems * sizeof(uint16_t); // half precision
                expected_B_size = B_elems * sizeof(uint16_t);
                expected_bias_size = n * sizeof(uint16_t);
            } else { // bf16
                expected_A_size = A_elems * sizeof(bfloat16_t);
                expected_B_size = B_elems * sizeof(bfloat16_t);
                expected_bias_size = n * sizeof(bfloat16_t);
            }

            if (A_bytes.size() == expected_A_size && B_bytes.size() == expected_B_size) {
                // Load CUDA data preserving precision based on target dtype
                if (dtype_info.dtype == DType::FP32) {
                    // Preserve full fp32 precision - no quantization!
                    const float* cuda_A = reinterpret_cast<const float*>(A_bytes.data());
                    const float* cuda_B = reinterpret_cast<const float*>(B_bytes.data());
                    std::memcpy(A_data_f32.data(), cuda_A, A_elems * sizeof(float));
                    std::memcpy(B_data_f32.data(), cuda_B, B_elems * sizeof(float));
                } else if (dtype_info.dtype == DType::FP16) {
                    const uint16_t* cuda_A = reinterpret_cast<const uint16_t*>(A_bytes.data());
                    const uint16_t* cuda_B = reinterpret_cast<const uint16_t*>(B_bytes.data());
                    for (size_t i = 0; i < A_elems; ++i) A_data_bf16[i] = cuda_A[i];
                    for (size_t i = 0; i < B_elems; ++i) B_data_bf16[i] = cuda_B[i];
                } else { // bf16 - direct copy
                    std::memcpy(A_data_bf16.data(), A_bytes.data(), A_bytes.size());
                    std::memcpy(B_data_bf16.data(), B_bytes.data(), B_bytes.size());
                }

                loaded_cuda_inputs = true;
                if (use_f32_storage) {
                    std::cout << "✅ Loaded CUDA reference A/B for gemm from: " << cuda_case_dir
                             << " (preserved full fp32 precision)" << std::endl;
                } else {
                    std::cout << "✅ Loaded CUDA reference A/B for gemm from: " << cuda_case_dir
                             << " (converted from " << dtype_info.dtype_str << " to bf16)" << std::endl;
                }

                // Load bias if available
                if (use_bias && file_exists(cuda_case_dir / "bias.bin")) {
                    std::vector<uint8_t> bias_bytes = read_bytes(cuda_case_dir / "bias.bin");
                    if (bias_bytes.size() == expected_bias_size) {
                        if (dtype_info.dtype == DType::FP32) {
                            bias_data_f32.resize(n);
                            const float* cuda_bias = reinterpret_cast<const float*>(bias_bytes.data());
                            std::memcpy(bias_data_f32.data(), cuda_bias, n * sizeof(float));
                        } else if (dtype_info.dtype == DType::FP16) {
                            bias_data_bf16.resize(n);
                            const uint16_t* cuda_bias = reinterpret_cast<const uint16_t*>(bias_bytes.data());
                            for (size_t i = 0; i < n; ++i) bias_data_bf16[i] = cuda_bias[i];
                        } else { // bf16
                            bias_data_bf16.resize(n);
                            std::memcpy(bias_data_bf16.data(), bias_bytes.data(), bias_bytes.size());
                        }
                    }
                }
            } else {
                std::cout << "❌ Size mismatch: A expected " << expected_A_size << " got " << A_bytes.size()
                         << ", B expected " << expected_B_size << " got " << B_bytes.size() << std::endl;
            }
        }
    }

    if (!loaded_cuda_inputs) {
        // Generate random test data in appropriate format
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        if (use_f32_storage) {
            for (auto& v : A_data_f32) v = dist(rng);
            for (auto& v : B_data_f32) v = dist(rng);
            if (use_bias) {
                bias_data_f32.resize(n);
                for (auto& v : bias_data_f32) v = dist(rng);
            }
        } else {
            for (auto& v : A_data_bf16) v = float_to_bf16(dist(rng));
            for (auto& v : B_data_bf16) v = float_to_bf16(dist(rng));
            if (use_bias) {
                bias_data_bf16.resize(n);
                for (auto& v : bias_data_bf16) v = float_to_bf16(dist(rng));
            }
        }
    }

    // Initialize Metal GEMM system
    if (!initialize_metal_gemm()) {
        throw std::runtime_error("Failed to initialize Metal GEMM system");
    }

    // Create Metal device and command queue for GEMM
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Prepare computation data based on storage format
    std::vector<float> A_f32, B_f32, C_f32, bias_f32;

    if (use_f32_storage) {
        // Already in f32, no conversion needed!
        A_f32 = A_data_f32;
        B_f32 = B_data_f32;
        C_f32.resize(C_elems, 0.0f);
        if (use_bias && !bias_data_f32.empty()) {
            bias_f32 = bias_data_f32;
        }
    } else {
        // Convert bf16 to f32 for computation
        A_f32.resize(A_elems);
        B_f32.resize(B_elems);
        C_f32.resize(C_elems, 0.0f);
        for (size_t i = 0; i < A_elems; ++i) A_f32[i] = bf16_to_float(A_data_bf16[i]);
        for (size_t i = 0; i < B_elems; ++i) B_f32[i] = bf16_to_float(B_data_bf16[i]);
        if (use_bias && !bias_data_bf16.empty()) {
            bias_f32.resize(n);
            for (size_t i = 0; i < n; ++i) bias_f32[i] = bf16_to_float(bias_data_bf16[i]);
        }
    }

    // Use pure f32 GEMM kernel
    metal_gemm_float32(
        device, commandQueue,
        A_f32.data(), B_f32.data(),
        use_bias ? bias_f32.data() : nullptr,
        C_f32.data(),
        m, n, k, nullptr, 0, transa, transb
    );

    if (use_f32_storage) {
        std::cout << "Used pure f32 Metal GEMM kernel with native f32 inputs (no quantization)" << std::endl;
        C_data_f32 = C_f32;  // Keep f32 results
    } else {
        std::cout << "Used pure f32 Metal GEMM kernel (converted bf16 inputs to f32)" << std::endl;
        // Convert f32 results back to bf16 for bf16 case
        for (size_t i = 0; i < C_elems; ++i) {
            C_data_bf16[i] = float_to_bf16(C_f32[i]);
        }
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("gemm")) {
        auto dir = artifacts::ensure_dir_for_case("gemm", case_id + "_metal");

        if (dtype_info.dtype == DType::FP32) {
            // Save pure f32 data directly (no bf16 conversion)
            artifacts::write_vector_bin(dir, "A", A_data_f32);
            artifacts::write_vector_bin(dir, "B", B_data_f32);
            artifacts::write_vector_bin(dir, "C", C_data_f32);  // Pure f32 results!

            if (use_bias && !bias_data_f32.empty()) {
                artifacts::write_vector_bin(dir, "bias", bias_data_f32);
            }
        } else {
            // Save as bf16 for other dtypes
            artifacts::write_host_bin(dir, "A", A_data_bf16.data(), A_data_bf16.size());
            artifacts::write_host_bin(dir, "B", B_data_bf16.data(), B_data_bf16.size());
            artifacts::write_host_bin(dir, "C", C_data_bf16.data(), C_data_bf16.size());

            if (use_bias && !bias_data_bf16.empty()) {
                artifacts::write_host_bin(dir, "bias", bias_data_bf16.data(), bias_data_bf16.size());
            }
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"gemm\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (transa ? "true" : "false")
             << ", \"transb\": " << (transb ? "true" : "false")
             << ", \"use_bias\": " << (use_bias ? "true" : "false") << "},\n"
             << "\"dtype_map\": {\"A\": \"" << dtype_info.dtype_str
             << "\", \"B\": \"" << dtype_info.dtype_str
             << "\", \"C\": \"" << dtype_info.dtype_str << "\"";

        if (use_bias) {
            meta << ", \"bias\": \"" << dtype_info.dtype_str << "\"";
        }

        meta << "},\n\"shape_map\": {";

        int A_dim0 = transa ? k : m;
        int A_dim1 = transa ? m : k;
        int B_dim0 = transb ? n : k;
        int B_dim1 = transb ? k : n;

        meta << "\"A\": [" << A_dim0 << ", " << A_dim1 << "], "
             << "\"B\": [" << B_dim0 << ", " << B_dim1 << "], "
             << "\"C\": [" << m << ", " << n << "]";

        if (use_bias) {
            meta << ", \"bias\": [" << n << "]";
        }

        meta << "}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal GEMM completed successfully" << std::endl;
}

} // namespace ops