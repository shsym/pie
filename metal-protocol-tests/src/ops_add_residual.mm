// Per-op Metal wrapper for Add Residual
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
#include "metal_add_residual.hpp"
#include "dtype_utils.hpp"
#include "workspace_utils.hpp"

namespace ops {

void run_add_residual_metal(const std::string& case_id, const AddResidualConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("add_residual", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for add_residual/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Add Residual: tokens=" << num_tokens << ", hidden_size=" << hidden_size
              << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Calculate tensor size
    const size_t tensor_size = static_cast<size_t>(num_tokens) * hidden_size;

    // Generate test data in bf16
    std::vector<bfloat16_t> input_data(tensor_size);
    std::vector<bfloat16_t> residual_data(tensor_size);
    std::vector<bfloat16_t> output_data(tensor_size);

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
        auto cuda_case_dir = cuda_base_dir / "add_residual" / case_id;
        if (file_exists(cuda_case_dir / "input_orig.bin") && file_exists(cuda_case_dir / "residual.bin")) {
            std::vector<uint8_t> input_bytes = read_bytes(cuda_case_dir / "input_orig.bin");
            std::vector<uint8_t> residual_bytes = read_bytes(cuda_case_dir / "residual.bin");

            if (input_bytes.size() == tensor_size * sizeof(bfloat16_t) &&
                residual_bytes.size() == tensor_size * sizeof(bfloat16_t)) {
                std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());
                std::memcpy(residual_data.data(), residual_bytes.data(), residual_bytes.size());
                loaded_cuda_inputs = true;
                std::cout << "✅ Loaded CUDA reference input/residual for add_residual from: " << cuda_case_dir << std::endl;
            }
        }
    }

    if (!loaded_cuda_inputs) {
        // Generate random test data
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto& v : input_data) v = float_to_bf16(dist(rng));
        for (auto& v : residual_data) v = float_to_bf16(dist(rng));
    }

    // Initialize Metal add_residual system
    if (!initialize_metal_add_residual()) {
        throw std::runtime_error("Failed to initialize Metal add_residual system");
    }

    // Create Metal device and command queue
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Convert bf16 to fp32 for fp32 kernel
        std::vector<float> input_fp32(tensor_size), residual_fp32(tensor_size), output_fp32(tensor_size);

        for (size_t i = 0; i < tensor_size; ++i) {
            input_fp32[i] = bf16_to_float(input_data[i]);
            residual_fp32[i] = bf16_to_float(residual_data[i]);
        }

        metal_add_residual_float32(
            device, commandQueue,
            input_fp32.data(), residual_fp32.data(), output_fp32.data(),
            tensor_size
        );

        // Convert results back to bf16 for consistency
        for (size_t i = 0; i < tensor_size; ++i) {
            output_data[i] = float_to_bf16(output_fp32[i]);
        }
    }
    else {
        // Default bf16 path (includes fp16 fallback)
        metal_add_residual_bfloat16(
            device, commandQueue,
            input_data.data(), residual_data.data(), output_data.data(),
            tensor_size
        );

        if (dtype_info.dtype == DType::FP16) {
            std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
        }
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("add_residual")) {
        auto dir = artifacts::ensure_dir_for_case("add_residual", case_id + "_metal");

        if (dtype_info.dtype == DType::FP32) {
            // Convert and save as fp32
            std::vector<float> input_fp32(tensor_size), residual_fp32(tensor_size), output_fp32(tensor_size);
            for (size_t i = 0; i < tensor_size; ++i) {
                input_fp32[i] = bf16_to_float(input_data[i]);
                residual_fp32[i] = bf16_to_float(residual_data[i]);
                output_fp32[i] = bf16_to_float(output_data[i]);
            }

            artifacts::write_vector_bin(dir, "input_orig", input_fp32);
            artifacts::write_vector_bin(dir, "residual", residual_fp32);
            artifacts::write_vector_bin(dir, "output", output_fp32);
        } else {
            // Save as bf16
            artifacts::write_host_bin(dir, "input_orig", input_data.data(), input_data.size());
            artifacts::write_host_bin(dir, "residual", residual_data.data(), residual_data.size());
            artifacts::write_host_bin(dir, "output", output_data.data(), output_data.size());
        }

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"add_residual\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens << ", \"hidden_size\": " << hidden_size << "},\n"
             << "\"dtype_map\": {\"input_orig\": \"" << dtype_info.dtype_str
             << "\", \"residual\": \"" << dtype_info.dtype_str
             << "\", \"output\": \"" << dtype_info.dtype_str << "\"},\n"
             << "\"shape_map\": {\"input_orig\": [" << num_tokens << ", " << hidden_size << "], "
             << "\"residual\": [" << num_tokens << ", " << hidden_size << "], "
             << "\"output\": [" << num_tokens << ", " << hidden_size << "]}";

        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Add Residual completed successfully" << std::endl;
}

} // namespace ops