// GPU op implementations that call into the same CUDA kernels/utilities
// used by backend/backend-cuda

#include "ops.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <vector>
#include <sstream>
#include <random>
#include <type_traits>

#include "common.cuh"           // embed<T,I>() declaration
#include "kernels.cuh"          // silu_and_mul, add_residual kernels
#include "artifacts.hpp"        // artifact writer (device -> host)
#include "flashinfer/norm.cuh"     // RMSNorm implementation
#include "flashinfer/pos_enc.cuh"   // RoPE implementation
#include "flashinfer/sampling.cuh"  // TopKMaskLogits
#include "flashinfer/page.cuh"      // Paged KV cache operations
#include "flashinfer/attention/prefill.cuh"      // Paged KV cache operations
#include "flashinfer/attention/scheduler.cuh" // FlashInfer attention scheduler
#include "flashinfer_ops.cuh"           // FlashInfer ops, including BatchPrefillHandler

namespace ops {

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

// Host-side float -> target type converter to safely initialize __half/bf16 buffers
template <typename T>
static inline T f2t(float v) {
    return static_cast<T>(v);
}

template <>
inline __half f2t<__half>(float v) {
    return __float2half(v);
}

template <>
inline __nv_bfloat16 f2t<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

void run_cast_type(const std::string& case_id,
                   const CastTypeConfig& cfg,
                   uint64_t seed) {
    const int num_elements = cfg.num_elements;
    const std::string& input_dtype = cfg.input_dtype;
    const std::string& output_dtype = cfg.output_dtype;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // For simplicity, implement fp32 -> fp16 casting as an example
    // The Metal backend will need to implement all combinations
    if (input_dtype == "fp32" && output_dtype == "fp16") {
        std::vector<float> h_input(num_elements);
        for (auto& v : h_input) v = dist(rng);

        float* d_input = nullptr;
        __half* d_output = nullptr;
        check_cuda(cudaMalloc(&d_input, num_elements * sizeof(float)));
        check_cuda(cudaMalloc(&d_output, num_elements * sizeof(__half)));

        check_cuda(cudaMemcpyAsync(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Call cast_type from common.cuh
        cast_type<float, __half>(d_input, d_output, num_elements, stream);

        check_cuda(cudaStreamSynchronize(stream));

        // Write artifacts
        if (artifacts::op_enabled("cast_type")) {
            auto dir = artifacts::ensure_dir_for_case("cast_type", case_id);

            artifacts::write_device_bin(dir, "input", d_input, num_elements);
            artifacts::write_device_bin(dir, "output", d_output, num_elements);

            std::ostringstream meta;
            meta << "\"version\": \"1\",\n"
                 << "\"op\": \"cast_type\",\n"
                 << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
                 << "\"config\": {\"num_elements\": " << num_elements
                 << ", \"input_dtype\": " << artifacts::json_escape(input_dtype)
                 << ", \"output_dtype\": " << artifacts::json_escape(output_dtype) << "},\n"
                 << "\"dtype_map\": {\"input\": \"fp32\", \"output\": \"fp16\"},\n"
                 << "\"shape_map\": {\"input\": [" << num_elements
                 << "], \"output\": [" << num_elements << "]}";
            artifacts::write_meta_json(dir, meta.str());
        }

        cudaFree(d_output);
        cudaFree(d_input);
    }

    cudaStreamDestroy(stream);
}



// Automatic multi-dtype runner
void run_all_dtypes_for_operation(const std::string& op_name, const std::string& base_case_id,
                                 const void* config_ptr, uint64_t seed) {
    if (op_name == "gemm") {
        const auto* cfg = static_cast<const GemmConfig*>(config_ptr);
        std::cout << "Running GEMM with all supported data types...\n";
        run_gemm_typed<float>(base_case_id, *cfg, seed);
        run_gemm_typed<__half>(base_case_id, *cfg, seed);
        run_gemm_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "embedding_lookup") {
        const auto* cfg = static_cast<const EmbeddingConfig*>(config_ptr);
        std::cout << "Running embedding lookup with all supported data types...\n";
        run_embedding_lookup_typed<float, int32_t>(base_case_id, *cfg, seed);
        run_embedding_lookup_typed<__half, int32_t>(base_case_id, *cfg, seed);
        run_embedding_lookup_typed<__nv_bfloat16, int32_t>(base_case_id, *cfg, seed);
    } else if (op_name == "extract_k_values") {
        const auto* cfg = static_cast<const ExtractKConfig*>(config_ptr);
        std::cout << "Running extract_k_values with all supported data types...\n";
        run_extract_k_values_typed<float>(base_case_id, *cfg, seed);
        run_extract_k_values_typed<__half>(base_case_id, *cfg, seed);
        run_extract_k_values_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "rms_norm") {
        const auto* cfg = static_cast<const RMSNormConfig*>(config_ptr);
        std::cout << "Running RMS Norm with all supported data types...\n";
        run_rms_norm_typed<float>(base_case_id, *cfg, seed);
        run_rms_norm_typed<__half>(base_case_id, *cfg, seed);
        run_rms_norm_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "silu_and_mul") {
        const auto* cfg = static_cast<const SiLUAndMulConfig*>(config_ptr);
        std::cout << "Running SiLU and Mul with all supported data types...\n";
        run_silu_and_mul_typed<float>(base_case_id, *cfg, seed);
        run_silu_and_mul_typed<__half>(base_case_id, *cfg, seed);
        run_silu_and_mul_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "rope") {
        const auto* cfg = static_cast<const RoPEConfig*>(config_ptr);
        std::cout << "Running RoPE with all supported data types...\n";
        run_rope_typed<float>(base_case_id, *cfg, seed);
        run_rope_typed<__half>(base_case_id, *cfg, seed);
        run_rope_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "add_residual") {
        const auto* cfg = static_cast<const AddResidualConfig*>(config_ptr);
        std::cout << "Running add_residual with all supported data types...\n";
        run_add_residual_typed<float>(base_case_id, *cfg, seed);
        run_add_residual_typed<__half>(base_case_id, *cfg, seed);
        run_add_residual_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "softmax") {
        const auto* cfg = static_cast<const SoftmaxConfig*>(config_ptr);
        std::cout << "Running softmax with supported data types (FlashInfer limitation: float only)...\n";
        run_softmax_typed<float>(base_case_id, *cfg, seed);
    } else if (op_name == "topk_mask_logits") {
        const auto* cfg = static_cast<const TopKMaskConfig*>(config_ptr);
        std::cout << "Running topk_mask_logits with supported data types (FlashInfer limitation: float only)...\n";
        run_topk_mask_logits_typed<float>(base_case_id, *cfg, seed);
    }
}

} // namespace ops
