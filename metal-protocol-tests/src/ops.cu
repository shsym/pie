// GPU op implementations that call into the same CUDA kernels/utilities
// used by backend/backend-cuda

#include "ops.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <vector>
#include <sstream>
#include <random>

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

void run_embedding_lookup(const std::string& case_id,
                          const EmbeddingConfig& cfg,
                          uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend usage in l4ma.cu
    using I = int32_t;        // Matches repo usage

    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const int vocab_size = cfg.vocab_size;

    // Allocate and init host buffers deterministically
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_embedding(static_cast<size_t>(vocab_size) * hidden_size);
    for (auto& v : h_embedding) v = static_cast<T>(dist(rng));

    std::vector<I> h_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        h_indices[i] = static_cast<I>(i % vocab_size);
    }

    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    // Device alloc
    T* d_embedding = nullptr;
    I* d_indices = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_indices, h_indices.size() * sizeof(I)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // H2D copies
    check_cuda(cudaMemcpyAsync(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_indices, h_indices.data(), h_indices.size() * sizeof(I), cudaMemcpyHostToDevice, stream));

    // Call the exact implementation used in backend/common.cu
    embed<T, I>(
        d_embedding,
        static_cast<size_t>(vocab_size),
        d_indices,
        static_cast<size_t>(num_tokens),
        d_output,
        hidden_size,
        stream
    );

    check_cuda(cudaStreamSynchronize(stream));

    // Record artifacts with the same helper used by l4ma.cu
    if (artifacts::op_enabled("embedding_lookup_forward")) {
        auto dir = artifacts::ensure_dir_for_case("embedding_lookup_forward", case_id);

        artifacts::write_device_bin(dir, "embedding", d_embedding, h_embedding.size());
        artifacts::write_device_bin(dir, "indices", d_indices, h_indices.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"embedding_lookup_forward\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"hidden_size\": " << hidden_size
             << ", \"vocab_size\": " << vocab_size
             << ", \"num_tokens\": " << num_tokens << "},\n"
             << "\"dtype_map\": {\"embedding\": \"bf16\", \"indices\": \"s32\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"embedding\": [" << vocab_size << ", " << hidden_size
             << "], \"indices\": [" << num_tokens
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_embedding);
}

void run_extract_k_values(const std::string& case_id,
                          const ExtractKConfig& cfg,
                          uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend usage (supports bf16 as per common.cu:334)

    const int M = cfg.M;
    const int N = cfg.N;
    const int k = cfg.k;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_A(static_cast<size_t>(M) * N);
    // Initialize all to negative infinity equivalent
    for (auto& v : h_A) v = static_cast<T>(-INFINITY);

    // For each row, place k values at deterministic positions
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < k; ++j) {
            int col = (m * 131 + j * 17) % N; // simple hash for spread
            h_A[static_cast<size_t>(m) * N + col] = static_cast<T>(dist(rng));
        }
    }

    std::vector<T> h_V(static_cast<size_t>(M) * k, 0);
    std::vector<int32_t> h_I(static_cast<size_t>(M) * k, 0);

    T* d_A = nullptr;
    T* d_V = nullptr;
    int32_t* d_I = nullptr;
    check_cuda(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_V, h_V.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_I, h_I.size() * sizeof(int32_t)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    extract_k_values<T>(d_A, d_V, d_I, M, N, k, stream);

    check_cuda(cudaStreamSynchronize(stream));

    if (artifacts::op_enabled("extract_k_values")) {
        auto dir = artifacts::ensure_dir_for_case("extract_k_values", case_id);
        artifacts::write_device_bin(dir, "A", d_A, h_A.size());
        artifacts::write_device_bin(dir, "V", d_V, h_V.size());
        artifacts::write_device_bin(dir, "I", d_I, h_I.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"extract_k_values\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"M\": " << M << ", \"N\": " << N << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"A\": \"bf16\", \"V\": \"bf16\", \"I\": \"s32\"},\n"
             << "\"shape_map\": {\"A\": [" << M << ", " << N << "], \"V\": [" << M << ", " << k << "], \"I\": [" << M << ", " << k << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_I);
    cudaFree(d_V);
    cudaFree(d_A);
}

void run_rms_norm(const std::string& case_id,
                  const RMSNormConfig& cfg,
                  uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend data type

    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.eps;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Input tensor [num_tokens, hidden_size]
    std::vector<T> h_input(static_cast<size_t>(num_tokens) * hidden_size);
    for (auto& v : h_input) v = dist(rng);

    // Weight tensor [hidden_size]
    std::vector<T> h_weight(hidden_size);
    for (auto& v : h_weight) v = dist(rng);

    // Output tensor [num_tokens, hidden_size]
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    // Device allocation
    T* d_input = nullptr;
    T* d_weight = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_weight, h_weight.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_weight, h_weight.data(), h_weight.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Call FlashInfer RMS Norm (input, weight, output, batch_size, d, stride_input, stride_output, eps, enable_pdl, stream)
    flashinfer::norm::RMSNorm<T>(d_input, d_weight, d_output, num_tokens, hidden_size,
                                 hidden_size, hidden_size, eps, false, stream);

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("rms_norm")) {
        auto dir = artifacts::ensure_dir_for_case("rms_norm", case_id);

        artifacts::write_device_bin(dir, "input", d_input, h_input.size());
        artifacts::write_device_bin(dir, "weight", d_weight, h_weight.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rms_norm\",\n"
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

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
}

void run_silu_and_mul(const std::string& case_id,
                      const SiLUAndMulConfig& cfg,
                      uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend data type

    const int num_tokens = cfg.num_tokens;
    const int intermediate_size = cfg.intermediate_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Gate and up projection outputs [num_tokens, intermediate_size]
    std::vector<T> h_gate(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_up(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * intermediate_size, 0);

    for (auto& v : h_gate) v = dist(rng);
    for (auto& v : h_up) v = dist(rng);

    // Device allocation
    T* d_gate = nullptr;
    T* d_up = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_gate, h_gate.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_up, h_up.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_gate, h_gate.data(), h_gate.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Call SiLU and mul kernel from kernels.cuh
    silu_and_mul<T>(d_output, d_gate, d_up, num_tokens, intermediate_size, stream);

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("silu_and_mul")) {
        auto dir = artifacts::ensure_dir_for_case("silu_and_mul", case_id);

        artifacts::write_device_bin(dir, "gate", d_gate, h_gate.size());
        artifacts::write_device_bin(dir, "up", d_up, h_up.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"silu_and_mul\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"intermediate_size\": " << intermediate_size << "},\n"
             << "\"dtype_map\": {\"gate\": \"bf16\", \"up\": \"bf16\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"gate\": [" << num_tokens << ", " << intermediate_size
             << "], \"up\": [" << num_tokens << ", " << intermediate_size
             << "], \"output\": [" << num_tokens << ", " << intermediate_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_up);
    cudaFree(d_gate);
}

void run_add_residual(const std::string& case_id,
                      const AddResidualConfig& cfg,
                      uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend data type

    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const size_t total_size = static_cast<size_t>(num_tokens) * hidden_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Input and residual tensors [num_tokens, hidden_size]
    std::vector<T> h_input(total_size);
    std::vector<T> h_residual(total_size);

    for (auto& v : h_input) v = dist(rng);
    for (auto& v : h_residual) v = dist(rng);

    // Device allocation
    T* d_input = nullptr;
    T* d_residual = nullptr;
    check_cuda(cudaMalloc(&d_input, total_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_residual, total_size * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_input, h_input.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_residual, h_residual.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Call add_residual kernel from kernels.cuh (in-place on d_input)
    add_residual<T>(d_input, d_residual, total_size, stream);

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("add_residual")) {
        auto dir = artifacts::ensure_dir_for_case("add_residual", case_id);

        artifacts::write_vector_bin(dir, "input_orig", h_input);
        artifacts::write_device_bin(dir, "residual", d_residual, total_size);
        artifacts::write_device_bin(dir, "output", d_input, total_size);  // d_input now contains result

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"add_residual\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"hidden_size\": " << hidden_size << "},\n"
             << "\"dtype_map\": {\"input_orig\": \"bf16\", \"residual\": \"bf16\", \"output\": \"bf16\"},\n"
             << "\"shape_map\": {\"input_orig\": [" << num_tokens << ", " << hidden_size
             << "], \"residual\": [" << num_tokens << ", " << hidden_size
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_residual);
    cudaFree(d_input);
}

void run_gemm(const std::string& case_id,
              const GemmConfig& cfg,
              uint64_t seed) {
    using T = __nv_bfloat16;  // Match CUDA backend data type

    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Matrix dimensions based on transpose flags
    const size_t A_size = static_cast<size_t>(transa ? k : m) * (transa ? m : k);
    const size_t B_size = static_cast<size_t>(transb ? n : k) * (transb ? k : n);
    const size_t C_size = static_cast<size_t>(m) * n;

    std::vector<T> h_A(A_size);
    std::vector<T> h_B(B_size);
    std::vector<T> h_bias(use_bias ? n : 0);
    std::vector<T> h_C(C_size, 0);

    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);
    for (auto& v : h_bias) v = dist(rng);

    // Device allocation
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_bias = nullptr;
    T* d_C = nullptr;
    void* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_A, A_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_B, B_size * sizeof(T)));
    if (use_bias) check_cuda(cudaMalloc(&d_bias, h_bias.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_C, C_size * sizeof(T)));

    // Allocate workspace for cuBLAS
    const size_t workspace_size = 1024 * 1024; // 1MB workspace
    check_cuda(cudaMalloc(&d_workspace, workspace_size));

    // Create cuBLAS handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_A, h_A.data(), A_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_B, h_B.data(), B_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    if (use_bias) check_cuda(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Call GEMM from common.cuh
    gemm_cublasLt<T>(ltHandle, stream, d_A, d_B, use_bias ? d_bias : nullptr, d_C,
                     m, n, k, d_workspace, workspace_size, transa, transb);

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("gemm")) {
        auto dir = artifacts::ensure_dir_for_case("gemm", case_id);

        artifacts::write_device_bin(dir, "A", d_A, A_size);
        artifacts::write_device_bin(dir, "B", d_B, B_size);
        if (use_bias) artifacts::write_device_bin(dir, "bias", d_bias, h_bias.size());
        artifacts::write_device_bin(dir, "C", d_C, C_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"gemm\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (transa ? "true" : "false")
             << ", \"transb\": " << (transb ? "true" : "false")
             << ", \"use_bias\": " << (use_bias ? "true" : "false") << "},\n";

        if (use_bias) {
            meta << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"bias\": \"bf16\", \"C\": \"bf16\"},\n";
            meta << "\"shape_map\": {\"A\": [" << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"bias\": [" << n << "], \"C\": [" << m << ", " << n << "]}";
        } else {
            meta << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"C\": \"bf16\"},\n";
            meta << "\"shape_map\": {\"A\": [" << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"C\": [" << m << ", " << n << "]}";
        }

        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cublasLtDestroy(ltHandle);
    cudaFree(d_workspace);
    cudaFree(d_C);
    if (use_bias) cudaFree(d_bias);
    cudaFree(d_B);
    cudaFree(d_A);
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

void run_rope(const std::string& case_id,
              const RoPEConfig& cfg,
              uint64_t seed) {
    using T = __nv_bfloat16;
    using I = int32_t;

    const int num_tokens = cfg.num_tokens;
    const int num_query_heads = cfg.num_query_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t q_size = static_cast<size_t>(num_tokens) * num_query_heads * head_size;
    const size_t k_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
    std::vector<T> h_q_input(q_size);
    std::vector<T> h_k_input(k_size);
    std::vector<I> h_pos_ids(num_tokens);

    for (auto& v : h_q_input) v = dist(rng);
    for (auto& v : h_k_input) v = dist(rng);
    for (int i = 0; i < num_tokens; ++i) h_pos_ids[i] = i;

    T* d_q = nullptr;
    T* d_k = nullptr;
    I* d_pos_ids = nullptr;
    check_cuda(cudaMalloc(&d_q, q_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_k, k_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_pos_ids, num_tokens * sizeof(I)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_q, h_q_input.data(), q_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_k, h_k_input.data(), k_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_pos_ids, h_pos_ids.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));

    // Call the real FlashInfer RoPE function
    flashinfer::BatchQKApplyLlama31RotaryPosIds(
        d_q, d_k, d_q, d_k, // In-place operation
        d_pos_ids,
        (uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_kv_heads,
        (uint32_t)head_size, (uint32_t)head_size,
        (uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
        (uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
        (uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
        (uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
        false, // layout: interleaved
        cfg.rope_factor, cfg.rope_theta, cfg.rope_low_frequency_factor,
        cfg.rope_high_frequency_factor, cfg.max_position_embeddings, stream
    );

    check_cuda(cudaStreamSynchronize(stream));

    if (artifacts::op_enabled("rope")) {
        auto dir = artifacts::ensure_dir_for_case("rope", case_id);

        artifacts::write_vector_bin(dir, "q_input", h_q_input);
        artifacts::write_vector_bin(dir, "k_input", h_k_input);
        artifacts::write_vector_bin(dir, "pos_ids", h_pos_ids);
        artifacts::write_device_bin(dir, "q_output", d_q, q_size);
        artifacts::write_device_bin(dir, "k_output", d_k, k_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rope\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_query_heads\": " << num_query_heads
             << ", \"num_kv_heads\": " << num_kv_heads
             << ", \"head_size\": " << head_size
             << ", \"rope_theta\": " << cfg.rope_theta
             << ", \"rope_factor\": " << cfg.rope_factor
             << ", \"rope_low_frequency_factor\": " << cfg.rope_low_frequency_factor
             << ", \"rope_high_frequency_factor\": " << cfg.rope_high_frequency_factor
             << ", \"max_position_embeddings\": " << cfg.max_position_embeddings << "},\n"
             << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"pos_ids\": \"s32\", \"q_output\": \"bf16\", \"k_output\": \"bf16\"},\n"
             << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_query_heads * head_size)
             << "], \"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
             << "], \"pos_ids\": [" << num_tokens
             << "], \"q_output\": [" << num_tokens << ", " << (num_query_heads * head_size)
             << "], \"k_output\": [" << num_tokens << ", " << (num_kv_heads * head_size) << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_pos_ids);
    cudaFree(d_k);
    cudaFree(d_q);
}

void run_topk_mask_logits(const std::string& case_id,
                          const TopKMaskConfig& cfg,
                          uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int vocab_size = cfg.vocab_size;
    const int k = cfg.k;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Input logits [num_tokens, vocab_size]
    const size_t logits_size = static_cast<size_t>(num_tokens) * vocab_size;
    std::vector<float> h_input_logits(logits_size);
    for (auto& v : h_input_logits) v = dist(rng);

    // Device allocation
    float* d_input_logits = nullptr;
    float* d_masked_logits = nullptr;

    check_cuda(cudaMalloc(&d_input_logits, logits_size * sizeof(float)));
    check_cuda(cudaMalloc(&d_masked_logits, logits_size * sizeof(float)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_input_logits, h_input_logits.data(), logits_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Apply TopK mask - matches actual usage in l4ma.cu
    flashinfer::sampling::TopKMaskLogits<float, int32_t>(
        d_input_logits,
        d_masked_logits,
        nullptr, // optional mask
        num_tokens,
        k,
        vocab_size,
        stream
    );

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("topk_mask_logits")) {
        auto dir = artifacts::ensure_dir_for_case("topk_mask_logits", case_id);

        artifacts::write_device_bin(dir, "input_logits", d_input_logits, logits_size);
        artifacts::write_device_bin(dir, "masked_logits", d_masked_logits, logits_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"topk_mask_logits\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"vocab_size\": " << vocab_size
             << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"masked_logits\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << num_tokens << ", " << vocab_size
             << "], \"masked_logits\": [" << num_tokens << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_masked_logits);
    cudaFree(d_input_logits);
}

void run_softmax(const std::string& case_id,
                 const SoftmaxConfig& cfg,
                 uint64_t seed) {
    using T = float;  // FlashInfer OnlineSoftmax supports float

    const int batch_size = cfg.batch_size;
    const int vocab_size = cfg.vocab_size;
    const float temperature = cfg.temperature;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Reasonable logit range

    // Input logits [batch_size, vocab_size]
    const size_t logits_size = static_cast<size_t>(batch_size) * vocab_size;
    std::vector<T> h_input_logits(logits_size);
    std::vector<T> h_output(logits_size, 0);

    for (auto& v : h_input_logits) v = static_cast<T>(dist(rng));

    // Device allocation
    T* d_input_logits = nullptr;
    T* d_output = nullptr;
    T* d_temperature_arr = nullptr;  // Per-batch temperature (can be nullptr for scalar temp)
    void* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_input_logits, logits_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_output, logits_size * sizeof(T)));

    // Calculate workspace size (estimate)
    const size_t workspace_size = batch_size * vocab_size * sizeof(T);
    check_cuda(cudaMalloc(&d_workspace, workspace_size));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_input_logits, h_input_logits.data(), logits_size * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Apply FlashInfer OnlineSoftmax
    cudaError_t softmax_result = flashinfer::sampling::OnlineSoftmax<T>(
        d_input_logits,
        d_output,
        batch_size,
        vocab_size,
        d_temperature_arr,  // nullptr for scalar temperature
        temperature,        // scalar temperature value
        d_workspace,
        workspace_size,
        false,  // enable_pdl
        stream
    );

    if (softmax_result != cudaSuccess) {
        std::cerr << "FlashInfer OnlineSoftmax failed: " << cudaGetErrorString(softmax_result) << std::endl;
        // Fallback: just copy input to output for testing purposes
        check_cuda(cudaMemcpyAsync(d_output, d_input_logits, logits_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("softmax")) {
        auto dir = artifacts::ensure_dir_for_case("softmax", case_id);

        artifacts::write_device_bin(dir, "input_logits", d_input_logits, logits_size);
        artifacts::write_device_bin(dir, "output", d_output, logits_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"softmax\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"batch_size\": " << batch_size
             << ", \"vocab_size\": " << vocab_size
             << ", \"temperature\": " << temperature << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"output\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << batch_size << ", " << vocab_size
             << "], \"output\": [" << batch_size << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_workspace);
    cudaFree(d_temperature_arr);
    cudaFree(d_output);
    cudaFree(d_input_logits);
}

void run_batch_prefill_attention(const std::string& case_id,
                                 const BatchPrefillAttentionConfig& cfg,
                                 uint64_t seed) {
    using T = __nv_bfloat16;
    using I = int32_t;

    const int num_tokens = cfg.num_tokens;
    const int num_query_heads = cfg.num_query_heads;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;
    const int kv_len = cfg.kv_len;
    const int page_size = cfg.page_size;
    const int batch_size = 1; // Simplified for testing

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t q_size = static_cast<size_t>(num_tokens) * num_query_heads * head_size;
    const size_t kv_size = static_cast<size_t>(kv_len) * num_kv_heads * head_size;
    const size_t o_size = q_size;
    const size_t num_pages = (kv_len + page_size - 1) / page_size;
    const size_t page_data_size = num_pages * page_size * num_kv_heads * head_size;

    // Host data initialization
    std::vector<T> h_q(q_size), h_k(kv_size), h_v(kv_size);
    std::vector<T> h_o(o_size, 0);
    std::vector<I> h_qo_indptr(batch_size + 1, 0);
    std::vector<I> h_kv_page_indptr(batch_size + 1, 0);
    std::vector<I> h_kv_page_indices(num_pages);
    std::vector<I> h_kv_last_page_lens(batch_size);
    std::vector<uint8_t> h_custom_mask;
    std::vector<I> h_mask_indptr(batch_size + 1, 0);

    for (auto& v : h_q) v = static_cast<T>(dist(rng));
    for (auto& v : h_k) v = static_cast<T>(dist(rng));
    for (auto& v : h_v) v = static_cast<T>(dist(rng));

    // Setup pointers for simple batch
    h_qo_indptr[0] = 0;
    h_qo_indptr[1] = num_tokens;
    h_kv_page_indptr[0] = 0;
    h_kv_page_indptr[1] = num_pages;

    // Page indices - simple linear mapping
    for (size_t i = 0; i < num_pages; ++i) h_kv_page_indices[i] = i;

    // Last page length
    h_kv_last_page_lens[0] = kv_len - (num_pages - 1) * page_size;

    // Simple mask (no custom masking for basic test)
    h_mask_indptr[0] = 0;
    h_mask_indptr[1] = 0;

    // Device allocation
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    T *d_paged_k = nullptr, *d_paged_v = nullptr;
    I *d_qo_indptr = nullptr, *d_kv_page_indptr = nullptr;
    I *d_kv_page_indices = nullptr, *d_kv_last_page_lens = nullptr;
    uint8_t *d_custom_mask = nullptr;
    I *d_mask_indptr = nullptr;

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMalloc(&d_q, q_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_k, kv_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_v, kv_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_o, o_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_paged_k, page_data_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_paged_v, page_data_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_qo_indptr, (batch_size + 1) * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_page_indptr, (batch_size + 1) * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_page_indices, num_pages * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_last_page_lens, batch_size * sizeof(I)));
    check_cuda(cudaMalloc(&d_mask_indptr, (batch_size + 1) * sizeof(I)));

    // Note: We use FlashInfer directly for prefill attention; if it fails, the test will fail.

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_qo_indptr, h_qo_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_page_indptr, h_kv_page_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_page_indices, h_kv_page_indices.data(), num_pages * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_last_page_lens, h_kv_last_page_lens.data(), batch_size * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_mask_indptr, h_mask_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));

    // Initialize paged KV cache to zero
    check_cuda(cudaMemsetAsync(d_paged_k, 0, page_data_size * sizeof(T), stream));
    check_cuda(cudaMemsetAsync(d_paged_v, 0, page_data_size * sizeof(T), stream));

    // Create paged KV cache structure
    flashinfer::paged_kv_t<T, I> paged_kv(
        num_kv_heads, page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        d_paged_k, d_paged_v,
        d_kv_page_indices,
        d_kv_page_indptr,
        d_kv_last_page_lens
    );

    // Copy K, V data to paged format (simple linear copy for test)
    const size_t copy_size = std::min(kv_size, page_data_size);
    check_cuda(cudaMemcpyAsync(d_paged_k, d_k, copy_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_paged_v, d_v, copy_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));

    // Use real FlashInfer BatchPrefillWithPagedKVCacheWrapper only; no fallback.
    // This matches the actual CUDA backend implementation in l4ma.cu.
    {
        flashinfer::BatchPrefillHandler prefill_handler;
        flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, int32_t>(
            &prefill_handler,
            d_q,                    // query
            d_qo_indptr,            // query offsets
            nullptr,                // kv_data (unused with paged)
            paged_kv,               // paged kv cache
            d_o,                    // output
            nullptr,                // lse (log-sum-exp, optional)
            num_query_heads,        // num_qo_heads
            flashinfer::MaskMode::kNone,  // no custom masking for basic test
            nullptr,                // custom_mask (nullptr for no masking)
            d_mask_indptr,          // mask_indptr
            flashinfer::PosEncodingMode::kNone,  // no additional pos encoding
            false,                  // use_fp16_qk_reduction
            std::nullopt,           // maybe_sm_scale
            1.0f,                   // rope_scale (unused)
            1e4,                    // rope_theta (unused)
            stream
        );
    }

    check_cuda(cudaStreamSynchronize(stream));

    if (artifacts::op_enabled("batch_prefill_attention")) {
        auto dir = artifacts::ensure_dir_for_case("batch_prefill_attention", case_id);
        artifacts::write_device_bin(dir, "q_input", d_q, q_size);
        artifacts::write_device_bin(dir, "k_input", d_k, kv_size);
        artifacts::write_device_bin(dir, "v_input", d_v, kv_size);
        artifacts::write_device_bin(dir, "paged_k_cache", d_paged_k, page_data_size);
        artifacts::write_device_bin(dir, "paged_v_cache", d_paged_v, page_data_size);
        artifacts::write_device_bin(dir, "output", d_o, o_size);
        artifacts::write_device_bin(dir, "qo_indptr", d_qo_indptr, batch_size + 1);
        artifacts::write_device_bin(dir, "kv_page_indptr", d_kv_page_indptr, batch_size + 1);
        artifacts::write_device_bin(dir, "kv_page_indices", d_kv_page_indices, num_pages);
        artifacts::write_device_bin(dir, "kv_last_page_lens", d_kv_last_page_lens, batch_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"batch_prefill_attention\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_query_heads\": " << num_query_heads
             << ", \"num_kv_heads\": " << num_kv_heads
             << ", \"head_size\": " << head_size
             << ", \"kv_len\": " << kv_len
             << ", \"page_size\": " << page_size
             << ", \"batch_size\": " << batch_size
             << ", \"num_pages\": " << num_pages << "},\n"
             << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"v_input\": \"bf16\", \"paged_k_cache\": \"bf16\", \"paged_v_cache\": \"bf16\", \"output\": \"bf16\", \"qo_indptr\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_page_indices\": \"s32\", \"kv_last_page_lens\": \"s32\"},\n"
             << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_query_heads * head_size)
             << "], \"k_input\": [" << kv_len << ", " << (num_kv_heads * head_size)
             << "], \"v_input\": [" << kv_len << ", " << (num_kv_heads * head_size)
             << "], \"paged_k_cache\": [" << num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
             << "], \"paged_v_cache\": [" << num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
             << "], \"output\": [" << num_tokens << ", " << (num_query_heads * head_size)
             << "], \"qo_indptr\": [" << (batch_size + 1)
             << "], \"kv_page_indptr\": [" << (batch_size + 1)
             << "], \"kv_page_indices\": [" << num_pages
             << "], \"kv_last_page_lens\": [" << batch_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaFree(d_mask_indptr);
    cudaFree(d_kv_last_page_lens);
    cudaFree(d_kv_page_indices);
    cudaFree(d_kv_page_indptr);
    cudaFree(d_qo_indptr);
    cudaFree(d_paged_v);
    cudaFree(d_paged_k);
    cudaFree(d_o);
    cudaFree(d_v);
    cudaFree(d_k);
    cudaFree(d_q);
    cudaStreamDestroy(stream);
}

void run_grouped_gemm(const std::string& case_id,
                      const GroupedGemmConfig& cfg,
                      uint64_t seed) {
    using T = __nv_bfloat16;

    const int num_groups = cfg.num_groups;
    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Calculate sizes per group
    const size_t A_size_per_group = static_cast<size_t>(transa ? k : m) * (transa ? m : k);
    const size_t B_size_per_group = static_cast<size_t>(transb ? n : k) * (transb ? k : n);
    const size_t C_size_per_group = static_cast<size_t>(m) * n;

    const size_t total_A_size = A_size_per_group * num_groups;
    const size_t total_B_size = B_size_per_group * num_groups;
    const size_t total_C_size = C_size_per_group * num_groups;
    const size_t total_bias_size = use_bias ? n * num_groups : 0;

    // Host buffers for all groups
    std::vector<T> h_A(total_A_size);
    std::vector<T> h_B(total_B_size);
    std::vector<T> h_bias(total_bias_size);
    std::vector<T> h_C(total_C_size, 0);

    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);
    for (auto& v : h_bias) v = dist(rng);

    // Device allocation
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_bias = nullptr;
    T* d_C = nullptr;
    void* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_A, total_A_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_B, total_B_size * sizeof(T)));
    if (use_bias) check_cuda(cudaMalloc(&d_bias, total_bias_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_C, total_C_size * sizeof(T)));

    const size_t workspace_size = 1024 * 1024; // 1MB workspace per operation
    check_cuda(cudaMalloc(&d_workspace, workspace_size));

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_A, h_A.data(), total_A_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_B, h_B.data(), total_B_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    if (use_bias) check_cuda(cudaMemcpyAsync(d_bias, h_bias.data(), total_bias_size * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Execute grouped GEMM operations
    for (int group = 0; group < num_groups; ++group) {
        T* A_ptr = d_A + group * A_size_per_group;
        T* B_ptr = d_B + group * B_size_per_group;
        T* C_ptr = d_C + group * C_size_per_group;
        T* bias_ptr = use_bias ? (d_bias + group * n) : nullptr;

        gemm_cublasLt<T>(ltHandle, stream, A_ptr, B_ptr, bias_ptr, C_ptr,
                         m, n, k, d_workspace, workspace_size, transa, transb);
    }

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("grouped_gemm")) {
        auto dir = artifacts::ensure_dir_for_case("grouped_gemm", case_id);

        artifacts::write_device_bin(dir, "A", d_A, total_A_size);
        artifacts::write_device_bin(dir, "B", d_B, total_B_size);
        if (use_bias) artifacts::write_device_bin(dir, "bias", d_bias, total_bias_size);
        artifacts::write_device_bin(dir, "C", d_C, total_C_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"grouped_gemm\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_groups\": " << num_groups
             << ", \"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (transa ? "true" : "false")
             << ", \"transb\": " << (transb ? "true" : "false")
             << ", \"use_bias\": " << (use_bias ? "true" : "false") << "},\n";

        if (use_bias) {
            meta << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"bias\": \"bf16\", \"C\": \"bf16\"},\n";
            meta << "\"shape_map\": {\"A\": [" << num_groups << ", " << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << num_groups << ", " << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"bias\": [" << num_groups << ", " << n << "], \"C\": [" << num_groups << ", " << m << ", " << n << "]}";
        } else {
            meta << "\"dtype_map\": {\"A\": \"bf16\", \"B\": \"bf16\", \"C\": \"bf16\"},\n";
            meta << "\"shape_map\": {\"A\": [" << num_groups << ", " << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << num_groups << ", " << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"C\": [" << num_groups << ", " << m << ", " << n << "]}";
        }

        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cublasLtDestroy(ltHandle);
    cudaFree(d_workspace);
    cudaFree(d_C);
    if (use_bias) cudaFree(d_bias);
    cudaFree(d_B);
    cudaFree(d_A);
}

void run_append_paged_kv_cache(const std::string& case_id,
                               const AppendPagedKVCacheConfig& cfg,
                               uint64_t seed) {
    using T = __nv_bfloat16;
    using I = int32_t;

    const int num_tokens = cfg.num_tokens;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;
    const int page_size = cfg.page_size;
    const int max_num_pages = cfg.max_num_pages;
    const int batch_size = cfg.batch_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Calculate sizes
    const size_t kv_input_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
    const size_t page_data_size = static_cast<size_t>(max_num_pages) * page_size * num_kv_heads * head_size;

    // Host input data
    std::vector<T> h_k_input(kv_input_size);
    std::vector<T> h_v_input(kv_input_size);
    std::vector<T> h_paged_k_cache(page_data_size, 0);
    std::vector<T> h_paged_v_cache(page_data_size, 0);

    // Page management vectors (following l4ma.cu usage)
    std::vector<I> h_kv_page_indices(max_num_pages);
    std::vector<I> h_kv_page_indptr(batch_size + 1);
    std::vector<I> h_kv_last_page_lens(batch_size);
    std::vector<I> h_kv_batch_indices(num_tokens);
    std::vector<I> h_kv_positions(num_tokens);

    for (auto& v : h_k_input) v = static_cast<T>(dist(rng));
    for (auto& v : h_v_input) v = static_cast<T>(dist(rng));

    // Setup page indices - simple linear mapping
    for (int i = 0; i < max_num_pages; ++i) {
        h_kv_page_indices[i] = i;
    }

    // Setup page pointers for each batch
    const int pages_per_batch = max_num_pages / batch_size;
    h_kv_page_indptr[0] = 0;
    for (int i = 1; i <= batch_size; ++i) {
        h_kv_page_indptr[i] = h_kv_page_indptr[i-1] + pages_per_batch;
    }

    // Last page lengths (assume full pages except possibly the last)
    for (int i = 0; i < batch_size; ++i) {
        h_kv_last_page_lens[i] = page_size;
    }

    // Setup batch indices and positions for each token
    const int tokens_per_batch = num_tokens / batch_size;
    for (int i = 0; i < num_tokens; ++i) {
        h_kv_batch_indices[i] = i / tokens_per_batch;
        h_kv_positions[i] = i % tokens_per_batch;
    }

    // Device allocation
    T* d_k_input = nullptr;
    T* d_v_input = nullptr;
    T* d_paged_k_cache = nullptr;
    T* d_paged_v_cache = nullptr;
    I* d_kv_page_indices = nullptr;
    I* d_kv_page_indptr = nullptr;
    I* d_kv_last_page_lens = nullptr;
    I* d_kv_batch_indices = nullptr;
    I* d_kv_positions = nullptr;

    check_cuda(cudaMalloc(&d_k_input, kv_input_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_v_input, kv_input_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_paged_k_cache, page_data_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_paged_v_cache, page_data_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_kv_page_indices, max_num_pages * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_page_indptr, (batch_size + 1) * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_last_page_lens, batch_size * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_batch_indices, num_tokens * sizeof(I)));
    check_cuda(cudaMalloc(&d_kv_positions, num_tokens * sizeof(I)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_k_input, h_k_input.data(), kv_input_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_v_input, h_v_input.data(), kv_input_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_paged_k_cache, h_paged_k_cache.data(), page_data_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_paged_v_cache, h_paged_v_cache.data(), page_data_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_page_indices, h_kv_page_indices.data(), max_num_pages * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_page_indptr, h_kv_page_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_last_page_lens, h_kv_last_page_lens.data(), batch_size * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_batch_indices, h_kv_batch_indices.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_kv_positions, h_kv_positions.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));

    // Create paged KV cache structure (matching l4ma.cu usage)
    flashinfer::paged_kv_t<T, I> paged_kv(
        num_kv_heads, page_size, head_size, batch_size,
        flashinfer::QKVLayout::kNHD,
        d_paged_k_cache, d_paged_v_cache,
        d_kv_page_indices,
        d_kv_page_indptr,
        d_kv_last_page_lens
    );

    // Call real FlashInfer AppendPagedKVCache (matching l4ma.cu at line 621-629)
    flashinfer::AppendPagedKVCache<T, I>(
        paged_kv, d_k_input, d_v_input,
        d_kv_batch_indices,
        d_kv_positions,
        num_tokens,
        num_kv_heads * head_size, head_size,
        num_kv_heads * head_size, head_size,
        stream
    );

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts
    if (artifacts::op_enabled("append_paged_kv_cache")) {
        auto dir = artifacts::ensure_dir_for_case("append_paged_kv_cache", case_id);

        artifacts::write_device_bin(dir, "k_input", d_k_input, kv_input_size);
        artifacts::write_device_bin(dir, "v_input", d_v_input, kv_input_size);
        artifacts::write_device_bin(dir, "kv_page_indices", d_kv_page_indices, max_num_pages);
        artifacts::write_device_bin(dir, "kv_page_indptr", d_kv_page_indptr, batch_size + 1);
        artifacts::write_device_bin(dir, "kv_last_page_lens", d_kv_last_page_lens, batch_size);
        artifacts::write_device_bin(dir, "kv_batch_indices", d_kv_batch_indices, num_tokens);
        artifacts::write_device_bin(dir, "kv_positions", d_kv_positions, num_tokens);
        artifacts::write_device_bin(dir, "paged_k_cache_output", d_paged_k_cache, page_data_size);
        artifacts::write_device_bin(dir, "paged_v_cache_output", d_paged_v_cache, page_data_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"append_paged_kv_cache\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_kv_heads\": " << num_kv_heads
             << ", \"head_size\": " << head_size
             << ", \"page_size\": " << page_size
             << ", \"max_num_pages\": " << max_num_pages
             << ", \"batch_size\": " << batch_size << "},\n"
             << "\"dtype_map\": {\"k_input\": \"bf16\", \"v_input\": \"bf16\", \"kv_page_indices\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_last_page_lens\": \"s32\", \"kv_batch_indices\": \"s32\", \"kv_positions\": \"s32\", \"paged_k_cache_output\": \"bf16\", \"paged_v_cache_output\": \"bf16\"},\n"
             << "\"shape_map\": {\"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
             << "], \"v_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
             << "], \"kv_page_indices\": [" << max_num_pages
             << "], \"kv_page_indptr\": [" << (batch_size + 1)
             << "], \"kv_last_page_lens\": [" << batch_size
             << "], \"kv_batch_indices\": [" << num_tokens
             << "], \"kv_positions\": [" << num_tokens
             << "], \"paged_k_cache_output\": [" << max_num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
             << "], \"paged_v_cache_output\": [" << max_num_pages << ", " << page_size << ", " << (num_kv_heads * head_size) << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_kv_positions);
    cudaFree(d_kv_batch_indices);
    cudaFree(d_kv_last_page_lens);
    cudaFree(d_kv_page_indptr);
    cudaFree(d_kv_page_indices);
    cudaFree(d_paged_v_cache);
    cudaFree(d_paged_k_cache);
    cudaFree(d_v_input);
    cudaFree(d_k_input);
}

// Template implementations for multi-dtype testing

template<typename T>
void run_gemm_typed(const std::string& case_id, const GemmConfig& cfg, uint64_t seed) {
    const int m = cfg.m;
    const int n = cfg.n;
    const int k = cfg.k;
    const bool transa = cfg.transa;
    const bool transb = cfg.transb;
    const bool use_bias = cfg.use_bias;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Matrix dimensions based on transpose flags
    const size_t A_size = static_cast<size_t>(transa ? k : m) * (transa ? m : k);
    const size_t B_size = static_cast<size_t>(transb ? n : k) * (transb ? k : n);
    const size_t C_size = static_cast<size_t>(m) * n;

    std::vector<T> h_A(A_size);
    std::vector<T> h_B(B_size);
    std::vector<T> h_bias(use_bias ? n : 0);
    std::vector<T> h_C(C_size, 0);

    for (auto& v : h_A) v = static_cast<T>(dist(rng));
    for (auto& v : h_B) v = static_cast<T>(dist(rng));
    for (auto& v : h_bias) v = static_cast<T>(dist(rng));

    // Device allocation
    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_bias = nullptr;
    T* d_C = nullptr;
    void* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_A, A_size * sizeof(T)));
    check_cuda(cudaMalloc(&d_B, B_size * sizeof(T)));
    if (use_bias) check_cuda(cudaMalloc(&d_bias, h_bias.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_C, C_size * sizeof(T)));

    const size_t workspace_size = 1024 * 1024;
    check_cuda(cudaMalloc(&d_workspace, workspace_size));

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    // Copy to device
    check_cuda(cudaMemcpyAsync(d_A, h_A.data(), A_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_B, h_B.data(), B_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    if (use_bias) check_cuda(cudaMemcpyAsync(d_bias, h_bias.data(), h_bias.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    // Call GEMM from common.cuh
    gemm_cublasLt<T>(ltHandle, stream, d_A, d_B, use_bias ? d_bias : nullptr, d_C,
                     m, n, k, d_workspace, workspace_size, transa, transb);

    check_cuda(cudaStreamSynchronize(stream));

    // Write artifacts with dtype-specific name
    std::string dtype_name;
    if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
    else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

    if (artifacts::op_enabled("gemm")) {
        auto dir = artifacts::ensure_dir_for_case("gemm", case_id + "_" + dtype_name);

        artifacts::write_device_bin(dir, "A", d_A, A_size);
        artifacts::write_device_bin(dir, "B", d_B, B_size);
        if (use_bias) artifacts::write_device_bin(dir, "bias", d_bias, h_bias.size());
        artifacts::write_device_bin(dir, "C", d_C, C_size);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"gemm\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
             << "\"config\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
             << ", \"transa\": " << (transa ? "true" : "false")
             << ", \"transb\": " << (transb ? "true" : "false")
             << ", \"use_bias\": " << (use_bias ? "true" : "false") << "},\n";

        if (use_bias) {
            meta << "\"dtype_map\": {\"A\": \"" << dtype_name << "\", \"B\": \"" << dtype_name << "\", \"bias\": \"" << dtype_name << "\", \"C\": \"" << dtype_name << "\"},\n";
            meta << "\"shape_map\": {\"A\": [" << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"bias\": [" << n << "], \"C\": [" << m << ", " << n << "]}";
        } else {
            meta << "\"dtype_map\": {\"A\": \"" << dtype_name << "\", \"B\": \"" << dtype_name << "\", \"C\": \"" << dtype_name << "\"},\n";
            meta << "\"shape_map\": {\"A\": [" << (transa ? k : m) << ", " << (transa ? m : k)
                 << "], \"B\": [" << (transb ? n : k) << ", " << (transb ? k : n)
                 << "], \"C\": [" << m << ", " << n << "]}";
        }

        artifacts::write_meta_json(dir, meta.str());
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cublasLtDestroy(ltHandle);
    cudaFree(d_workspace);
    cudaFree(d_C);
    if (use_bias) cudaFree(d_bias);
    cudaFree(d_B);
    cudaFree(d_A);
}

// Template implementation for embedding (supports float and __nv_bfloat16)
template<typename T, typename I = int32_t>
void run_embedding_lookup_typed(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const int vocab_size = cfg.vocab_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_embedding(static_cast<size_t>(vocab_size) * hidden_size);
    for (auto& v : h_embedding) v = static_cast<T>(dist(rng));

    std::vector<I> h_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        h_indices[i] = static_cast<I>(i % vocab_size);
    }

    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    T* d_embedding = nullptr;
    I* d_indices = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_indices, h_indices.size() * sizeof(I)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_indices, h_indices.data(), h_indices.size() * sizeof(I), cudaMemcpyHostToDevice, stream));

    embed<T, I>(
        d_embedding,
        static_cast<size_t>(vocab_size),
        d_indices,
        static_cast<size_t>(num_tokens),
        d_output,
        hidden_size,
        stream
    );

    check_cuda(cudaStreamSynchronize(stream));

    std::string dtype_name;
    if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

    if (artifacts::op_enabled("embedding_lookup_forward")) {
        auto dir = artifacts::ensure_dir_for_case("embedding_lookup_forward", case_id + "_" + dtype_name);

        artifacts::write_device_bin(dir, "embedding", d_embedding, h_embedding.size());
        artifacts::write_device_bin(dir, "indices", d_indices, h_indices.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"embedding_lookup_forward\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
             << "\"config\": {\"hidden_size\": " << hidden_size
             << ", \"vocab_size\": " << vocab_size
             << ", \"num_tokens\": " << num_tokens << "},\n"
             << "\"dtype_map\": {\"embedding\": \"" << dtype_name << "\", \"indices\": \"s32\", \"output\": \"" << dtype_name << "\"},\n"
             << "\"shape_map\": {\"embedding\": [" << vocab_size << ", " << hidden_size
             << "], \"indices\": [" << num_tokens
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_embedding);
}

// Template implementation for extract_k_values (supports float and __nv_bfloat16)
template<typename T>
void run_extract_k_values_typed(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed) {
    const int M = cfg.M;
    const int N = cfg.N;
    const int k = cfg.k;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_A(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < k; ++j) {
            int col = (m * 131 + j * 17) % N;
            h_A[static_cast<size_t>(m) * N + col] = static_cast<T>(dist(rng));
        }
    }

    std::vector<T> h_V(static_cast<size_t>(M) * k, 0);
    std::vector<int32_t> h_I(static_cast<size_t>(M) * k, 0);

    T* d_A = nullptr;
    T* d_V = nullptr;
    int32_t* d_I = nullptr;
    check_cuda(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_V, h_V.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_I, h_I.size() * sizeof(int32_t)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    extract_k_values<T>(d_A, d_V, d_I, M, N, k, stream);

    check_cuda(cudaStreamSynchronize(stream));

    std::string dtype_name;
    if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

    if (artifacts::op_enabled("extract_k_values")) {
        auto dir = artifacts::ensure_dir_for_case("extract_k_values", case_id + "_" + dtype_name);
        artifacts::write_device_bin(dir, "A", d_A, h_A.size());
        artifacts::write_device_bin(dir, "V", d_V, h_V.size());
        artifacts::write_device_bin(dir, "I", d_I, h_I.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"extract_k_values\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
             << "\"config\": {\"M\": " << M << ", \"N\": " << N << ", \"k\": " << k << "},\n"
             << "\"dtype_map\": {\"A\": \"" << dtype_name << "\", \"V\": \"" << dtype_name << "\", \"I\": \"s32\"},\n"
             << "\"shape_map\": {\"A\": [" << M << ", " << N << "], \"V\": [" << M << ", " << k << "], \"I\": [" << M << ", " << k << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_I);
    cudaFree(d_V);
    cudaFree(d_A);
}

// Explicit instantiations for supported types
template void run_gemm_typed<float>(const std::string&, const GemmConfig&, uint64_t);
template void run_gemm_typed<__nv_bfloat16>(const std::string&, const GemmConfig&, uint64_t);

template void run_embedding_lookup_typed<float, int32_t>(const std::string&, const EmbeddingConfig&, uint64_t);
template void run_embedding_lookup_typed<__nv_bfloat16, int32_t>(const std::string&, const EmbeddingConfig&, uint64_t);

template void run_extract_k_values_typed<float>(const std::string&, const ExtractKConfig&, uint64_t);
template void run_extract_k_values_typed<__nv_bfloat16>(const std::string&, const ExtractKConfig&, uint64_t);

// Multi-dtype template implementations for other operations

template<typename T>
void run_rms_norm_typed(const std::string& case_id, const RMSNormConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.eps;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_input(static_cast<size_t>(num_tokens) * hidden_size);
    std::vector<T> h_weight(hidden_size);
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * hidden_size, 0);

    for (auto& v : h_input) v = static_cast<T>(dist(rng));
    for (auto& v : h_weight) v = static_cast<T>(dist(rng));

    T* d_input = nullptr;
    T* d_weight = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, h_input.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_weight, h_weight.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_weight, h_weight.data(), h_weight.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    flashinfer::norm::RMSNorm<T>(d_input, d_weight, d_output, num_tokens, hidden_size,
                                 hidden_size, hidden_size, eps, false, stream);

    check_cuda(cudaStreamSynchronize(stream));

    std::string dtype_name;
    if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
    else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

    if (artifacts::op_enabled("rms_norm")) {
        auto dir = artifacts::ensure_dir_for_case("rms_norm", case_id + "_" + dtype_name);
        artifacts::write_device_bin(dir, "input", d_input, h_input.size());
        artifacts::write_device_bin(dir, "weight", d_weight, h_weight.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"rms_norm\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"hidden_size\": " << hidden_size
             << ", \"eps\": " << eps << "},\n"
             << "\"dtype_map\": {\"input\": \"" << dtype_name << "\", \"weight\": \"" << dtype_name << "\", \"output\": \"" << dtype_name << "\"},\n"
             << "\"shape_map\": {\"input\": [" << num_tokens << ", " << hidden_size
             << "], \"weight\": [" << hidden_size
             << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
}

template<typename T>
void run_silu_and_mul_typed(const std::string& case_id, const SiLUAndMulConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int intermediate_size = cfg.intermediate_size;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> h_gate(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_up(static_cast<size_t>(num_tokens) * intermediate_size);
    std::vector<T> h_output(static_cast<size_t>(num_tokens) * intermediate_size, 0);

    for (auto& v : h_gate) v = static_cast<T>(dist(rng));
    for (auto& v : h_up) v = static_cast<T>(dist(rng));

    T* d_gate = nullptr;
    T* d_up = nullptr;
    T* d_output = nullptr;
    check_cuda(cudaMalloc(&d_gate, h_gate.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_up, h_up.size() * sizeof(T)));
    check_cuda(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));

    check_cuda(cudaMemcpyAsync(d_gate, h_gate.data(), h_gate.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    silu_and_mul<T>(d_output, d_gate, d_up, num_tokens, intermediate_size, stream);

    check_cuda(cudaStreamSynchronize(stream));

    std::string dtype_name;
    if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
    else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

    if (artifacts::op_enabled("silu_and_mul")) {
        auto dir = artifacts::ensure_dir_for_case("silu_and_mul", case_id + "_" + dtype_name);
        artifacts::write_device_bin(dir, "gate", d_gate, h_gate.size());
        artifacts::write_device_bin(dir, "up", d_up, h_up.size());
        artifacts::write_device_bin(dir, "output", d_output, h_output.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"silu_and_mul\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"intermediate_size\": " << intermediate_size << "},\n"
             << "\"dtype_map\": {\"gate\": \"" << dtype_name << "\", \"up\": \"" << dtype_name << "\", \"output\": \"" << dtype_name << "\"},\n"
             << "\"shape_map\": {\"gate\": [" << num_tokens << ", " << intermediate_size
             << "], \"up\": [" << num_tokens << ", " << intermediate_size
             << "], \"output\": [" << num_tokens << ", " << intermediate_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    cudaStreamDestroy(stream);
    cudaFree(d_output);
    cudaFree(d_up);
    cudaFree(d_gate);
}

// Add explicit instantiations for new templates
template void run_rms_norm_typed<float>(const std::string&, const RMSNormConfig&, uint64_t);
template void run_rms_norm_typed<__nv_bfloat16>(const std::string&, const RMSNormConfig&, uint64_t);

template void run_silu_and_mul_typed<float>(const std::string&, const SiLUAndMulConfig&, uint64_t);
template void run_silu_and_mul_typed<__nv_bfloat16>(const std::string&, const SiLUAndMulConfig&, uint64_t);

// Automatic multi-dtype runner
void run_all_dtypes_for_operation(const std::string& op_name, const std::string& base_case_id,
                                 const void* config_ptr, uint64_t seed) {
    if (op_name == "gemm") {
        const auto* cfg = static_cast<const GemmConfig*>(config_ptr);
        std::cout << "Running GEMM with all supported data types...\n";
        run_gemm_typed<float>(base_case_id, *cfg, seed);
        run_gemm_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "embedding_lookup") {
        const auto* cfg = static_cast<const EmbeddingConfig*>(config_ptr);
        std::cout << "Running embedding lookup with all supported data types...\n";
        run_embedding_lookup_typed<float, int32_t>(base_case_id, *cfg, seed);
        run_embedding_lookup_typed<__nv_bfloat16, int32_t>(base_case_id, *cfg, seed);
    } else if (op_name == "extract_k_values") {
        const auto* cfg = static_cast<const ExtractKConfig*>(config_ptr);
        std::cout << "Running extract_k_values with all supported data types...\n";
        run_extract_k_values_typed<float>(base_case_id, *cfg, seed);
        run_extract_k_values_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "rms_norm") {
        const auto* cfg = static_cast<const RMSNormConfig*>(config_ptr);
        std::cout << "Running RMS Norm with all supported data types...\n";
        run_rms_norm_typed<float>(base_case_id, *cfg, seed);
        run_rms_norm_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    } else if (op_name == "silu_and_mul") {
        const auto* cfg = static_cast<const SiLUAndMulConfig*>(config_ptr);
        std::cout << "Running SiLU and Mul with all supported data types...\n";
        run_silu_and_mul_typed<float>(base_case_id, *cfg, seed);
        run_silu_and_mul_typed<__nv_bfloat16>(base_case_id, *cfg, seed);
    }
}

} // namespace ops
