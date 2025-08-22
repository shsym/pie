#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "../../backend/backend-cuda/src/common.cuh"
#include <random>
#include <sstream>

namespace ops {

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

} // namespace ops
