#include "../ops.hpp"
#include "ops_common.cuh"
#include "common.cuh"
#include "artifacts.hpp"
#include "../test_kernels/test_kernels.cuh"
#include <random>
#include <sstream>
#include <vector>
#include <type_traits>

namespace ops {

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

	for (auto& v : h_A) v = f2t<T>(dist(rng));
	for (auto& v : h_B) v = f2t<T>(dist(rng));
	for (auto& v : h_bias) v = f2t<T>(dist(rng));

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

// Explicit instantiations
template void run_gemm_typed<float>(const std::string&, const GemmConfig&, uint64_t);
template void run_gemm_typed<__half>(const std::string&, const GemmConfig&, uint64_t);
template void run_gemm_typed<__nv_bfloat16>(const std::string&, const GemmConfig&, uint64_t);

} // namespace ops
