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
void run_extract_k_values_typed(const std::string& case_id, const ExtractKConfig& cfg, uint64_t seed) {
	const int M = cfg.M;
	const int N = cfg.N;
	const int k = cfg.k;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<T> h_A(static_cast<size_t>(M) * N);
	// Initialize all to negative infinity equivalent, then place k values per row
	for (auto& v : h_A) v = f2t<T>(-INFINITY);
	for (int m = 0; m < M; ++m) {
		for (int j = 0; j < k; ++j) {
			int col = (m * 131 + j * 17) % N;
			h_A[static_cast<size_t>(m) * N + col] = f2t<T>(dist(rng));
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

	// Call the implementation - use test-local version for __half
	if constexpr (std::is_same_v<T, __half>) {
		extract_k_values_test_local<T>(d_A, d_V, d_I, M, N, k, stream);
	} else {
		// Use backend implementation for supported types
		extract_k_values<T>(d_A, d_V, d_I, M, N, k, stream);
	}

	check_cuda(cudaStreamSynchronize(stream));

	std::string dtype_name;
	if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
	else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
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

void run_extract_k_values(const std::string& case_id,
						  const ExtractKConfig& cfg,
						  uint64_t seed) {
	using T = __nv_bfloat16;

	const int M = cfg.M;
	const int N = cfg.N;
	const int k = cfg.k;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<T> h_A(static_cast<size_t>(M) * N);
	for (auto& v : h_A) v = static_cast<T>(-INFINITY);
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

// Explicit instantiations
template void run_extract_k_values_typed<float>(const std::string&, const ExtractKConfig&, uint64_t);
template void run_extract_k_values_typed<__half>(const std::string&, const ExtractKConfig&, uint64_t);
template void run_extract_k_values_typed<__nv_bfloat16>(const std::string&, const ExtractKConfig&, uint64_t);

} // namespace ops
