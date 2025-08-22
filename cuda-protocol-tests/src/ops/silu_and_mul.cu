#include "../ops.hpp"
#include "ops_common.cuh"
#include "kernels.cuh"
#include <random>
#include <sstream>
#include <vector>
#include <type_traits>
#include "artifacts.hpp"

namespace ops {

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

	// Call kernel
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

	// Cleanup
	cudaStreamDestroy(stream);
	cudaFree(d_output);
	cudaFree(d_up);
	cudaFree(d_gate);
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

// Explicit instantiations
template void run_silu_and_mul_typed<float>(const std::string&, const SiLUAndMulConfig&, uint64_t);
template void run_silu_and_mul_typed<__half>(const std::string&, const SiLUAndMulConfig&, uint64_t);
template void run_silu_and_mul_typed<__nv_bfloat16>(const std::string&, const SiLUAndMulConfig&, uint64_t);

} // namespace ops
