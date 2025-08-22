#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "kernels.cuh"
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

namespace ops {

template<typename T>
void run_add_residual_typed(const std::string& case_id, const AddResidualConfig& cfg, uint64_t seed) {
	const int num_tokens = cfg.num_tokens;
	const int hidden_size = cfg.hidden_size;
	const size_t total_size = static_cast<size_t>(num_tokens) * hidden_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// Host buffers
	std::vector<T> h_input(total_size);
	std::vector<T> h_residual(total_size);
	for (auto& v : h_input) v = static_cast<T>(dist(rng));
	for (auto& v : h_residual) v = static_cast<T>(dist(rng));

	// Device buffers
	T* d_input = nullptr;
	T* d_residual = nullptr;
	check_cuda(cudaMalloc(&d_input, total_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_residual, total_size * sizeof(T)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	check_cuda(cudaMemcpyAsync(d_input, h_input.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_residual, h_residual.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));

	// In-place add
	add_residual<T>(d_input, d_residual, total_size, stream);
	check_cuda(cudaStreamSynchronize(stream));

	std::string dtype_name;
	if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
	else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
	else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

	if (artifacts::op_enabled("add_residual")) {
		auto dir = artifacts::ensure_dir_for_case("add_residual", case_id + "_" + dtype_name);
		artifacts::write_vector_bin(dir, "input_orig", h_input);
		artifacts::write_device_bin(dir, "residual", d_residual, total_size);
		artifacts::write_device_bin(dir, "output", d_input, total_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"add_residual\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"hidden_size\": " << hidden_size << "},\n"
			 << "\"dtype_map\": {\"input_orig\": \"" << dtype_name << "\", \"residual\": \"" << dtype_name << "\", \"output\": \"" << dtype_name << "\"},\n"
			 << "\"shape_map\": {\"input_orig\": [" << num_tokens << ", " << hidden_size
			 << "], \"residual\": [" << num_tokens << ", " << hidden_size
			 << "], \"output\": [" << num_tokens << ", " << hidden_size << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	cudaStreamDestroy(stream);
	cudaFree(d_residual);
	cudaFree(d_input);
}

void run_add_residual(const std::string& case_id, const AddResidualConfig& cfg, uint64_t seed) {
	using T = __nv_bfloat16;
	const int num_tokens = cfg.num_tokens;
	const int hidden_size = cfg.hidden_size;
	const size_t total_size = static_cast<size_t>(num_tokens) * hidden_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<T> h_input(total_size);
	std::vector<T> h_residual(total_size);
	for (auto& v : h_input) v = dist(rng);
	for (auto& v : h_residual) v = dist(rng);

	T* d_input = nullptr;
	T* d_residual = nullptr;
	check_cuda(cudaMalloc(&d_input, total_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_residual, total_size * sizeof(T)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	check_cuda(cudaMemcpyAsync(d_input, h_input.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_residual, h_residual.data(), total_size * sizeof(T), cudaMemcpyHostToDevice, stream));

	add_residual<T>(d_input, d_residual, total_size, stream);
	check_cuda(cudaStreamSynchronize(stream));

	if (artifacts::op_enabled("add_residual")) {
		auto dir = artifacts::ensure_dir_for_case("add_residual", case_id);
		artifacts::write_vector_bin(dir, "input_orig", h_input);
		artifacts::write_device_bin(dir, "residual", d_residual, total_size);
		artifacts::write_device_bin(dir, "output", d_input, total_size);

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

	cudaStreamDestroy(stream);
	cudaFree(d_residual);
	cudaFree(d_input);
}

// Explicit instantiations
template void run_add_residual_typed<float>(const std::string&, const AddResidualConfig&, uint64_t);
template void run_add_residual_typed<__half>(const std::string&, const AddResidualConfig&, uint64_t);
template void run_add_residual_typed<__nv_bfloat16>(const std::string&, const AddResidualConfig&, uint64_t);

} // namespace ops
