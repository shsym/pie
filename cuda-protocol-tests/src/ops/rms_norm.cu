#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/norm.cuh"
#include <random>
#include <sstream>
#include <vector>
#include <type_traits>

namespace ops {

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

	// Call FlashInfer RMS Norm
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

// Explicit instantiations
template void run_rms_norm_typed<float>(const std::string&, const RMSNormConfig&, uint64_t);
template void run_rms_norm_typed<__half>(const std::string&, const RMSNormConfig&, uint64_t);
template void run_rms_norm_typed<__nv_bfloat16>(const std::string&, const RMSNormConfig&, uint64_t);

} // namespace ops
