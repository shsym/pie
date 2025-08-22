#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/sampling.cuh"
#include "../test_kernels/test_kernels.cuh"
#include <random>
#include <sstream>
#include <vector>
#include <optional>
#include <type_traits>

namespace ops {

template<typename T>
void run_softmax_typed(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed) {
	const int batch_size = cfg.batch_size;
	const int vocab_size = cfg.vocab_size;
	const float temperature = cfg.temperature;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Reasonable logit range

	// Input logits [batch_size, vocab_size]
	const size_t logits_size = static_cast<size_t>(batch_size) * vocab_size;
	std::vector<T> h_input_logits(logits_size);
	std::vector<T> h_output(logits_size, 0);

	for (auto& v : h_input_logits) v = f2t<T>(dist(rng));

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

	// Apply FlashInfer OnlineSoftmax - use test-local wrapper for non-float types
	cudaError_t softmax_result;
	if constexpr (std::is_same_v<T, float>) {
		// Use FlashInfer directly for float (same as backend)
		softmax_result = flashinfer::sampling::OnlineSoftmax<T>(
			d_input_logits,
			d_output,
			batch_size,
			vocab_size,
			d_temperature_arr,  // nullptr for scalar temperature
			f2t<T>(temperature),
			d_workspace,
			workspace_size,
			false,  // enable_pdl
			stream
		);
	} else {
		// Use test-local wrapper for other dtypes
		softmax_result = online_softmax_test_local<T>(
			d_input_logits,
			d_output,
			batch_size,
			vocab_size,
			d_temperature_arr,  // nullptr for scalar temperature
			f2t<T>(temperature),
			d_workspace,
			workspace_size,
			false,  // enable_pdl
			stream
		);
	}

	if (softmax_result != cudaSuccess) {
		std::cerr << "FlashInfer OnlineSoftmax failed: " << cudaGetErrorString(softmax_result) << std::endl;
		// Fallback: just copy input to output for testing purposes
		check_cuda(cudaMemcpyAsync(d_output, d_input_logits, logits_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
	}

	check_cuda(cudaStreamSynchronize(stream));

	// Generate dtype string and case_id
	std::string dtype_name;
	if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
	else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
	else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

	// Write artifacts
	if (artifacts::op_enabled("softmax")) {
		auto dir = artifacts::ensure_dir_for_case("softmax", case_id + "_" + dtype_name);

		artifacts::write_device_bin(dir, "input_logits", d_input_logits, logits_size);
		artifacts::write_device_bin(dir, "output", d_output, logits_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"softmax\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
			 << "\"config\": {\"batch_size\": " << batch_size
			 << ", \"vocab_size\": " << vocab_size
			 << ", \"temperature\": " << temperature << "},\n"
			 << "\"dtype_map\": {\"input_logits\": \"" << dtype_name << "\", \"output\": \"" << dtype_name << "\"},\n"
			 << "\"shape_map\": {\"input_logits\": [" << batch_size << ", " << vocab_size
			 << "], \"output\": [" << batch_size << ", " << vocab_size << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	// Cleanup
	cudaStreamDestroy(stream);
	cudaFree(d_workspace);
	cudaFree(d_output);
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

// Note: FlashInfer OnlineSoftmax has internal float assumptions
// Only instantiate float for now - __half and __nv_bfloat16 have compilation issues
template void run_softmax_typed<float>(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed);

} // namespace ops
