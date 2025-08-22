#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/sampling.cuh"
#include "../test_kernels/test_kernels.cuh"
#include <random>
#include <sstream>
#include <vector>
#include <type_traits>

namespace ops {

template<typename T>
void run_topk_mask_logits_typed(const std::string& case_id, const TopKMaskConfig& cfg, uint64_t seed) {
	const int num_tokens = cfg.num_tokens;
	const int vocab_size = cfg.vocab_size;
	const int k = cfg.k;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// Input logits [num_tokens, vocab_size]
	const size_t logits_size = static_cast<size_t>(num_tokens) * vocab_size;
	std::vector<T> h_input_logits(logits_size);
	for (auto& v : h_input_logits) v = f2t<T>(dist(rng));

	// Device allocation
	T* d_input_logits = nullptr;
	T* d_masked_logits = nullptr;
	check_cuda(cudaMalloc(&d_input_logits, logits_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_masked_logits, logits_size * sizeof(T)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	// Copy to device
	check_cuda(cudaMemcpyAsync(d_input_logits, h_input_logits.data(), logits_size * sizeof(T), cudaMemcpyHostToDevice, stream));

	// Apply TopK mask - use same context as backend: TopKMaskLogits<T, int32_t>
	cudaError_t result;
	if constexpr (std::is_same_v<T, float>) {
		// Use FlashInfer directly for float (same as backend)
		result = flashinfer::sampling::TopKMaskLogits<T, int32_t>(
			d_input_logits,
			d_masked_logits,
			nullptr, // optional mask
			num_tokens,
			k,
			vocab_size,
			stream
		);
	} else {
		// Use test-local wrapper for other dtypes
		result = topk_mask_logits_test_local<T>(
			d_input_logits,
			d_masked_logits,
			nullptr, // optional mask
			num_tokens,
			k,
			vocab_size,
			stream
		);
	}

	check_cuda(cudaStreamSynchronize(stream));

	// Generate dtype string and case_id
	std::string dtype_name;
	if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
	else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
	else if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype_name = "bf16";

	// Write artifacts
	if (artifacts::op_enabled("topk_mask_logits")) {
		auto dir = artifacts::ensure_dir_for_case("topk_mask_logits", case_id + "_" + dtype_name);

		artifacts::write_device_bin(dir, "input_logits", d_input_logits, logits_size);
		artifacts::write_device_bin(dir, "masked_logits", d_masked_logits, logits_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"topk_mask_logits\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id + "_" + dtype_name) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"vocab_size\": " << vocab_size
			 << ", \"k\": " << k << "},\n"
			 << "\"dtype_map\": {\"input_logits\": \"" << dtype_name << "\", \"masked_logits\": \"" << dtype_name << "\"},\n"
			 << "\"shape_map\": {\"input_logits\": [" << num_tokens << ", " << vocab_size
			 << "], \"masked_logits\": [" << num_tokens << ", " << vocab_size << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	// Cleanup
	cudaStreamDestroy(stream);
	cudaFree(d_masked_logits);
	cudaFree(d_input_logits);
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

// Note: FlashInfer TopKMaskLogits has internal float assumptions
// Only instantiate float for now - __half and __nv_bfloat16 have compilation issues
template void run_topk_mask_logits_typed<float>(const std::string& case_id, const TopKMaskConfig& cfg, uint64_t seed);

} // namespace ops
