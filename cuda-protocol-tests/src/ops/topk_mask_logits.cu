#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/sampling.cuh"
#include <random>
#include <sstream>
#include <vector>

namespace ops {

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

} // namespace ops
