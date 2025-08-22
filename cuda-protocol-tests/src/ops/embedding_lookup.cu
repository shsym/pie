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

// Typed implementation (float, bf16) and the untyped wrapper

template<typename T, typename I>
void run_embedding_lookup_typed(const std::string& case_id, const EmbeddingConfig& cfg, uint64_t seed) {
	const int num_tokens = cfg.num_tokens;
	const int hidden_size = cfg.hidden_size;
	const int vocab_size = cfg.vocab_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<T> h_embedding(static_cast<size_t>(vocab_size) * hidden_size);
	for (auto& v : h_embedding) v = f2t<T>(dist(rng));

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

	// Call the implementation - use test-local version for __half
	if constexpr (std::is_same_v<T, __half>) {
		embed_test_local<T, I>(
			d_embedding,
			static_cast<size_t>(vocab_size),
			d_indices,
			static_cast<size_t>(num_tokens),
			d_output,
			hidden_size,
			stream
		);
	} else {
		// Use backend implementation for supported types
		embed<T, I>(
			d_embedding,
			static_cast<size_t>(vocab_size),
			d_indices,
			static_cast<size_t>(num_tokens),
			d_output,
			hidden_size,
			stream
		);
	}

	check_cuda(cudaStreamSynchronize(stream));

	std::string dtype_name;
	if constexpr (std::is_same_v<T, float>) dtype_name = "fp32";
	else if constexpr (std::is_same_v<T, __half>) dtype_name = "fp16";
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

	// Cleanup
	cudaStreamDestroy(stream);
	cudaFree(d_output);
	cudaFree(d_indices);
	cudaFree(d_embedding);
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

// Explicit instantiations
template void run_embedding_lookup_typed<float, int32_t>(const std::string&, const EmbeddingConfig&, uint64_t);
template void run_embedding_lookup_typed<__half, int32_t>(const std::string&, const EmbeddingConfig&, uint64_t);
template void run_embedding_lookup_typed<__nv_bfloat16, int32_t>(const std::string&, const EmbeddingConfig&, uint64_t);

} // namespace ops
