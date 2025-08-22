#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/pos_enc.cuh"
#include <random>
#include <sstream>
#include <vector>

namespace ops {

void run_rope(const std::string& case_id,
			  const RoPEConfig& cfg,
			  uint64_t seed) {
	using T = __nv_bfloat16;
	using I = int32_t;

	const int num_tokens = cfg.num_tokens;
	const int num_query_heads = cfg.num_query_heads;
	const int num_kv_heads = cfg.num_kv_heads;
	const int head_size = cfg.head_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const size_t q_size = static_cast<size_t>(num_tokens) * num_query_heads * head_size;
	const size_t k_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
	std::vector<T> h_q_input(q_size);
	std::vector<T> h_k_input(k_size);
	std::vector<I> h_pos_ids(num_tokens);

	for (auto& v : h_q_input) v = dist(rng);
	for (auto& v : h_k_input) v = dist(rng);
	for (int i = 0; i < num_tokens; ++i) h_pos_ids[i] = i;

	T* d_q = nullptr;
	T* d_k = nullptr;
	I* d_pos_ids = nullptr;
	check_cuda(cudaMalloc(&d_q, q_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_k, k_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_pos_ids, num_tokens * sizeof(I)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	check_cuda(cudaMemcpyAsync(d_q, h_q_input.data(), q_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_k, h_k_input.data(), k_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_pos_ids, h_pos_ids.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));

	// Call the real FlashInfer RoPE function
	flashinfer::BatchQKApplyLlama31RotaryPosIds(
		d_q, d_k, d_q, d_k, // In-place operation
		d_pos_ids,
		(uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_kv_heads,
		(uint32_t)head_size, (uint32_t)head_size,
		(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
		(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
		(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
		(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
		false, // layout: interleaved
		cfg.rope_factor, cfg.rope_theta, cfg.rope_low_frequency_factor,
		cfg.rope_high_frequency_factor, cfg.max_position_embeddings, stream
	);

	check_cuda(cudaStreamSynchronize(stream));

	if (artifacts::op_enabled("rope")) {
		auto dir = artifacts::ensure_dir_for_case("rope", case_id);

		artifacts::write_vector_bin(dir, "q_input", h_q_input);
		artifacts::write_vector_bin(dir, "k_input", h_k_input);
		artifacts::write_vector_bin(dir, "pos_ids", h_pos_ids);
		artifacts::write_device_bin(dir, "q_output", d_q, q_size);
		artifacts::write_device_bin(dir, "k_output", d_k, k_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"rope\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"num_query_heads\": " << num_query_heads
			 << ", \"num_kv_heads\": " << num_kv_heads
			 << ", \"head_size\": " << head_size
			 << ", \"rope_theta\": " << cfg.rope_theta
			 << ", \"rope_factor\": " << cfg.rope_factor
			 << ", \"rope_low_frequency_factor\": " << cfg.rope_low_frequency_factor
			 << ", \"rope_high_frequency_factor\": " << cfg.rope_high_frequency_factor
			 << ", \"max_position_embeddings\": " << cfg.max_position_embeddings << "},\n"
			 << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"pos_ids\": \"s32\", \"q_output\": \"bf16\", \"k_output\": \"bf16\"},\n"
			 << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
			 << "], \"pos_ids\": [" << num_tokens
			 << "], \"q_output\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"k_output\": [" << num_tokens << ", " << (num_kv_heads * head_size) << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	cudaStreamDestroy(stream);
	cudaFree(d_pos_ids);
	cudaFree(d_k);
	cudaFree(d_q);
}

// Template implementation for multi-dtype support
template <typename T>
void run_rope_typed(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed) {
	using I = int32_t;
	
	const int num_tokens = cfg.num_tokens;
	const int num_query_heads = cfg.num_query_heads;
	const int num_kv_heads = cfg.num_kv_heads;
	const int head_size = cfg.head_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const size_t q_size = static_cast<size_t>(num_tokens) * num_query_heads * head_size;
	const size_t k_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
	std::vector<T> h_q_input(q_size);
	std::vector<T> h_k_input(k_size);
	std::vector<I> h_pos_ids(num_tokens);

	for (auto& v : h_q_input) v = f2t<T>(dist(rng));
	for (auto& v : h_k_input) v = f2t<T>(dist(rng));
	for (int i = 0; i < num_tokens; ++i) h_pos_ids[i] = i;

	T* d_q = nullptr;
	T* d_k = nullptr;
	I* d_pos_ids = nullptr;
	check_cuda(cudaMalloc(&d_q, q_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_k, k_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_pos_ids, num_tokens * sizeof(I)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	check_cuda(cudaMemcpyAsync(d_q, h_q_input.data(), q_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_k, h_k_input.data(), k_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_pos_ids, h_pos_ids.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));

	// Call the FlashInfer RoPE function - template specialization needed for different types
	if constexpr (std::is_same_v<T, __nv_bfloat16>) {
		flashinfer::BatchQKApplyLlama31RotaryPosIds(
			d_q, d_k, d_q, d_k, // In-place operation
			d_pos_ids,
			(uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_kv_heads,
			(uint32_t)head_size, (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			false, // layout: interleaved
			cfg.rope_factor, cfg.rope_theta, cfg.rope_low_frequency_factor,
			cfg.rope_high_frequency_factor, cfg.max_position_embeddings, stream
		);
	} else if constexpr (std::is_same_v<T, __half>) {
		flashinfer::BatchQKApplyLlama31RotaryPosIds(
			d_q, d_k, d_q, d_k, // In-place operation
			d_pos_ids,
			(uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_kv_heads,
			(uint32_t)head_size, (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			false, // layout: interleaved
			cfg.rope_factor, cfg.rope_theta, cfg.rope_low_frequency_factor,
			cfg.rope_high_frequency_factor, cfg.max_position_embeddings, stream
		);
	} else if constexpr (std::is_same_v<T, float>) {
		flashinfer::BatchQKApplyLlama31RotaryPosIds(
			d_q, d_k, d_q, d_k, // In-place operation
			d_pos_ids,
			(uint32_t)num_tokens, (uint32_t)num_query_heads, (uint32_t)num_kv_heads,
			(uint32_t)head_size, (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_query_heads * head_size), (uint32_t)head_size,
			(uint32_t)(num_kv_heads * head_size), (uint32_t)head_size,
			false, // layout: interleaved
			cfg.rope_factor, cfg.rope_theta, cfg.rope_low_frequency_factor,
			cfg.rope_high_frequency_factor, cfg.max_position_embeddings, stream
		);
	}

	check_cuda(cudaStreamSynchronize(stream));

	if (artifacts::op_enabled("rope")) {
		// Generate case_id with dtype suffix
		std::string dtype_suffix;
		if constexpr (std::is_same_v<T, float>) {
			dtype_suffix = "_fp32";
		} else if constexpr (std::is_same_v<T, __half>) {
			dtype_suffix = "_fp16";
		} else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
			dtype_suffix = "_bf16";
		}
		std::string full_case_id = case_id + dtype_suffix;
		
		auto dir = artifacts::ensure_dir_for_case("rope", full_case_id);

		artifacts::write_vector_bin(dir, "q_input", h_q_input);
		artifacts::write_vector_bin(dir, "k_input", h_k_input);
		artifacts::write_vector_bin(dir, "pos_ids", h_pos_ids);
		artifacts::write_device_bin(dir, "q_output", d_q, q_size);
		artifacts::write_device_bin(dir, "k_output", d_k, k_size);

		// Generate dtype string for meta.json
		std::string dtype_str;
		if constexpr (std::is_same_v<T, float>) {
			dtype_str = "fp32";
		} else if constexpr (std::is_same_v<T, __half>) {
			dtype_str = "fp16";
		} else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
			dtype_str = "bf16";
		}

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"rope\",\n"
			 << "\"case_id\": " << artifacts::json_escape(full_case_id) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"num_query_heads\": " << num_query_heads
			 << ", \"num_kv_heads\": " << num_kv_heads
			 << ", \"head_size\": " << head_size
			 << ", \"rope_theta\": " << cfg.rope_theta
			 << ", \"rope_factor\": " << cfg.rope_factor
			 << ", \"rope_low_frequency_factor\": " << cfg.rope_low_frequency_factor
			 << ", \"rope_high_frequency_factor\": " << cfg.rope_high_frequency_factor
			 << ", \"max_position_embeddings\": " << cfg.max_position_embeddings << "},\n"
			 << "\"dtype_map\": {\"q_input\": \"" << dtype_str << "\", \"k_input\": \"" << dtype_str << "\", \"pos_ids\": \"s32\", \"q_output\": \"" << dtype_str << "\", \"k_output\": \"" << dtype_str << "\"},\n"
			 << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
			 << "], \"pos_ids\": [" << num_tokens
			 << "], \"q_output\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"k_output\": [" << num_tokens << ", " << (num_kv_heads * head_size) << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	cudaStreamDestroy(stream);
	cudaFree(d_pos_ids);
	cudaFree(d_k);
	cudaFree(d_q);
}

// Explicit instantiations
template void run_rope_typed<float>(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed);
template void run_rope_typed<__half>(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed);
template void run_rope_typed<__nv_bfloat16>(const std::string& case_id, const RoPEConfig& cfg, uint64_t seed);

} // namespace ops
