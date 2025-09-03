#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/page.cuh"
#include <random>
#include <sstream>

namespace ops {

template<typename T>
void run_append_paged_kv_cache_typed(const std::string& case_id,
							   const AppendPagedKVCacheConfig& cfg,
							   uint64_t seed) {
	using I = int32_t;

	const int num_tokens = cfg.num_tokens;
	const int num_kv_heads = cfg.num_kv_heads;
	const int head_size = cfg.head_size;
	const int page_size = cfg.page_size;
	const int max_num_pages = cfg.max_num_pages;
	const int batch_size = cfg.batch_size;

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// Calculate sizes
	const size_t kv_input_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
	const size_t page_data_size = static_cast<size_t>(max_num_pages) * page_size * num_kv_heads * head_size;

	// Host input data
	std::vector<T> h_k_input(kv_input_size);
	std::vector<T> h_v_input(kv_input_size);
	std::vector<T> h_paged_k_cache(page_data_size, 0);
	std::vector<T> h_paged_v_cache(page_data_size, 0);

	// Page management vectors (following l4ma.cu usage)
	std::vector<I> h_kv_page_indices(max_num_pages);
	std::vector<I> h_kv_page_indptr(batch_size + 1);
	std::vector<I> h_kv_last_page_lens(batch_size);
	std::vector<I> h_kv_batch_indices(num_tokens);
	std::vector<I> h_kv_positions(num_tokens);

	for (auto& v : h_k_input) v = static_cast<T>(dist(rng));
	for (auto& v : h_v_input) v = static_cast<T>(dist(rng));

	// Setup page indices - simple linear mapping
	for (int i = 0; i < max_num_pages; ++i) {
		h_kv_page_indices[i] = i;
	}

	// Setup page pointers for each batch
	const int pages_per_batch = max_num_pages / batch_size;
	h_kv_page_indptr[0] = 0;
	for (int i = 1; i <= batch_size; ++i) {
		h_kv_page_indptr[i] = h_kv_page_indptr[i-1] + pages_per_batch;
	}

	// Last page lengths (assume full pages except possibly the last)
	for (int i = 0; i < batch_size; ++i) {
		h_kv_last_page_lens[i] = page_size;
	}

	// Setup batch indices and positions for each token
	const int tokens_per_batch = num_tokens / batch_size;
	for (int i = 0; i < num_tokens; ++i) {
		h_kv_batch_indices[i] = i / tokens_per_batch;
		h_kv_positions[i] = i % tokens_per_batch;
	}

	// Device allocation
	T* d_k_input = nullptr;
	T* d_v_input = nullptr;
	T* d_paged_k_cache = nullptr;
	T* d_paged_v_cache = nullptr;
	I* d_kv_page_indices = nullptr;
	I* d_kv_page_indptr = nullptr;
	I* d_kv_last_page_lens = nullptr;
	I* d_kv_batch_indices = nullptr;
	I* d_kv_positions = nullptr;

	check_cuda(cudaMalloc(&d_k_input, kv_input_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_v_input, kv_input_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_paged_k_cache, page_data_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_paged_v_cache, page_data_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_kv_page_indices, max_num_pages * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_page_indptr, (batch_size + 1) * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_last_page_lens, batch_size * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_batch_indices, num_tokens * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_positions, num_tokens * sizeof(I)));

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	// Copy to device
	check_cuda(cudaMemcpyAsync(d_k_input, h_k_input.data(), kv_input_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_v_input, h_v_input.data(), kv_input_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_paged_k_cache, h_paged_k_cache.data(), page_data_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_paged_v_cache, h_paged_v_cache.data(), page_data_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_page_indices, h_kv_page_indices.data(), max_num_pages * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_page_indptr, h_kv_page_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_last_page_lens, h_kv_last_page_lens.data(), batch_size * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_batch_indices, h_kv_batch_indices.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_positions, h_kv_positions.data(), num_tokens * sizeof(I), cudaMemcpyHostToDevice, stream));

	// Create paged KV cache structure (matching l4ma.cu usage)
	flashinfer::paged_kv_t<T, I> paged_kv(
		num_kv_heads, page_size, head_size, batch_size,
		flashinfer::QKVLayout::kNHD,
		d_paged_k_cache, d_paged_v_cache,
		d_kv_page_indices,
		d_kv_page_indptr,
		d_kv_last_page_lens
	);

	// Call real FlashInfer AppendPagedKVCache (matching l4ma.cu)
	flashinfer::AppendPagedKVCache<T, I>(
		paged_kv, d_k_input, d_v_input,
		d_kv_batch_indices,
		d_kv_positions,
		num_tokens,
		num_kv_heads * head_size, head_size,
		num_kv_heads * head_size, head_size,
		stream
	);

	check_cuda(cudaStreamSynchronize(stream));

	// Write artifacts
	if (artifacts::op_enabled("append_paged_kv_cache")) {
		auto dir = artifacts::ensure_dir_for_case("append_paged_kv_cache", case_id);

		artifacts::write_device_bin(dir, "k_input", d_k_input, kv_input_size);
		artifacts::write_device_bin(dir, "v_input", d_v_input, kv_input_size);
		artifacts::write_device_bin(dir, "kv_page_indices", d_kv_page_indices, max_num_pages);
		artifacts::write_device_bin(dir, "kv_page_indptr", d_kv_page_indptr, batch_size + 1);
		artifacts::write_device_bin(dir, "kv_last_page_lens", d_kv_last_page_lens, batch_size);
		artifacts::write_device_bin(dir, "kv_batch_indices", d_kv_batch_indices, num_tokens);
		artifacts::write_device_bin(dir, "kv_positions", d_kv_positions, num_tokens);
		artifacts::write_device_bin(dir, "paged_k_cache_output", d_paged_k_cache, page_data_size);
		artifacts::write_device_bin(dir, "paged_v_cache_output", d_paged_v_cache, page_data_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"append_paged_kv_cache\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"num_kv_heads\": " << num_kv_heads
			 << ", \"head_size\": " << head_size
			 << ", \"page_size\": " << page_size
			 << ", \"max_num_pages\": " << max_num_pages
			 << ", \"batch_size\": " << batch_size << "},\n"
			 << "\"dtype_map\": {\"k_input\": \"bf16\", \"v_input\": \"bf16\", \"kv_page_indices\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_last_page_lens\": \"s32\", \"kv_batch_indices\": \"s32\", \"kv_positions\": \"s32\", \"paged_k_cache_output\": \"bf16\", \"paged_v_cache_output\": \"bf16\"},\n"
			 << "\"shape_map\": {\"k_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
			 << "], \"v_input\": [" << num_tokens << ", " << (num_kv_heads * head_size)
			 << "], \"kv_page_indices\": [" << max_num_pages
			 << "], \"kv_page_indptr\": [" << (batch_size + 1)
			 << "], \"kv_last_page_lens\": [" << batch_size
			 << "], \"kv_batch_indices\": [" << num_tokens
			 << "], \"kv_positions\": [" << num_tokens
			 << "], \"paged_k_cache_output\": [" << max_num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
			 << "], \"paged_v_cache_output\": [" << max_num_pages << ", " << page_size << ", " << (num_kv_heads * head_size) << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	// Cleanup
	cudaStreamDestroy(stream);
	cudaFree(d_kv_positions);
	cudaFree(d_kv_batch_indices);
	cudaFree(d_kv_last_page_lens);
	cudaFree(d_kv_page_indptr);
	cudaFree(d_kv_page_indices);
	cudaFree(d_paged_v_cache);
	cudaFree(d_paged_k_cache);
	cudaFree(d_v_input);
	cudaFree(d_k_input);
}

// Non-templated wrapper that calls the templated version with bf16
void run_append_paged_kv_cache(const std::string& case_id,
							   const AppendPagedKVCacheConfig& cfg,
							   uint64_t seed) {
	run_append_paged_kv_cache_typed<__nv_bfloat16>(case_id, cfg, seed);
}

// Explicit template instantiations (FlashInfer only supports 16-bit types for KV operations)
template void run_append_paged_kv_cache_typed<__half>(const std::string&, const AppendPagedKVCacheConfig&, uint64_t);
template void run_append_paged_kv_cache_typed<__nv_bfloat16>(const std::string&, const AppendPagedKVCacheConfig&, uint64_t);

} // namespace ops
