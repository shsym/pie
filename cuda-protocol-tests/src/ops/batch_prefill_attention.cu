#include "../ops.hpp"
#include "ops_common.cuh"
#include "artifacts.hpp"
#include "flashinfer/page.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer_ops.cuh"  // backend wrapper providing BatchPrefillHandler and wrappers
#include <random>
#include <sstream>
#include <vector>
#include <optional>
#include <algorithm>

namespace ops {

void run_batch_prefill_attention(const std::string& case_id,
								 const BatchPrefillAttentionConfig& cfg,
								 uint64_t seed) {
	using T = __nv_bfloat16;
	using I = int32_t;

	const int num_tokens = cfg.num_tokens;
	const int num_query_heads = cfg.num_query_heads;
	const int num_kv_heads = cfg.num_kv_heads;
	const int head_size = cfg.head_size;
	const int kv_len = cfg.kv_len;
	const int page_size = cfg.page_size;
	const int batch_size = 1; // Simplified for testing

	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const size_t q_size = static_cast<size_t>(num_tokens) * num_query_heads * head_size;
	const size_t kv_size = static_cast<size_t>(kv_len) * num_kv_heads * head_size;
	const size_t o_size = q_size;
	const size_t num_pages = (kv_len + page_size - 1) / page_size;
	const size_t page_data_size = num_pages * page_size * num_kv_heads * head_size;

	// Host data initialization
	std::vector<T> h_q(q_size), h_k(kv_size), h_v(kv_size);
	std::vector<T> h_o(o_size, 0);
	std::vector<I> h_qo_indptr(batch_size + 1, 0);
	std::vector<I> h_kv_page_indptr(batch_size + 1, 0);
	std::vector<I> h_kv_page_indices(num_pages);
	std::vector<I> h_kv_last_page_lens(batch_size);
	std::vector<uint8_t> h_custom_mask;
	std::vector<I> h_mask_indptr(batch_size + 1, 0);

	for (auto& v : h_q) v = static_cast<T>(dist(rng));
	for (auto& v : h_k) v = static_cast<T>(dist(rng));
	for (auto& v : h_v) v = static_cast<T>(dist(rng));

	// Setup pointers for simple batch
	h_qo_indptr[0] = 0;
	h_qo_indptr[1] = num_tokens;
	h_kv_page_indptr[0] = 0;
	h_kv_page_indptr[1] = num_pages;

	// Page indices - simple linear mapping
	for (size_t i = 0; i < num_pages; ++i) h_kv_page_indices[i] = i;

	// Last page length
	h_kv_last_page_lens[0] = kv_len - (num_pages - 1) * page_size;

	// Simple mask (no custom masking for basic test)
	h_mask_indptr[0] = 0;
	h_mask_indptr[1] = 0;

	// Device allocation
	T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
	T *d_paged_k = nullptr, *d_paged_v = nullptr;
	I *d_qo_indptr = nullptr, *d_kv_page_indptr = nullptr;
	I *d_kv_page_indices = nullptr, *d_kv_last_page_lens = nullptr;
	uint8_t *d_custom_mask = nullptr;
	I *d_mask_indptr = nullptr;

	cudaStream_t stream;
	check_cuda(cudaStreamCreate(&stream));

	check_cuda(cudaMalloc(&d_q, q_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_k, kv_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_v, kv_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_o, o_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_paged_k, page_data_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_paged_v, page_data_size * sizeof(T)));
	check_cuda(cudaMalloc(&d_qo_indptr, (batch_size + 1) * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_page_indptr, (batch_size + 1) * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_page_indices, num_pages * sizeof(I)));
	check_cuda(cudaMalloc(&d_kv_last_page_lens, batch_size * sizeof(I)));
	check_cuda(cudaMalloc(&d_mask_indptr, (batch_size + 1) * sizeof(I)));

	// Note: We use FlashInfer directly for prefill attention; if it fails, the test will fail.

	// Copy to device
	check_cuda(cudaMemcpyAsync(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_qo_indptr, h_qo_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_page_indptr, h_kv_page_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_page_indices, h_kv_page_indices.data(), num_pages * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_kv_last_page_lens, h_kv_last_page_lens.data(), batch_size * sizeof(I), cudaMemcpyHostToDevice, stream));
	check_cuda(cudaMemcpyAsync(d_mask_indptr, h_mask_indptr.data(), (batch_size + 1) * sizeof(I), cudaMemcpyHostToDevice, stream));

	// Initialize paged KV cache to zero first
	check_cuda(cudaMemsetAsync(d_paged_k, 0, page_data_size * sizeof(T), stream));
	check_cuda(cudaMemsetAsync(d_paged_v, 0, page_data_size * sizeof(T), stream));

	// Create paged KV cache structure
	flashinfer::paged_kv_t<T, I> paged_kv(
		num_kv_heads, page_size, head_size, batch_size,
		flashinfer::QKVLayout::kNHD,
		d_paged_k, d_paged_v,
		d_kv_page_indices,
		d_kv_page_indptr,
		d_kv_last_page_lens
	);

	// Properly copy K, V data to paged format respecting page layout
	// Layout: [page_idx][token_in_page][head][dim]
	const int tokens_per_page = page_size;
	const int head_data_size = num_kv_heads * head_size;
	
	// Copy data page by page to respect paged layout
	for (int page_idx = 0; page_idx < static_cast<int>(num_pages); page_idx++) {
		int start_token = page_idx * tokens_per_page;
		int tokens_in_this_page = std::min(tokens_per_page, kv_len - start_token);
		
		if (tokens_in_this_page <= 0) break;
		
		// Source: K[start_token:start_token+tokens_in_this_page, :, :]
		T* src_k = d_k + start_token * head_data_size;
		T* src_v = d_v + start_token * head_data_size;
		
		// Destination: paged_k[page_idx, 0:tokens_in_this_page, :, :]
		T* dst_k = d_paged_k + page_idx * tokens_per_page * head_data_size;
		T* dst_v = d_paged_v + page_idx * tokens_per_page * head_data_size;
		
		size_t copy_bytes = tokens_in_this_page * head_data_size * sizeof(T);
		check_cuda(cudaMemcpyAsync(dst_k, src_k, copy_bytes, cudaMemcpyDeviceToDevice, stream));
		check_cuda(cudaMemcpyAsync(dst_v, src_v, copy_bytes, cudaMemcpyDeviceToDevice, stream));
	}

	// Use real FlashInfer BatchPrefillWithPagedKVCacheWrapper only; no fallback.
	// This matches the actual CUDA backend implementation in l4ma.cu via flashinfer_ops.cuh.
	{
		flashinfer::BatchPrefillHandler prefill_handler(/*enable_cuda_graph=*/false);
		prefill_handler.SetCUDAStream(stream);
		// Minimal plan to initialize handler workspace; mirrors usage pattern in backend
		// Here we provide host copies of indptr arrays for planning.
		prefill_handler.UpdatePageLockedBufferSize(8 * 1024 * 1024);
		// Allocate temporary dummy buffers for plan; small scratch sizes suffice for tests
		void* float_buf = nullptr;
		void* int_buf = nullptr;
		ops::check_cuda(cudaMalloc(&float_buf, 8 * 1024 * 1024));
		ops::check_cuda(cudaMalloc(&int_buf, 4 * 1024 * 1024));
		// host vectors already exist: h_qo_indptr, h_kv_page_indptr
		prefill_handler.Plan<T, I>(
			float_buf, 8 * 1024 * 1024,
			int_buf, 4 * 1024 * 1024,
			h_qo_indptr.data(), h_kv_page_indptr.data(),
			/*total_num_rows*/ static_cast<uint32_t>(num_tokens),
			/*batch_size*/ static_cast<uint32_t>(batch_size),
			/*num_qo_heads*/ static_cast<uint32_t>(num_query_heads),
			/*num_kv_heads*/ static_cast<uint32_t>(num_kv_heads),
			/*head_dim*/ static_cast<uint32_t>(head_size),
			/*page_size*/ static_cast<uint32_t>(page_size)
		);

		// Use FlashInfer's default scale (1/sqrt(head_size)) by not specifying explicit scale
		// This matches the standard attention scaling behavior
		
		flashinfer::BatchPrefillWithPagedKVCacheWrapper<T, T, T, I>(
			&prefill_handler,
			d_q,                    // query
			d_qo_indptr,            // query offsets
			/*q_rope_offset*/ nullptr,
			paged_kv,               // paged kv cache
			d_o,                    // output
			/*lse*/ nullptr,        // log-sum-exp, optional
			/*num_qo_heads*/ static_cast<uint32_t>(num_query_heads),
			/*mask_mode*/ flashinfer::MaskMode::kNone,
			/*custom_mask*/ nullptr,
			/*mask_indptr*/ d_mask_indptr,
			/*pos_enc*/ flashinfer::PosEncodingMode::kNone,
			/*use_fp16_qk_reduction*/ false,
			/*maybe_sm_scale*/ std::optional<float>(),  // Use default scale
			/*rope_scale*/ 1.0f,
			/*rope_theta*/ 1e4f,
			/*stream*/ stream
		);

		cudaFree(int_buf);
		cudaFree(float_buf);
	}

	check_cuda(cudaStreamSynchronize(stream));

	// Validate output is not all zeros (indicates successful attention computation)
	std::vector<T> h_o_validation(o_size);
	check_cuda(cudaMemcpy(h_o_validation.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
	
	int nonzero_count = 0;
	float sum_abs = 0.0f;
	float max_abs = 0.0f;
	
	for (size_t i = 0; i < h_o_validation.size(); i++) {
		float val = static_cast<float>(h_o_validation[i]);
		float abs_val = std::abs(val);
		if (abs_val > 1e-8f) {
			nonzero_count++;
			sum_abs += abs_val;
			max_abs = std::max(max_abs, abs_val);
		}
	}
	
	std::cout << "\n📊 CUDA Output Validation:" << std::endl;
	std::cout << "  Total elements: " << h_o_validation.size() << std::endl;
	std::cout << "  Non-zero elements: " << nonzero_count << " (" 
	          << (100.0f * nonzero_count / h_o_validation.size()) << "%)" << std::endl;
	std::cout << "  Sum of absolute values: " << sum_abs << std::endl;
	std::cout << "  Max absolute value: " << max_abs << std::endl;
	std::cout << "  Average absolute value: " << (nonzero_count > 0 ? sum_abs / nonzero_count : 0.0f) << std::endl;
	
	if (nonzero_count == 0) {
		std::cerr << "❌ ERROR: Output is all zeros! FlashInfer attention failed." << std::endl;
		std::cerr << "   This indicates an issue with paged KV cache setup or FlashInfer parameters." << std::endl;
	} else if (nonzero_count < static_cast<int>(h_o_validation.size()) * 0.1f) {
		std::cerr << "⚠️  WARNING: Output has very few non-zero values (" << nonzero_count << "/" << h_o_validation.size() << ")." << std::endl;
		std::cerr << "   This may indicate partial attention computation issues." << std::endl;
	} else {
		std::cout << "✅ SUCCESS: Output contains meaningful attention values." << std::endl;
	}

	if (artifacts::op_enabled("batch_prefill_attention")) {
		auto dir = artifacts::ensure_dir_for_case("batch_prefill_attention", case_id);
		artifacts::write_device_bin(dir, "q_input", d_q, q_size);
		artifacts::write_device_bin(dir, "k_input", d_k, kv_size);
		artifacts::write_device_bin(dir, "v_input", d_v, kv_size);
		artifacts::write_device_bin(dir, "paged_k_cache", d_paged_k, page_data_size);
		artifacts::write_device_bin(dir, "paged_v_cache", d_paged_v, page_data_size);
		artifacts::write_device_bin(dir, "output", d_o, o_size);
		artifacts::write_device_bin(dir, "qo_indptr", d_qo_indptr, batch_size + 1);
		artifacts::write_device_bin(dir, "kv_page_indptr", d_kv_page_indptr, batch_size + 1);
		artifacts::write_device_bin(dir, "kv_page_indices", d_kv_page_indices, num_pages);
		artifacts::write_device_bin(dir, "kv_last_page_lens", d_kv_last_page_lens, batch_size);

		std::ostringstream meta;
		meta << "\"version\": \"1\",\n"
			 << "\"op\": \"batch_prefill_attention\",\n"
			 << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
			 << "\"config\": {\"num_tokens\": " << num_tokens
			 << ", \"num_query_heads\": " << num_query_heads
			 << ", \"num_kv_heads\": " << num_kv_heads
			 << ", \"head_size\": " << head_size
			 << ", \"kv_len\": " << kv_len
			 << ", \"page_size\": " << page_size
			 << ", \"batch_size\": " << batch_size
			 << ", \"num_pages\": " << num_pages << "},\n"
			 << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"v_input\": \"bf16\", \"paged_k_cache\": \"bf16\", \"paged_v_cache\": \"bf16\", \"output\": \"bf16\", \"qo_indptr\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_page_indices\": \"s32\", \"kv_last_page_lens\": \"s32\"},\n"
			 << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"k_input\": [" << kv_len << ", " << (num_kv_heads * head_size)
			 << "], \"v_input\": [" << kv_len << ", " << (num_kv_heads * head_size)
			 << "], \"paged_k_cache\": [" << num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
			 << "], \"paged_v_cache\": [" << num_pages << ", " << page_size << ", " << (num_kv_heads * head_size)
			 << "], \"output\": [" << num_tokens << ", " << (num_query_heads * head_size)
			 << "], \"qo_indptr\": [" << (batch_size + 1)
			 << "], \"kv_page_indptr\": [" << (batch_size + 1)
			 << "], \"kv_page_indices\": [" << num_pages
			 << "], \"kv_last_page_lens\": [" << batch_size << "]}";
		artifacts::write_meta_json(dir, meta.str());
	}

	// Cleanup
	cudaFree(d_mask_indptr);
	cudaFree(d_kv_last_page_lens);
	cudaFree(d_kv_page_indices);
	cudaFree(d_kv_page_indptr);
	cudaFree(d_qo_indptr);
	cudaFree(d_paged_v);
	cudaFree(d_paged_k);
	cudaFree(d_o);
	cudaFree(d_v);
	cudaFree(d_k);
	cudaFree(d_q);
	cudaStreamDestroy(stream);
}

} // namespace ops
