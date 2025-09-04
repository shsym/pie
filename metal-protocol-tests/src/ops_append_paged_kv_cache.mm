// Per-op Metal wrapper for Append Paged KV Cache
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <filesystem>
#include <fstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_append_paged_kv_cache.hpp"
#include "dtype_utils.hpp"
#include "workspace_utils.hpp"

namespace ops {

void run_append_paged_kv_cache_metal(const std::string& case_id, const AppendPagedKVCacheConfig& cfg, uint64_t seed) {
    const int num_tokens = cfg.num_tokens;
    const int num_kv_heads = cfg.num_kv_heads;
    const int head_size = cfg.head_size;
    const int page_size = cfg.page_size;
    const int max_num_pages = cfg.max_num_pages;
    const int batch_size = cfg.batch_size;

    // Detect target dtype from CUDA reference meta.json
    auto dtype_info = detect_dtype_from_meta("append_paged_kv_cache", case_id);
    if (!dtype_info.success) {
        std::cerr << "ERROR: meta.json not found for append_paged_kv_cache/" << case_id
                  << ". Use --write-meta-from-cli to generate metadata first." << std::endl;
        return;
    }

    std::cout << "Running Metal Append Paged KV Cache: tokens=" << num_tokens
              << ", kv_heads=" << num_kv_heads << ", head_size=" << head_size
              << ", page_size=" << page_size << ", max_pages=" << max_num_pages
              << ", batch_size=" << batch_size << ", dtype=" << dtype_info.dtype_str << std::endl;

    // Calculate tensor sizes
    const size_t kv_data_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_size;
    const size_t page_cache_size = static_cast<size_t>(max_num_pages) * page_size * num_kv_heads * head_size;

    // Generate test data
    std::vector<bfloat16_t> k_data(kv_data_size);
    std::vector<bfloat16_t> v_data(kv_data_size);
    std::vector<bfloat16_t> paged_k_cache(page_cache_size);
    std::vector<bfloat16_t> paged_v_cache(page_cache_size);
    // Page management data structures (matching CUDA implementation)
    std::vector<int32_t> page_indices(max_num_pages);
    std::vector<int32_t> page_indptr(batch_size + 1);
    std::vector<int32_t> last_page_lens(batch_size);
    std::vector<int32_t> batch_indices(num_tokens);
    std::vector<int32_t> positions(num_tokens);

    // Attempt to load CUDA reference inputs for apples-to-apples comparison
    auto file_exists = [](const std::filesystem::path& p) -> bool {
        std::error_code ec; return std::filesystem::exists(p, ec);
    };
    auto read_bytes = [](const std::filesystem::path& p) -> std::vector<uint8_t> {
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs.is_open()) return {};
        ifs.seekg(0, std::ios::end);
        std::streamsize size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(static_cast<size_t>(std::max<int64_t>(0, size)));
        if (size > 0) ifs.read(reinterpret_cast<char*>(buf.data()), size);
        return buf;
    };

    bool loaded_cuda_inputs = false;
    auto cuda_base_dir = workspace_utils::get_cuda_artifacts_dir();
    if (!cuda_base_dir.empty()) {
        auto cuda_case_dir = cuda_base_dir / "append_paged_kv_cache" / case_id;
        if (file_exists(cuda_case_dir / "k_input.bin") &&
            file_exists(cuda_case_dir / "v_input.bin") &&
            file_exists(cuda_case_dir / "kv_page_indices.bin") &&
            file_exists(cuda_case_dir / "kv_page_indptr.bin") &&
            file_exists(cuda_case_dir / "kv_batch_indices.bin") &&
            file_exists(cuda_case_dir / "kv_positions.bin") &&
            file_exists(cuda_case_dir / "kv_last_page_lens.bin")) {

            std::vector<uint8_t> k_bytes = read_bytes(cuda_case_dir / "k_input.bin");
            std::vector<uint8_t> v_bytes = read_bytes(cuda_case_dir / "v_input.bin");
            std::vector<uint8_t> page_indices_bytes = read_bytes(cuda_case_dir / "kv_page_indices.bin");
            std::vector<uint8_t> page_indptr_bytes = read_bytes(cuda_case_dir / "kv_page_indptr.bin");
            std::vector<uint8_t> batch_indices_bytes = read_bytes(cuda_case_dir / "kv_batch_indices.bin");
            std::vector<uint8_t> positions_bytes = read_bytes(cuda_case_dir / "kv_positions.bin");
            std::vector<uint8_t> last_page_lens_bytes = read_bytes(cuda_case_dir / "kv_last_page_lens.bin");

            if (k_bytes.size() == kv_data_size * sizeof(bfloat16_t) &&
                v_bytes.size() == kv_data_size * sizeof(bfloat16_t) &&
                page_indices_bytes.size() == max_num_pages * sizeof(int32_t) &&
                page_indptr_bytes.size() == (batch_size + 1) * sizeof(int32_t) &&
                batch_indices_bytes.size() == num_tokens * sizeof(int32_t) &&
                positions_bytes.size() == num_tokens * sizeof(int32_t) &&
                last_page_lens_bytes.size() == batch_size * sizeof(int32_t)) {

                std::memcpy(k_data.data(), k_bytes.data(), k_bytes.size());
                std::memcpy(v_data.data(), v_bytes.data(), v_bytes.size());
                std::memcpy(page_indices.data(), page_indices_bytes.data(), page_indices_bytes.size());
                std::memcpy(page_indptr.data(), page_indptr_bytes.data(), page_indptr_bytes.size());
                std::memcpy(batch_indices.data(), batch_indices_bytes.data(), batch_indices_bytes.size());
                std::memcpy(positions.data(), positions_bytes.data(), positions_bytes.size());
                std::memcpy(last_page_lens.data(), last_page_lens_bytes.data(), last_page_lens_bytes.size());
                loaded_cuda_inputs = true;

                std::cout << "✅ Loaded CUDA reference data for append_paged_kv_cache from: " << cuda_case_dir << std::endl;

                // Load existing paged cache if available
                if (file_exists(cuda_case_dir / "paged_k_cache.bin") &&
                    file_exists(cuda_case_dir / "paged_v_cache.bin")) {
                    std::vector<uint8_t> pk_bytes = read_bytes(cuda_case_dir / "paged_k_cache.bin");
                    std::vector<uint8_t> pv_bytes = read_bytes(cuda_case_dir / "paged_v_cache.bin");

                    if (pk_bytes.size() == page_cache_size * sizeof(bfloat16_t) &&
                        pv_bytes.size() == page_cache_size * sizeof(bfloat16_t)) {
                        std::memcpy(paged_k_cache.data(), pk_bytes.data(), pk_bytes.size());
                        std::memcpy(paged_v_cache.data(), pv_bytes.data(), pv_bytes.size());
                    }
                }
            }
        }
    }

    if (!loaded_cuda_inputs) {
        // Generate random test data following CUDA pattern
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto& v : k_data) v = float_to_bf16(dist(rng));
        for (auto& v : v_data) v = float_to_bf16(dist(rng));

        // Initialize paged cache to zeros
        std::fill(paged_k_cache.begin(), paged_k_cache.end(), float_to_bf16(0.0f));
        std::fill(paged_v_cache.begin(), paged_v_cache.end(), float_to_bf16(0.0f));

        // Setup page indices - simple linear mapping like CUDA
        for (int i = 0; i < max_num_pages; ++i) {
            page_indices[i] = i;
        }

        // Setup page pointers for each batch
        const int pages_per_batch = max_num_pages / batch_size;
        page_indptr[0] = 0;
        for (int i = 1; i <= batch_size; ++i) {
            page_indptr[i] = page_indptr[i-1] + pages_per_batch;
        }

        // Last page lengths (assume full pages)
        for (int i = 0; i < batch_size; ++i) {
            last_page_lens[i] = page_size;
        }

        // Setup batch indices and positions for each token
        const int tokens_per_batch = num_tokens / batch_size;
        for (int i = 0; i < num_tokens; ++i) {
            batch_indices[i] = i / tokens_per_batch;
            positions[i] = i % tokens_per_batch;
        }
    }

    int result = 0;

    // Route to appropriate kernel based on detected dtype
    if (dtype_info.dtype == DType::FP32) {
        // Use native fp32 kernel with fp32 data
        std::vector<float> k_fp32(kv_data_size), v_fp32(kv_data_size);
        std::vector<float> paged_k_fp32(page_cache_size), paged_v_fp32(page_cache_size);

        // Convert bf16 data to fp32
        for (size_t i = 0; i < kv_data_size; ++i) {
            k_fp32[i] = bf16_to_float(k_data[i]);
            v_fp32[i] = bf16_to_float(v_data[i]);
        }
        for (size_t i = 0; i < page_cache_size; ++i) {
            paged_k_fp32[i] = bf16_to_float(paged_k_cache[i]);
            paged_v_fp32[i] = bf16_to_float(paged_v_cache[i]);
        }

        // Initialize Metal system
        if (!initialize_metal_append_paged_kv_cache()) {
            throw std::runtime_error("Failed to initialize Metal append paged KV cache system");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        metal_append_paged_kv_cache_float32(
            device, commandQueue,
            k_fp32.data(), v_fp32.data(),
            paged_k_fp32.data(), paged_v_fp32.data(),
            reinterpret_cast<const uint32_t*>(batch_indices.data()),
            reinterpret_cast<const uint32_t*>(positions.data()),
            reinterpret_cast<const uint32_t*>(page_indices.data()),
            reinterpret_cast<const uint32_t*>(page_indptr.data()),
            reinterpret_cast<const uint32_t*>(last_page_lens.data()),
            num_tokens, num_kv_heads, head_size, page_size, max_num_pages, batch_size
        );

        // Convert results back to bf16 for consistency
        for (size_t i = 0; i < page_cache_size; ++i) {
            paged_k_cache[i] = float_to_bf16(paged_k_fp32[i]);
            paged_v_cache[i] = float_to_bf16(paged_v_fp32[i]);
        }
    }
    else {
        // Initialize Metal system
        if (!initialize_metal_append_paged_kv_cache()) {
            throw std::runtime_error("Failed to initialize Metal append paged KV cache system");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        metal_append_paged_kv_cache_bfloat16(
            device, commandQueue,
            k_data.data(), v_data.data(),
            paged_k_cache.data(), paged_v_cache.data(),
            reinterpret_cast<const uint32_t*>(batch_indices.data()),
            reinterpret_cast<const uint32_t*>(positions.data()),
            reinterpret_cast<const uint32_t*>(page_indices.data()),
            reinterpret_cast<const uint32_t*>(page_indptr.data()),
            reinterpret_cast<const uint32_t*>(last_page_lens.data()),
            num_tokens, num_kv_heads, head_size, page_size, max_num_pages, batch_size
        );

        if (dtype_info.dtype == DType::FP16) {
            std::cout << "Note: Using bf16 kernel for fp16 request (fp16 kernel not yet implemented)" << std::endl;
        }
    }

    if (result != 0) {
        throw std::runtime_error("Metal Append Paged KV Cache execution failed with code: " + std::to_string(result));
    }

    // Write artifacts for comparison with CUDA
    if (artifacts::op_enabled("append_paged_kv_cache")) {
        auto dir = artifacts::ensure_dir_for_case("append_paged_kv_cache", case_id + "_metal");

        if (dtype_info.dtype == DType::FP32) {
            // Convert and save as fp32
            std::vector<float> k_fp32(kv_data_size), v_fp32(kv_data_size);
            std::vector<float> paged_k_fp32(page_cache_size), paged_v_fp32(page_cache_size);

            for (size_t i = 0; i < kv_data_size; ++i) {
                k_fp32[i] = bf16_to_float(k_data[i]);
                v_fp32[i] = bf16_to_float(v_data[i]);
            }
            for (size_t i = 0; i < page_cache_size; ++i) {
                paged_k_fp32[i] = bf16_to_float(paged_k_cache[i]);
                paged_v_fp32[i] = bf16_to_float(paged_v_cache[i]);
            }

            artifacts::write_vector_bin(dir, "k_input", k_fp32);
            artifacts::write_vector_bin(dir, "v_input", v_fp32);
            artifacts::write_vector_bin(dir, "paged_k_cache_output", paged_k_fp32);
            artifacts::write_vector_bin(dir, "paged_v_cache_output", paged_v_fp32);
        } else {
            // Save as bf16
            artifacts::write_host_bin(dir, "k_input", k_data.data(), k_data.size());
            artifacts::write_host_bin(dir, "v_input", v_data.data(), v_data.size());
            artifacts::write_host_bin(dir, "paged_k_cache_output", paged_k_cache.data(), paged_k_cache.size());
            artifacts::write_host_bin(dir, "paged_v_cache_output", paged_v_cache.data(), paged_v_cache.size());
        }

        // Save all the page management data with correct names
        artifacts::write_host_bin(dir, "kv_page_indices", page_indices.data(), page_indices.size());
        artifacts::write_host_bin(dir, "kv_page_indptr", page_indptr.data(), page_indptr.size());
        artifacts::write_host_bin(dir, "kv_last_page_lens", last_page_lens.data(), last_page_lens.size());
        artifacts::write_host_bin(dir, "kv_batch_indices", batch_indices.data(), batch_indices.size());
        artifacts::write_host_bin(dir, "kv_positions", positions.data(), positions.size());

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"append_paged_kv_cache\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"num_tokens\": " << num_tokens
             << ", \"num_kv_heads\": " << num_kv_heads
             << ", \"head_size\": " << head_size
             << ", \"page_size\": " << page_size
             << ", \"max_num_pages\": " << max_num_pages
             << ", \"batch_size\": " << batch_size << "},\n"
             << "\"dtype_map\": {\"k\": \"" << dtype_info.dtype_str
             << "\", \"v\": \"" << dtype_info.dtype_str
             << "\", \"paged_k_cache\": \"" << dtype_info.dtype_str
             << "\", \"paged_v_cache\": \"" << dtype_info.dtype_str
             << "\", \"page_indices\": \"int32\"},\n"
             << "\"shape_map\": {\"k\": [" << num_tokens << ", " << num_kv_heads << ", " << head_size << "], "
             << "\"v\": [" << num_tokens << ", " << num_kv_heads << ", " << head_size << "], "
             << "\"paged_k_cache\": [" << max_num_pages << ", " << page_size << ", " << num_kv_heads << ", " << head_size << "], "
             << "\"paged_v_cache\": [" << max_num_pages << ", " << page_size << ", " << num_kv_heads << ", " << head_size << "], "
             << "\"page_indices\": [" << batch_size << "]}";

        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Append Paged KV Cache completed successfully" << std::endl;
}

} // namespace ops