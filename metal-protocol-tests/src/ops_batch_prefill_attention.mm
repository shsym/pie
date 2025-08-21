// Per-op Metal wrapper for Batch Prefill Attention
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <chrono>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_helpers.hpp"
#include "metal_batch_prefill_attention.hpp"

namespace ops {

void run_batch_prefill_attention_metal(const std::string& case_id, const BatchPrefillAttentionConfig& cfg, uint64_t seed) {
    using T = bfloat16_t;  // Metal host-side bfloat16

    const int num_tokens = cfg.num_tokens;         // qo tokens
    const int num_query_heads = cfg.num_query_heads;
    const int head_size = cfg.head_size;
    const int head_dim = num_query_heads * head_size;
    const int page_size = cfg.page_size;

    std::cout << "\n🚀 Running Metal Batch Prefill Attention: tokens=" << num_tokens
              << ", query_heads=" << num_query_heads
              << ", head_size=" << head_size
              << ", head_dim=" << head_dim
              << ", page_size=" << page_size << std::endl;

    // Resolve CUDA artifacts base directory
    std::filesystem::path cuda_base_dir;
    if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
        cuda_base_dir = std::filesystem::path(envp);
    } else {
        std::filesystem::path this_file(__FILE__);
        auto project_root = this_file.parent_path().parent_path(); // .../metal-protocol-tests
        cuda_base_dir = project_root / "tests" / "artifacts";
    }
    std::filesystem::path cuda_case_dir = cuda_base_dir / "batch_prefill_attention" / case_id;

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
    auto read_vec_s32 = [&](const std::filesystem::path& p) -> std::vector<int32_t> {
        std::vector<uint8_t> bytes = read_bytes(p);
        size_t n = bytes.size() / sizeof(int32_t);
        std::vector<int32_t> v(n);
        if (n) memcpy(v.data(), bytes.data(), n * sizeof(int32_t));
        return v;
    };
    auto read_vec_bf16 = [&](const std::filesystem::path& p, size_t expected = 0) -> std::vector<T> {
        std::vector<uint8_t> bytes = read_bytes(p);
        size_t n = bytes.size() / sizeof(T);
        if (expected && n != expected) {
            std::cerr << "Warning: " << p << " element count mismatch; expected " << expected << ", got " << n << std::endl;
        }
        std::vector<T> v(n);
        if (n) memcpy(v.data(), bytes.data(), n * sizeof(T));
        return v;
    };

    // Allocate containers
    std::vector<T> q_input(static_cast<size_t>(num_tokens) * head_dim);
    std::vector<T> paged_k_cache;
    std::vector<T> paged_v_cache;
    std::vector<T> output(static_cast<size_t>(num_tokens) * head_dim);
    std::vector<int32_t> qo_indptr{0, num_tokens};
    std::vector<int32_t> kv_page_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;

    // Try to load CUDA reference inputs if available, otherwise generate test data
    bool use_cuda_artifacts = false;
    if (file_exists(cuda_case_dir)) {
        auto q_input_p = cuda_case_dir / "q_input.bin";
        auto pkv_p = cuda_case_dir / "paged_k_cache.bin";
        auto pvv_p = cuda_case_dir / "paged_v_cache.bin";
        auto qo_indptr_p = cuda_case_dir / "qo_indptr.bin";
        auto kv_page_indptr_p = cuda_case_dir / "kv_page_indptr.bin";
        auto kv_page_indices_p = cuda_case_dir / "kv_page_indices.bin";
        auto kv_last_page_lens_p = cuda_case_dir / "kv_last_page_lens.bin";

        // Check if all required CUDA reference files exist
        std::vector<std::filesystem::path> required_files = {
            q_input_p, pkv_p, pvv_p, qo_indptr_p, kv_page_indptr_p, kv_page_indices_p, kv_last_page_lens_p
        };

        bool all_files_exist = true;
        for (const auto& file : required_files) {
            if (!file_exists(file)) { all_files_exist = false; break; }
        }

        if (all_files_exist) {
            // Load CUDA reference inputs
            qo_indptr = read_vec_s32(qo_indptr_p);
            kv_page_indptr = read_vec_s32(kv_page_indptr_p);
            kv_page_indices = read_vec_s32(kv_page_indices_p);
            kv_last_page_lens = read_vec_s32(kv_last_page_lens_p);
            q_input = read_vec_bf16(q_input_p, static_cast<size_t>(num_tokens) * head_dim);
            paged_k_cache = read_vec_bf16(pkv_p);
            paged_v_cache = read_vec_bf16(pvv_p);
            use_cuda_artifacts = true;
            std::cout << "✅ Loaded CUDA reference inputs from: " << cuda_case_dir << std::endl;
        }
    }

    if (!use_cuda_artifacts) {
        // Generate synthetic test data for FlashInfer interface testing
        std::cout << "🔧 Generating synthetic test data for FlashInfer interface testing" << std::endl;

        // Set up paging structure to match CUDA reference expectations
        const int num_pages = (cfg.kv_len + page_size - 1) / page_size;  // Round up
        qo_indptr = {0, num_tokens};
        kv_page_indptr = {0, num_pages};
        kv_page_indices.resize(num_pages);
        for (int i = 0; i < num_pages; ++i) kv_page_indices[i] = i;
        const int last_page_len = cfg.kv_len - (num_pages - 1) * page_size;
        kv_last_page_lens = {last_page_len};

        // Generate random test data using seed
        std::mt19937_64 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        q_input.resize(static_cast<size_t>(num_tokens) * head_dim);
        for (size_t i = 0; i < q_input.size(); ++i) q_input[i] = float_to_bf16(dist(gen));

        size_t cache_size = static_cast<size_t>(num_pages) * page_size * head_dim;
        paged_k_cache.resize(cache_size);
        paged_v_cache.resize(cache_size);
        for (auto& val : paged_k_cache) val = float_to_bf16(dist(gen));
        for (auto& val : paged_v_cache) val = float_to_bf16(dist(gen));
    }
    std::fill(output.begin(), output.end(), static_cast<T>(0));

    print_vec_stats("q_input", q_input, 16);
    print_vec_stats("paged_k_cache", paged_k_cache, 16);
    print_vec_stats("paged_v_cache", paged_v_cache, 16);

    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    std::cout << "\n🔍 DEBUG: Host-side parameter values:" << std::endl;
    std::cout << "  head_size: " << head_size << std::endl;
    std::cout << "  sqrt(head_size): " << sqrtf(static_cast<float>(head_size)) << std::endl;
    std::cout << "  scale = 1/sqrt(head_size): " << scale << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;
    std::cout << "  page_size: " << page_size << std::endl;
    std::cout << "  kv_page_indices.size(): " << kv_page_indices.size() << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        metal::batch_prefill_attention::batch_prefill_attention_unified_bf16(
            q_input.data(),
            paged_k_cache.data(),
            paged_v_cache.data(),
            qo_indptr.data(),
            kv_page_indptr.data(),
            kv_page_indices.data(),
            kv_last_page_lens.data(),
            output.data(),
            num_tokens, head_dim, head_size, page_size, scale,
            static_cast<int>(kv_page_indices.size())
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "\n⏱️  Metal kernel execution time: " << elapsed.count() << " ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Metal kernel execution failed: " << e.what() << std::endl;
        throw;
    }

    print_vec_stats("output", output);

    int num_pages_actual = static_cast<int>(kv_page_indices.size());
    int kv_len_actual = 0;
    if (!kv_last_page_lens.empty()) {
        kv_len_actual = (num_pages_actual > 0)
            ? (num_pages_actual - 1) * page_size + kv_last_page_lens[0]
            : 0;
    }

    if (artifacts::op_enabled("batch_prefill_attention")) {
        auto dir = artifacts::ensure_dir_for_case("batch_prefill_attention", case_id + "_metal");

        artifacts::write_host_bin(dir, "q_input", q_input.data(), q_input.size());
        artifacts::write_host_bin(dir, "k_input", paged_k_cache.data(), paged_k_cache.size());
        artifacts::write_host_bin(dir, "v_input", paged_v_cache.data(), paged_v_cache.size());
        artifacts::write_host_bin(dir, "paged_k_cache", paged_k_cache.data(), paged_k_cache.size());
        artifacts::write_host_bin(dir, "paged_v_cache", paged_v_cache.data(), paged_v_cache.size());
        artifacts::write_host_bin(dir, "output", output.data(), output.size());
        artifacts::write_host_bin(dir, "qo_indptr", qo_indptr.data(), qo_indptr.size());
        artifacts::write_host_bin(dir, "kv_page_indptr", kv_page_indptr.data(), kv_page_indptr.size());
        artifacts::write_host_bin(dir, "kv_page_indices", kv_page_indices.data(), kv_page_indices.size());
        artifacts::write_host_bin(dir, "kv_last_page_lens", kv_last_page_lens.data(), kv_last_page_lens.size());

       std::ostringstream meta;
       meta << "\"version\": \"1\",\n"
           << "\"op\": \"batch_prefill_attention\",\n"
           << "\"backend\": \"metal\",\n"
           << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
           << "\"config\": {\"num_tokens\": " << num_tokens
           << ", \"num_query_heads\": " << num_query_heads
           << ", \"num_kv_heads\": " << cfg.num_kv_heads
           << ", \"head_size\": " << head_size
           << ", \"kv_len\": " << (kv_len_actual > 0 ? kv_len_actual : cfg.kv_len)
           << ", \"page_size\": " << page_size
           << ", \"batch_size\": 1, \"num_pages\": " << num_pages_actual << "},\n"
           << "\"dtype_map\": {\"q_input\": \"bf16\", \"k_input\": \"bf16\", \"v_input\": \"bf16\", \"paged_k_cache\": \"bf16\", \"paged_v_cache\": \"bf16\", \"output\": \"bf16\", \"qo_indptr\": \"s32\", \"kv_page_indptr\": \"s32\", \"kv_page_indices\": \"s32\", \"kv_last_page_lens\": \"s32\"},\n"
           << "\"shape_map\": {\"q_input\": [" << num_tokens << ", " << head_dim << "], "
           << "\"k_input\": [" << (num_pages_actual * page_size) << ", " << head_dim << "], "
           << "\"v_input\": [" << (num_pages_actual * page_size) << ", " << head_dim << "], "
           << "\"paged_k_cache\": [" << num_pages_actual << ", " << page_size << ", " << head_dim << "], "
           << "\"paged_v_cache\": [" << num_pages_actual << ", " << page_size << ", " << head_dim << "], "
           << "\"output\": [" << num_tokens << ", " << head_dim << "], "
           << "\"qo_indptr\": [2], \"kv_page_indptr\": [2], \"kv_page_indices\": [" << num_pages_actual << "], \"kv_last_page_lens\": [1]}";
       artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "\n✅ Metal Batch Prefill Attention completed successfully" << std::endl;
}

} // namespace ops
