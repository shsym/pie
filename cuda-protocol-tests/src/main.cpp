#include "ops.hpp"
#include "artifacts.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

namespace {

struct Args {
    std::string op;
    std::string case_id = "auto";
    int num_tokens = 128;
    int hidden_size = 4096;
    int vocab_size = 32000;
    int intermediate_size = 11008;
    int M = 128;
    int N = 32000;
    int k = 50;
    int m = 128;
    int n = 4096;
    int num_elements = 1024;
    float eps = 1e-5f;
    bool transa = false;
    bool transb = true;
    bool use_bias = false;
    std::string dtype = "bf16";
    std::string input_dtype = "fp32";
    std::string output_dtype = "fp16";
    float rope_theta = 1e4f;
    float rope_factor = 1.0f;
    float rope_low_frequency_factor = 1.0f;
    float rope_high_frequency_factor = 1.0f;
    int max_position_embeddings = 8192;
    int num_heads = 32;
    int head_size = 128;
    int num_query_heads = 32;
    int num_kv_heads = 32;
    int kv_len = 2048;
    int page_size = 16;
    int num_groups = 4;
    int max_num_pages = 8;
    int batch_size = 2;
    float temperature = 1.0f;
    uint64_t seed = 0x12345678ULL;
};

bool parse_int(const char* s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_u64(const char* s, uint64_t& out) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

bool parse_float(const char* s, float& out) {
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (!end || *end != '\0') return false;
    out = v;
    return true;
}

bool parse_bool(const char* s, bool& out) {
    std::string str(s);
    for (auto& c : str) c = std::tolower(c);
    if (str == "true" || str == "1" || str == "yes") {
        out = true;
        return true;
    } else if (str == "false" || str == "0" || str == "no") {
        out = false;
        return true;
    }
    return false;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&](int& i) -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (flag == "--op") {
            if (const char* v = next(i)) a.op = v; else throw std::runtime_error("--op requires value");
        } else if (flag == "--case") {
            if (const char* v = next(i)) a.case_id = v; else throw std::runtime_error("--case requires value");
        } else if (flag == "--num_tokens") {
            const char* v = next(i); if (!v || !parse_int(v, a.num_tokens)) throw std::runtime_error("--num_tokens int");
        } else if (flag == "--hidden_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.hidden_size)) throw std::runtime_error("--hidden_size int");
        } else if (flag == "--vocab_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.vocab_size)) throw std::runtime_error("--vocab_size int");
        } else if (flag == "--M") {
            const char* v = next(i); if (!v || !parse_int(v, a.M)) throw std::runtime_error("--M int");
        } else if (flag == "--N") {
            const char* v = next(i); if (!v || !parse_int(v, a.N)) throw std::runtime_error("--N int");
        } else if (flag == "--k") {
            const char* v = next(i); if (!v || !parse_int(v, a.k)) throw std::runtime_error("--k int");
        } else if (flag == "--intermediate_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.intermediate_size)) throw std::runtime_error("--intermediate_size int");
        } else if (flag == "--m") {
            const char* v = next(i); if (!v || !parse_int(v, a.m)) throw std::runtime_error("--m int");
        } else if (flag == "--n") {
            const char* v = next(i); if (!v || !parse_int(v, a.n)) throw std::runtime_error("--n int");
        } else if (flag == "--num_elements") {
            const char* v = next(i); if (!v || !parse_int(v, a.num_elements)) throw std::runtime_error("--num_elements int");
        } else if (flag == "--eps") {
            const char* v = next(i); if (!v || !parse_float(v, a.eps)) throw std::runtime_error("--eps float");
        } else if (flag == "--transa") {
            const char* v = next(i); if (!v || !parse_bool(v, a.transa)) throw std::runtime_error("--transa bool");
        } else if (flag == "--transb") {
            const char* v = next(i); if (!v || !parse_bool(v, a.transb)) throw std::runtime_error("--transb bool");
        } else if (flag == "--use_bias") {
            const char* v = next(i); if (!v || !parse_bool(v, a.use_bias)) throw std::runtime_error("--use_bias bool");
        } else if (flag == "--input_dtype") {
            if (const char* v = next(i)) a.input_dtype = v; else throw std::runtime_error("--input_dtype requires value");
        } else if (flag == "--output_dtype") {
            if (const char* v = next(i)) a.output_dtype = v; else throw std::runtime_error("--output_dtype requires value");
        } else if (flag == "--seed") {
            const char* v = next(i); if (!v || !parse_u64(v, a.seed)) throw std::runtime_error("--seed u64");
        } else if (flag == "-h" || flag == "--help") {
            std::cout << "CUDA Protocol Tests - Generate Golden Reference Data\\n"
                         "Usage:\\n"
                         "  cuda_protocol_tests --op OP [--case ID] [OPTIONS...]\\n"
                         "\\n"
                         "This tool generates CUDA golden reference artifacts for Metal validation.\\n"
                         "\\n"
                         "Operations:\\n"
                         "  cuda_protocol_tests --op embedding_lookup [--case ID] [--num_tokens N] [--hidden_size D] [--vocab_size V] [--seed S]\\n"
                         "  cuda_protocol_tests --op extract_k_values [--case ID] [--M rows] [--N cols] [--k per_row] [--seed S]\\n"
                         "  cuda_protocol_tests --op rms_norm [--case ID] [--num_tokens N] [--hidden_size D] [--eps E] [--seed S]\\n"
                         "  cuda_protocol_tests --op silu_and_mul [--case ID] [--num_tokens N] [--intermediate_size I] [--seed S]\\n"
                         "  cuda_protocol_tests --op gemm [--case ID] [--m M] [--n N] [--k K] [--transa T] [--transb T] [--use_bias B] [--seed S]\\n"
                         "  cuda_protocol_tests --op rope [--case ID] [--num_tokens N] [--num_heads H] [--head_size S] [--rope_theta T] [--rope_factor F] [--seed S]\\n"
                         "  cuda_protocol_tests --op softmax [--case ID] [--batch_size B] [--vocab_size V] [--temperature T] [--seed S]\\n"
                         "  cuda_protocol_tests --op batch_prefill_attention [--case ID] [--num_tokens N] [--num_query_heads QH] [--num_kv_heads KH] [--head_size S] [--kv_len L] [--page_size P] [--seed S]\\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown flag: " + flag);
        }
    }
    if (a.op.empty()) throw std::runtime_error("--op is required");
    return a;
}

}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        // Ensure artifact writing is on
        if (!artifacts::get_env_flag("PIE_WRITE_ARTIFACTS", false)) {
            std::cerr << "Note: enabling PIE_WRITE_ARTIFACTS=1 for this run" << std::endl;
            setenv("PIE_WRITE_ARTIFACTS", "1", 1);
        }

        // Derive or use provided case id
        std::string case_id = args.case_id;
        if (case_id == "auto") {
            case_id = args.op + std::string("_N") + std::to_string(args.num_tokens)
                    + "_D" + std::to_string(args.hidden_size);
        }

        // Route operations (CUDA only)
        if (args.op == "embedding_lookup") {
            ops::EmbeddingConfig cfg{args.num_tokens, args.hidden_size, args.vocab_size};
            ops::run_embedding_lookup(case_id, cfg, args.seed);
        } else if (args.op == "extract_k_values") {
            ops::ExtractKConfig cfg{args.M, args.N, args.k};
            ops::run_extract_k_values(case_id, cfg, args.seed);
        } else if (args.op == "rms_norm") {
            ops::RMSNormConfig cfg{args.num_tokens, args.hidden_size, args.eps};
            ops::run_rms_norm(case_id, cfg, args.seed);
        } else if (args.op == "silu_and_mul") {
            ops::SiLUAndMulConfig cfg{args.num_tokens, args.intermediate_size};
            ops::run_silu_and_mul(case_id, cfg, args.seed);
        } else if (args.op == "gemm") {
            ops::GemmConfig cfg{args.m, args.n, args.k, args.transa, args.transb, args.use_bias};
            ops::run_gemm(case_id, cfg, args.seed);
        } else if (args.op == "rope") {
            ops::RoPEConfig cfg{args.num_tokens, args.num_query_heads, args.num_kv_heads, args.head_size, args.rope_theta, args.rope_factor, args.rope_low_frequency_factor, args.rope_high_frequency_factor, args.max_position_embeddings};
            ops::run_rope(case_id, cfg, args.seed);
        } else if (args.op == "topk_mask_logits") {
            ops::TopKMaskConfig cfg{args.num_tokens, args.vocab_size, args.k};
            ops::run_topk_mask_logits(case_id, cfg, args.seed);
        } else if (args.op == "softmax") {
            ops::SoftmaxConfig cfg{args.batch_size, args.vocab_size, args.temperature};
            ops::run_softmax(case_id, cfg, args.seed);
        } else if (args.op == "batch_prefill_attention") {
            ops::BatchPrefillAttentionConfig cfg{args.num_tokens, args.num_query_heads, args.num_kv_heads, args.head_size, args.kv_len, args.page_size};
            ops::run_batch_prefill_attention(case_id, cfg, args.seed);
        } else if (args.op == "grouped_gemm") {
            ops::GroupedGemmConfig cfg{args.num_groups, args.m, args.n, args.k, args.transa, args.transb, args.use_bias};
            ops::run_grouped_gemm(case_id, cfg, args.seed);
        } else if (args.op == "append_paged_kv_cache") {
            ops::AppendPagedKVCacheConfig cfg{args.num_tokens, args.num_kv_heads, args.head_size, args.page_size, args.max_num_pages, args.batch_size};
            ops::run_append_paged_kv_cache(case_id, cfg, args.seed);
        } else if (args.op == "gemm_all_dtypes") {
            ops::GemmConfig cfg{args.m, args.n, args.k, args.transa, args.transb, args.use_bias};
            ops::run_all_dtypes_for_operation("gemm", case_id, &cfg, args.seed);
        } else if (args.op == "embedding_lookup_all_dtypes") {
            ops::EmbeddingConfig cfg{args.num_tokens, args.hidden_size, args.vocab_size};
            ops::run_all_dtypes_for_operation("embedding_lookup", case_id, &cfg, args.seed);
        } else if (args.op == "extract_k_values_all_dtypes") {
            ops::ExtractKConfig cfg{args.M, args.N, args.k};
            ops::run_all_dtypes_for_operation("extract_k_values", case_id, &cfg, args.seed);
        } else {
            std::cerr << "Unsupported op: " << args.op << "\\n";
            return 2;
        }

        std::cout << "CUDA golden reference artifacts written under: tests/artifacts/" << args.op << "/" << case_id << "\\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return 1;
    }
}