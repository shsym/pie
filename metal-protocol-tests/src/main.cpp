#include "ops.hpp"
#include "artifacts.hpp"
#include "workspace_utils.hpp" // host-only artifact helpers for writing/reading artifacts

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

struct Args {
    std::string op;
    std::string case_id = "auto";
    bool auto_compare = true;           // Automatically compare with CUDA reference
    std::string cuda_artifacts_dir = "tests/artifacts";  // CUDA reference directory
    bool write_meta_from_cli = false;   // Emit minimal CUDA meta.json from CLI params and exit
    int num_tokens = 128;          // Realistic sequence length
    int hidden_size = 4096;        // Llama 7B hidden size
    int vocab_size = 32000;        // Llama vocab size
    int intermediate_size = 11008;  // Llama 7B intermediate size (FFN)
    int M = 128;                   // Batch tokens
    int N = 32000;                 // Vocab size for topk
    int k = 50;                    // Top-k value
    int m = 128;                   // GEMM batch tokens
    int n = 4096;                  // GEMM output dim
    int num_elements = 1024;
    float eps = 1e-5f;
    bool transa = false;
    bool transb = true;
    bool use_bias = false;
    std::string dtype = "bf16";        // Default data type (bf16, fp16, fp32)
    std::string input_dtype = "fp32";
    std::string output_dtype = "fp16";
    // RoPE parameters
    float rope_theta = 1e4f;
    float rope_factor = 1.0f;
    float rope_low_frequency_factor = 1.0f;
    float rope_high_frequency_factor = 1.0f;
    int max_position_embeddings = 8192;
    int num_heads = 32;             // Llama 7B num heads
    int head_size = 128;            // Llama 7B head size (4096/32)
    // Attention parameters
    int num_query_heads = 32;       // Llama 7B query heads
    int num_kv_heads = 32;          // Llama 7B KV heads (same as query for 7B)
    int kv_len = 2048;              // Realistic KV cache length
    int page_size = 16;             // FlashInfer page size
    // Grouped GEMM parameters
    int num_groups = 4;
    // Paged KV cache parameters
    int max_num_pages = 8;
    int batch_size = 2;
    float temperature = 1.0f;           // Softmax temperature parameter
    uint64_t seed = 0x12345678ULL;
};

bool parse_int(const char* s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

// Emit a minimal CUDA-style meta.json based on current Args for the given op/case
void emit_cuda_meta_from_cli(const Args& args) {
    // Map operation names where CUDA artifacts differ
    std::string cuda_op_name = args.op == "embedding_lookup" ? "embedding_lookup_forward" : args.op;
    std::filesystem::path dir = std::filesystem::path(args.cuda_artifacts_dir) / cuda_op_name / args.case_id;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    (void)ec;
    std::filesystem::path meta = dir / "meta.json";
    std::ostringstream os;

    // Generate new format for batch_prefill_attention, old format for others
    if (args.op == "batch_prefill_attention") {
        // Calculate dimensions
        int head_dim = args.num_query_heads * args.head_size;
        int kv_head_dim = args.num_kv_heads * args.head_size;
        int num_pages = (args.kv_len + args.page_size - 1) / args.page_size; // Ceiling division

        os << "{\n";
        os << "  \"version\": \"1\",\n";
        os << "  \"op\": \"" << cuda_op_name << "\",\n";
        os << "  \"backend\": \"cuda\",\n";
        os << "  \"case_id\": \"" << args.case_id << "\",\n";
        os << "  \"config\": {\n";
        os << "    \"num_tokens\": " << args.num_tokens << ",\n";
        os << "    \"num_query_heads\": " << args.num_query_heads << ",\n";
        os << "    \"num_kv_heads\": " << args.num_kv_heads << ",\n";
        os << "    \"head_size\": " << args.head_size << ",\n";
        os << "    \"kv_len\": " << args.kv_len << ",\n";
        os << "    \"page_size\": " << args.page_size << ",\n";
        os << "    \"batch_size\": 1,\n";
        os << "    \"num_pages\": " << num_pages << "\n";
        os << "  },\n";
        os << "  \"dtype_map\": {\n";
        os << "    \"q_input\": \"" << args.dtype << "\",\n";
        os << "    \"k_input\": \"" << args.dtype << "\",\n";
        os << "    \"v_input\": \"" << args.dtype << "\",\n";
        os << "    \"paged_k_cache\": \"" << args.dtype << "\",\n";
        os << "    \"paged_v_cache\": \"" << args.dtype << "\",\n";
        os << "    \"output\": \"" << args.dtype << "\",\n";
        os << "    \"qo_indptr\": \"s32\",\n";
        os << "    \"kv_page_indptr\": \"s32\",\n";
        os << "    \"kv_page_indices\": \"s32\",\n";
        os << "    \"kv_last_page_lens\": \"s32\"\n";
        os << "  },\n";
        os << "  \"shape_map\": {\n";
        os << "    \"q_input\": [" << args.num_tokens << ", " << head_dim << "],\n";
        os << "    \"k_input\": [" << (num_pages * args.page_size) << ", " << kv_head_dim << "],\n";
        os << "    \"v_input\": [" << (num_pages * args.page_size) << ", " << kv_head_dim << "],\n";
        os << "    \"paged_k_cache\": [" << num_pages << ", " << args.page_size << ", " << kv_head_dim << "],\n";
        os << "    \"paged_v_cache\": [" << num_pages << ", " << args.page_size << ", " << kv_head_dim << "],\n";
        os << "    \"output\": [" << args.num_tokens << ", " << head_dim << "],\n";
        os << "    \"qo_indptr\": [2],\n";
        os << "    \"kv_page_indptr\": [2],\n";
        os << "    \"kv_page_indices\": [" << num_pages << "],\n";
        os << "    \"kv_last_page_lens\": [1]\n";
        os << "  }\n";
        os << "}\n";
    } else {
        // Legacy format for other operations
        os << "{\n";
        os << "  \"op\": \"" << cuda_op_name << "\",\n";
        os << "  \"case_id\": \"" << args.case_id << "\",\n";
        os << "  \"dtype\": \"" << args.dtype << "\",\n";
        os << "  \"num_tokens\": " << args.num_tokens << ",\n";
        os << "  \"hidden_size\": " << args.hidden_size << ",\n";
        os << "  \"vocab_size\": " << args.vocab_size << ",\n";
        os << "  \"intermediate_size\": " << args.intermediate_size << ",\n";
        os << "  \"M\": " << args.M << ",\n";
        os << "  \"N\": " << args.N << ",\n";
        os << "  \"k\": " << args.k << ",\n";
        os << "  \"m\": " << args.m << ",\n";
        os << "  \"n\": " << args.n << ",\n";
        os << "  \"batch_size\": " << args.batch_size << ",\n";
        os << "  \"num_heads\": " << args.num_heads << ",\n";
        os << "  \"head_size\": " << args.head_size << ",\n";
        os << "  \"num_query_heads\": " << args.num_query_heads << ",\n";
        os << "  \"num_kv_heads\": " << args.num_kv_heads << ",\n";
        os << "  \"kv_len\": " << args.kv_len << ",\n";
        os << "  \"page_size\": " << args.page_size << ",\n";
        os << "  \"num_groups\": " << args.num_groups << ",\n";
        os << "  \"max_num_pages\": " << args.max_num_pages << ",\n";
        os << "  \"eps\": " << args.eps << ",\n";
        os << "  \"temperature\": " << args.temperature << "\n";
        os << "}\n";
    }

    std::ofstream ofs(meta);
    ofs << os.str();
    std::cout << "Wrote CUDA-style meta.json to " << meta << " with dtype=" << args.dtype << std::endl;
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

// Simple JSON value extraction (for basic config reading)
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.length();
    while (pos < json.length() && std::isspace(json[pos])) pos++;

    if (pos >= json.length() || json[pos] != '"') return "";
    pos++; // skip opening quote

    size_t end = json.find('"', pos);
    if (end == std::string::npos) return "";

    return json.substr(pos, end - pos);
}

int extract_json_int(const std::string& json, const std::string& key, int default_val) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.length();
    while (pos < json.length() && std::isspace(json[pos])) pos++;

    std::string num_str;
    while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '-')) {
        num_str += json[pos];
        pos++;
    }

    if (num_str.empty()) return default_val;

    try {
        return std::stoi(num_str);
    } catch (...) {
        return default_val;
    }
}

float extract_json_float(const std::string& json, const std::string& key, float default_val) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos += search.length();
    while (pos < json.length() && std::isspace(json[pos])) pos++;

    std::string num_str;
    while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '.' || json[pos] == '-' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+')) {
        num_str += json[pos];
        pos++;
    }

    if (num_str.empty()) return default_val;

    try {
        return std::stof(num_str);
    } catch (...) {
        return default_val;
    }
}

// Read CUDA reference metadata and override parameters if auto_compare is enabled
void override_with_cuda_metadata(Args& args, const std::string& cuda_artifacts_dir) {
    if (!args.auto_compare) return;

    // Map operation names to their CUDA artifact directory names
    std::string cuda_op_name = args.op;
    if (args.op == "embedding_lookup") cuda_op_name = "embedding_lookup_forward";

    std::filesystem::path cuda_meta_path = std::filesystem::path(cuda_artifacts_dir) / cuda_op_name / args.case_id / "meta.json";

    if (!std::filesystem::exists(cuda_meta_path)) {
        // Caller should have enforced meta presence already. Keep this as a hard error path.
        throw std::runtime_error(std::string("Required CUDA meta.json not found at ") + cuda_meta_path.string());
    }

    try {
        std::ifstream file(cuda_meta_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_content = buffer.str();

        std::cout << "Reading CUDA reference metadata from " << cuda_meta_path << std::endl;

        // Extract config parameters from JSON
        args.num_tokens = extract_json_int(json_content, "num_tokens", args.num_tokens);
        args.hidden_size = extract_json_int(json_content, "hidden_size", args.hidden_size);
        args.vocab_size = extract_json_int(json_content, "vocab_size", args.vocab_size);
        args.intermediate_size = extract_json_int(json_content, "intermediate_size", args.intermediate_size);

        // Extract M, N, k from nested "config" object if present (new CUDA artifact format)
        std::string config_search = "\"config\":";
        size_t config_pos = json_content.find(config_search);
        if (config_pos != std::string::npos) {
            // Find the config object content between braces
            size_t config_start = json_content.find('{', config_pos + config_search.length());
            if (config_start != std::string::npos) {
                size_t config_end = json_content.find('}', config_start);
                if (config_end != std::string::npos) {
                    std::string config_content = json_content.substr(config_start, config_end - config_start + 1);
                    args.M = extract_json_int(config_content, "M", args.M);
                    args.N = extract_json_int(config_content, "N", args.N);
                    args.k = extract_json_int(config_content, "k", args.k);
                }
            }
        } else {
            // Fallback to root level for backward compatibility
            args.M = extract_json_int(json_content, "M", args.M);
            args.N = extract_json_int(json_content, "N", args.N);
            args.k = extract_json_int(json_content, "k", args.k);
        }
        args.m = extract_json_int(json_content, "m", args.m);
        args.n = extract_json_int(json_content, "n", args.n);
        args.batch_size = extract_json_int(json_content, "batch_size", args.batch_size);
        args.num_heads = extract_json_int(json_content, "num_heads", args.num_heads);
        args.head_size = extract_json_int(json_content, "head_size", args.head_size);
        args.num_query_heads = extract_json_int(json_content, "num_query_heads", args.num_query_heads);
        args.num_kv_heads = extract_json_int(json_content, "num_kv_heads", args.num_kv_heads);
        args.kv_len = extract_json_int(json_content, "kv_len", args.kv_len);
        args.page_size = extract_json_int(json_content, "page_size", args.page_size);
        args.num_groups = extract_json_int(json_content, "num_groups", args.num_groups);
        args.max_num_pages = extract_json_int(json_content, "max_num_pages", args.max_num_pages);

        args.eps = extract_json_float(json_content, "eps", args.eps);
        args.temperature = extract_json_float(json_content, "temperature", args.temperature);
        args.rope_theta = extract_json_float(json_content, "rope_theta", args.rope_theta);
        args.rope_factor = extract_json_float(json_content, "rope_factor", args.rope_factor);
        args.rope_low_frequency_factor = extract_json_float(json_content, "rope_low_frequency_factor", args.rope_low_frequency_factor);
        args.rope_high_frequency_factor = extract_json_float(json_content, "rope_high_frequency_factor", args.rope_high_frequency_factor);

        std::cout << "Loaded parameters from CUDA reference:" << std::endl;
        std::cout << "  num_tokens=" << args.num_tokens << ", hidden_size=" << args.hidden_size << ", vocab_size=" << args.vocab_size << std::endl;
        std::cout << "  batch_size=" << args.batch_size << ", temperature=" << args.temperature << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to read CUDA metadata from " << cuda_meta_path << ": " << e.what() << std::endl;
        std::cerr << "Using command-line parameters instead" << std::endl;
    }
}

// Try to auto-select a CUDA case if requested case doesn't exist or is 'auto'
void maybe_autoselect_cuda_case(Args& args) {
    if (!args.auto_compare) return;
    // Map op name to CUDA artifacts op folder
    std::string cuda_op_name = args.op;
    if (args.op == "embedding_lookup") cuda_op_name = "embedding_lookup_forward";

    std::filesystem::path op_dir = std::filesystem::path(args.cuda_artifacts_dir) / cuda_op_name;
    if (!std::filesystem::exists(op_dir) || !std::filesystem::is_directory(op_dir)) {
        return; // Nothing to auto-select from
    }

    auto meta_for = [&](const std::string& case_name) {
        return op_dir / case_name / "meta.json";
    };

    auto exists_meta = [&](const std::string& case_name) {
        auto p = meta_for(case_name);
        return std::filesystem::exists(p);
    };

    bool need_select = (args.case_id == "auto") || !exists_meta(args.case_id);
    if (!need_select) return;

    // Prefer 'test_unified' if present, else the first directory entry with meta.json
    if (exists_meta("test_unified")) {
        std::cout << "Auto-selecting CUDA reference case: test_unified" << std::endl;
        args.case_id = "test_unified";
        return;
    }

    for (auto& entry : std::filesystem::directory_iterator(op_dir)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        if (exists_meta(name)) {
            std::cout << "Auto-selecting CUDA reference case: " << name << std::endl;
            args.case_id = name;
            return;
        }
    }
    // If we got here, no meta.json cases found; leave args.case_id as-is
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
        } else if (flag == "--no-compare") {
            a.auto_compare = false;
        } else if (flag == "--cuda-artifacts-dir") {
            if (const char* v = next(i)) a.cuda_artifacts_dir = v; else throw std::runtime_error("--cuda-artifacts-dir requires value");
        } else if (flag == "--write-meta-from-cli") {
            a.write_meta_from_cli = true;
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
        } else if (flag == "--rope_theta") {
            const char* v = next(i); if (!v || !parse_float(v, a.rope_theta)) throw std::runtime_error("--rope_theta float");
        } else if (flag == "--rope_factor") {
            const char* v = next(i); if (!v || !parse_float(v, a.rope_factor)) throw std::runtime_error("--rope_factor float");
        } else if (flag == "--rope_low_frequency_factor") {
            const char* v = next(i); if (!v || !parse_float(v, a.rope_low_frequency_factor)) throw std::runtime_error("--rope_low_frequency_factor float");
        } else if (flag == "--rope_high_frequency_factor") {
            const char* v = next(i); if (!v || !parse_float(v, a.rope_high_frequency_factor)) throw std::runtime_error("--rope_high_frequency_factor float");
        } else if (flag == "--max_position_embeddings") {
            const char* v = next(i); if (!v || !parse_int(v, a.max_position_embeddings)) throw std::runtime_error("--max_position_embeddings int");
        } else if (flag == "--num_heads") {
            const char* v = next(i); if (!v || !parse_int(v, a.num_heads)) throw std::runtime_error("--num_heads int");
        } else if (flag == "--head_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.head_size)) throw std::runtime_error("--head_size int");
        } else if (flag == "--num_query_heads") {
            const char* v = next(i); if (!v || !parse_int(v, a.num_query_heads)) throw std::runtime_error("--num_query_heads int");
        } else if (flag == "--num_kv_heads") {
            const char* v = next(i); if (!v || !parse_int(v, a.num_kv_heads)) throw std::runtime_error("--num_kv_heads int");
        } else if (flag == "--kv_len") {
            const char* v = next(i); if (!v || !parse_int(v, a.kv_len)) throw std::runtime_error("--kv_len int");
        } else if (flag == "--page_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.page_size)) throw std::runtime_error("--page_size int");
        } else if (flag == "--temperature") {
            const char* v = next(i); if (!v || !parse_float(v, a.temperature)) throw std::runtime_error("--temperature float");
        } else if (flag == "--batch_size") {
            const char* v = next(i); if (!v || !parse_int(v, a.batch_size)) throw std::runtime_error("--batch_size int");
        } else if (flag == "--seed") {
            const char* v = next(i); if (!v || !parse_u64(v, a.seed)) throw std::runtime_error("--seed u64");
        } else if (flag == "--dtype") {
            if (const char* v = next(i)) a.dtype = v; else throw std::runtime_error("--dtype requires value");
        } else if (flag == "-h" || flag == "--help") {
            std::cout << "Usage:\\n"
                         "  metal_protocol_tests --op OP [--case ID] [OPTIONS...]\n"
                         "\\n"
                         "Metal Protocol Tests - Test Metal backend against CUDA golden reference\\n"
                         "\\n"
                         "Options:\\n"
                         "  --op OP                     Operation to test (required)\\n"
                         "  --case ID                   Test case identifier (default: auto)\\n"
                         "  --no-compare                Skip automatic CUDA vs Metal comparison\\n"
                         "  --write-meta-from-cli       Emit CUDA-style meta.json based on CLI and exit (for ad-hoc)\n"
                         "  --cuda-artifacts-dir DIR    Directory containing CUDA reference artifacts\\n"
                         "                              (default: ../cuda-protocol-tests/tests/artifacts)\\n"
                         "\\n"
                         "Operations:\\n"
                         "  metal_protocol_tests --op embedding_lookup [--case ID] [--num_tokens N] [--hidden_size D] [--vocab_size V] [--seed S]\\n"
                         "  metal_protocol_tests --op extract_k_values [--case ID] [--M rows] [--N cols] [--k per_row] [--seed S]\\n"
                         "  metal_protocol_tests --op rms_norm [--case ID] [--num_tokens N] [--hidden_size D] [--eps E] [--seed S]\\n"
                         "  metal_protocol_tests --op silu_and_mul [--case ID] [--num_tokens N] [--intermediate_size I] [--seed S]\\n"
                         "  metal_protocol_tests --op add_residual [--case ID] [--num_tokens N] [--hidden_size D] [--seed S]\\n"
                         "  metal_protocol_tests --op gemm [--case ID] [--m M] [--n N] [--k K] [--transa T] [--transb T] [--use_bias B] [--seed S]\\n"
                         "  metal_protocol_tests --op cast_type [--case ID] [--num_elements N] [--input_dtype T] [--output_dtype T] [--seed S]\\n"
                         "  metal_protocol_tests --op rope [--case ID] [--num_tokens N] [--num_heads H] [--head_size S] [--rope_theta T] [--rope_factor F] [--seed S]\\n"
                         "  metal_protocol_tests --op topk_mask_logits [--case ID] [--num_tokens N] [--vocab_size V] [--k K] [--seed S]\\n"
                         "  metal_protocol_tests --op softmax [--case ID] [--batch_size B] [--vocab_size V] [--temperature T] [--seed S]\\n"
                         "  metal_protocol_tests --op batch_prefill_attention [--case ID] [--num_tokens N] [--num_query_heads QH] [--num_kv_heads KH] [--head_size S] [--kv_len L] [--page_size P] [--seed S]\\n";
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
    bool comparison_ok = true; // Track CUDA vs Metal comparison result
    // Resolve comparator script path relative to the executable directory so CWD doesn't matter
    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::filesystem::path comparator_script_path = (exe_dir / "../scripts/compare_artifacts.py").lexically_normal();
    const std::string comparator_script = comparator_script_path.string();

        // Fix cuda_artifacts_dir to be absolute if it's still the default relative path
        if (args.cuda_artifacts_dir == "tests/artifacts") {
            auto cuda_artifacts_path = workspace_utils::get_cuda_artifacts_dir();
            args.cuda_artifacts_dir = cuda_artifacts_path.string();
            std::cout << "Using CUDA artifacts from: " << args.cuda_artifacts_dir << std::endl;
        }
        // Also export for per-op wrappers that look up CUDA inputs directly
        setenv("PIE_CUDA_ARTIFACTS_DIR", args.cuda_artifacts_dir.c_str(), 1);

    // If requested case doesn't exist (or is 'auto'), try to auto-select an available CUDA reference case
    maybe_autoselect_cuda_case(args);

    // Handle meta emission for ad-hoc runs, or enforce presence of CUDA meta when comparing
    if (args.auto_compare) {
        // Map op name for CUDA artifacts
        std::string cuda_op_name = args.op == "embedding_lookup" ? "embedding_lookup_forward" : args.op;
        std::filesystem::path cuda_meta_path = std::filesystem::path(args.cuda_artifacts_dir) / cuda_op_name / args.case_id / "meta.json";
        if (args.write_meta_from_cli) {
            emit_cuda_meta_from_cli(args);
            std::cout << "Exiting after writing meta (no execution performed)." << std::endl;
            return 0;
        }
        if (!std::filesystem::exists(cuda_meta_path)) {
            throw std::runtime_error(std::string("Missing required CUDA meta.json for ") + cuda_op_name + "/" + args.case_id +
                                     ". Run with --write-meta-from-cli to generate one for ad-hoc runs, or provide valid artifacts under --cuda-artifacts-dir.");
        }
    }

    // Override parameters with CUDA reference metadata (enforced above)
    override_with_cuda_metadata(args, args.cuda_artifacts_dir);

        // Resolve a default artifacts base inside the build directory (next to the executable)
        // so runs don't spill artifacts into the repo root even if CWD changes.
        std::filesystem::path default_metal_artifacts_base_path = (exe_dir / "tests/artifacts").lexically_normal();
        const std::string default_metal_artifacts_base = default_metal_artifacts_base_path.string();
        // If PIE_ARTIFACTS_DIR is not set, set it to our default under the build dir
        if (std::getenv("PIE_ARTIFACTS_DIR") == nullptr) {
            setenv("PIE_ARTIFACTS_DIR", default_metal_artifacts_base.c_str(), 1);
        }
        // Ensure artifact writing is on unless explicitly disabled
        if (!artifacts::get_env_flag("PIE_WRITE_ARTIFACTS", false)) {
            std::cerr << "Note: enabling PIE_WRITE_ARTIFACTS=1 for this run" << std::endl;
            setenv("PIE_WRITE_ARTIFACTS", "1", 1);
        }

        // Derive or use provided case id
        std::string case_id = args.case_id;
        if (case_id == "auto") {
            // Compose auto case id from op and params for easy grouping
            case_id = args.op + std::string("_N") + std::to_string(args.num_tokens)
                    + "_D" + std::to_string(args.hidden_size);
        }

        // Auto-comparison function
    auto run_comparison = [&](const std::string& op, const std::string& case_id) {
            if (!args.auto_compare) {
                return;
            }

            std::cout << "\\n=== Automatic Comparison with CUDA Reference ===" << std::endl;

            // Construct comparison command
            std::string comparison_cmd = std::string("python ") + comparator_script;
            comparison_cmd += " --op " + op;
            comparison_cmd += " --case " + case_id;
            comparison_cmd += " --cuda-base " + args.cuda_artifacts_dir;
            // Use the resolved artifacts base (env may override our default)
            const char* env_metal_base = std::getenv("PIE_ARTIFACTS_DIR");
            const std::string metal_artifacts_base = env_metal_base ? std::string(env_metal_base) : default_metal_artifacts_base;
            comparison_cmd += " --metal-base " + metal_artifacts_base;
            // Skip --verbose to reduce output clutter

            std::cout << "Running: " << comparison_cmd << std::endl;

            int result = std::system(comparison_cmd.c_str());
            if (result == 0) {
                std::cout << "✅ Metal implementation matches CUDA reference!" << std::endl;
            } else {
                std::cout << "❌ Metal implementation differs from CUDA reference. See details above." << std::endl;
                comparison_ok = false;
            }
        };

        // Route operations (Metal-only harness)
        if (args.op == "embedding_lookup") {
            ops::EmbeddingConfig cfg{args.num_tokens, args.hidden_size, args.vocab_size};
            ops::run_embedding_lookup_metal(case_id, cfg, args.seed);
            run_comparison("embedding_lookup_forward", case_id);
        } else if (args.op == "extract_k_values") {
            ops::ExtractKConfig cfg{args.M, args.N, args.k};
            ops::run_extract_k_values_metal(case_id, cfg, args.seed);
            run_comparison("extract_k_values", case_id);
        } else if (args.op == "rms_norm") {
            ops::RMSNormConfig cfg{args.num_tokens, args.hidden_size, args.eps};
            ops::run_rms_norm_metal(case_id, cfg, args.seed);
            run_comparison("rms_norm", case_id);
        } else if (args.op == "silu_and_mul") {
            ops::SiLUAndMulConfig cfg{args.num_tokens, args.intermediate_size};
            ops::run_silu_and_mul_metal(case_id, cfg, args.seed);
            run_comparison("silu_and_mul", case_id);
        } else if (args.op == "gemm") {
            ops::GemmConfig cfg{args.m, args.n, args.k, args.transa, args.transb, args.use_bias};
            ops::run_gemm_metal(case_id, cfg, args.seed);
            run_comparison("gemm", case_id);
        } else if (args.op == "add_residual") {
            ops::AddResidualConfig cfg{args.num_tokens, args.hidden_size};
            ops::run_add_residual_metal(case_id, cfg, args.seed);
            run_comparison("add_residual", case_id);
    } else if (args.op == "cast_type") {
            ops::CastTypeConfig cfg{args.num_elements, args.input_dtype, args.output_dtype};
            std::cerr << "Error: cast_type not implemented for Metal backend yet" << std::endl;
            return 1;
        } else if (args.op == "rope") {
            ops::RoPEConfig cfg{args.num_tokens, args.num_query_heads, args.num_kv_heads, args.head_size, args.rope_theta, args.rope_factor, args.rope_low_frequency_factor, args.rope_high_frequency_factor, args.max_position_embeddings};
            ops::run_rope_metal(args.case_id, cfg, args.seed);
            run_comparison("rope", args.case_id);
        } else if (args.op == "topk_mask_logits") {
            ops::TopKMaskConfig cfg{args.num_tokens, args.vocab_size, args.k};
            ops::run_topk_mask_logits_metal(args.case_id, cfg, args.seed);
            run_comparison("topk_mask_logits", args.case_id);
        } else if (args.op == "softmax") {
            ops::SoftmaxConfig cfg{args.batch_size, args.vocab_size, args.temperature};
            ops::run_softmax_metal(case_id, cfg, args.seed);
            run_comparison("softmax", case_id);
        } else if (args.op == "batch_prefill_attention") {
            ops::BatchPrefillAttentionConfig cfg{args.num_tokens, args.num_query_heads, args.num_kv_heads, args.head_size, args.kv_len, args.page_size};
            ops::run_batch_prefill_attention_metal(case_id, cfg, args.seed);
            run_comparison("batch_prefill_attention", case_id);
        } else if (args.op == "grouped_gemm") {
            ops::GroupedGemmConfig cfg{args.num_groups, args.m, args.n, args.k, args.transa, args.transb, args.use_bias};
            ops::run_grouped_gemm_metal(args.case_id, cfg, args.seed);
            run_comparison("grouped_gemm", args.case_id);
        } else if (args.op == "append_paged_kv_cache") {
            ops::AppendPagedKVCacheConfig cfg{args.num_tokens, args.num_kv_heads, args.head_size, args.page_size, args.max_num_pages, args.batch_size};
            ops::run_append_paged_kv_cache_metal(args.case_id, cfg, args.seed);
            run_comparison("append_paged_kv_cache", args.case_id);
    } else if (args.op == "embedding_lookup_all_dtypes") {
            ops::EmbeddingConfig cfg{args.num_tokens, args.hidden_size, args.vocab_size};
            std::cerr << "Error: embedding_lookup_all_dtypes not implemented for Metal backend yet" << std::endl;
            return 1;
        } else if (args.op == "extract_k_values_all_dtypes") {
            ops::ExtractKConfig cfg{args.M, args.N, args.k};
            std::cerr << "Error: extract_k_values_all_dtypes not implemented for Metal backend yet" << std::endl;
            return 1;
        } else {
            std::cerr << "Unsupported op: " << args.op << "\n";
            return 2;
        }

        // If comparison failed, propagate failure via exit code so scripts can detect it
        if (args.auto_compare && !comparison_ok) {
            std::cerr << "Comparison with CUDA reference failed" << "\n";
            return 2;
        }

    // Report where artifacts were written
    const char* env_metal_base2 = std::getenv("PIE_ARTIFACTS_DIR");
    const std::string metal_artifacts_base2 = env_metal_base2 ? std::string(env_metal_base2) : default_metal_artifacts_base;
    std::cout << "Artifacts directory base: " << metal_artifacts_base2
          << " (Metal runs append _metal to case folder)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
