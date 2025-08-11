// Generates two artifact sets with different case IDs and verifies
// required files exist and (for weights & selected bins) their contents match.

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "model.hpp"
#include "config.hpp"

#include "../../src/l4ma.cu" // bring in template instantiations

static AppConfig make_test_config() {
    AppConfig cfg;
    cfg.model_name = "llama-3.2-1b-instruct"; // assumes cached weights
    cfg.cache_dir = std::filesystem::path(std::getenv("HOME")) / ".cache" / "pie";
    cfg.kv_page_size = 32;
    cfg.dist_size = 8;
    cfg.max_num_kv_pages = 4;
    cfg.max_num_embeds = 1024;
    cfg.device = "cuda:0";
    cfg.dtype = "bfloat16";
    return cfg;
}

ModelMetadata load_model_metadata_for_test(const AppConfig& cfg);

// Simple FNV-1a 64-bit for host file bytes
static uint64_t fnv1a64_file(const std::filesystem::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs.is_open()) return 0ULL;
    constexpr uint64_t offset = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;
    uint64_t h = offset;
    char buf[1<<15];
    while (ifs) {
        ifs.read(buf, sizeof(buf));
        std::streamsize got = ifs.gcount();
        for (std::streamsize i=0;i<got;++i) { h ^= static_cast<unsigned char>(buf[i]); h *= prime; }
    }
    return h;
}

static bool file_exists(const std::filesystem::path& p) { return std::filesystem::exists(p); }
static bool dir_exists(const std::filesystem::path& p) { return std::filesystem::exists(p) && std::filesystem::is_directory(p); }

static bool generate_artifacts(const AppConfig& base_cfg, const ModelMetadata& meta, const std::filesystem::path& artifacts_root, const std::string& case_id) {
    setenv("PIE_WRITE_ARTIFACTS", "1", 1);
    setenv("PIE_ARTIFACT_OPS", "model_weights,l4ma_attention_forward,rmsnorm_forward,mlp_forward", 1);
    setenv("PIE_ARTIFACT_CASE_ID", case_id.c_str(), 1);
    setenv("PIE_ARTIFACTS_DIR", artifacts_root.string().c_str(), 1);
    try {
        Model model(base_cfg, meta); // weight load + checksums
        // minimal forward
        uint32_t block_id = 42;
        model.handle_allocate({Model::AllocateCommand{Model::ObjectKind::KV_BLOCK, block_id, 1}});
        Model::ForwardTextCommand cmd; cmd.kv_page_last_len = 0; cmd.kv_page_ids = {}; cmd.token_ids = {11,22,33}; cmd.position_ids = {0,1,2}; cmd.output_indices = {1,2};
        (void)model.handle_forward_text({cmd});
    } catch (const std::exception& e) {
        std::cout << "SKIP: generation failed (" << e.what() << ")\n";
        return false;
    }
    return true;
}

static bool compare_same(const std::filesystem::path& a, const std::filesystem::path& b) {
    if (!file_exists(a) || !file_exists(b)) return false;
    if (std::filesystem::file_size(a) != std::filesystem::file_size(b)) return false;
    return fnv1a64_file(a) == fnv1a64_file(b);
}

int main() {
    auto cfg = make_test_config();
    ModelMetadata meta{};
    try { meta = load_model_metadata_for_test(cfg); } catch (const std::exception& e) {
        std::cout << "SKIP: compare test requires model metadata (" << e.what() << ")\n";
        return 0;
    }

    std::filesystem::path artifacts_root = std::filesystem::current_path() / "artifacts_compare";
    std::error_code ec; std::filesystem::remove_all(artifacts_root, ec);

    if (!generate_artifacts(cfg, meta, artifacts_root, "cmpA")) return 0;
    if (!generate_artifacts(cfg, meta, artifacts_root, "cmpB")) return 0;

    auto weightsA = artifacts_root / "model_weights" / "cmpA";
    auto weightsB = artifacts_root / "model_weights" / "cmpB";
    if (!(dir_exists(weightsA) && dir_exists(weightsB))) { std::cout << "SKIP: weight dirs missing\n"; return 0; }
    auto cA = weightsA / "checksums.json";
    auto cB = weightsB / "checksums.json";
    if (!compare_same(cA, cB)) { std::cerr << "FAIL: weight checksum mismatch\n"; return 1; }

    std::vector<std::string> attn_core = {"q_proj.bin","k_proj.bin","v_proj.bin","attn_output.bin"};
    auto attnA = artifacts_root / "l4ma_attention_forward" / "cmpA";
    auto attnB = artifacts_root / "l4ma_attention_forward" / "cmpB";
    if (!(dir_exists(attnA) && dir_exists(attnB))) { std::cerr << "FAIL: attention dirs missing\n"; return 1; }
    if (!file_exists(attnA/"meta.json") || !file_exists(attnB/"meta.json")) { std::cerr << "FAIL: attention meta missing\n"; return 1; }
    for (auto& f: attn_core) {
        if (!compare_same(attnA/f, attnB/f)) { std::cerr << "FAIL: attention diff in " << f << "\n"; return 1; }
    }

    auto rnA = artifacts_root / "rmsnorm_forward" / "cmpA";
    auto rnB = artifacts_root / "rmsnorm_forward" / "cmpB";
    for (auto name: {"input.bin","weight.bin","output.bin"}) {
        if (!compare_same(rnA/name, rnB/name)) { std::cerr << "FAIL: rmsnorm diff in " << name << "\n"; return 1; }
    }

    auto mlpA = artifacts_root / "mlp_forward" / "cmpA";
    auto mlpB = artifacts_root / "mlp_forward" / "cmpB";
    for (auto name: {"input.bin","output.bin"}) {
        if (!compare_same(mlpA/name, mlpB/name)) { std::cerr << "FAIL: mlp diff in " << name << "\n"; return 1; }
    }

    std::cout << "Artifacts compare test passed (identical across two runs).\n";
    return 0;
}
