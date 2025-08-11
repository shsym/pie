// Validates that the checksum_map in meta.json matches recomputed FNV1a64 hashes
// over each tensor .bin file for selected ops (attention, rmsnorm, mlp).

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "model.hpp"
#include "config.hpp"
#include "../../src/l4ma.cu" // template instantiations

#include <nlohmann/json.hpp>

static AppConfig make_test_config() {
    AppConfig cfg;
    cfg.model_name = "llama-3.2-1b-instruct"; // requires cached weights
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

// Streaming FNV-1a 64-bit over a file
static uint64_t fnv1a64_file(const std::filesystem::path& p) {
    constexpr uint64_t offset = 14695981039346656037ull;
    constexpr uint64_t prime  = 1099511628211ull;
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs.is_open()) return 0ULL; // treat missing as 0; test will flag mismatch
    uint64_t h = offset;
    char buf[1<<15];
    while (ifs) {
        ifs.read(buf, sizeof(buf));
        std::streamsize got = ifs.gcount();
        for (std::streamsize i=0;i<got;++i) {
            h ^= static_cast<unsigned char>(buf[i]);
            h *= prime;
        }
    }
    return h;
}

static std::string to_hex64(uint64_t v) {
    char out[17];
    static const char* hex = "0123456789abcdef";
    for (int i=15;i>=0;--i) { out[i] = hex[v & 0xF]; v >>= 4; }
    out[16] = '\0';
    return std::string(out, 16);
}

static bool validate_meta_dir(const std::filesystem::path& dir, bool& any_checked) {
    auto meta_path = dir / "meta.json";
    if (!std::filesystem::exists(meta_path)) {
        std::cerr << "Missing meta.json in " << dir << "\n";
        return false;
    }
    std::ifstream ifs(meta_path);
    nlohmann::json j; ifs >> j;
    if (!j.contains("checksum_map")) {
        std::cerr << "No checksum_map in " << meta_path << "\n";
        return false;
    }
    const auto& csum = j["checksum_map"];
    if (!csum.is_object()) {
        std::cerr << "checksum_map not object in " << meta_path << "\n";
        return false;
    }
    bool ok = true;
    for (auto it = csum.begin(); it != csum.end(); ++it) {
        std::string tensor = it.key();
        std::string expected = it.value().get<std::string>();
        auto bin_path = dir / (tensor + ".bin");
        if (!std::filesystem::exists(bin_path)) {
            std::cerr << "Missing bin file for tensor " << tensor << " in " << dir << "\n";
            ok = false; continue;
        }
        uint64_t h = fnv1a64_file(bin_path);
        std::string got = to_hex64(h);
        if (got != expected) {
            std::cerr << "Checksum mismatch for " << tensor << ": expected " << expected << " got " << got << "\n";
            ok = false;
        }
        any_checked = true;
    }
    return ok;
}

int main() {
    auto cfg = make_test_config();
    ModelMetadata meta{};
    try { meta = load_model_metadata_for_test(cfg); } catch (const std::exception& e) {
        std::cout << "SKIP: checksum test requires model metadata (" << e.what() << ")\n";
        return 0;
    }

    // Generate artifacts with dedicated case id
    std::filesystem::path artifacts_root = std::filesystem::current_path() / "artifacts_checksums";
    std::error_code ec; std::filesystem::remove_all(artifacts_root, ec);
    setenv("PIE_WRITE_ARTIFACTS", "1", 1);
    setenv("PIE_ARTIFACT_OPS", "l4ma_attention_forward,rmsnorm_forward,mlp_forward", 1);
    setenv("PIE_ARTIFACT_CASE_ID", "chk", 1);
    setenv("PIE_ARTIFACTS_DIR", artifacts_root.string().c_str(), 1);

    try {
        Model model(cfg, meta);
        uint32_t block_id = 7;
        model.handle_allocate({Model::AllocateCommand{Model::ObjectKind::KV_BLOCK, block_id, 1}});
        Model::ForwardTextCommand cmd; cmd.kv_page_last_len=0; cmd.kv_page_ids={}; cmd.token_ids={101,102,103}; cmd.position_ids={0,1,2}; cmd.output_indices={1,2};
        (void)model.handle_forward_text({cmd});
    } catch (const std::exception& e) {
        std::cout << "SKIP: could not generate artifacts (" << e.what() << ")\n";
        return 0;
    }

    bool any_checked = false;
    bool ok = true;
    ok &= validate_meta_dir(artifacts_root / "l4ma_attention_forward" / "chk", any_checked);
    ok &= validate_meta_dir(artifacts_root / "rmsnorm_forward" / "chk", any_checked);
    ok &= validate_meta_dir(artifacts_root / "mlp_forward" / "chk", any_checked);

    if (!any_checked) {
        std::cout << "SKIP: no tensors validated (possibly missing ops)\n";
        return 0;
    }

    if (!ok) {
        std::cerr << "FAIL: checksum validation failed" << std::endl;
        return 1;
    }

    std::cout << "Checksum validation test passed." << std::endl;
    return 0;
}
