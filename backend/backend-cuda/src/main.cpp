#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

#include "utils.hpp" // For base64_decode and get_user_cache_dir
#include <toml++/toml.hpp>


#include "config.hpp"
#include "server.hpp"

#include <CLI/CLI.hpp>
#include <iostream>


// ======================================================================================
// MARK: - Parsing and Helper Functions
// ======================================================================================

inline std::map<int, std::vector<uint8_t>> load_merge_rules(const std::filesystem::path& path) {
    std::map<int, std::vector<uint8_t>> merge_rules;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to load vocabulary file: " + path.string());

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) continue;

        std::string::size_type pos = line.find(' ');
        if (pos == std::string::npos) throw std::runtime_error("Error on line " + std::to_string(line_number) + ": expected 2 parts, but found 1.");
        
        std::string b64_token = line.substr(0, pos);
        std::string rank_str = line.substr(pos + 1);

        try {
            merge_rules[std::stoi(rank_str)] = utils::base64_decode(b64_token);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    return merge_rules;
}

inline ModelMetadata parse_model_metadata(const std::filesystem::path& metadata_path) {
    if (!std::filesystem::exists(metadata_path)) {
        throw std::runtime_error("Metadata file not found at: " + metadata_path.string());
    }
    
    const auto metadata_dir = metadata_path.parent_path();

    toml::table tbl;
    try {
        tbl = toml::parse_file(metadata_path.string());
    } catch (const toml::parse_error& err) {
        throw std::runtime_error("Failed to parse model metadata TOML '" + metadata_path.string() + "': " + std::string(err.what()));
    }

    ModelMetadata metadata;

    // Helper to get a required value from a TOML table.
    auto get_required = [&](const toml::table& t, const std::string& key, const std::string& section) {
        auto node = t[key];
        if (!node) {
            throw std::runtime_error("Missing required key '" + key + "' in section '[" + section + "]' of TOML file: " + metadata_path.string());
        }
        return node;
    };
    
    // --- Parse top-level keys ---
    metadata.name = get_required(tbl, "name", "top-level").value_or("");
    metadata.description = get_required(tbl, "description", "top-level").value_or("");
    metadata.version = get_required(tbl, "version", "top-level").value_or("");

    if (auto params_node = get_required(tbl, "parameters", "top-level").as_array()) {
        for (const auto& elem : *params_node) {
            metadata.parameters.push_back(elem.value_or(""));
        }
    }

    // --- Parse Architecture ---
    auto arch_data = get_required(tbl, "architecture", "top-level").as_table();
    auto rope_data = get_required(*arch_data, "rope", "architecture").as_table();
    
    metadata.architecture = {
        .type = get_required(*arch_data, "type", "architecture").value_or(""),
        .num_layers = get_required(*arch_data, "num_layers", "architecture").value_or(0),
        .num_query_heads = get_required(*arch_data, "num_query_heads", "architecture").value_or(0),
        .num_key_value_heads = get_required(*arch_data, "num_key_value_heads", "architecture").value_or(0),
        .head_size = get_required(*arch_data, "head_size", "architecture").value_or(0),
        .hidden_size = get_required(*arch_data, "hidden_size", "architecture").value_or(0),
        .intermediate_size = get_required(*arch_data, "intermediate_size", "architecture").value_or(0),
        .vocab_size = get_required(*arch_data, "vocab_size", "architecture").value_or(0),
        .use_qkv_bias = false, // Hardcoded
        .rms_norm_eps = 1e-05f, // Hardcoded
        .rope_factor = static_cast<float>(get_required(*rope_data, "factor", "architecture.rope").value_or(0.0)),
        .rope_high_frequency_factor = static_cast<float>(get_required(*rope_data, "high_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_low_frequency_factor = static_cast<float>(get_required(*rope_data, "low_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_theta = static_cast<float>(get_required(*rope_data, "theta", "architecture.rope").value_or(0.0)),
    };

    // --- Parse Tokenizer ---
    auto tokenizer_data = get_required(tbl, "tokenizer", "top-level").as_table();
    std::string vocab_file = get_required(*tokenizer_data, "vocabulary_file", "tokenizer").value_or("");
    auto vocab_full_path = metadata_dir / vocab_file;
    
    metadata.tokenizer.type = get_required(*tokenizer_data, "type", "tokenizer").value_or("");
    metadata.tokenizer.split_regex = get_required(*tokenizer_data, "split_regex", "tokenizer").value_or("");
    metadata.tokenizer.merge_table = load_merge_rules(vocab_full_path);

    if (auto special_tokens_node = get_required(*tokenizer_data, "special_tokens", "tokenizer").as_table()) {
        for (const auto& [k, v] : *special_tokens_node) {
            if (auto id = v.value<int>()) {
                metadata.tokenizer.special_tokens[*id] = std::string(k);
            } else {
                std::cerr << "Failed to parse special token ID for key: " << k << std::endl;
            }
        }
    }

    // --- Parse Template ---
    auto template_data = get_required(tbl, "template", "top-level").as_table();
    metadata.template_type = get_required(*template_data, "type", "template").value_or("");
    metadata.template_content = get_required(*template_data, "content", "template").value_or("");

    return metadata;
}

inline std::filesystem::path get_cache_dir(const std::optional<std::string>& cli_path, const toml::table& config_table) {
    if (cli_path && !cli_path->empty()) return *cli_path;
    if (auto node = config_table["cache_dir"]; node.is_string()) return node.as_string()->get();
    if (const char* env_pie_home = std::getenv("PIE_HOME")) {
        if (std::strlen(env_pie_home) > 0) return env_pie_home;
    }
    return utils::get_user_cache_dir() / "pie";
}



int main(int argc, char* argv[]) {
    // --- 1. Argument Parsing Setup ---
    CLI::App app{"PIE C++ Backend Service"};
    app.allow_config_extras(true);

    AppConfig cli_config;
    std::optional<std::string> config_filepath_opt;
    std::optional<std::string> cache_dir_opt;
    
    app.add_option("--config", config_filepath_opt, "Path to a TOML configuration file.")->check(CLI::ExistingFile);
    app.add_option("--host", cli_config.host, "The hostname to bind to.");
    app.add_option("--port", cli_config.port, "The port number to listen on.");
    app.add_option("--controller-host", cli_config.controller_host, "The controller hostname.");
    app.add_option("--controller-port", cli_config.controller_port, "The controller port number.");
    app.add_option("--auth-token", cli_config.auth_token, "The authentication token for the controller.");
    app.add_option("--model", cli_config.model_name, "The model name (e.g., 'llama-3.2-1b-instruct').");
    app.add_option("--version", cli_config.version, "The version of the model.");
    app.add_option("--cache-dir", cache_dir_opt, "The directory for caching models.");
    app.add_option("--kv-page-size", cli_config.kv_page_size, "The KV page size.");
    app.add_option("--dist-size", cli_config.dist_size, "The distribution size.");
    app.add_option("--max-num-kv-pages", cli_config.max_num_kv_pages, "The maximum number of KV pages.");
    app.add_option("--max-num-embeds", cli_config.max_num_embeds, "The maximum number of embeddings.");
    app.add_option("--device", cli_config.device, "The device to use (e.g., 'cuda:0').");
    app.add_option("--dtype", cli_config.dtype, "The data type (e.g., 'bfloat16').");

    CLI11_PARSE(app, argc, argv);

    try {
        // --- 2. Configuration Loading and Merging ---
        toml::table config_from_file;
        if (config_filepath_opt) {
            try {
                config_from_file = toml::parse_file(*config_filepath_opt);
            } catch (const toml::parse_error& err) {
                std::cerr << "Error decoding TOML file '" << *config_filepath_opt << "': " << err.what() << std::endl;
                return 1;
            }
        }

        AppConfig final_config;
        // (Configuration merging logic remains the same)
        final_config.host = app.count("--host") > 0 ? cli_config.host : config_from_file["host"].value_or(final_config.host);
        final_config.port = app.count("--port") > 0 ? cli_config.port : config_from_file["port"].value_or(final_config.port);
        final_config.controller_host = app.count("--controller-host") > 0 ? cli_config.controller_host : config_from_file["controller_host"].value_or(final_config.controller_host);
        final_config.controller_port = app.count("--controller-port") > 0 ? cli_config.controller_port : config_from_file["controller_port"].value_or(final_config.controller_port);
        
        if (app.count("--model") > 0) final_config.model_name = cli_config.model_name;
        else if (auto node = config_from_file["model"]; node.is_string()) final_config.model_name = node.as_string()->get();

        if (app.count("--auth-token") > 0) final_config.auth_token = cli_config.auth_token;
        else if (auto node = config_from_file["auth_token"]; node.is_string()) final_config.auth_token = node.as_string()->get();

        final_config.version = app.count("--version") > 0 ? cli_config.version : config_from_file["version"].value_or(final_config.version);
        final_config.cache_dir = get_cache_dir(cache_dir_opt, config_from_file);
        // (Other config assignments remain the same)

        std::cout << final_config << std::endl;

        // Parse the model metadata
        // path = final_config.cache_dir / f{final_config.model_name}-{final_config.version}.toml
        std::filesystem::path metadata_path = final_config.cache_dir / final_config.model_name / (final_config.model_name + "-" + final_config.version + ".toml");
        if (!std::filesystem::exists(metadata_path)) {
            std::cerr << "Metadata file not found at: " << metadata_path.string() << std::endl;
            return 1;
        }
        ModelMetadata model_metadata = parse_model_metadata(metadata_path);
        std::cout << model_metadata << std::endl;


        // --- 3. Create and Run the Server ---
        
        // The Server constructor handles all the loading.
        Server server(final_config, model_metadata);
        
        // The run() method contains the main application loop.
        server.run();
        
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server shutdown initiated." << std::endl;
    return 0;
}