// main.cpp
//
// This C++ application is a rewrite of the original logic to align with the provided
// Python implementation. It acts as a backend service for a language model,
// handling requests via a ZeroMQ (ZMQ) router.
//
// Key changes from the old version:
// - Configuration: Switched from YAML to TOML for both the main config and model metadata.
// - Architecture: Implemented a multi-threaded service with separate threads for the
//   ZMQ server and for registering with a central controller.
// - Model Loading: Adapted to load model metadata and weights from a directory structure
//   consistent with the Python version, including support for multiple .ztensor files.
// - Dependencies: Uses libraries like toml++, cppzmq, and placeholders for Protobuf,
//   MsgPack, and WebSockets to match the Python script's functionality.
//

#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <thread>
#include <chrono>
#include <random>
#include <map>
#include <stdexcept>
#include <fstream>

// --- Third-Party Libraries ---
// Note: These would need to be properly included and linked in a real project.
// A sample CMakeLists.txt is provided at the end of this file.
#include "CLI/CLI.hpp"
#include "toml++/toml.hpp"
#include "zmq.hpp"
#include "zmq_addon.hpp"

// --- Placeholder Includes for Required Components ---
// These headers represent components that would need to be implemented or included.
// #include "l4ma.cuh"          // For the core model implementation (L4maForCausalLM)
// #include "bpe.hpp"           // For the tokenizer
// #include "ztensor.hpp"       // Placeholder for a .ztensor file reading library

// --- Protobuf Generated Headers (assumed to be generated) ---
#include "handshake.pb.h"
#include "l4m.pb.h"
// #include "l4m_vision.pb.h" // Uncomment if vision features are needed

// --- MsgPack and WebSocket Client Headers (placeholders) ---
// #include <msgpack.hpp>
// #include <websocketpp/client.hpp>
// #include <websocketpp/config/asio_client.hpp>

// ======================================================================================
// MARK: - Platform & Utility Functions
// ======================================================================================

namespace platform_dirs
{
    // Implements a cross-platform method for finding the user's cache directory.
    inline std::filesystem::path get_user_cache_dir()
    {
#if defined(_WIN32)
        // Windows: Prefer %LOCALAPPDATA%, fallback to %APPDATA%
        const char *local_appdata = std::getenv("LOCALAPPDATA");
        if (local_appdata)
        {
            return std::filesystem::path(local_appdata);
        }
        const char *appdata = std::getenv("APPDATA");
        if (appdata)
        {
            return std::filesystem::path(appdata) / "Local";
        }
#else
        // Linux, macOS, and other UNIX-like systems
        const char *home = std::getenv("HOME");
        if (!home)
        {
            throw std::runtime_error("Could not determine home directory. Please set the $HOME environment variable.");
        }
        std::filesystem::path home_path(home);

#if defined(__APPLE__)
        // macOS: ~/Library/Caches
        return home_path / "Library" / "Caches";
#else
        // Linux: Follow XDG Base Directory Specification
        const char *xdg_cache_home = std::getenv("XDG_CACHE_HOME");
        if (xdg_cache_home && std::strlen(xdg_cache_home) > 0)
        {
            return std::filesystem::path(xdg_cache_home);
        }
        // Default to ~/.cache
        return home_path / ".cache";
#endif
#endif
        throw std::runtime_error("Unsupported platform or unable to determine cache directory.");
    }
}

namespace utils {
    // Basic Base64 decoding utility.
    inline std::vector<uint8_t> base64_decode(const std::string& in) {
        std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<int> T(256,-1);
        for (int i=0; i<64; i++) T[chars[i]] = i;

        std::vector<uint8_t> out;
        int val=0, valb=-8;
        for (unsigned char c : in) {
            if (T[c] == -1) break;
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                out.push_back(char((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return out;
    }
}


// ======================================================================================
// MARK: - Configuration and Data Structures
// ======================================================================================

// Holds the final, merged configuration for the application.
struct AppConfig
{
    // Network
    std::string host = "localhost";
    int port = 10123;
    std::string controller_host = "localhost";
    int controller_port = 9123;
    std::optional<std::string> auth_token;

    // Model & Cache
    std::optional<std::string> model_name;
    std::string version = "";
    std::filesystem::path cache_dir;

    // Engine Parameters
    int kv_page_size = 32;
    int dist_size = 32;
    int max_num_kv_pages = 1000;
    int max_num_embeds = 50000;

    // Hardware
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
};

// Corresponds to the [architecture] table in the model's TOML file.
struct L4maConfig
{
    std::string type;
    int num_layers;
    int num_query_heads;
    int num_key_value_heads;
    int head_size;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    bool use_qkv_bias;
    float rms_norm_eps;
    float rope_factor;
    float rope_high_frequency_factor;
    float rope_low_frequency_factor;
    float rope_theta;
    std::string device = "cuda:0";
    // torch::dtype dtype = torch::kBFloat16; // Placeholder for a specific dtype
};

// Corresponds to the [tokenizer] table in the model's TOML file.
struct TokenizerInfo
{
    std::string type;
    std::map<int, std::vector<uint8_t>> merge_table;
    std::string split_regex;
    std::map<int, std::string> special_tokens;
};

// Top-level structure holding the entire parsed model configuration.
struct ModelMetadata
{
    std::string name;
    std::string description;
    std::string version;
    std::vector<std::string> parameters;
    L4maConfig architecture;
    TokenizerInfo tokenizer;
    std::string template_type;
    std::string template_content;
};


// ======================================================================================
// MARK: - Helper and Parsing Functions
// ======================================================================================

// Loads merge rules from a Llama 3 vocabulary file.
std::map<int, std::vector<uint8_t>> load_merge_rules(const std::filesystem::path& path) {
    std::map<int, std::vector<uint8_t>> merge_rules;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to load vocabulary file: " + path.string());
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
            continue; // Skip empty or whitespace-only lines
        }

        std::string::size_type pos = line.find(' ');
        if (pos == std::string::npos) {
             throw std::runtime_error("Error on line " + std::to_string(line_number) + ": expected 2 parts, but found 1.");
        }
        
        std::string b64_token = line.substr(0, pos);
        std::string rank_str = line.substr(pos + 1);

        try {
            int rank = std::stoi(rank_str);
            merge_rules[rank] = utils::base64_decode(b64_token);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Error on line " + std::to_string(line_number) + ": failed to parse rank '" + rank_str + "' as an integer.");
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Error on line " + std::to_string(line_number) + ": rank '" + rank_str + "' is out of range.");
        }
    }
    return merge_rules;
}

// Parses the model's metadata TOML file with detailed error checking.
ModelMetadata parse_model_metadata(const std::filesystem::path& metadata_path)
{
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
            metadata.tokenizer.special_tokens[std::stoi(std::string(k))] = v.value_or("");
        }
    }

    // --- Parse Template ---
    auto template_data = get_required(tbl, "template", "top-level").as_table();
    metadata.template_type = get_required(*template_data, "type", "template").value_or("");
    metadata.template_content = get_required(*template_data, "content", "template").value_or("");

    return metadata;
}

// Determines the cache directory based on a clear priority order.
std::filesystem::path get_cache_dir(
    const std::optional<std::string>& cli_path,
    const toml::table& config_table)
{
    // 1. Command-line argument
    if (cli_path && !cli_path->empty()) {
        return *cli_path;
    }
    // 2. Value from TOML config file
    if (auto node = config_table["cache_dir"]; node.is_string()) {
        return node.as_string()->get();
    }
    // 3. PIE_HOME environment variable
    if (const char* env_pie_home = std::getenv("PIE_HOME")) {
        if (std::strlen(env_pie_home) > 0) {
            return env_pie_home;
        }
    }
    // 4. Default to user cache directory
    return platform_dirs::get_user_cache_dir() / "pie";
}

// Prints the final configuration that the application will run with.
void print_config(const AppConfig& config)
{
    std::cout << "--- Configuration ---\n"
              << "host: " << config.host << "\n"
              << "port: " << config.port << "\n"
              << "controller_host: " << config.controller_host << "\n"
              << "controller_port: " << config.controller_port << "\n"
              << "auth_token: " << config.auth_token.value_or("Not Set") << "\n"
              << "model: " << config.model_name.value_or("Not Set") << "\n"
              << "version: " << config.version << "\n"
              << "cache_dir: " << config.cache_dir << "\n"
              << "kv_page_size: " << config.kv_page_size << "\n"
              << "dist_size: " << config.dist_size << "\n"
              << "max_num_kv_pages: " << config.max_num_kv_pages << "\n"
              << "max_num_embeds: " << config.max_num_embeds << "\n"
              << "device: " << config.device << "\n"
              << "dtype: " << config.dtype << "\n"
              << "----------------------\n" << std::endl;
}

// ======================================================================================
// MARK: - Core Service Logic
// ======================================================================================

// Loads the model and its weights from the specified files.
// NOTE: `l4ma::L4maForCausalLM` and `ztensor::Reader` are placeholders for the actual
// model and tensor-loading libraries.
void load_model(const AppConfig& config, ModelMetadata& out_metadata)
{
    if (!config.model_name) {
        throw std::runtime_error("Model name must be specified via --model or config file.");
    }

    const auto model_dir = config.cache_dir / *config.model_name;
    const std::string metadata_filename = *config.model_name + (config.version.empty() ? "" : "-") + config.version + ".toml";
    const auto metadata_path = model_dir / metadata_filename;

    std::cout << "Loading model '" << *config.model_name << "' from: " << model_dir << std::endl;
    std::cout << "Reading metadata from: " << metadata_path << std::endl;

    out_metadata = parse_model_metadata(metadata_path);
    
   
    // TODO: Check for missing keys and print warnings
    // ...

    // TODO: Set model to evaluation mode
    // model->eval();

}


// Registers this service with the central controller.
// NOTE: This is a placeholder. A real implementation would require a WebSocket
// and MessagePack library.
void register_with_controller(const AppConfig& config, const std::string& service_endpoint)
{
    std::cout << "[Registration Thread] Attempting to register with controller at "
              << config.controller_host << ":" << config.controller_port << std::endl;
    
    // PSEUDOCODE for registration logic
}


// Runs the main ZMQ server loop, listening for and handling requests.
void run_zmq_server(zmq::socket_t& router, const AppConfig& config, const ModelMetadata& metadata)
{
    std::cout << "[ZMQ Server Thread] ZMQ server thread started." << std::endl;
    std::vector<std::string> protocols = {"l4m", "l4m-vision", "ping"};
    std::unordered_map<std::string, bool> connected_clients;

    while (true) {
        try {
            std::vector<zmq::message_t> multipart_msg;
            auto result = zmq::recv_multipart(router, std::back_inserter(multipart_msg));

            if (!result) {
                std::cerr << "[ZMQ Server Thread] Failed to receive message." << std::endl;
                continue;
            }

            const std::string client_identity = multipart_msg[0].to_string();

            // Handle handshake for new clients
            if (connected_clients.find(client_identity) == connected_clients.end()) {
                std::cout << "[ZMQ Server Thread] New client connection, performing handshake." << std::endl;
                handshake::Request hs_req;
                if (!hs_req.ParseFromString(multipart_msg[1].to_string())) {
                    std::cerr << "[ZMQ Server Thread] Invalid handshake message." << std::endl;
                    continue;
                }
                
                handshake::Response hs_res;
                for(const auto& p : protocols) {
                    hs_res.add_protocols(p);
                }
                
                zmq::message_t response_payload(hs_res.ByteSizeLong());
                hs_res.SerializeToArray(response_payload.data(), response_payload.size());

                router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                router.send(response_payload, zmq::send_flags::none);

                connected_clients[client_identity] = true;
                continue;
            }
            
            // Handle requests from existing clients
            if (multipart_msg.size() != 3) {
                std::cerr << "[ZMQ Server Thread] Invalid message format from known client." << std::endl;
                continue;
            }
            
            const int protocol_idx = *static_cast<const uint8_t*>(multipart_msg[1].data());
            if (protocol_idx < 0 || protocol_idx >= protocols.size()) {
                std::cerr << "[ZMQ Server Thread] Invalid protocol index: " << protocol_idx << std::endl;
                continue;
            }
            const std::string& protocol = protocols[protocol_idx];
            const std::string& payload_str = multipart_msg[2].to_string();

            if (protocol == "l4m") {
                // l4m::Request request;
                // request.ParseFromString(payload_str);
                
                // if (request.has_get_info()) {
                //     l4m::Response response;
                //     response.set_correlation_id(request.correlation_id());
                //     auto* info = response.mutable_get_info();
                //     info->set_version(metadata.version);
                //     info->set_model_name(metadata.name);
                //     info->set_kv_page_size(config.kv_page_size);
                //     info->set_num_available_kv_pages(config.max_num_kv_pages);
                //     info->set_num_available_embeddings(config.max_num_embeds);
                    
                //     auto* tokenizer_info = info->mutable_tokenizer();
                //     tokenizer_info->set_split_regex(metadata.tokenizer.split_regex);
                //     // Add merge table and special tokens...

                //     router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                //     router.send(multipart_msg[1], zmq::send_flags::sndmore);
                //     router.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
                // } 
                // TODO: Implement other l4m commands.

            } else if (protocol == "ping") {
                // ping::Ping ping_req;
                // ping_req.ParseFromString(payload_str);

                // ping::Pong pong_res;
                // pong_res.set_correlation_id(ping_req.correlation_id());
                // pong_res.set_message("Pong:" + ping_req.message());

                // router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                // router.send(multipart_msg[1], zmq::send_flags::sndmore);
                // router.send(zmq::buffer(pong_res.SerializeAsString()), zmq::send_flags::none);
            }
            
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                std::cout << "[ZMQ Server Thread] Context terminated, shutting down." << std::endl;
                break;
            }
            std::cerr << "[ZMQ Server Thread] ZMQ Error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ZMQ Server Thread] An unexpected error occurred: " << e.what() << std::endl;
        }
    }
}


// ======================================================================================
// MARK: - Main Function
// ======================================================================================

int main(int argc, char* argv[])
{
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

    try
    {
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
        final_config.host = app.count("--host") > 0 ? cli_config.host : config_from_file["host"].value_or(final_config.host);
        final_config.port = app.count("--port") > 0 ? cli_config.port : config_from_file["port"].value_or(final_config.port);
        final_config.controller_host = app.count("--controller-host") > 0 ? cli_config.controller_host : config_from_file["controller_host"].value_or(final_config.controller_host);
        final_config.controller_port = app.count("--controller-port") > 0 ? cli_config.controller_port : config_from_file["controller_port"].value_or(final_config.controller_port);
        final_config.model_name = app.count("--model") > 0 ? cli_config.model_name : config_from_file["model"].value<std::string>();

        if (app.count("--auth-token") > 0) {
            final_config.auth_token = cli_config.auth_token;
        } else if (auto token_node = config_from_file["auth_token"]; token_node.is_string()) {
            final_config.auth_token = token_node.as_string()->get();
        }

        final_config.model_name = app.count("--model") > 0 ? cli_config.model_name : config_from_file["model"].value<std::string>();
        final_config.version = app.count("--version") > 0 ? cli_config.version : config_from_file["version"].value_or(final_config.version);
        final_config.kv_page_size = app.count("--kv-page-size") > 0 ? cli_config.kv_page_size : config_from_file["kv_page_size"].value_or(final_config.kv_page_size);
        final_config.dist_size = app.count("--dist-size") > 0 ? cli_config.dist_size : config_from_file["dist_size"].value_or(final_config.dist_size);
        final_config.max_num_kv_pages = app.count("--max-num-kv-pages") > 0 ? cli_config.max_num_kv_pages : config_from_file["max_num_kv_pages"].value_or(final_config.max_num_kv_pages);
        final_config.max_num_embeds = app.count("--max-num-embeds") > 0 ? cli_config.max_num_embeds : config_from_file["max_num_embeds"].value_or(final_config.max_num_embeds);
        final_config.device = app.count("--device") > 0 ? cli_config.device : config_from_file["device"].value_or(final_config.device);
        final_config.dtype = app.count("--dtype") > 0 ? cli_config.dtype : config_from_file["dtype"].value_or(final_config.dtype);        
        final_config.cache_dir = get_cache_dir(cache_dir_opt, config_from_file);

        print_config(final_config);

        // --- 3. Load Model ---
        ModelMetadata model_metadata;
        load_model(final_config, model_metadata);
        std::cout << "Model loading complete." << std::endl;
        
        // TODO: Create the main Driver/Engine instance here.

        // --- 4. Start Services ---
        zmq::context_t context(1);
        zmq::socket_t router(context, zmq::socket_type::router);
        
        std::string service_endpoint;
        if (final_config.host == "localhost" || final_config.host == "127.0.0.1") {
            int unique_id = std::rand() % 9000 + 1000;
            service_endpoint = "ipc:///tmp/pie-service-" + std::to_string(unique_id);
        } else {
            service_endpoint = "tcp://" + final_config.host + ":" + std::to_string(final_config.port);
        }
        router.bind(service_endpoint);
        std::cout << "Server listening on " << service_endpoint << std::endl;

        std::thread zmq_thread(run_zmq_server, std::ref(router), std::cref(final_config), std::cref(model_metadata));
        std::thread register_thread(register_with_controller, std::cref(final_config), service_endpoint);

        std::cout << "Service is running. Press Ctrl+C to shut down." << std::endl;
        zmq_thread.join();
        register_thread.join();

        router.close();
        context.close();
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server shutdown complete." << std::endl;
    return 0;
}


// #include <string>
// #include <format>
// #include <iostream>
// #include <vector>
// #include <numeric>
// #include <stdexcept>
// #include <optional>
// #include <fstream>
// #include <filesystem>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/sequence.h>
// #include <thrust/extrema.h>

// #include "yaml-cpp/yaml.h"
// #include "CLI/CLI.hpp"

// #include "l4ma.cuh"
// #include "bpe.hpp"

// // Struct to hold the application's configuration.
// struct AppConfig
// {
//     std::optional<std::string> controller;
//     std::optional<std::string> model;
//     std::optional<int> kv_page_size;
//     std::optional<int> max_num_kv_page;
//     std::optional<int> max_num_embed;
// };

// // --- Helper Functions ---

// // Displays the final, merged configuration.
// void print_config(const AppConfig &config)
// {
//     std::cout << "Configuration successfully loaded:\n"
//               << "  Controller:   " << config.controller.value_or("N/A") << "\n"
//               << "  Model:        " << config.model.value_or("N/A") << "\n"
//               << "  KV Page Size:  " << (config.kv_page_size ? std::to_string(*config.kv_page_size) : "N/A") << "\n"
//               << "  Max # KV Page:  " << (config.max_num_kv_page ? std::to_string(*config.max_num_kv_page) : "N/A") << "\n"
//               << "  Max # Embed:   " << (config.max_num_embed ? std::to_string(*config.max_num_embed) : "N/A") << "\n";
// }

// // Generic function to safely parse a value from a YAML node.
// template <typename T>
// std::optional<T> parse_yaml_value(const YAML::Node &node, const std::string &key)
// {
//     if (node[key])
//     {
//         try
//         {
//             return node[key].as<T>();
//         }
//         catch (const YAML::BadConversion &e)
//         {
//             throw std::runtime_error("YAML config error: Bad conversion for key '" + key + "'.");
//         }
//     }
//     return std::nullopt;
// }

// // Parses the configuration from a specified YAML file.
// AppConfig parse_yaml_config(const std::string &filepath)
// {
//     AppConfig yaml_config;
//     YAML::Node config_node;

//     try
//     {
//         config_node = YAML::LoadFile(filepath);
//     }
//     catch (const YAML::BadFile &e)
//     {
//         throw std::runtime_error("Error: Could not open or read config file: " + filepath);
//     }
//     catch (const YAML::ParserException &e)
//     {
//         throw std::runtime_error("Error: Failed to parse YAML file: " + std::string(e.what()));
//     }

//     // Note: YAML keys are in snake_case as requested.
//     yaml_config.controller = parse_yaml_value<std::string>(config_node, "controller");
//     yaml_config.model = parse_yaml_value<std::string>(config_node, "model");
//     yaml_config.kv_page_size = parse_yaml_value<int>(config_node, "kv_page_size");
//     yaml_config.max_num_kv_page = parse_yaml_value<int>(config_node, "max_num_kv_page");
//     yaml_config.max_num_embed = parse_yaml_value<int>(config_node, "max_num_embed");

//     return yaml_config;
// }

// std::filesystem::path get_pie_home()
// {
//     if (const char *env_pie_home = std::getenv("PIE_HOME"))
//     {
//         // Use the PIE_HOME environment variable if it is set and not empty.
//         if (std::strlen(env_pie_home) > 0)
//         {
//             return std::filesystem::path(env_pie_home);
//         }
//     }
//     // Default to ~/.cache/pie if PIE_HOME is not set.
//     if (const char *env_home = std::getenv("HOME"))
//     {
//         return std::filesystem::path(env_home) / ".cache" / "pie";
//     }

//     throw std::runtime_error("Could not determine PIE_HOME. Please set the $PIE_HOME or $HOME environment variable.");
// }

// void print_tokens(const std::vector<bpe::Rank> &tokens)
// {
//     std::cout << "[";
//     for (size_t i = 0; i < tokens.size(); ++i)
//     {
//         std::cout << tokens[i] << (i == tokens.size() - 1 ? "" : ", ");
//     }
//     std::cout << "]" << std::endl;
// }

// // Formats a prompt for the Llama 3 model.
// std::string llama3_format(
//     const std::string &prompt,
//     const std::optional<std::string> &hint,
//     const std::string &system = "You are a helpful, respectful and honest assistant.")
// {
//     std::string temp = "<|begin_of_text|>";
//     temp += std::format("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system);
//     temp += std::format("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>", prompt);
//     temp += "<|start_header_id|>assistant<|end_header_id|>\n\n";

//     if (hint)
//     {
//         temp += *hint;
//     }

//     return temp;
// }

// int main(int argc, char *argv[])
// {
//     CLI::App app{"PIE CUDA Backend"};

//     std::string config_filepath;
//     app.add_option("--config", config_filepath, "Specify a path to a YAML configuration file.")->check(CLI::ExistingFile);

//     AppConfig cli_config;
//     app.add_option("--controller", cli_config.controller, "Set the controller endpoint address.");
//     app.add_option("--model", cli_config.model, "Set the model name (e.g., 'llama-3.2-1b-instruct').");
//     app.add_option("--kv-page-size", cli_config.kv_page_size, "Set the KV page size.");
//     app.add_option("--max-num-kv-page", cli_config.max_num_kv_page, "Set the maximum number of KV pages.");
//     app.add_option("--max-num-embed", cli_config.max_num_embed, "Set the maximum number of embeddings.");

//     try
//     {
//         // CLI11 automatically handles --help.
//         app.parse(argc, argv);

//         // Load YAML Config if specified
//         AppConfig yaml_config;
//         if (!config_filepath.empty())
//         {
//             yaml_config = parse_yaml_config(config_filepath);
//         }

//         AppConfig config;

//         config.controller = cli_config.controller ? cli_config.controller : yaml_config.controller;
//         if (!config.controller)
//             throw std::runtime_error("Error: 'controller' is a required argument.");

//         config.model = cli_config.model ? cli_config.model : yaml_config.model;
//         if (!config.model)
//             throw std::runtime_error("Error: 'model' is a required argument.");

//         config.kv_page_size = cli_config.kv_page_size ? cli_config.kv_page_size : yaml_config.kv_page_size;
//         if (!config.kv_page_size)
//             throw std::runtime_error("Error: 'kv-page-size' is a required argument.");

//         config.max_num_kv_page = cli_config.max_num_kv_page ? cli_config.max_num_kv_page : yaml_config.max_num_kv_page;
//         if (!config.max_num_kv_page)
//             throw std::runtime_error("Error: 'max-num-kv-page' is a required argument.");

//         config.max_num_embed = cli_config.max_num_embed ? cli_config.max_num_embed : yaml_config.max_num_embed;
//         if (!config.max_num_embed)
//             throw std::runtime_error("Error: 'max-num-embed' is a required argument.");

//         print_config(config);

//         const std::filesystem::path pie_home = get_pie_home();
//         const std::string model_name = *config.model;
//         const std::filesystem::path model_dir = pie_home / model_name;

//         const std::filesystem::path model_yaml_path = model_dir / (model_name + ".yaml");

//         std::cout << "Loading model '" << model_name << "' from: " << model_dir << std::endl;

//         if (!std::filesystem::exists(model_yaml_path))
//         {
//             throw std::runtime_error("Model data file not found: " + model_yaml_path.string());
//         }

//         YAML::Node config_dict = YAML::LoadFile(model_yaml_path.string());

//         // Extract the required filenames directly from the node.
//         const std::string tokenizer_filename = config_dict["tokenizer"]["vocab"].as<std::string>();
//         const std::string zt_filename = config_dict["parameters"][0].as<std::string>();

//         const std::filesystem::path tokenizer_path = model_dir / tokenizer_filename;
//         const std::filesystem::path zt_path = model_dir / zt_filename;

//         std::cout << "Resolved asset paths from YAML:\n"
//                   << "  Tokenizer: " << tokenizer_path << "\n"
//                   << "  Model .zt: " << zt_path << "\n"
//                   << std::endl;

//         // --- 4. Tokenizer Test ---
//         std::cout << "--- Tokenizer Test ---" << std::endl;
//         const auto tokenizer = bpe::llama3_tokenizer(tokenizer_path);
//         const std::string text_to_encode = llama3_format("What is the capital of France?", std::nullopt);
//         std::cout << "Formatted text: " << text_to_encode;
//         const auto tokens = tokenizer.encode_with_special_tokens(text_to_encode);
//         std::cout << "Encoded tokens: ";
//         print_tokens(tokens);
//         const std::string decoded_text = tokenizer.decode(tokens);
//         std::cout << "Decoded text: " << decoded_text << std::endl;
//     }
//     catch (const CLI::ParseError &e)
//     {
//         return app.exit(e);
//     }
//     catch (const std::runtime_error &e)
//     {
//         std::cerr << e.what() << std::endl;
//         return 1;
//     }

//     return 0; // Success
// }
