#include <string>
#include <format>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <optional>
#include <fstream>
#include <filesystem>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include "yaml-cpp/yaml.h"
#include "CLI/CLI.hpp"

#include "l4ma.cuh"
#include "bpe.hpp"

// Struct to hold the application's configuration.
struct AppConfig
{
    std::optional<std::string> controller;
    std::optional<std::string> model;
    std::optional<int> kv_page_size;
    std::optional<int> max_num_kv_page;
    std::optional<int> max_num_embed;
};

// --- Helper Functions ---

// Displays the final, merged configuration.
void print_config(const AppConfig &config)
{
    std::cout << "Configuration successfully loaded:\n"
              << "  Controller:   " << config.controller.value_or("N/A") << "\n"
              << "  Model:        " << config.model.value_or("N/A") << "\n"
              << "  KV Page Size:  " << (config.kv_page_size ? std::to_string(*config.kv_page_size) : "N/A") << "\n"
              << "  Max # KV Page:  " << (config.max_num_kv_page ? std::to_string(*config.max_num_kv_page) : "N/A") << "\n"
              << "  Max # Embed:   " << (config.max_num_embed ? std::to_string(*config.max_num_embed) : "N/A") << "\n";
}

// Generic function to safely parse a value from a YAML node.
template <typename T>
std::optional<T> parse_yaml_value(const YAML::Node &node, const std::string &key)
{
    if (node[key])
    {
        try
        {
            return node[key].as<T>();
        }
        catch (const YAML::BadConversion &e)
        {
            throw std::runtime_error("YAML config error: Bad conversion for key '" + key + "'.");
        }
    }
    return std::nullopt;
}

// Parses the configuration from a specified YAML file.
AppConfig parse_yaml_config(const std::string &filepath)
{
    AppConfig yaml_config;
    YAML::Node config_node;

    try
    {
        config_node = YAML::LoadFile(filepath);
    }
    catch (const YAML::BadFile &e)
    {
        throw std::runtime_error("Error: Could not open or read config file: " + filepath);
    }
    catch (const YAML::ParserException &e)
    {
        throw std::runtime_error("Error: Failed to parse YAML file: " + std::string(e.what()));
    }

    // Note: YAML keys are in snake_case as requested.
    yaml_config.controller = parse_yaml_value<std::string>(config_node, "controller");
    yaml_config.model = parse_yaml_value<std::string>(config_node, "model");
    yaml_config.kv_page_size = parse_yaml_value<int>(config_node, "kv_page_size");
    yaml_config.max_num_kv_page = parse_yaml_value<int>(config_node, "max_num_kv_page");
    yaml_config.max_num_embed = parse_yaml_value<int>(config_node, "max_num_embed");

    return yaml_config;
}

std::filesystem::path get_pie_home()
{
    if (const char *env_pie_home = std::getenv("PIE_HOME"))
    {
        // Use the PIE_HOME environment variable if it is set and not empty.
        if (std::strlen(env_pie_home) > 0)
        {
            return std::filesystem::path(env_pie_home);
        }
    }
    // Default to ~/.cache/pie if PIE_HOME is not set.
    if (const char *env_home = std::getenv("HOME"))
    {
        return std::filesystem::path(env_home) / ".cache" / "pie";
    }

    throw std::runtime_error("Could not determine PIE_HOME. Please set the $PIE_HOME or $HOME environment variable.");
}

void print_tokens(const std::vector<bpe::Rank> &tokens)
{
    std::cout << "[";
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        std::cout << tokens[i] << (i == tokens.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// Formats a prompt for the Llama 3 model.
std::string llama3_format(
    const std::string &prompt,
    const std::optional<std::string> &hint,
    const std::string &system = "You are a helpful, respectful and honest assistant.")
{
    std::string temp = "<|begin_of_text|>";
    temp += std::format("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system);
    temp += std::format("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>", prompt);
    temp += "<|start_header_id|>assistant<|end_header_id|>\n\n";

    if (hint)
    {
        temp += *hint;
    }

    return temp;
}

int main(int argc, char *argv[])
{
    CLI::App app{"PIE CUDA Backend"};

    std::string config_filepath;
    app.add_option("--config", config_filepath, "Specify a path to a YAML configuration file.")->check(CLI::ExistingFile);

    AppConfig cli_config;
    app.add_option("--controller", cli_config.controller, "Set the controller endpoint address.");
    app.add_option("--model", cli_config.model, "Set the model name (e.g., 'llama-3.2-1b-instruct').");
    app.add_option("--kv-page-size", cli_config.kv_page_size, "Set the KV page size.");
    app.add_option("--max-num-kv-page", cli_config.max_num_kv_page, "Set the maximum number of KV pages.");
    app.add_option("--max-num-embed", cli_config.max_num_embed, "Set the maximum number of embeddings.");

    try
    {
        // CLI11 automatically handles --help.
        app.parse(argc, argv);

        // Load YAML Config if specified
        AppConfig yaml_config;
        if (!config_filepath.empty())
        {
            yaml_config = parse_yaml_config(config_filepath);
        }

        AppConfig config;

        config.controller = cli_config.controller ? cli_config.controller : yaml_config.controller;
        if (!config.controller)
            throw std::runtime_error("Error: 'controller' is a required argument.");

        config.model = cli_config.model ? cli_config.model : yaml_config.model;
        if (!config.model)
            throw std::runtime_error("Error: 'model' is a required argument.");

        config.kv_page_size = cli_config.kv_page_size ? cli_config.kv_page_size : yaml_config.kv_page_size;
        if (!config.kv_page_size)
            throw std::runtime_error("Error: 'kv-page-size' is a required argument.");

        config.max_num_kv_page = cli_config.max_num_kv_page ? cli_config.max_num_kv_page : yaml_config.max_num_kv_page;
        if (!config.max_num_kv_page)
            throw std::runtime_error("Error: 'max-num-kv-page' is a required argument.");

        config.max_num_embed = cli_config.max_num_embed ? cli_config.max_num_embed : yaml_config.max_num_embed;
        if (!config.max_num_embed)
            throw std::runtime_error("Error: 'max-num-embed' is a required argument.");

        print_config(config);

        const std::filesystem::path pie_home = get_pie_home();
        const std::string model_name = *config.model;
        const std::filesystem::path model_dir = pie_home / model_name;

        const std::filesystem::path model_yaml_path = model_dir / (model_name + ".yaml");

        std::cout << "Loading model '" << model_name << "' from: " << model_dir << std::endl;

        if (!std::filesystem::exists(model_yaml_path))
        {
            throw std::runtime_error("Model data file not found: " + model_yaml_path.string());
        }

        YAML::Node config_dict = YAML::LoadFile(model_yaml_path.string());

        // Extract the required filenames directly from the node.
        const std::string tokenizer_filename = config_dict["tokenizer"]["vocab"].as<std::string>();
        const std::string zt_filename = config_dict["parameters"][0].as<std::string>();

        const std::filesystem::path tokenizer_path = model_dir / tokenizer_filename;
        const std::filesystem::path zt_path = model_dir / zt_filename;

        std::cout << "Resolved asset paths from YAML:\n"
                  << "  Tokenizer: " << tokenizer_path << "\n"
                  << "  Model .zt: " << zt_path << "\n"
                  << std::endl;

        // --- 4. Tokenizer Test ---
        std::cout << "--- Tokenizer Test ---" << std::endl;
        const auto tokenizer = bpe::llama3_tokenizer(tokenizer_path);
        const std::string text_to_encode = llama3_format("What is the capital of France?", std::nullopt);
        std::cout << "Formatted text: " << text_to_encode;
        const auto tokens = tokenizer.encode_with_special_tokens(text_to_encode);
        std::cout << "Encoded tokens: ";
        print_tokens(tokens);
        const std::string decoded_text = tokenizer.decode(tokens);
        std::cout << "Decoded text: " << decoded_text << std::endl;
    }
    catch (const CLI::ParseError &e)
    {
        return app.exit(e);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0; // Success
}

// int main()
// {
//     std::cout << "hello world!" << std::endl;

//     /// tokenizer test

//     std::string model_path = "/home/ingim/Workspace/model-index/meta-llama--Llama-3.2-1B-Instruct/tokenizer.model";
//     auto tokenizer = bpe::llama3_tokenizer(model_path);

//     std::string text = llama3_format("What is the capital of France?", std::nullopt);

//     std::cout << "Original text: " << text << std::endl;

//     // Encode the text
//     auto tokens = tokenizer.encode_with_special_tokens(text);
//     std::cout << "Encoded tokens: ";
//     print_tokens(tokens);

//     // Decode the tokens
//     std::string decoded_text = tokenizer.decode(tokens);
//     std::cout << "Decoded text: " << decoded_text << std::endl;

//     // --- Print ztensor metadata for llama1b.zt ---
//     std::string pie_home;
//     const char *env_pie_home = std::getenv("PIE_HOME");
//     if (env_pie_home && env_pie_home[0] != '\0')
//     {
//         pie_home = env_pie_home;
//     }
//     else
//     {
//         const char *home = std::getenv("HOME");
//         if (!home)
//         {
//             std::cerr << "Could not determine $HOME for PIE_HOME fallback." << std::endl;
//             return 1;
//         }
//         pie_home = std::string(home) + "/.cache/pie";
//     }
//     std::string zt_path = pie_home + "/llama1b.zt";

//     // set config_path to "./l4ma.yaml"
//     std::string config_path = "../../l4ma.yaml";
//     const int MAX_TOTAL_TOKENS = 2048;

//     try
//     {
//         // --- 2. Load Model ---
//         std::cout << "Loading model from files..." << std::endl;
//         // Use the static factory method to load config, weights, and construct the model
//         auto model = L4maModel<__nv_bfloat16>::from_files(config_path, zt_path);
//         std::cout << "Model loaded successfully." << std::endl;

//         // Extract config details needed for setup
//         // IMPORTANT: The model class should expose its config. For this example, we re-load it.
//         // In a better design, model.config() would be a public method.
//         L4maConfig config = model.get_config();
//         config.print();

//         // construct input_ids from tokens
//         thrust::device_vector<uint32_t> input_ids(tokens.begin(), tokens.end());

//         // create a uninitalized vector with size equal to the number of len(input_ids) * config.hidden_size
//         thrust::device_vector<__nv_bfloat16> embed_output(input_ids.size() * config.hidden_size);

//         model.embed_input_ids(input_ids, embed_output);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "\nAn error occurred: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }
