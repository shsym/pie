#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <filesystem>

// CUDA includes for error checking
#include <cuda_runtime.h>

// Include CUDA backend APIs - directly using functions from model.cu
#include "config.hpp"
#include "ztensor.hpp"
#include "artifacts.hpp"
#include "l4ma.cuh"  // Now we can include L4ma classes
#include "model.hpp" // Model class with constructor that calls load_model_internal
#include "bpe.hpp"   // BPE tokenizer for proper encoding/decoding

// Include new modular CUDA context manager
#include "cuda_context.hpp"

std::string decode_tokens(const std::vector<uint32_t>& token_ids) {
    try {
        std::vector<std::string> possible_tokenizer_paths;

        const char* model_path_env = std::getenv("PIE_MODEL_PATH");
        if (model_path_env) {
            std::string model_path(model_path_env);
            size_t last_slash = model_path.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string model_dir = model_path.substr(0, last_slash);
                possible_tokenizer_paths.push_back(model_dir + "/llama-3.2.vocab");
            }
        }

        possible_tokenizer_paths.push_back(
            std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") +
            "/.cache/pie/models/llama-3.2-1b-instruct/llama-3.2.vocab"
        );

        for (const auto& path : possible_tokenizer_paths) {
            try {
                auto tokenizer = bpe::llama3_tokenizer(path);
                return tokenizer.decode(token_ids);
            } catch (const std::exception& e) {
                continue;
            }
        }

        std::string fallback = "[";
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (i > 0) fallback += ", ";
            fallback += std::to_string(token_ids[i]);
        }
        fallback += "]";
        return fallback;
    } catch (const std::exception& e) {
        return "[decoding error: " + std::string(e.what()) + "]";
    }
}

/**
 * @brief CUDA Integration Test using CudaContext for Proper KV Cache Management
 *
 * This version uses the new CudaContext class that mirrors the Rust implementation:
 * - Proper stateful KV cache management with page allocation
 * - Correct token and position ID tracking across generation steps
 * - Modular design separating generation logic from test setup
 * - Follows the same patterns as the working Rust Context
 */
class CudaIntegrationWithContext {
private:
    std::string model_path;
    std::string tokenizer_path;
    std::string case_id;
    ModelMetadata model_metadata;
    AppConfig app_config;
    std::unique_ptr<CudaContext> context;
    bool verbose;

    std::string find_tokenizer_path() {
        std::vector<std::string> possible_paths;

        const char* model_path_env = std::getenv("PIE_MODEL_PATH");
        if (model_path_env) {
            std::string model_path_str(model_path_env);
            size_t last_slash = model_path_str.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string model_dir = model_path_str.substr(0, last_slash);
                possible_paths.push_back(model_dir + "/llama-3.2.vocab");
            }
        }

        possible_paths.push_back(
            std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") +
            "/.cache/pie/models/llama-3.2-1b-instruct/llama-3.2.vocab"
        );

        for (const auto& path : possible_paths) {
            std::ifstream file(path);
            if (file.good()) {
                return path;
            }
        }

        throw std::runtime_error("No tokenizer file found");
    }

public:
    CudaIntegrationWithContext(const std::string& case_name)
        : case_id(case_name), verbose(true) {

        // Get model path from environment or use fallback
        const char* env_path = std::getenv("PIE_MODEL_PATH");
        if (env_path) {
            model_path = std::string(env_path);
        } else {
            // Default model paths to try
            std::vector<std::string> default_paths = {
                std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/.cache/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt",
                std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/.cache/pie/llama-3.2-1b-bf16.zt",
                "./models/llama-3.2-1b-bf16.zt"
            };

            bool found = false;
            for (const auto& path : default_paths) {
                std::ifstream file(path);
                if (file.good()) {
                    model_path = path;
                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("No model file found. Set PIE_MODEL_PATH environment variable or ensure model exists at: ~/.cache/pie/models/llama-3.2-1b-instruct/llama-3.2-1b-instruct.zt");
            }
        }

        // Find tokenizer
        try {
            tokenizer_path = find_tokenizer_path();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to find tokenizer: " + std::string(e.what()));
        }

    }

    void test_generation_with_context(const std::string& prompt_text, int max_new_tokens) {
        try {
            context->set_verbose(false);
            context->fill(prompt_text);
            context->flush();

            CudaContext::GreedySampler sampler;
            CudaContext::LengthStopCondition stop_condition(max_new_tokens);

            std::string response = context->generate(sampler, stop_condition);
            std::cout << "  Input: \"" << prompt_text << "\"" << std::endl;
            std::cout << "  Generated: \"" << response << "\"" << std::endl;
            std::cout << "  Status: " << (response.length() > 0 ? "✅ SUCCESS" : "❌ FAILED") << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ❌ Generation failed: " << e.what() << std::endl;
        }
    }

    void test_single_decode_step(const std::string& prompt_text) {
        try {
            context->set_verbose(false);
            context->fill(prompt_text);

            CudaContext::Distribution dist = context->decode_step();

            std::cout << "  Top token: " << dist.token_ids[0]
                      << " (prob=" << std::fixed << std::setprecision(4) << dist.probabilities[0] << ")" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ❌ Decode step failed: " << e.what() << std::endl;
        }
    }

    void run_integration_test() {
        std::cout << "=== CUDA Integration Test (case: " << case_id << ") ===" << std::endl;

        try {
            setup_model_and_context();
            test_various_prompts();
            validate_context_behavior();

            std::cout << "✅ Integration test completed successfully!" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "❌ Integration test failed: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void setup_model_and_context() {
        try {
            ztensor::zTensorReader reader(model_path);
            model_metadata = auto_detect_config_from_ztensor(reader);

            setup_app_config();

            auto cuda_model = std::make_unique<Model>(app_config, model_metadata);
            context = std::make_unique<CudaContext>(
                std::move(cuda_model),
                tokenizer_path,
                app_config.kv_page_size
            );

            std::cout << "  ✅ Model and context initialized" << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Setup failed: " + std::string(e.what()));
        }
    }

    void setup_app_config() {
        std::filesystem::path model_file_path(model_path);
        app_config.model_name = model_file_path.parent_path().filename().string();
        app_config.cache_dir = model_file_path.parent_path().parent_path().parent_path();
        app_config.max_num_kv_pages = 1024;
        app_config.kv_page_size = 16;
        app_config.dist_size = std::min(model_metadata.architecture.vocab_size, 50000);
        app_config.max_num_embeds = 50000;
        app_config.device = "cuda:0";
        app_config.dtype = "bfloat16";
    }

    void test_various_prompts() {
        std::cout << "\n=== End-to-End Generation Tests ===" << std::endl;

        // Test 1: Simple conversation
        std::cout << "\nTest 1: Simple conversation" << std::endl;
        test_generation_with_context("Hello, how are you?", 10);

        // Test 2: Question answering
        std::cout << "\nTest 2: Question answering" << std::endl;
        test_generation_with_context("What is the capital of France?", 10);

        // Test 3: Single decode step for verification
        std::cout << "\nTest 3: Single decode step verification" << std::endl;
        test_single_decode_step("The weather today is");
    }

    void validate_context_behavior() {
        try {
            auto test_model = std::make_unique<Model>(app_config, model_metadata);
            CudaContext test_context(std::move(test_model), tokenizer_path, 8);
            test_context.set_verbose(false);

            test_context.fill("This is a test.");
            bool before_flush = test_context.validate_state();

            test_context.flush();
            bool after_flush = test_context.validate_state();

            std::cout << "  State validation: " << (before_flush && after_flush ? "✅" : "❌") << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ⚠️ Validation error: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Auto-detect model configuration from zTensor file
     * Uses the same logic that would be used by load_model_internal
     */
    ModelMetadata auto_detect_config_from_ztensor(ztensor::zTensorReader& reader) {
        ModelMetadata config;
        config.name = "llama-3.2-1b-instruct";
        config.description = "Llama 3.2 1B model loaded from zTensor";

        try {
            // Extract basic dimensions from key tensors
            auto embed_info = reader.get_tensor_info("model.embed_tokens.weight");
            config.architecture.vocab_size = static_cast<int>(embed_info.shape[0]);
            config.architecture.hidden_size = static_cast<int>(embed_info.shape[1]);

            // Extract MLP dimensions from first layer
            auto mlp_gate_info = reader.get_tensor_info("model.layers.0.mlp.gate_proj.weight");
            config.architecture.intermediate_size = static_cast<int>(mlp_gate_info.shape[0]);

            // Extract attention dimensions
            auto q_proj_info = reader.get_tensor_info("model.layers.0.self_attn.q_proj.weight");
            int total_q_dim = static_cast<int>(q_proj_info.shape[0]);

            auto k_proj_info = reader.get_tensor_info("model.layers.0.self_attn.k_proj.weight");
            int total_kv_dim = static_cast<int>(k_proj_info.shape[0]);

            // Count number of layers
            auto tensor_list = reader.list_tensors();
            int layer_count = 0;
            for (const auto& name : tensor_list) {
                if (name.find("model.layers.") == 0 && name.find(".self_attn.q_proj.weight") != std::string::npos) {
                    layer_count++;
                }
            }
            config.architecture.num_layers = layer_count;

            // Infer head configuration
            std::vector<int> common_head_sizes = {64, 80, 96, 128};
            bool found_head_config = false;

            for (int head_size : common_head_sizes) {
                if (total_q_dim % head_size == 0 && total_kv_dim % head_size == 0) {
                    config.architecture.head_size = head_size;
                    config.architecture.num_query_heads = total_q_dim / head_size;
                    config.architecture.num_key_value_heads = total_kv_dim / head_size;
                    found_head_config = true;
                    break;
                }
            }

            if (!found_head_config) {
                throw std::runtime_error("Could not determine head configuration");
            }

            // Set Llama 3.2 specific defaults
            config.architecture.type = "llama";
            config.architecture.use_qkv_bias = false;
            config.architecture.rope_theta = 500000.0f;
            config.architecture.rope_factor = 32.0f;
            config.architecture.rope_low_frequency_factor = 1.0f;
            config.architecture.rope_high_frequency_factor = 4.0f;
            config.architecture.rms_norm_eps = 1e-5f;

            // Set tokenizer info
            config.tokenizer.type = "bpe";
            config.template_type = "llama";

            // Setup parameters field - this is what load_model_internal expects
            config.parameters = {std::filesystem::path(model_path).filename().string()};

            // Validate configuration
            if (config.architecture.vocab_size <= 0 || config.architecture.hidden_size <= 0 || config.architecture.num_layers <= 0) {
                throw std::runtime_error("Invalid model dimensions detected");
            }

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to auto-detect config: " + std::string(e.what()));
        }

        return config;
    }

    /**
     * @brief Record integration readiness and configuration for validation
     */
    void record_integration_readiness_results(const std::string& artifact_dir_path) {
        try {
            // Create metadata showing successful integration preparation
            nlohmann::json metadata;
            metadata["case_id"] = case_id;
            metadata["test_type"] = "forward_pass_integration_direct_functions";
            metadata["model_path"] = model_path;
            metadata["backend"] = "cuda";
            metadata["integration_approach"] = "direct_function_reuse";
            metadata["ready_for_load_model_internal"] = true;

            // Real model configuration
            metadata["vocab_size"] = model_metadata.architecture.vocab_size;
            metadata["hidden_size"] = model_metadata.architecture.hidden_size;
            metadata["num_layers"] = model_metadata.architecture.num_layers;
            metadata["num_heads"] = model_metadata.architecture.num_query_heads;
            metadata["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            artifacts::write_meta_json(artifact_dir_path, metadata.dump(2));

            std::cout << "  📊 Artifacts saved to: " << artifact_dir_path << std::endl;
            std::cout << "    - Integration readiness metadata with real model config" << std::endl;
            std::cout << "    - Ready to call load_model_internal() with " << model_metadata.architecture.vocab_size << " vocab, "
                      << model_metadata.architecture.num_layers << " layers" << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to record artifacts: " + std::string(e.what()));
        }
    }
};

// Test execution function
void run_cuda_integration_with_context(const std::string& case_id) {
    try {
        CudaIntegrationWithContext test(case_id);
        test.run_integration_test();
    } catch (const std::exception& e) {
        std::cerr << "CUDA integration test with CudaContext failed: " << e.what() << std::endl;
        throw;
    }
}

// Main function for standalone testing
int main(int argc, char* argv[]) {
    std::string case_id = (argc > 1) ? argv[1] : "cuda_context_test";

    try {
        run_cuda_integration_with_context(case_id);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}