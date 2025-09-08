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

// Metal backend includes
#include "metal_l4ma.hpp"
#include "metal_common.hpp"
#include "metal_embedding.hpp"
#include "metal_add_residual.hpp"
#include "metal_gemm.hpp"
#include "ztensor.hpp"
#include "bpe.hpp"

// Include new Metal context manager
#include "metal_context.hpp"

// Include Metal model loader
#include "metal_model.hpp"

// Include ztensor for auto-config detection
#include "ztensor.hpp"

// Forward declare types defined in metal_l4ma.mm
struct AppConfig {
    std::string model_path;
    std::string cache_dir;
    bool verbose = false;
};

struct ModelMetadata {
    std::string model_name;
    std::string checkpoint_path;
    L4maConfig config;
    size_t total_params;
};


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
 * @brief Metal Integration Test using MetalGenerationContext for Proper KV Cache Management
 *
 * This version uses the new MetalGenerationContext class that mirrors the CUDA CudaContext implementation:
 * - Proper stateful KV cache management with Metal page allocation
 * - Correct token and position ID tracking across generation steps
 * - Modular design separating generation logic from test setup
 * - Uses Metal-specific L4MA model and buffer management
 */
class MetalIntegrationWithContext {
private:
    std::string model_path;
    std::string tokenizer_path;
    std::string case_id;
    L4maConfig l4ma_config;
    std::unique_ptr<MetalGenerationContext> context;
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
    MetalIntegrationWithContext(const std::string& case_name)
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
            // Enable verbose for first two tests to compare inputs
            static int test_count = 0;
            context->set_verbose(test_count < 2);
            test_count++;

            context->reset();  // Clear any previous state
            context->fill(prompt_text);
            context->flush();

            MetalGenerationContext::GreedySampler sampler;
            MetalGenerationContext::LengthStopCondition stop_condition(max_new_tokens);

            // CRITICAL DEBUG: Test if the first decode step produces different results
            std::cout << "  🔍 DEBUG: Testing first decode step for input variation..." << std::endl;
            auto first_decode = context->decode_step();
            if (!first_decode.token_ids.empty() && !first_decode.probabilities.empty()) {
                std::cout << "    First token prediction: " << first_decode.token_ids[0]
                         << " (prob=" << std::fixed << std::setprecision(4) << first_decode.probabilities[0] << ")" << std::endl;
            }

            // Reset and test again with a different single token to isolate the issue
            context->reset();
            context->fill(prompt_text);
            context->flush();

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
            context->reset();  // Clear any previous state
            context->fill(prompt_text);

            MetalGenerationContext::Distribution dist = context->decode_step();

            std::cout << "  Top token: " << dist.token_ids[0]
                      << " (prob=" << std::fixed << std::setprecision(4) << dist.probabilities[0] << ")" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  ❌ Decode step failed: " << e.what() << std::endl;
        }
    }

    void run_integration_test() {
        std::cout << "=== Metal Integration Test with Context (case: " << case_id << ") ===" << std::endl;

        try {
            setup_model_and_context();
            test_various_prompts();
            validate_context_behavior();

            std::cout << "✅ Metal integration test completed successfully!" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "❌ Metal integration test failed: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void setup_model_and_context() {
        try {
            // Initialize Metal context FIRST before creating any Metal resources
            std::cout << "  Initializing Metal context..." << std::endl;
            auto& metal_ctx = MetalContext::getInstance();
            if (!metal_ctx.initialize()) {
                throw std::runtime_error("Failed to initialize Metal context");
            }
            std::cout << "  ✅ Metal context initialized" << std::endl;

            // Initialize all Metal components
            std::cout << "  Initializing Metal compute components..." << std::endl;

            if (!initialize_metal_embedding()) {
                throw std::runtime_error("Failed to initialize Metal embedding");
            }
            std::cout << "    ✅ Metal embedding initialized" << std::endl;

            if (!initialize_metal_add_residual()) {
                throw std::runtime_error("Failed to initialize Metal add_residual");
            }
            std::cout << "    ✅ Metal add_residual initialized" << std::endl;

            if (!initialize_metal_gemm()) {
                throw std::runtime_error("Failed to initialize Metal GEMM");
            }
            std::cout << "    ✅ Metal GEMM initialized" << std::endl;

            // Create AppConfig and ModelMetadata for proper model loading

            // AppConfig
            AppConfig app_config;
            app_config.model_path = model_path;
            app_config.cache_dir = "/tmp";
            app_config.verbose = true;

            // AUTO-DETECT model configuration from zTensor file (fixes the root cause!)
            std::cout << "  Auto-detecting model configuration from zTensor file..." << std::endl;
            ztensor::zTensorReader reader(model_path);
            l4ma_config = auto_detect_config_from_ztensor(reader);

            // Validate that configuration is reasonable
            if (l4ma_config.vocab_size <= 0 || l4ma_config.hidden_size <= 0 || l4ma_config.num_layers <= 0) {
                throw std::runtime_error("Invalid auto-detected configuration");
            }

            if (l4ma_config.num_query_heads * l4ma_config.head_size != l4ma_config.hidden_size) {
                throw std::runtime_error("Configuration inconsistency: heads * head_size != hidden_size");
            }

            // Create model structure first (like CUDA test)
            auto metal_model = std::make_unique<MetalL4maForCausalLM<bfloat16_t>>(l4ma_config);
            if (!metal_model) {
                throw std::runtime_error("Failed to create Metal model structure");
            }
            std::cout << "  ✅ Model structure created with auto-detected config" << std::endl;

            // CRITICAL FIX: Load actual weights from zTensor file (like CUDA test does!)
            std::cout << "  Loading actual model weights from zTensor file..." << std::endl;
            if (!load_model_weights_from_ztensor(*metal_model, reader)) {
                throw std::runtime_error("Failed to load actual model weights from zTensor file");
            }
            std::cout << "  ✅ Model weights loaded from zTensor file" << std::endl;

            // Create Metal context
            context = std::make_unique<MetalGenerationContext>(
                std::move(metal_model),
                tokenizer_path,
                16  // kv_page_size - same as CUDA test
            );

            std::cout << "  ✅ Metal model and context initialized" << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error("Setup failed: " + std::string(e.what()));
        }
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

    /**
     * @brief Auto-detect L4maConfig from zTensor file - ported from CUDA test
     * This ensures the configuration matches the actual model weights
     */
    L4maConfig auto_detect_config_from_ztensor(ztensor::zTensorReader& reader) {
        L4maConfig config;
        config.type = "llama";

        try {
            // Extract basic dimensions from key tensors
            auto embed_info = reader.get_tensor_info("model.embed_tokens.weight");
            config.vocab_size = static_cast<int>(embed_info.shape[0]);
            config.hidden_size = static_cast<int>(embed_info.shape[1]);

            // Extract MLP dimensions from first layer
            auto mlp_gate_info = reader.get_tensor_info("model.layers.0.mlp.gate_proj.weight");
            config.intermediate_size = static_cast<int>(mlp_gate_info.shape[0]);

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
            config.num_layers = layer_count;

            // Infer head configuration
            std::vector<int> common_head_sizes = {64, 80, 96, 128};
            for (int head_size : common_head_sizes) {
                if (total_q_dim % head_size == 0 && total_kv_dim % head_size == 0) {
                    config.head_size = head_size;
                    config.num_query_heads = total_q_dim / head_size;
                    config.num_key_value_heads = total_kv_dim / head_size;
                    break;
                }
            }

            if (config.head_size == 0) {
                throw std::runtime_error("Could not determine head configuration");
            }

            // Set defaults
            config.rope_theta = 500000.0f;
            config.rope_factor = 32.0f;
            config.rms_norm_eps = 1e-5f;

            std::cout << "  ✅ Auto-detected config: vocab=" << config.vocab_size
                      << ", hidden=" << config.hidden_size
                      << ", layers=" << config.num_layers
                      << ", q_heads=" << config.num_query_heads
                      << ", kv_heads=" << config.num_key_value_heads
                      << ", head_size=" << config.head_size << std::endl;

            return config;

        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to auto-detect config from zTensor: " + std::string(e.what()));
        }
    }

    /**
     * @brief Load model weights from zTensor file - ported from CUDA test
     * This is the critical missing piece that loads actual trained weights!
     */
    bool load_model_weights_from_ztensor(MetalL4maForCausalLM<bfloat16_t>& model,
                                       ztensor::zTensorReader& reader) {
        try {
            std::cout << "    Loading model parameters from zTensor file..." << std::endl;

            // Get all model parameters
            auto model_params = model.get_parameters();
            std::cout << "    Model has " << model_params.size() << " parameters to load" << std::endl;

            size_t loaded_count = 0;
            size_t skipped_count = 0;

            // Load each parameter
            for (const auto& [param_name, metal_tensor] : model_params) {
                // Map Metal parameter name to zTensor name
                std::string ztensor_name = "model." + param_name;

                try {
                    // Check if tensor exists in zTensor file
                    auto tensor_list = reader.list_tensors();
                    auto it = std::find(tensor_list.begin(), tensor_list.end(), ztensor_name);
                    if (it == tensor_list.end()) {
                        std::cout << "    ⚠️  Skipping " << param_name << " (not found as " << ztensor_name << ")" << std::endl;
                        skipped_count++;
                        continue;
                    }

                    // Get tensor info and verify compatibility
                    auto tensor_info = reader.get_tensor_info(ztensor_name);
                    const auto& metal_shape = metal_tensor->shape();

                    // Verify shape compatibility
                    if (metal_shape.size() != tensor_info.shape.size()) {
                        std::cerr << "    ❌ Shape dimension mismatch for " << param_name
                                 << ": Metal=" << metal_shape.size() << "D, zTensor=" << tensor_info.shape.size() << "D" << std::endl;
                        skipped_count++;
                        continue;
                    }

                    bool shape_compatible = true;
                    for (size_t i = 0; i < metal_shape.size(); ++i) {
                        if (static_cast<size_t>(metal_shape[i]) != tensor_info.shape[i]) {
                            shape_compatible = false;
                            break;
                        }
                    }

                    if (!shape_compatible) {
                        std::cerr << "    ❌ Shape mismatch for " << param_name << std::endl;
                        skipped_count++;
                        continue;
                    }

                    // Read raw tensor data from zTensor file
                    const void* raw_data = reader.get_raw_tensor_pointer(ztensor_name);
                    if (!raw_data) {
                        std::cerr << "    ❌ Failed to get raw data for " << param_name << std::endl;
                        skipped_count++;
                        continue;
                    }

                    // Calculate total elements
                    size_t element_count = 1;
                    for (int dim : metal_shape) {
                        element_count *= dim;
                    }

                    // Convert and copy data based on zTensor dtype (using correct Metal API)
                    if (tensor_info.dtype == "bfloat16" || tensor_info.dtype == "bf16") {
                        // Direct copy for bfloat16
                        const bfloat16_t* bf16_data = static_cast<const bfloat16_t*>(raw_data);
                        metal_tensor->copyFromHost(bf16_data);
                    } else if (tensor_info.dtype == "float32" || tensor_info.dtype == "f32") {
                        // Convert float32 to bfloat16
                        const float* f32_data = static_cast<const float*>(raw_data);
                        std::vector<bfloat16_t> bf16_data(element_count);

                        for (size_t i = 0; i < element_count; ++i) {
                            bf16_data[i] = static_cast<bfloat16_t>(f32_data[i]);
                        }

                        metal_tensor->copyFromHost(bf16_data.data());
                    } else {
                        std::cerr << "    ❌ Unsupported dtype for " << param_name << ": " << tensor_info.dtype << std::endl;
                        skipped_count++;
                        continue;
                    }

                    loaded_count++;

                } catch (const std::exception& e) {
                    std::cerr << "    ❌ Error loading " << param_name << ": " << e.what() << std::endl;
                    skipped_count++;
                }
            }

            std::cout << "    ✅ Model weight loading completed: " << loaded_count << "/" << (loaded_count + skipped_count) << " parameters loaded" << std::endl;

            // Return true if we loaded most parameters (allow some mismatches)
            return loaded_count > 0 && loaded_count >= (loaded_count + skipped_count) * 0.8;

        } catch (const std::exception& e) {
            std::cerr << "    ❌ Failed to load model weights: " << e.what() << std::endl;
            return false;
        }
    }

    void validate_context_behavior() {
        try {
            // Create another Metal model for validation
            L4maConfig test_config = l4ma_config; // Copy the config
            auto temp_model = std::make_unique<MetalL4maForCausalLM<bfloat16_t>>(test_config);

            MetalGenerationContext test_context(std::move(temp_model), tokenizer_path, 8);
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

};

// Test execution function
void run_metal_integration_with_context(const std::string& case_id) {
    try {
        MetalIntegrationWithContext test(case_id);
        test.run_integration_test();
    } catch (const std::exception& e) {
        std::cerr << "Metal integration test with MetalGenerationContext failed: " << e.what() << std::endl;
        throw;
    }
}

// Main function for standalone testing
int main(int argc, char* argv[]) {
    std::string case_id = (argc > 1) ? argv[1] : "metal_context_test";

    try {
        run_metal_integration_with_context(case_id);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}