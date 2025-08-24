#include "metal_model.hpp"
#include "metal_l4ma.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <sstream>
#include <functional>

// Configuration structures (matching CUDA backend)
struct AppConfig {
    std::string model_path;
    std::string cache_dir;
    bool verbose = false;
    int32_t kv_page_size = 16;
    int32_t dist_size = 64;
    size_t max_num_kv_pages = 14000;
    size_t max_num_embeds = 50000;
};

// L4maConfig is already defined in metal_common.hpp, so we don't redefine it

struct ModelMetadata {
    std::string model_name;
    std::string checkpoint_path;
    L4maConfig config;
    size_t total_params;
};

// Test configuration that mimics typical PIE usage
struct TestConfig {
    std::string model_name = "llama-3.2-1b-instruct";
    std::string cache_dir = "~/.cache/pie/models/";
    int32_t kv_page_size = 16;
    int32_t dist_size = 64;
    size_t max_num_kv_pages = 1000;  // Reduced for testing
    size_t max_num_embeds = 1000;    // Reduced for testing
    bool verbose = true;
};

// Mock model metadata for testing
ModelMetadata create_test_metadata() {
    ModelMetadata metadata;
    metadata.model_name = "llama-3.2-1b-instruct";
    metadata.checkpoint_path = "/test/model/path";
    metadata.total_params = 1248013312; // ~1.2B parameters
    
    // L4MA config matching typical 1B model
    metadata.config.vocab_size = 128256;
    metadata.config.hidden_size = 2048;
    metadata.config.intermediate_size = 8192;
    metadata.config.num_layers = 16;
    metadata.config.num_query_heads = 32;      // Correct field name
    metadata.config.num_key_value_heads = 8;
    metadata.config.head_size = 64;
    metadata.config.use_qkv_bias = false;
    metadata.config.rms_norm_eps = 1e-6f;
    metadata.config.rope_theta = 10000.0f;
    metadata.config.rope_factor = 1.0f;
    metadata.config.rope_high_frequency_factor = 4.0f;
    metadata.config.rope_low_frequency_factor = 1.0f;
    
    return metadata;
}

// Test utility functions
class MetalModelTester {
public:
    static bool test_metal_availability() {
        std::cout << "\n=== Testing Metal Availability ===" << std::endl;
        
        if (!MetalModelFactory::isMetalAvailable()) {
            std::cout << "❌ Metal not available on this system" << std::endl;
            return false;
        }
        
        auto device_info = MetalModelFactory::getDeviceInfo();
        std::cout << "✅ Metal available" << std::endl;
        std::cout << "  Device: " << device_info.name << std::endl;
        std::cout << "  Max Buffer: " << (device_info.max_buffer_length / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Max Threadgroup Memory: " << device_info.max_threadgroup_memory_length << " bytes" << std::endl;
        std::cout << "  Supports bfloat16: " << (device_info.supports_bfloat16 ? "Yes" : "No") << std::endl;
        
        return true;
    }
    
    static bool test_model_creation() {
        std::cout << "\n=== Testing Model Creation ===" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = test_config.verbose,
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto model = MetalModelFactory::createModel(config, metadata);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "✅ Model created successfully" << std::endl;
            std::cout << "  Creation time: " << duration.count() << " ms" << std::endl;
            std::cout << "  Model ready for inference" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Model creation failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    static bool test_memory_allocation() {
        std::cout << "\n=== Testing Memory Allocation ===" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = test_config.verbose,
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            auto model = MetalModelFactory::createModel(config, metadata);
            
            // Test KV block allocation (typical inference pattern)
            std::vector<MetalModel::AllocateCommand> alloc_commands;
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::KV_BLOCK,
                .object_id_offset = 0,
                .count = 10
            });
            
            // Test embedding allocation
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::EMB,
                .object_id_offset = 0,
                .count = 5
            });
            
            // Test distribution allocation
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::DIST,
                .object_id_offset = 0,
                .count = 5
            });
            
            model->handle_allocate(alloc_commands);
            std::cout << "✅ Allocated KV blocks, embeddings, and distributions" << std::endl;
            
            // Test deallocation
            std::vector<MetalModel::DeallocateCommand> dealloc_commands;
            dealloc_commands.push_back({
                .kind = MetalModel::ObjectKind::KV_BLOCK,
                .object_id_offset = 5,
                .count = 5
            });
            
            model->handle_deallocate(dealloc_commands);
            std::cout << "✅ Deallocated partial KV blocks" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Memory allocation test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    static bool test_text_embedding() {
        std::cout << "\n=== Testing Text Embedding ===" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = test_config.verbose,
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            auto model = MetalModelFactory::createModel(config, metadata);
            
            // Allocate embeddings first
            std::vector<MetalModel::AllocateCommand> alloc_commands;
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::EMB,
                .object_id_offset = 0,
                .count = 3
            });
            model->handle_allocate(alloc_commands);
            
            // Test text embedding (typical tokens for \"Hello, how are you?\")
            std::vector<MetalModel::EmbedTextCommand> embed_commands;
            embed_commands.push_back({.embedding_id = 0, .token_id = 15339, .position_id = 0}); // \"Hello\"
            embed_commands.push_back({.embedding_id = 1, .token_id = 11, .position_id = 1});    // \",\"
            embed_commands.push_back({.embedding_id = 2, .token_id = 1268, .position_id = 2});  // \"how\"
            
            model->handle_embed_text(embed_commands);
            std::cout << "✅ Embedded text tokens successfully" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Text embedding test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    static bool test_full_inference_workflow() {
        std::cout << "\n=== Testing Full Inference Workflow ===" << std::endl;
        std::cout << "(Simulating PIE CLI text completion request)" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = test_config.verbose,
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            auto model = MetalModelFactory::createModel(config, metadata);
            
            // Enable profiling for performance monitoring
            MetalModelProfiler::enableProfiling(true);
            MetalModelProfiler::resetProfiling();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 1. Allocate resources (typical PIE workflow)
            std::vector<MetalModel::AllocateCommand> alloc_commands;
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::KV_BLOCK,
                .object_id_offset = 0,
                .count = 5  // Allocate KV blocks for context
            });
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::EMB,
                .object_id_offset = 0,
                .count = 10  // Embeddings for input tokens
            });
            alloc_commands.push_back({
                .kind = MetalModel::ObjectKind::DIST,
                .object_id_offset = 0,
                .count = 5   // Distributions for output tokens
            });
            model->handle_allocate(alloc_commands);
            
            // 2. Embed input text: \"What is machine learning?\"
            std::vector<MetalModel::EmbedTextCommand> embed_commands;
            embed_commands.push_back({.embedding_id = 0, .token_id = 3923, .position_id = 0});  // \"What\"
            embed_commands.push_back({.embedding_id = 1, .token_id = 374, .position_id = 1});   // \"is\"
            embed_commands.push_back({.embedding_id = 2, .token_id = 5780, .position_id = 2});  // \"machine\"
            embed_commands.push_back({.embedding_id = 3, .token_id = 6975, .position_id = 3});  // \"learning\"
            embed_commands.push_back({.embedding_id = 4, .token_id = 30, .position_id = 4});    // \"?\"
            model->handle_embed_text(embed_commands);
            
            // 3. Fill KV cache blocks with context
            std::vector<MetalModel::FillBlockCommand> fill_commands;
            MetalModel::FillBlockCommand fill_cmd;
            fill_cmd.last_block_len = 5;
            fill_cmd.context_block_ids = {0, 1, 2, 3, 4};
            fill_cmd.input_embedding_ids = {0, 1, 2, 3, 4};
            fill_cmd.output_embedding_ids = {4}; // Generate distribution for last token
            fill_commands.push_back(fill_cmd);
            model->handle_fill_block(fill_commands);
            
            // 4. Decode token distributions (autoregressive generation)
            std::vector<MetalModel::DecodeTokenDistributionCommand> decode_commands;
            decode_commands.push_back({.embedding_id = 4, .distribution_id = 0});
            model->handle_decode_token_distribution(decode_commands);
            
            // 5. Sample top-k tokens for text completion
            std::vector<MetalModel::SampleTopKCommand> sample_commands;
            sample_commands.push_back({.distribution_id = 0, .k = 10});
            auto sample_results = model->handle_sample_top_k(sample_commands);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Verify results
            if (!sample_results.empty() && !sample_results[0].token_ids.empty()) {
                std::cout << "✅ Full inference workflow completed successfully" << std::endl;
                std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
                std::cout << "  Top-k results (k=" << sample_results[0].token_ids.size() << "):" << std::endl;
                
                for (size_t i = 0; i < std::min(size_t(5), sample_results[0].token_ids.size()); ++i) {
                    std::cout << "    Token " << sample_results[0].token_ids[i] 
                             << " (prob: " << sample_results[0].probabilities[i] << ")" << std::endl;
                }
                
                // Print profiling report
                std::cout << "\nPerformance Profile:" << std::endl;
                MetalModelProfiler::printProfilingReport();
                
                return true;
            } else {
                std::cout << "❌ No valid results from inference" << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ Full inference workflow failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    static bool test_forward_text_api() {
        std::cout << "\n=== Testing Forward Text API ===" << std::endl;
        std::cout << "(Direct API similar to WASM inferlets)" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = test_config.verbose,
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            auto model = MetalModelFactory::createModel(config, metadata);
            
            // Test the higher-level forward_text API
            std::vector<MetalModel::ForwardTextCommand> forward_commands;
            MetalModel::ForwardTextCommand cmd;
            cmd.kv_page_last_len = 4;
            cmd.kv_page_ids = {0, 1, 2};
            cmd.token_ids = {3923, 374, 5780, 6975};    // \"What is machine learning\"
            cmd.position_ids = {0, 1, 2, 3};
            cmd.output_indices = {3}; // Generate output for last token
            forward_commands.push_back(cmd);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = model->handle_forward_text(forward_commands);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (!results.empty() && !results[0].empty() && !results[0][0].token_ids.empty()) {
                std::cout << "✅ Forward text API completed successfully" << std::endl;
                std::cout << "  Inference time: " << duration.count() << " ms" << std::endl;
                std::cout << "  Output distributions: " << results[0].size() << std::endl;
                std::cout << "  Top tokens for position 0:" << std::endl;
                
                const auto& dist = results[0][0];
                for (size_t i = 0; i < std::min(size_t(5), dist.token_ids.size()); ++i) {
                    std::cout << "    Token " << dist.token_ids[i] 
                             << " (prob: " << dist.probabilities[i] << ")" << std::endl;
                }
                
                return true;
            } else {
                std::cout << "❌ Forward text API returned no results" << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ Forward text API test failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    static bool test_performance_benchmarks() {
        std::cout << "\n=== Performance Benchmarks ===" << std::endl;
        std::cout << "(Measuring key metrics for PIE deployment)" << std::endl;
        
        try {
            TestConfig test_config;
            AppConfig config{
                .model_path = test_config.cache_dir + test_config.model_name,
                .cache_dir = test_config.cache_dir,
                .verbose = false, // Reduce output for benchmarking
                .kv_page_size = test_config.kv_page_size,
                .dist_size = test_config.dist_size,
                .max_num_kv_pages = test_config.max_num_kv_pages,
                .max_num_embeds = test_config.max_num_embeds
            };
            
            ModelMetadata metadata = create_test_metadata();
            
            // Benchmark model creation time
            auto start_time = std::chrono::high_resolution_clock::now();
            auto model = MetalModelFactory::createModel(config, metadata);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Benchmark inference with different sequence lengths
            std::vector<int> seq_lengths = {1, 4, 8, 16, 32};
            
            std::cout << "Model Creation Time: " << creation_time.count() << " ms" << std::endl;
            std::cout << "\nInference Times by Sequence Length:" << std::endl;
            
            for (int seq_len : seq_lengths) {
                std::vector<MetalModel::ForwardTextCommand> forward_commands;
                MetalModel::ForwardTextCommand cmd;
                cmd.kv_page_last_len = seq_len;
                
                // Generate sequence of token IDs
                for (int i = 0; i < seq_len; ++i) {
                    cmd.kv_page_ids.push_back(i);
                    cmd.token_ids.push_back(1000 + i); // Dummy tokens
                    cmd.position_ids.push_back(i);
                }
                cmd.output_indices = {static_cast<uint32_t>(seq_len - 1)}; // Output for last token
                forward_commands.push_back(cmd);
                
                // Warm up
                model->handle_forward_text(forward_commands);
                
                // Benchmark multiple runs
                const int num_runs = 5;
                auto bench_start = std::chrono::high_resolution_clock::now();
                
                for (int run = 0; run < num_runs; ++run) {
                    model->handle_forward_text(forward_commands);
                }
                
                auto bench_end = std::chrono::high_resolution_clock::now();
                auto avg_time = std::chrono::duration_cast<std::chrono::microseconds>(bench_end - bench_start) / num_runs;
                
                std::cout << "  Seq length " << seq_len << ": " << avg_time.count() << " μs" << std::endl;
            }
            
            std::cout << "✅ Performance benchmarking completed" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Performance benchmark failed: " << e.what() << std::endl;
            return false;
        }
    }
};

// Main test runner
int main() {
    std::cout << "=== Metal Model Integration Test Suite ===" << std::endl;
    std::cout << "Testing Metal backend for PIE inference engine" << std::endl;
    
    bool all_tests_passed = true;
    std::vector<std::pair<std::string, std::function<bool()>>> tests = {
        {"Metal Availability", MetalModelTester::test_metal_availability},
        {"Model Creation", MetalModelTester::test_model_creation},
        {"Memory Allocation", MetalModelTester::test_memory_allocation},
        {"Text Embedding", MetalModelTester::test_text_embedding},
        {"Full Inference Workflow", MetalModelTester::test_full_inference_workflow},
        {"Forward Text API", MetalModelTester::test_forward_text_api},
        {"Performance Benchmarks", MetalModelTester::test_performance_benchmarks}
    };
    
    int passed = 0;
    int total = tests.size();
    
    for (const auto& [name, test_func] : tests) {
        try {
            if (test_func()) {
                passed++;
            } else {
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cout << "❌ Test '" << name << "' threw exception: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << " tests" << std::endl;
    
    if (all_tests_passed) {
        std::cout << "🎉 All Metal model integration tests passed!" << std::endl;
        std::cout << "Metal backend is ready for PIE deployment" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed" << std::endl;
        return 1;
    }
}