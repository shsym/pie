#include "metal_model.hpp"
#include "metal_l4ma.hpp"
#include "metal_common.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>

// Test configuration structures (must match backend interface)
struct AppConfig {
    std::string model_path = "/tmp/test_model";
    std::string cache_dir = "/tmp/cache";
    bool verbose = false;
    int32_t kv_page_size = 16;
    int32_t dist_size = 64;
    size_t max_num_kv_pages = 14000;
    size_t max_num_embeds = 50000;
};

struct ModelMetadata {
    std::string model_name = "L4MA";
    std::string checkpoint_path = "/tmp/test_model.ckpt";
    L4maConfig config;
    size_t total_params = 125000000; // 125M parameters
};

// Test configuration - initialize with realistic values
L4maConfig create_test_l4ma_config() {
    L4maConfig config;
    config.type = "l4ma";
    config.hidden_size = 1024;
    config.intermediate_size = 2816;
    config.num_query_heads = 8;
    config.num_key_value_heads = 4;
    config.head_size = 128;
    config.num_layers = 4;
    config.vocab_size = 32000;
    config.use_qkv_bias = false;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.rope_factor = 1.0f;
    config.rope_high_frequency_factor = 1.0f;
    config.rope_low_frequency_factor = 1.0f;
    return config;
}

/**
 * @brief Full inference pipeline test that matches CUDA backend patterns
 * 
 * Tests the complete Metal model forward pass including:
 * - Model initialization and loading
 * - Token embedding and position encoding
 * - Multi-layer transformer processing
 * - Final softmax and top-k sampling
 * - Memory management and cleanup
 * - Performance validation
 */
class MetalFullInferencePipelineTest {
public:
    MetalFullInferencePipelineTest() {
        config_.config = create_test_l4ma_config();
        app_config_.verbose = true;
    }
    
    bool run() {
        std::cout << "=== Metal Full Inference Pipeline Test ===" << std::endl;
        
        if (!initialize_metal_backend()) {
            std::cerr << "Failed to initialize Metal backend" << std::endl;
            return false;
        }
        
        if (!test_model_initialization()) {
            std::cerr << "Model initialization test failed" << std::endl;
            return false;
        }
        
        if (!test_single_token_forward()) {
            std::cerr << "Single token forward test failed" << std::endl;
            return false;
        }
        
        if (!test_batch_token_forward()) {
            std::cerr << "Batch token forward test failed" << std::endl;
            return false;
        }
        
        if (!test_sequence_generation()) {
            std::cerr << "Sequence generation test failed" << std::endl;
            return false;
        }
        
        if (!test_kv_cache_management()) {
            std::cerr << "KV cache management test failed" << std::endl;
            return false;
        }
        
        if (!validate_performance_metrics()) {
            std::cerr << "Performance validation failed" << std::endl;
            return false;
        }
        
        std::cout << "✅ All full inference pipeline tests passed!" << std::endl;
        return true;
    }
    
private:
    AppConfig app_config_;
    ModelMetadata config_;
    std::unique_ptr<MetalModel> model_;
    
    bool initialize_metal_backend() {
        std::cout << "Initializing Metal backend..." << std::endl;
        
        auto& context = MetalContext::getInstance();
        if (!context.initialize()) {
            return false;
        }
        
        // Verify Metal device capabilities
        auto device_info = MetalModelFactory::getDeviceInfo();
        std::cout << "Metal device: " << device_info.name << std::endl;
        std::cout << "Max buffer length: " << device_info.max_buffer_length / 1024 / 1024 << " MB" << std::endl;
        std::cout << "bfloat16 support: " << (device_info.supports_bfloat16 ? "Yes" : "No") << std::endl;
        
        if (!MetalModelFactory::isMetalAvailable()) {
            std::cerr << "Metal not available on this system" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool test_model_initialization() {
        std::cout << "Testing model initialization..." << std::endl;
        
        try {
            // Create mock model weights for testing
            if (!create_mock_model_weights()) {
                return false;
            }
            
            // Initialize model using factory
            model_ = MetalModelFactory::createModel(app_config_, config_);
            if (!model_) {
                std::cerr << "Failed to create Metal model" << std::endl;
                return false;
            }
            
            // Validate model state
            if (!MetalModelDiagnostics::validateModelState(*model_)) {
                auto error = MetalModelDiagnostics::getLastError();
                std::cerr << "Model validation failed: " << error.description << std::endl;
                return false;
            }
            
            std::cout << "✓ Model initialized successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Model initialization exception: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool test_single_token_forward() {
        std::cout << "Testing single token forward pass..." << std::endl;
        
        // Prepare single token input
        MetalModel::ForwardTextCommand command;
        command.kv_page_last_len = 0;
        command.kv_page_ids = {};
        command.token_ids = {1}; // Single token (BOS)
        command.position_ids = {0};
        command.brle_masks = {{}};
        command.output_indices = {0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute forward pass
        auto results = model_->handle_forward_text({command});
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Validate results
        if (results.empty() || results[0].empty()) {
            std::cerr << "Empty results from forward pass" << std::endl;
            return false;
        }
        
        const auto& distribution = results[0][0];
        if (distribution.token_ids.size() != distribution.probabilities.size()) {
            std::cerr << "Mismatched token_ids and probabilities sizes" << std::endl;
            return false;
        }
        
        // Validate probability distribution
        float prob_sum = 0.0f;
        for (float prob : distribution.probabilities) {
            if (prob < 0.0f || prob > 1.0f) {
                std::cerr << "Invalid probability value: " << prob << std::endl;
                return false;
            }
            prob_sum += prob;
        }
        
        if (std::abs(prob_sum - 1.0f) > 0.01f) {
            std::cerr << "Probability distribution doesn't sum to 1.0: " << prob_sum << std::endl;
            return false;
        }
        
        std::cout << "✓ Single token forward pass: " << duration.count() << "μs" << std::endl;
        std::cout << "  Generated " << distribution.token_ids.size() << " top-k tokens" << std::endl;
        return true;
    }
    
    bool test_batch_token_forward() {
        std::cout << "Testing batch token forward pass..." << std::endl;
        
        // Prepare batch input (4 sequences)
        std::vector<MetalModel::ForwardTextCommand> commands;
        for (int i = 0; i < 4; ++i) {
            MetalModel::ForwardTextCommand command;
            command.kv_page_last_len = 0;
            command.kv_page_ids = {};
            command.token_ids = {1, 2, 3}; // 3 tokens per sequence
            command.position_ids = {0, 1, 2};
            command.brle_masks = {{}, {}, {}};
            command.output_indices = {2}; // Output distribution for last token
            commands.push_back(command);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute batch forward pass
        auto results = model_->handle_forward_text(commands);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Validate batch results
        if (results.size() != 4) {
            std::cerr << "Expected 4 batch results, got " << results.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].empty()) {
                std::cerr << "Empty result for batch item " << i << std::endl;
                return false;
            }
            
            const auto& distribution = results[i][0];
            if (distribution.token_ids.empty() || distribution.probabilities.empty()) {
                std::cerr << "Empty distribution for batch item " << i << std::endl;
                return false;
            }
        }
        
        std::cout << "✓ Batch token forward pass: " << duration.count() << "μs" << std::endl;
        std::cout << "  Processed 4 sequences with 3 tokens each" << std::endl;
        return true;
    }
    
    bool test_sequence_generation() {
        std::cout << "Testing sequence generation..." << std::endl;
        
        // Generate a sequence of 10 tokens
        std::vector<uint32_t> generated_tokens = {1}; // Start with BOS
        std::vector<uint32_t> kv_page_ids;
        
        for (int step = 0; step < 9; ++step) {
            MetalModel::ForwardTextCommand command;
            command.kv_page_last_len = generated_tokens.size() % app_config_.kv_page_size;
            command.kv_page_ids = kv_page_ids;
            command.token_ids = {generated_tokens.back()};
            command.position_ids = {static_cast<uint32_t>(generated_tokens.size() - 1)};
            command.brle_masks = {{}};
            command.output_indices = {0};
            
            auto results = model_->handle_forward_text({command});
            if (results.empty() || results[0].empty()) {
                std::cerr << "Failed to generate token at step " << step << std::endl;
                return false;
            }
            
            // Select top token (greedy decoding for test)
            const auto& distribution = results[0][0];
            if (distribution.token_ids.empty()) {
                std::cerr << "Empty distribution at generation step " << step << std::endl;
                return false;
            }
            
            uint32_t next_token = distribution.token_ids[0];
            generated_tokens.push_back(next_token);
            
            // Update KV cache page tracking
            if (generated_tokens.size() % app_config_.kv_page_size == 1 && generated_tokens.size() > 1) {
                kv_page_ids.push_back(kv_page_ids.size());
            }
        }
        
        if (generated_tokens.size() != 10) {
            std::cerr << "Expected 10 generated tokens, got " << generated_tokens.size() << std::endl;
            return false;
        }
        
        std::cout << "✓ Sequence generation completed" << std::endl;
        std::cout << "  Generated sequence length: " << generated_tokens.size() << std::endl;
        std::cout << "  First 5 tokens: ";
        for (size_t i = 0; i < std::min(size_t(5), generated_tokens.size()); ++i) {
            std::cout << generated_tokens[i] << " ";
        }
        std::cout << std::endl;
        return true;
    }
    
    bool test_kv_cache_management() {
        std::cout << "Testing KV cache management..." << std::endl;
        
        // Test allocate/deallocate cycle
        std::vector<MetalModel::AllocateCommand> alloc_commands;
        MetalModel::AllocateCommand alloc_cmd;
        alloc_cmd.kind = MetalModel::ObjectKind::KV_BLOCK;
        alloc_cmd.object_id_offset = 0;
        alloc_cmd.count = 10;
        alloc_commands.push_back(alloc_cmd);
        
        model_->handle_allocate(alloc_commands);
        
        // Use allocated KV blocks
        MetalModel::ForwardTextCommand command;
        command.kv_page_last_len = 8;
        command.kv_page_ids = {0, 1, 2};
        command.token_ids = {1, 2, 3, 4};
        command.position_ids = {16, 17, 18, 19}; // Continue from previous context
        command.brle_masks = {{}, {}, {}, {}};
        command.output_indices = {3};
        
        auto results = model_->handle_forward_text({command});
        if (results.empty() || results[0].empty()) {
            std::cerr << "Failed forward pass with KV cache" << std::endl;
            return false;
        }
        
        // Test copy block operation
        std::vector<MetalModel::CopyBlockCommand> copy_commands;
        MetalModel::CopyBlockCommand copy_cmd;
        copy_cmd.source_block_id = 0;
        copy_cmd.destination_block_id = 5;
        copy_cmd.source_start = 0;
        copy_cmd.destination_start = 0;
        copy_cmd.length = 16;
        copy_commands.push_back(copy_cmd);
        
        model_->handle_copy_block(copy_commands);
        
        // Deallocate blocks
        std::vector<MetalModel::DeallocateCommand> dealloc_commands;
        MetalModel::DeallocateCommand dealloc_cmd;
        dealloc_cmd.kind = MetalModel::ObjectKind::KV_BLOCK;
        dealloc_cmd.object_id_offset = 0;
        dealloc_cmd.count = 10;
        dealloc_commands.push_back(dealloc_cmd);
        
        model_->handle_deallocate(dealloc_commands);
        
        std::cout << "✓ KV cache management completed" << std::endl;
        return true;
    }
    
    bool validate_performance_metrics() {
        std::cout << "Validating performance metrics..." << std::endl;
        
        // Run performance benchmark
        const int num_iterations = 10;
        const int batch_size = 4;
        const int seq_len = 64;
        
        std::vector<double> latencies;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Prepare batch
            std::vector<MetalModel::ForwardTextCommand> commands;
            for (int b = 0; b < batch_size; ++b) {
                MetalModel::ForwardTextCommand command;
                command.kv_page_last_len = 0;
                command.kv_page_ids = {};
                
                // Generate sequence tokens
                command.token_ids.resize(seq_len);
                command.position_ids.resize(seq_len);
                for (int i = 0; i < seq_len; ++i) {
                    command.token_ids[i] = (i + b * 1000) % config_.config.vocab_size;
                    command.position_ids[i] = i;
                }
                
                command.brle_masks.resize(seq_len);
                command.output_indices = {seq_len - 1};
                commands.push_back(command);
            }
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = model_->handle_forward_text(commands);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            if (results.size() != batch_size) {
                std::cerr << "Performance test failed at iteration " << iter << std::endl;
                return false;
            }
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        }
        
        // Calculate statistics
        double avg_latency = 0.0;
        for (double lat : latencies) {
            avg_latency += lat;
        }
        avg_latency /= latencies.size();
        
        double throughput = (batch_size * seq_len) / (avg_latency / 1000.0); // tokens per second
        
        std::cout << "✓ Performance metrics:" << std::endl;
        std::cout << "  Average latency: " << avg_latency << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput << " tokens/sec" << std::endl;
        
        // Performance thresholds (adjust based on hardware)
        if (avg_latency > 1000.0) { // 1 second threshold
            std::cerr << "Warning: High latency detected" << std::endl;
        }
        
        if (throughput < 100.0) { // 100 tokens/sec threshold
            std::cerr << "Warning: Low throughput detected" << std::endl;
        }
        
        // Print profiling report if verbose enabled
        if (app_config_.verbose) {
            MetalModelProfiler::printProfilingReport();
        }
        
        return true;
    }
    
    bool create_mock_model_weights() {
        // For testing purposes, we'll create minimal mock weights
        // In a real implementation, this would load from checkpoint files
        std::cout << "Creating mock model weights for testing..." << std::endl;
        
        // This is a placeholder - in practice, model weights would be loaded
        // from actual checkpoint files using MetalModelUtils::load_model_internal
        
        return true;
    }
};

int main() {
    std::cout << "Metal Full Inference Pipeline Test" << std::endl;
    std::cout << "===================================" << std::endl;
    
    MetalFullInferencePipelineTest test;
    
    bool success = test.run();
    
    if (success) {
        std::cout << "\n🎉 All inference pipeline tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some inference pipeline tests failed!" << std::endl;
        return 1;
    }
}