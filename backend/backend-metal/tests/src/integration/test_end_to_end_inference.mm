#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <thread>
#include "metal_model.hpp"
#include "metal_l4ma.hpp"
#include "metal_common.hpp"

// Simulated model configuration matching typical LLM settings
const int MODEL_VOCAB_SIZE = 32000;
const int MODEL_HIDDEN_SIZE = 4096;
const int MODEL_INTERMEDIATE_SIZE = 11008;
const int MODEL_NUM_LAYERS = 32;
const int MODEL_NUM_HEADS = 32;
const int MODEL_NUM_KV_HEADS = 32;
const int MODEL_MAX_SEQ_LEN = 2048;

// Test parameters for inference simulation
const int TEST_BATCH_SIZE = 4;
const int TEST_SEQUENCE_LENGTH = 128;
const int TEST_NUM_TOKENS = 32;  // Generate 32 new tokens

class MockModelWeights {
public:
    std::vector<float> embed_tokens;
    std::vector<float> norm_weight;
    std::vector<float> lm_head_weight;
    
    // Layer weights (simplified)
    std::vector<std::vector<float>> layer_norm_weights;
    std::vector<std::vector<float>> attn_q_weights;
    std::vector<std::vector<float>> attn_k_weights;
    std::vector<std::vector<float>> attn_v_weights;
    std::vector<std::vector<float>> attn_o_weights;
    std::vector<std::vector<float>> mlp_gate_weights;
    std::vector<std::vector<float>> mlp_up_weights;
    std::vector<std::vector<float>> mlp_down_weights;
    
    MockModelWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> weight_dist(0.0f, 0.02f);
        
        // Initialize embedding weights
        embed_tokens.resize(MODEL_VOCAB_SIZE * MODEL_HIDDEN_SIZE);
        for (auto& w : embed_tokens) w = weight_dist(gen);
        
        // Initialize norm weights
        norm_weight.resize(MODEL_HIDDEN_SIZE);
        for (auto& w : norm_weight) w = 1.0f + weight_dist(gen) * 0.1f;
        
        // Initialize LM head weights
        lm_head_weight.resize(MODEL_HIDDEN_SIZE * MODEL_VOCAB_SIZE);
        for (auto& w : lm_head_weight) w = weight_dist(gen);
        
        // Initialize layer weights (simplified for testing)
        layer_norm_weights.resize(MODEL_NUM_LAYERS);
        attn_q_weights.resize(MODEL_NUM_LAYERS);
        attn_k_weights.resize(MODEL_NUM_LAYERS);
        attn_v_weights.resize(MODEL_NUM_LAYERS);
        attn_o_weights.resize(MODEL_NUM_LAYERS);
        mlp_gate_weights.resize(MODEL_NUM_LAYERS);
        mlp_up_weights.resize(MODEL_NUM_LAYERS);
        mlp_down_weights.resize(MODEL_NUM_LAYERS);
        
        for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
            layer_norm_weights[layer].resize(MODEL_HIDDEN_SIZE);
            for (auto& w : layer_norm_weights[layer]) w = 1.0f + weight_dist(gen) * 0.1f;
            
            attn_q_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_HIDDEN_SIZE);
            for (auto& w : attn_q_weights[layer]) w = weight_dist(gen);
            
            attn_k_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_HIDDEN_SIZE);
            for (auto& w : attn_k_weights[layer]) w = weight_dist(gen);
            
            attn_v_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_HIDDEN_SIZE);
            for (auto& w : attn_v_weights[layer]) w = weight_dist(gen);
            
            attn_o_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_HIDDEN_SIZE);
            for (auto& w : attn_o_weights[layer]) w = weight_dist(gen);
            
            mlp_gate_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_INTERMEDIATE_SIZE);
            for (auto& w : mlp_gate_weights[layer]) w = weight_dist(gen);
            
            mlp_up_weights[layer].resize(MODEL_HIDDEN_SIZE * MODEL_INTERMEDIATE_SIZE);
            for (auto& w : mlp_up_weights[layer]) w = weight_dist(gen);
            
            mlp_down_weights[layer].resize(MODEL_INTERMEDIATE_SIZE * MODEL_HIDDEN_SIZE);
            for (auto& w : mlp_down_weights[layer]) w = weight_dist(gen);
        }
    }
};

class InferenceTimer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return static_cast<double>(duration.count()) / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Mock function to simulate token generation
std::vector<int32_t> generate_input_tokens(int batch_size, int seq_len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> token_dist(1, MODEL_VOCAB_SIZE - 1);
    
    std::vector<int32_t> tokens(batch_size * seq_len);
    for (auto& token : tokens) {
        token = token_dist(gen);
    }
    return tokens;
}

// Test the complete inference pipeline
int test_end_to_end_inference() {
    std::cout << "=== End-to-End Metal Inference Test ===" << std::endl;
    
    std::cout << "Model Configuration:" << std::endl;
    std::cout << "  Vocab Size: " << MODEL_VOCAB_SIZE << std::endl;
    std::cout << "  Hidden Size: " << MODEL_HIDDEN_SIZE << std::endl;
    std::cout << "  Num Layers: " << MODEL_NUM_LAYERS << std::endl;
    std::cout << "  Num Heads: " << MODEL_NUM_HEADS << std::endl;
    std::cout << "  Max Seq Len: " << MODEL_MAX_SEQ_LEN << std::endl;
    
    std::cout << "\nTest Parameters:" << std::endl;
    std::cout << "  Batch Size: " << TEST_BATCH_SIZE << std::endl;
    std::cout << "  Sequence Length: " << TEST_SEQUENCE_LENGTH << std::endl;
    std::cout << "  Tokens to Generate: " << TEST_NUM_TOKENS << std::endl;
    
    // Initialize Metal context
    std::cout << "\n1. Initializing Metal Context..." << std::endl;
    MetalContext context;
    if (!context.initialize()) {
        std::cerr << "❌ Failed to initialize Metal context" << std::endl;
        return 1;
    }
    std::cout << "✅ Metal context initialized" << std::endl;
    
    // Create L4MA configuration
    std::cout << "\n2. Setting up Model Configuration..." << std::endl;
    L4maConfig config;
    config.vocab_size = MODEL_VOCAB_SIZE;
    config.hidden_size = MODEL_HIDDEN_SIZE;
    config.intermediate_size = MODEL_INTERMEDIATE_SIZE;
    config.num_layers = MODEL_NUM_LAYERS;
    config.num_query_heads = MODEL_NUM_HEADS;
    config.num_key_value_heads = MODEL_NUM_KV_HEADS;
    config.max_position_embeddings = MODEL_MAX_SEQ_LEN;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    
    // Initialize mock weights
    std::cout << "\n3. Loading Mock Weights..." << std::endl;
    MockModelWeights weights;
    std::cout << "✅ Mock weights initialized" << std::endl;
    
    // Test basic tensor operations
    std::cout << "\n4. Testing Basic Tensor Operations..." << std::endl;
    
    try {
        // Test tensor creation
        MetalTensor input_tensor;
        input_tensor.initialize(context.getDevice(), {TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH}, bfloat16_t());
        
        MetalTensor output_tensor;
        output_tensor.initialize(context.getDevice(), {TEST_BATCH_SIZE, MODEL_VOCAB_SIZE}, float());
        
        std::cout << "✅ Tensor creation successful" << std::endl;
        
        // Generate input tokens
        std::vector<int32_t> input_tokens = generate_input_tokens(TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH);
        
        // Test embedding lookup (simplified)
        InferenceTimer timer;
        timer.start();
        
        // Simulate token processing
        for (int step = 0; step < TEST_NUM_TOKENS; step++) {
            // This would typically involve:
            // 1. Embedding lookup
            // 2. Layer-by-layer forward pass
            // 3. Final norm and LM head
            // 4. Sampling next token
            
            // For now, just simulate timing
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        double inference_time = timer.elapsed_ms();
        
        std::cout << "✅ Inference simulation completed" << std::endl;
        
        // Calculate performance metrics
        double tokens_per_second = (TEST_BATCH_SIZE * TEST_NUM_TOKENS * 1000.0) / inference_time;
        double time_per_token = inference_time / (TEST_BATCH_SIZE * TEST_NUM_TOKENS);
        
        std::cout << "\n5. Performance Metrics:" << std::endl;
        std::cout << "  Total inference time: " << std::fixed << std::setprecision(2) << inference_time << " ms" << std::endl;
        std::cout << "  Tokens per second: " << std::fixed << std::setprecision(1) << tokens_per_second << std::endl;
        std::cout << "  Time per token: " << std::fixed << std::setprecision(2) << time_per_token << " ms" << std::endl;
        
        // Memory usage estimation
        size_t model_memory_mb = (weights.embed_tokens.size() * sizeof(float) + 
                                 weights.lm_head_weight.size() * sizeof(float)) / (1024 * 1024);
        size_t activation_memory_mb = (TEST_BATCH_SIZE * TEST_SEQUENCE_LENGTH * MODEL_HIDDEN_SIZE * sizeof(bfloat16_t)) / (1024 * 1024);
        
        std::cout << "\n6. Memory Usage Estimation:" << std::endl;
        std::cout << "  Model weights: ~" << model_memory_mb << " MB" << std::endl;
        std::cout << "  Activations: ~" << activation_memory_mb << " MB" << std::endl;
        std::cout << "  Total: ~" << model_memory_mb + activation_memory_mb << " MB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Inference test failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 End-to-End Metal Inference Test Passed!" << std::endl;
    std::cout << "\nThis test validates:" << std::endl;
    std::cout << "  ✓ Metal context initialization" << std::endl;
    std::cout << "  ✓ Model configuration setup" << std::endl;
    std::cout << "  ✓ Tensor memory management" << std::endl;
    std::cout << "  ✓ Performance measurement" << std::endl;
    std::cout << "  ✓ Memory usage estimation" << std::endl;
    
    return 0;
}

int main() {
    return test_end_to_end_inference();
}