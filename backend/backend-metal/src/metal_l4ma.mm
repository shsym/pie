#include "metal_l4ma.hpp"
#include "metal_gemm_wrapper.hpp"
#include "metal_rmsnorm_wrapper.hpp"
#include "metal_rope_wrapper.hpp"
#include "metal_silu_and_mul_wrapper.hpp"
#include "metal_batch_prefill_attention_wrapper.hpp"
#include "metal_embedding.hpp"
#include "metal_topk_mask_logits.hpp"
#include "metal_softmax.hpp"
#include "metal_extract_k_values.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

// L4maConfig is defined in metal_common.hpp - no need to redefine
// Use the existing L4maConfig from metal_common.hpp

struct ModelMetadata {
    std::string model_name;
    std::string checkpoint_path;
    L4maConfig config;
    size_t total_params;
};

struct AppConfig {
    std::string model_path;
    std::string cache_dir;
    bool verbose = false;
};

// Template implementations for MetalL4maModel
template <typename T>
MetalL4maModel<T>::MetalL4maModel(const L4maConfig& config) : config_(config), norm_(config) {
    // Initialize embedding weights
    embed_tokens_weight_ = MetalTensor<T>({static_cast<size_t>(config_.vocab_size), 
                                         static_cast<size_t>(config_.hidden_size)});
    
    // Initialize decoder layers
    layers_.reserve(config_.num_layers);
    for (int i = 0; i < config_.num_layers; ++i) {
        layers_.emplace_back(config_);
    }
}

template <typename T>
void MetalL4maModel<T>::forward(MetalProfileScope profiler, MetalL4maBuffer<T>& buffer, 
                               MetalL4maKVCache<T>& kv_cache, T* final_norm_output) {
    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;
    
    // Allocate hidden states buffer
    auto hidden_states = buffer.template allocate<T>(num_tokens * hidden_size);
    
    // 1. Token embedding lookup using metal_embedding  
    auto& context = MetalContext::getInstance();
    metal_embedding_lookup_bfloat16(
        context.getDevice(), context.getCommandQueue(),
        reinterpret_cast<const bfloat16_t*>(embed_tokens_weight_.data()),
        config_.vocab_size,
        reinterpret_cast<const int32_t*>(buffer.input_ids.data()),
        num_tokens,
        reinterpret_cast<bfloat16_t*>(hidden_states.data()),
        config_.hidden_size
    );
    profiler.record("token_embedding");
    
    // 2. Forward through all decoder layers
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        auto layer_profiler = profiler.scope("layer_" + std::to_string(layer_idx));
        
        // Get KV cache for this layer
        auto [kv_cache_k, kv_cache_v] = kv_cache.get_layer_pointers(layer_idx);
        
        layers_[layer_idx].forward(layer_profiler, buffer, hidden_states.data(), kv_cache_k, kv_cache_v);
    }
    
    // 3. Final layer normalization
    norm_.forward(final_norm_output, hidden_states.data(), num_tokens, buffer.commandBuffer);
    profiler.record("final_norm");
    
    // Deallocate hidden states
    buffer.deallocate(hidden_states);
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maModel<T>::get_parameters() {
    std::map<std::string, MetalTensor<T>*> params;
    
    // Add embedding parameters
    params["embed_tokens.weight"] = &embed_tokens_weight_;
    
    // Add layer parameters
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        auto layer_params = layers_[layer_idx].get_parameters();
        for (auto& [name, tensor] : layer_params) {
            params["layers." + std::to_string(layer_idx) + "." + name] = tensor;
        }
    }
    
    // Add final norm parameters
    auto norm_params = norm_.get_parameters();
    for (auto& [name, tensor] : norm_params) {
        params["norm." + name] = tensor;
    }
    
    return params;
}

// Template implementations for MetalL4maForCausalLM
template <typename T>
MetalL4maForCausalLM<T>::MetalL4maForCausalLM(const L4maConfig& config) : config_(config), model_(config) {
}

template <typename T>
std::pair<std::vector<float>, std::vector<int32_t>> MetalL4maForCausalLM<T>::forward(
    MetalProfileScope profiler, MetalL4maBuffer<T>& buffer, MetalL4maKVCache<T>& kv_cache) {
    
    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;
    size_t vocab_size = config_.vocab_size;
    
    // Allocate final norm output
    auto final_norm_output = buffer.template allocate<T>(num_tokens * hidden_size);
    
    // 1. Forward through the model
    model_.forward(profiler.scope("model"), buffer, kv_cache, final_norm_output.data());
    
    // 2. Language model head (reuse embedding weights)
    auto logits = buffer.template allocate<T>(num_tokens * vocab_size);
    
    auto& context = MetalContext::getInstance();
    metal_gemm_bfloat16(
        context.getDevice(), context.getCommandQueue(),
        reinterpret_cast<const bfloat16_t*>(final_norm_output.data()),
        reinterpret_cast<const bfloat16_t*>(model_.get_embed_tokens_weight().data()),
        nullptr, // no bias
        reinterpret_cast<bfloat16_t*>(logits.data()),
        num_tokens, vocab_size, hidden_size,
        nullptr, 0, // workspace (unused for compatibility)
        false, false // no transpose
    );
    profiler.record("lm_head");
    
    // 3. Apply top-k mask and softmax to get final probabilities
    constexpr int top_k = 50; // Default top-k value
    auto topk_values = buffer.template allocate<float>(num_tokens * top_k);
    auto topk_indices = buffer.template allocate<int32_t>(num_tokens * top_k);
    
    // Apply top-k masking to logits
    int result = metal_topk_mask_logits_bfloat16(
        logits.data(), // in-place masking
        num_tokens,
        vocab_size,
        top_k
    );
    if (result != 0) {
        throw std::runtime_error("Failed to apply top-k masking");
    }
    profiler.record("topk_mask");
    
    // Apply softmax to get probabilities (convert bfloat16 to float temporarily)
    auto logits_float = buffer.template allocate<float>(num_tokens * vocab_size);
    MetalCast::bfloat16_to_float(context.getDevice(), context.getCommandQueue(),
                                reinterpret_cast<const bfloat16_t*>(logits.data()),
                                logits_float.data(), num_tokens * vocab_size);
    
    int softmax_result = metal_softmax_float(
        logits_float.data(),
        logits_float.data(), // in-place softmax
        num_tokens,
        vocab_size,
        1.0f // temperature
    );
    if (softmax_result != 0) {
        throw std::runtime_error("Failed to apply softmax");
    }
    
    // Convert back to bfloat16
    MetalCast::float_to_bfloat16(context.getDevice(), context.getCommandQueue(),
                                logits_float.data(),
                                reinterpret_cast<bfloat16_t*>(logits.data()),
                                num_tokens * vocab_size);
    
    buffer.deallocate(logits_float);
    profiler.record("softmax");
    
    // Extract top-k values and indices  
    int extract_result = metal_extract_k_values_bfloat16(
        reinterpret_cast<const bfloat16_t*>(logits.data()),
        topk_values.data(),
        topk_indices.data(),
        num_tokens,
        vocab_size,
        top_k
    );
    if (extract_result != 0) {
        throw std::runtime_error("Failed to extract top-k values");
    }
    profiler.record("extract_k_values");
    
    // Copy results to host vectors
    std::vector<float> result_values(num_tokens * top_k);
    std::vector<int32_t> result_indices(num_tokens * top_k);
    
    std::memcpy(result_values.data(), topk_values.data(), num_tokens * top_k * sizeof(float));
    std::memcpy(result_indices.data(), topk_indices.data(), num_tokens * top_k * sizeof(int32_t));
    
    // Deallocate intermediate tensors
    buffer.deallocate(final_norm_output);
    buffer.deallocate(logits);
    buffer.deallocate(topk_values);
    buffer.deallocate(topk_indices);
    
    return std::make_pair(std::move(result_values), std::move(result_indices));
}

template <typename T>
std::map<std::string, MetalTensor<T>*> MetalL4maForCausalLM<T>::get_parameters() {
    // For causal LM, we just return the model parameters
    // The language model head reuses the embedding weights
    return model_.get_parameters();
}

// Model loading utilities implementation
namespace MetalModelUtils {
    
    template<typename T>
    std::unique_ptr<MetalL4maForCausalLM<T>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata) {
        
        // Validate configuration
        if (!validate_model_config(metadata.config)) {
            std::cerr << "Invalid model configuration for Metal backend" << std::endl;
            return nullptr;
        }
        
        std::cout << "Loading Metal L4MA model: " << metadata.model_name << std::endl;
        std::cout << "Model config:" << std::endl;
        std::cout << "  vocab_size: " << metadata.config.vocab_size << std::endl;
        std::cout << "  hidden_size: " << metadata.config.hidden_size << std::endl;
        std::cout << "  num_layers: " << metadata.config.num_layers << std::endl;
        std::cout << "  num_heads: " << metadata.config.num_query_heads << std::endl;
        
        // Create model
        auto model = std::make_unique<MetalL4maForCausalLM<T>>(metadata.config);
        
        // Load parameters from checkpoint
        std::string weight_file = metadata.checkpoint_path + "/model.safetensors";
        
        if (config.verbose) {
            std::cout << "Loading weights from: " << weight_file << std::endl;
        }
        
        // Get all parameter tensors
        auto param_map = model->get_parameters();
        
        // TODO: Implement actual safetensors loading
        // For now, initialize with random values for testing
        std::cout << "Warning: Using random initialization (safetensors loading not implemented)" << std::endl;
        
        for (auto& [name, tensor] : param_map) {
            // Initialize with small random values
            size_t num_elements = 1;
            for (size_t dim : tensor->shape()) {
                num_elements *= dim;
            }
            
            std::vector<T> random_data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                // Simple random initialization
                random_data[i] = static_cast<T>((rand() / float(RAND_MAX) - 0.5f) * 0.1f);
            }
            
            if (!load_parameter_tensor(*tensor, random_data.data(), num_elements, name)) {
                std::cerr << "Failed to load parameter: " << name << std::endl;
                return nullptr;
            }
            
            if (config.verbose && name.find("embed_tokens") != std::string::npos) {
                std::cout << "Loaded parameter: " << name << " (" << num_elements << " elements)" << std::endl;
            }
        }
        
        std::cout << "Model loaded successfully with " << param_map.size() << " parameter tensors" << std::endl;
        return model;
    }
    
    template<typename T>
    bool load_parameter_tensor(MetalTensor<T>& target_tensor, const T* host_data, 
                              size_t num_elements, const std::string& tensor_name) {
        try {
            // Verify size matches
            size_t expected_elements = 1;
            for (size_t dim : target_tensor.shape()) {
                expected_elements *= dim;
            }
            
            if (expected_elements != num_elements) {
                std::cerr << "Size mismatch for tensor " << tensor_name 
                         << ": expected " << expected_elements 
                         << ", got " << num_elements << std::endl;
                return false;
            }
            
            // Copy data to Metal buffer
            target_tensor.copyFromHost(host_data);
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading tensor " << tensor_name << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    bool validate_model_config(const L4maConfig& config) {
        // Basic validation checks
        if (config.vocab_size <= 0 || config.hidden_size <= 0 || 
            config.num_layers <= 0 || config.num_query_heads <= 0) {
            std::cerr << "Invalid model dimensions" << std::endl;
            return false;
        }
        
        if (config.hidden_size % config.num_query_heads != 0) {
            std::cerr << "hidden_size must be divisible by num_query_heads" << std::endl;
            return false;
        }
        
        if (config.head_size != config.hidden_size / config.num_query_heads) {
            std::cerr << "head_size mismatch: expected " << (config.hidden_size / config.num_query_heads)
                     << ", got " << config.head_size << std::endl;
            return false;
        }
        
        // Check Metal-specific constraints
        auto& context = MetalContext::getInstance();
        if (!context.getDevice()) {
            std::cerr << "Metal device not available" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Explicit template instantiations
    template std::unique_ptr<MetalL4maForCausalLM<float>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata);
    template std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata);
        
    template bool load_parameter_tensor(MetalTensor<float>& target_tensor, 
                                       const float* host_data, size_t num_elements, 
                                       const std::string& tensor_name);
    template bool load_parameter_tensor(MetalTensor<bfloat16_t>& target_tensor, 
                                       const bfloat16_t* host_data, size_t num_elements, 
                                       const std::string& tensor_name);
}

// Explicit template instantiations for all model classes
template class MetalL4maRMSNorm<float>;
template class MetalL4maRMSNorm<bfloat16_t>;

template class MetalL4maMlp<float>;
template class MetalL4maMlp<bfloat16_t>;

template class MetalL4maAttention<float>;
template class MetalL4maAttention<bfloat16_t>;

template class MetalL4maDecoderLayer<float>;
template class MetalL4maDecoderLayer<bfloat16_t>;

template class MetalL4maModel<float>;
template class MetalL4maModel<bfloat16_t>;

template class MetalL4maForCausalLM<float>;
template class MetalL4maForCausalLM<bfloat16_t>;