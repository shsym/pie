#include "metal_l4ma.hpp"
#include "metal_gemm_wrapper.hpp"
#include "metal_rmsnorm_wrapper.hpp"
#include "metal_rope_wrapper.hpp"
#include "metal_silu_and_mul_wrapper.hpp"
#include "metal_batch_prefill_attention.hpp"
#include "metal_embedding.hpp"
#include "metal_topk_mask_logits.hpp"
#include "metal_softmax.hpp"
#include "metal_extract_k_values.hpp"
#include "../../backend-cuda/src/ztensor.hpp"
#include <cstring>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <utility>

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
void MetalL4maModel<T>::forward(MetalL4maBuffer<T>& buffer,
                               MetalL4maKVCache<T>& kv_cache, T* final_norm_output) {
    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;

    // Choose hidden states backing: use persistent workspace if available, otherwise stack-alloc
    T* hidden_ptr = nullptr;
    bool hidden_owns = false; // whether we must deallocate
    MetalTensor<T> hidden_states_owned; // only valid if hidden_owns=true
    if (auto* ws0 = buffer.getLayerWorkspace(0)) {
        // Use residual buffer as the persistent hidden state across layers
        hidden_ptr = ws0->residual_buffer.data();
    } else {
        hidden_states_owned = buffer.template allocate<T>(num_tokens * hidden_size);
        hidden_ptr = hidden_states_owned.data();
        hidden_owns = true;
    }

    // 1. Token embedding lookup using metal_embedding
    auto& context = MetalContext::getInstance();
    metal_embedding_lookup_bfloat16(
        context.getDevice(), context.getCommandQueue(),
        reinterpret_cast<const bfloat16_t*>(embed_tokens_weight_.data()),
        config_.vocab_size,
        reinterpret_cast<const int32_t*>(buffer.input_ids.data()),
        num_tokens,
        reinterpret_cast<bfloat16_t*>(hidden_ptr),
        config_.hidden_size
    );
    MetalProfiler::getInstance().record("token_embedding");

    // 2. Forward through all decoder layers
    MetalProfiler::getInstance().recordStart("attention");
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        // Get KV cache for this layer
        auto [kv_cache_k, kv_cache_v] = kv_cache.get_layer_pointers(layer_idx);
        layers_[layer_idx].forward(buffer, hidden_ptr, kv_cache_k, kv_cache_v, kv_cache.get_num_pages());
    }
    MetalProfiler::getInstance().recordEnd("attention");

    // 3. Final layer normalization
    norm_.forward(final_norm_output, hidden_ptr, num_tokens, buffer.commandBuffer);
    MetalProfiler::getInstance().record("final_norm");

    // Deallocate if we owned the temporary hidden buffer
    if (hidden_owns) {
        buffer.deallocate(hidden_states_owned);
        std::cout << "  ✅ Deallocated intermediate hidden states (stack)\n";
    } else {
        std::cout << "  ♻️  Reused persistent workspace for hidden states\n";
    }
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
    MetalL4maBuffer<T>& buffer, MetalL4maKVCache<T>& kv_cache) {

    size_t num_tokens = buffer.num_tokens;
    size_t hidden_size = config_.hidden_size;
    size_t vocab_size = config_.vocab_size;

    // Allocate final norm output
    auto final_norm_output = buffer.template allocate<T>(num_tokens * hidden_size);

    // 1. Forward through the model
    MetalProfiler::getInstance().recordStart("model");
    model_.forward(buffer, kv_cache, final_norm_output.data());
    MetalProfiler::getInstance().recordEnd("model");

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
    MetalProfiler::getInstance().record("lm_head");

    // DEBUG: Check input values to lm_head GEMM and the resulting logits
    if (num_tokens > 0) {
        // Check final_norm_output values (input to lm_head)
        std::cout << "🔍 [DEBUG] final_norm_output (input to lm_head, first 10):" << std::endl;
        float max_norm_val = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(hidden_size)); ++i) {
            float val = static_cast<float>(final_norm_output.data()[i]);
            std::cout << "  norm_out[" << i << "] = " << std::fixed << std::setprecision(6) << val << std::endl;
            max_norm_val = std::max(max_norm_val, std::abs(val));
        }
        std::cout << "  Max final_norm magnitude: " << max_norm_val << std::endl;

        // Check embedding weight values (used in lm_head)
        auto& embed_weights = model_.get_embed_tokens_weight();
        std::cout << "🔍 [DEBUG] embedding_weights (used in lm_head, first 10):" << std::endl;
        float max_weight_val = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(hidden_size)); ++i) {
            float val = static_cast<float>(embed_weights.data()[i]);
            std::cout << "  embed_weight[" << i << "] = " << std::fixed << std::setprecision(6) << val << std::endl;
            max_weight_val = std::max(max_weight_val, std::abs(val));
        }
        std::cout << "  Max embedding weight magnitude: " << max_weight_val << std::endl;

        // Check resulting logits
        bfloat16_t* logits_bf16 = reinterpret_cast<bfloat16_t*>(logits.data());
        std::cout << "🔍 [DEBUG] Raw logits after lm_head (first 10 values):" << std::endl;
        float max_logit = 0.0f;
        for (int i = 0; i < std::min(10, static_cast<int>(vocab_size)); ++i) {
            float val = static_cast<float>(logits_bf16[i]);
            std::cout << "  logits[" << i << "] = " << std::fixed << std::setprecision(6) << val << std::endl;
            max_logit = std::max(max_logit, std::abs(val));
        }
        std::cout << "  Max logit magnitude: " << max_logit << std::endl;
    }

    // 3. Apply top-k mask and softmax to get final probabilities
    constexpr int top_k = 50; // Default top-k value
    auto topk_values = buffer.template allocate<float>(num_tokens * top_k);
    auto topk_indices = buffer.template allocate<int32_t>(num_tokens * top_k);

    // Apply top-k masking to logits
    std::cout << "🔍 [PIPELINE] Applying topk_mask (k=" << top_k << ")" << std::endl;


    int result = metal_topk_mask_logits_bfloat16(
        logits.data(), // in-place masking
        num_tokens,
        vocab_size,
        top_k
    );
    if (result != 0) {
        throw std::runtime_error("Failed to apply top-k masking");
    }
    MetalProfiler::getInstance().record("topk_mask");

    // DEBUG: Check ALL logits after topk_mask using proper conversion
    std::cout << "  🔍 AFTER topk_mask kernel:" << std::endl;
    auto debug_after = buffer.template allocate<float>(vocab_size);  // Check ALL values
    MetalCast::bfloat16_to_float(context.getDevice(), context.getCommandQueue(),
                                reinterpret_cast<const bfloat16_t*>(logits.data()),
                                debug_after.data(), vocab_size);
    int finite_after = 0;
    float max_after = -1e9f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (debug_after.data()[i] > -65000.0f) {  // Not masked
            finite_after++;
            if (finite_after <= 3) {
                std::cout << "    Preserved[" << i << "] = " << debug_after.data()[i] << std::endl;
            }
            max_after = std::max(max_after, debug_after.data()[i]);
        }
    }
    std::cout << "    After: " << finite_after << "/" << top_k << " unmasked values, max=" << max_after << std::endl;

    // MEMORY DEBUG: Capture pointer and first few bytes after successful topk_mask
    void* logits_ptr_after_topk = logits.data();
    bfloat16_t* logits_bf16_after_topk = reinterpret_cast<bfloat16_t*>(logits.data());
    std::vector<uint16_t> first_bytes_after_topk;
    for (int i = 0; i < 10; ++i) {
        first_bytes_after_topk.push_back(static_cast<uint16_t>(logits_bf16_after_topk[i]));
    }
    std::cout << "  🔍 MEMORY: logits.data() = " << logits_ptr_after_topk << std::endl;
    std::cout << "  🔍 MEMORY: first 10 raw bytes = ";
    for (int i = 0; i < 10; ++i) {
        std::cout << "0x" << std::hex << first_bytes_after_topk[i] << std::dec;
        if (i < 9) std::cout << " ";
    }
    std::cout << std::endl;

    // Check the FULL logits.data() buffer for unmasked values
    int full_unmasked_count = 0;
    std::cout << "  🔍 MEMORY: Checking FULL logits.data() buffer for unmasked values..." << std::endl;
    for (size_t i = 0; i < vocab_size; ++i) {
        uint16_t raw_bf16 = static_cast<uint16_t>(logits_bf16_after_topk[i]);
        if (raw_bf16 != 0xc780) {
            full_unmasked_count++;
            if (full_unmasked_count <= 5) {
                float single_val;
                MetalCast::bfloat16_to_float(context.getDevice(), context.getCommandQueue(),
                                           &logits_bf16_after_topk[i], &single_val, 1);
                std::cout << "    logits.data()[" << i << "] = 0x" << std::hex << raw_bf16
                          << std::dec << " = " << single_val << std::endl;
            }
        }
    }
    std::cout << "  🔍 MEMORY: Found " << full_unmasked_count << " unmasked values in FULL logits.data()" << std::endl;

    // Verify topk_mask worked correctly
    if (num_tokens > 0) {
        bfloat16_t* logits_bf16 = reinterpret_cast<bfloat16_t*>(logits.data());
        int unmasked_count = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            uint16_t raw_bf16 = static_cast<uint16_t>(logits_bf16[i]);
            if (raw_bf16 != 0xc780) {
                unmasked_count++;
            }
        }
        std::cout << "  ✅ topk_mask: " << unmasked_count << "/" << top_k << " values preserved" << std::endl;
    }

    // Step 1: Extract top-k logits and indices (before softmax)
    std::cout << "🔍 [PIPELINE] Step 1: Extracting top-k masked logits" << std::endl;

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
    MetalProfiler::getInstance().record("extract_k_values");

    // Verify we extracted the expected logits
    if (num_tokens > 0) {
        int extracted_count = 0;
        float max_logit = -1e9f;
        for (int i = 0; i < top_k; ++i) {
            float val = topk_values.data()[i];
            if (val != 0.0f && val > -65000.0f) {  // Valid extracted logit
                extracted_count++;
                max_logit = std::max(max_logit, val);
            }
        }
        std::cout << "  ✅ Extracted " << extracted_count << "/" << top_k << " logits, max=" << max_logit << std::endl;
    }

    // Step 2: Apply softmax only to the extracted top-k logits
    std::cout << "🔍 [PIPELINE] Step 2: Applying softmax to top-k logits" << std::endl;

    int softmax_result = metal_softmax_float(
        topk_values.data(),
        topk_values.data(), // in-place softmax on k values only
        num_tokens,
        top_k, // Process only k values, not full vocab
        1.0f   // temperature
    );
    if (softmax_result != 0) {
        throw std::runtime_error("Failed to apply softmax to top-k values");
    }
    MetalProfiler::getInstance().record("softmax");

    // Verify softmax produced valid probabilities
    if (num_tokens > 0) {
        float prob_sum = 0.0f;
        float max_prob = 0.0f;
        int non_zero_probs = 0;

        for (int i = 0; i < top_k; ++i) {
            float val = topk_values.data()[i];
            prob_sum += val;
            if (val > 1e-10f) {
                non_zero_probs++;
                max_prob = std::max(max_prob, val);
            }
        }

        std::cout << "  ✅ Softmax: sum=" << std::fixed << std::setprecision(8) << prob_sum
                  << ", non_zero=" << non_zero_probs << ", max=" << max_prob << std::endl;
    }


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

    // Forward declarations
    std::string map_parameter_name_to_ztensor(const std::string& metal_name);

    template<typename T>
    void initialize_parameter_randomly(MetalTensor<T>& tensor, const std::string& param_name);

    template<typename T>
    bool verify_tensor_shape_compatibility(const MetalTensor<T>& metal_tensor,
                                          const ztensor::TensorInfo& ztensor_info,
                                          const std::string& metal_name,
                                          const std::string& ztensor_name);

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

        // Load parameters from zTensor checkpoint file
        std::string weight_file = metadata.checkpoint_path;

        if (config.verbose) {
            std::cout << "Loading weights from: " << weight_file << std::endl;
        }

        // Get all parameter tensors
        auto param_map = model->get_parameters();

        // Load weights from zTensor file
        try {
            ztensor::zTensorReader reader(weight_file);
            auto tensor_list = reader.list_tensors();

            if (config.verbose) {
                std::cout << "Found " << tensor_list.size() << " tensors in model file" << std::endl;
            }

            // Track loaded tensors for verification
            int loaded_count = 0;
            int total_count = param_map.size();

            for (auto& [param_name, tensor] : param_map) {
                // Map Metal parameter names to zTensor names
                std::string ztensor_name = map_parameter_name_to_ztensor(param_name);

                // Check if tensor exists in the zTensor file
                auto it = std::find(tensor_list.begin(), tensor_list.end(), ztensor_name);
                if (it == tensor_list.end()) {
                    if (config.verbose) {
                        std::cout << "Warning: Parameter '" << param_name
                                 << "' (mapped to '" << ztensor_name << "') not found in model file. "
                                 << "Using random initialization." << std::endl;
                    }

                    // Fall back to random initialization for missing parameters
                    initialize_parameter_randomly(*tensor, param_name);
                    continue;
                }

                // Load tensor info and data
                auto info = reader.get_tensor_info(ztensor_name);
                const void* raw_data = reader.get_raw_tensor_pointer(ztensor_name);

                // Verify shape compatibility
                if (!verify_tensor_shape_compatibility(*tensor, info, param_name, ztensor_name)) {
                    std::cerr << "Shape mismatch for parameter " << param_name << std::endl;
                    return nullptr;
                }

                // Calculate total elements
                size_t total_elements = 1;
                for (auto dim : info.shape) {
                    total_elements *= dim;
                }

                // Load data using copyFromMappedMemory
                tensor->copyFromMappedMemory(raw_data, total_elements);

                // Calculate and log weight statistics for verification
                if (config.verbose && total_elements > 0) {
                    // Cast to appropriate type based on tensor data type
                    if (info.dtype == "bfloat16") {
                        const uint16_t* bf16_data = static_cast<const uint16_t*>(raw_data);
                        float sum = 0.0f, min_val = FLT_MAX, max_val = -FLT_MAX;

                        for (size_t i = 0; i < total_elements; ++i) {
                            // Convert bfloat16 to float32 for statistics
                            uint32_t f32_bits = static_cast<uint32_t>(bf16_data[i]) << 16;
                            float value = *reinterpret_cast<const float*>(&f32_bits);
                            sum += value;
                            min_val = std::min(min_val, value);
                            max_val = std::max(max_val, value);
                        }
                        float mean_val = sum / total_elements;

                        std::cout << "  Weight stats: min=" << min_val << ", max=" << max_val
                                 << ", mean=" << mean_val << std::endl;
                    } else if (info.dtype == "float32") {
                        const float* f32_data = static_cast<const float*>(raw_data);
                        float sum = 0.0f, min_val = FLT_MAX, max_val = -FLT_MAX;

                        for (size_t i = 0; i < total_elements; ++i) {
                            sum += f32_data[i];
                            min_val = std::min(min_val, f32_data[i]);
                            max_val = std::max(max_val, f32_data[i]);
                        }
                        float mean_val = sum / total_elements;

                        std::cout << "  Weight stats: min=" << min_val << ", max=" << max_val
                                 << ", mean=" << mean_val << std::endl;
                    }
                }

                loaded_count++;
                if (config.verbose) {
                    std::cout << "Loaded parameter: " << param_name << " <- " << ztensor_name
                             << " (" << total_elements << " elements)" << std::endl;
                }
            }

            std::cout << "Successfully loaded " << loaded_count << "/" << total_count
                     << " parameters from " << weight_file << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error loading zTensor file: " << e.what() << std::endl;
            std::cerr << "Falling back to random initialization" << std::endl;

            // Fall back to random initialization for all parameters
            for (auto& [name, tensor] : param_map) {
                initialize_parameter_randomly(*tensor, name);
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

    /**
     * @brief Map Metal parameter names to zTensor names
     *
     * Converts parameter names from the Metal model format (e.g., "layers.0.self_attn.q_proj.weight")
     * to the format used in zTensor files (e.g., "model.layers.0.self_attn.q_proj.weight")
     */
    std::string map_parameter_name_to_ztensor(const std::string& metal_name) {
        // Add "model." prefix for most parameters
        if (metal_name.find("embed_tokens") == 0) {
            return "model." + metal_name;
        }

        if (metal_name.find("layers.") == 0) {
            return "model." + metal_name;
        }

        if (metal_name.find("norm.") == 0) {
            return "model." + metal_name;
        }

        // For other parameters, just add the prefix
        return "model." + metal_name;
    }

    /**
     * @brief Initialize parameter with random values (fallback)
     */
    template<typename T>
    void initialize_parameter_randomly(MetalTensor<T>& tensor, const std::string& param_name) {
        size_t num_elements = 1;
        for (size_t dim : tensor.shape()) {
            num_elements *= dim;
        }

        std::vector<T> random_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            // Simple random initialization
            random_data[i] = static_cast<T>((rand() / float(RAND_MAX) - 0.5f) * 0.1f);
        }

        tensor.copyFromHost(random_data.data());
    }

    /**
     * @brief Verify tensor shape compatibility between Metal and zTensor
     */
    template<typename T>
    bool verify_tensor_shape_compatibility(const MetalTensor<T>& metal_tensor,
                                          const ztensor::TensorInfo& ztensor_info,
                                          const std::string& metal_name,
                                          const std::string& ztensor_name) {
        const auto& metal_shape = metal_tensor.shape();
        const auto& ztensor_shape = ztensor_info.shape;

        if (metal_shape.size() != ztensor_shape.size()) {
            std::cerr << "Dimension mismatch for " << metal_name << " <- " << ztensor_name
                     << ": Metal has " << metal_shape.size() << " dims, zTensor has "
                     << ztensor_shape.size() << " dims" << std::endl;
            return false;
        }

        for (size_t i = 0; i < metal_shape.size(); ++i) {
            if (metal_shape[i] != static_cast<size_t>(ztensor_shape[i])) {
                std::cerr << "Shape mismatch for " << metal_name << " <- " << ztensor_name
                         << " at dimension " << i << ": Metal has " << metal_shape[i]
                         << ", zTensor has " << ztensor_shape[i] << std::endl;
                return false;
            }
        }

        return true;
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

    template void initialize_parameter_randomly(MetalTensor<float>& tensor, const std::string& param_name);
    template void initialize_parameter_randomly(MetalTensor<bfloat16_t>& tensor, const std::string& param_name);
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