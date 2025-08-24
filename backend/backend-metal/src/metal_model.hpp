#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include "metal_buffer.hpp"
#include "metal_kv_cache.hpp"
#include "metal_l4ma.hpp"
#include <memory>
#include <vector>
#include <map>
#include <cstdint>
#include <string>

// Forward declarations - these should match the CUDA backend exactly
struct AppConfig;
struct ModelMetadata;

/**
 * @brief Metal equivalent of the CUDA Model class
 * 
 * This class provides the exact same interface as the CUDA Model class from model.hpp,
 * ensuring seamless backend compatibility. All method signatures and semantics match
 * the CUDA implementation.
 */
class MetalModel {
public:
    // --- Public nested types matching CUDA Model exactly ---
    
    // Corresponds to l4m.ObjectKind enum
    enum class ObjectKind {
        UNSPECIFIED = 0,
        KV_BLOCK = 1,
        EMB = 2,
        DIST = 3,
    };

    // Corresponds to l4m.Allocate
    struct AllocateCommand {
        ObjectKind kind;
        uint32_t object_id_offset;
        uint32_t count;
    };

    // Corresponds to l4m.Deallocate
    struct DeallocateCommand {
        ObjectKind kind;
        uint32_t object_id_offset;
        uint32_t count;
    };

    // Corresponds to l4m.EmbedText
    struct EmbedTextCommand {
        uint32_t embedding_id;
        uint32_t token_id;
        uint32_t position_id;
    };

    // Corresponds to l4m.FillBlock
    struct FillBlockCommand {
        uint32_t last_block_len;
        std::vector<uint32_t> context_block_ids;
        std::vector<uint32_t> input_embedding_ids;
        std::vector<uint32_t> output_embedding_ids;
    };

    // Corresponds to l4m.MaskBlock
    struct MaskBlockCommand {
        uint32_t block_id;
        std::vector<bool> mask;
    };

    // Corresponds to l4m.CopyBlock
    struct CopyBlockCommand {
        uint32_t source_block_id;
        uint32_t destination_block_id;
        uint32_t source_start;
        uint32_t destination_start;
        uint32_t length;
    };

    // Corresponds to l4m.DecodeTokenDistribution
    struct DecodeTokenDistributionCommand {
        uint32_t embedding_id;
        uint32_t distribution_id;
    };

    // Corresponds to l4m.SampleTopKRequest
    struct SampleTopKCommand {
        uint32_t distribution_id;
        uint32_t k;
    };

    // Corresponds to l4m.SampleTopKResponse
    struct SampleTopKResult {
        std::vector<uint32_t> token_ids;
        std::vector<float> probabilities;
    };

    // Corresponds to l4m.ForwardText
    struct ForwardTextCommand {
        uint32_t kv_page_last_len;
        std::vector<uint32_t> kv_page_ids;
        std::vector<uint32_t> token_ids;
        std::vector<uint32_t> position_ids;
        std::vector<std::vector<uint32_t>> brle_masks; // raw BRLE buffers per token
        std::vector<uint32_t> output_indices; // indices within token_ids to produce distributions for
    };

    struct Distribution {
        std::vector<uint32_t> token_ids;
        std::vector<float> probabilities;
    };

    // ForwardText handler: returns a vector of items, each item containing a vector of distributions
    std::vector<std::vector<Distribution>> handle_forward_text(const std::vector<ForwardTextCommand>& commands);

    // --- Core Class Methods ---

    MetalModel(const AppConfig& config, const ModelMetadata& metadata);
    ~MetalModel();
    void run();

    // --- L4M Handler Methods (exactly matching CUDA interface) ---

    void handle_allocate(const std::vector<AllocateCommand>& commands);
    void handle_deallocate(const std::vector<DeallocateCommand>& commands);
    void handle_embed_text(const std::vector<EmbedTextCommand>& commands);
    void handle_fill_block(const std::vector<FillBlockCommand>& commands);
    void handle_mask_block(const std::vector<MaskBlockCommand>& commands);
    void handle_copy_block(const std::vector<CopyBlockCommand>& commands);
    void handle_decode_token_distribution(const std::vector<DecodeTokenDistributionCommand>& commands);
    std::vector<SampleTopKResult> handle_sample_top_k(const std::vector<SampleTopKCommand>& commands);

private:
    struct MetalModelImpl;
    std::unique_ptr<MetalModelImpl> pimpl;
};

/**
 * @brief Internal implementation class for MetalModel
 * 
 * Hidden implementation using PIMPL idiom to match CUDA backend structure.
 * Contains the actual Metal compute resources and model state.
 */
struct MetalModel::MetalModelImpl {
    // Model components (using bfloat16 to match CUDA backend)
    std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> model;
    std::unique_ptr<MetalL4maBuffer<bfloat16_t>> buffer;
    std::unique_ptr<MetalL4maKVCache<bfloat16_t>> kv_cache;
    
    // Forward declarations for nested structs
    struct Block;
    struct TextEmbed;
    struct Dist;
    
    // State management (matching CUDA backend exactly)
    std::map<uint32_t, Block> blocks;
    std::map<uint32_t, TextEmbed> embeds;
    std::map<uint32_t, Dist> dists;
    
    // Configuration
    int32_t kv_page_size;
    int32_t dist_size;
    
    // Metal command buffer for operations
    id<MTLCommandBuffer> commandBuffer;
    
    // Handler method implementations
    void handle_allocate(const std::vector<MetalModel::AllocateCommand>& commands);
    void handle_deallocate(const std::vector<MetalModel::DeallocateCommand>& commands);
    void handle_embed_text(const std::vector<MetalModel::EmbedTextCommand>& commands);
    void handle_fill_block(const std::vector<MetalModel::FillBlockCommand>& commands);
    void handle_mask_block(const std::vector<MetalModel::MaskBlockCommand>& commands);
    void handle_copy_block(const std::vector<MetalModel::CopyBlockCommand>& commands);
    void handle_decode_token_distribution(const std::vector<MetalModel::DecodeTokenDistributionCommand>& commands);
    std::vector<MetalModel::SampleTopKResult> handle_sample_top_k(const std::vector<MetalModel::SampleTopKCommand>& commands);
};

/**
 * @brief Metal model loading utilities
 * 
 * Functions to load and initialize Metal models from checkpoints,
 * matching the CUDA backend's model loading interface.
 */
namespace MetalModelLoader {
    
    /**
     * @brief Load Metal model from checkpoint files
     * Matches load_model_internal from CUDA backend
     */
    template<typename T>
    std::unique_ptr<MetalL4maForCausalLM<T>> load_model_internal(
        const AppConfig& config, 
        const ModelMetadata& metadata
    );
    
    /**
     * @brief Initialize Metal compute environment for model inference
     * Must be called before creating any MetalModel instances
     */
    bool initialize_metal_backend();
    
    /**
     * @brief Cleanup Metal compute environment
     * Should be called when shutting down the application
     */
    void cleanup_metal_backend();
    
    /**
     * @brief Validate model metadata for Metal backend compatibility
     */
    bool validate_model_metadata(const ModelMetadata& metadata);
}

/**
 * @brief Factory function to create MetalModel instances
 * 
 * This provides a consistent interface for creating models regardless of backend
 */
namespace MetalModelFactory {
    
    /**
     * @brief Create a MetalModel instance with the given configuration
     * Handles all initialization and resource allocation
     */
    std::unique_ptr<MetalModel> createModel(const AppConfig& config, const ModelMetadata& metadata);
    
    /**
     * @brief Check if Metal backend is available on this system
     */
    bool isMetalAvailable();
    
    /**
     * @brief Get Metal device information
     */
    struct MetalDeviceInfo {
        std::string name;
        size_t max_buffer_length;
        size_t max_threadgroup_memory_length;
        bool supports_bfloat16;
        bool supports_function_pointers;
    };
    
    MetalDeviceInfo getDeviceInfo();
}

/**
 * @brief Performance monitoring and debugging utilities
 */
namespace MetalModelProfiler {
    
    /**
     * @brief Global profiler for Metal model operations
     */
    extern MetalProfiler globalProfiler;
    
    /**
     * @brief Enable/disable global profiling
     */
    void enableProfiling(bool enabled);
    
    /**
     * @brief Print global profiling report
     */
    void printProfilingReport();
    
    /**
     * @brief Reset profiling counters
     */
    void resetProfiling();
}

/**
 * @brief Error handling and diagnostics
 */
namespace MetalModelDiagnostics {
    
    /**
     * @brief Validate Metal model state
     */
    bool validateModelState(const MetalModel& model);
    
    /**
     * @brief Get detailed error information
     */
    struct ErrorInfo {
        std::string error_code;
        std::string description;
        std::string suggestion;
        std::vector<std::string> diagnostic_data;
    };
    
    ErrorInfo getLastError();
    
    /**
     * @brief Enable verbose error reporting
     */
    void enableVerboseErrors(bool enabled);
}