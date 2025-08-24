#include "metal_model.hpp"
#include "metal_l4ma.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// Internal data structures matching CUDA backend
struct MetalModel::MetalModelImpl::Block {
    std::vector<uint32_t> position_ids;
    std::vector<bool> occupancy;

    Block() = default;
    Block(int32_t kv_page_size)
        : position_ids(kv_page_size, 0), occupancy(kv_page_size, false) {}
};

struct MetalModel::MetalModelImpl::TextEmbed {
    uint32_t token_id;
    uint32_t position_id;
};

struct MetalModel::MetalModelImpl::Dist {
    std::vector<float> probabilities;
    std::vector<int32_t> token_ids;
};

// Configuration structures (should match CUDA backend exactly)
struct AppConfig {
    std::string model_path;
    std::string cache_dir;
    bool verbose = false;
    int32_t kv_page_size = 16;
    int32_t dist_size = 64;
    size_t max_num_kv_pages = 14000;
    size_t max_num_embeds = 50000;
};

struct ModelMetadata {
    std::string model_name;
    std::string checkpoint_path;
    L4maConfig config;
    size_t total_params;
};

// Implementation of MetalModel::MetalModelImpl
void MetalModel::MetalModelImpl::handle_allocate(const std::vector<MetalModel::AllocateCommand>& commands) {
    for (const auto& cmd : commands) {
        switch (cmd.kind) {
            case MetalModel::ObjectKind::KV_BLOCK:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t block_id = cmd.object_id_offset + i;
                    blocks[block_id] = Block(kv_page_size);
                }
                break;
                
            case MetalModel::ObjectKind::EMB:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t embed_id = cmd.object_id_offset + i;
                    embeds[embed_id] = TextEmbed{};
                }
                break;
                
            case MetalModel::ObjectKind::DIST:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t dist_id = cmd.object_id_offset + i;
                    Dist new_dist;
                    new_dist.probabilities.resize(dist_size);
                    new_dist.token_ids.resize(dist_size);
                    dists[dist_id] = std::move(new_dist);
                }
                break;
                
            default:
                std::cerr << "Unknown allocation kind: " << static_cast<int>(cmd.kind) << std::endl;
                break;
        }
    }
}

void MetalModel::MetalModelImpl::handle_deallocate(const std::vector<MetalModel::DeallocateCommand>& commands) {
    for (const auto& cmd : commands) {
        switch (cmd.kind) {
            case MetalModel::ObjectKind::KV_BLOCK:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t block_id = cmd.object_id_offset + i;
                    blocks.erase(block_id);
                }
                break;
                
            case MetalModel::ObjectKind::EMB:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t embed_id = cmd.object_id_offset + i;
                    embeds.erase(embed_id);
                }
                break;
                
            case MetalModel::ObjectKind::DIST:
                for (uint32_t i = 0; i < cmd.count; ++i) {
                    uint32_t dist_id = cmd.object_id_offset + i;
                    dists.erase(dist_id);
                }
                break;
                
            default:
                std::cerr << "Unknown deallocation kind: " << static_cast<int>(cmd.kind) << std::endl;
                break;
        }
    }
}

void MetalModel::MetalModelImpl::handle_embed_text(const std::vector<MetalModel::EmbedTextCommand>& commands) {
    for (const auto& cmd : commands) {
        auto it = embeds.find(cmd.embedding_id);
        if (it != embeds.end()) {
            it->second.token_id = cmd.token_id;
            it->second.position_id = cmd.position_id;
        } else {
            std::cerr << "Embedding ID not found: " << cmd.embedding_id << std::endl;
        }
    }
}

void MetalModel::MetalModelImpl::handle_fill_block(const std::vector<MetalModel::FillBlockCommand>& commands) {
    for (const auto& cmd : commands) {
        // Setup buffer for input processing
        std::vector<int32_t> input_ids_temp, position_ids_temp;
        
        // Process input embeddings to fill KV cache blocks
        for (size_t i = 0; i < cmd.input_embedding_ids.size(); ++i) {
            uint32_t embed_id = cmd.input_embedding_ids[i];
            auto embed_it = embeds.find(embed_id);
            
            if (embed_it != embeds.end()) {
                // Add token and position to temporary vectors
                input_ids_temp.push_back(embed_it->second.token_id);
                position_ids_temp.push_back(embed_it->second.position_id);
            }
        }
        
        // Copy to buffer tensors
        buffer->input_ids.copyFromHost(input_ids_temp.data());
        buffer->position_ids.copyFromHost(position_ids_temp.data());
        
        // Update block page information
        for (uint32_t block_id : cmd.context_block_ids) {
            auto block_it = blocks.find(block_id);
            if (block_it != blocks.end()) {
                // Mark positions as occupied up to last_block_len
                for (uint32_t pos = 0; pos < cmd.last_block_len && pos < kv_page_size; ++pos) {
                    block_it->second.occupancy[pos] = true;
                }
            }
        }
        
        buffer->num_tokens = cmd.input_embedding_ids.size();
        
        // Copy page indices to buffer
        buffer->kv_page_indices.copyFromHost(reinterpret_cast<const int32_t*>(cmd.context_block_ids.data()));
        
        // Set up page pointers for KV cache
        std::vector<uint32_t> page_indptr;
        page_indptr.push_back(0);
        page_indptr.push_back(static_cast<uint32_t>(cmd.context_block_ids.size()));
        buffer->kv_page_indptr.copyFromHost(reinterpret_cast<const int32_t*>(page_indptr.data()));
        
        // Set up query output pointers
        std::vector<uint32_t> qo_indptr;
        qo_indptr.push_back(0);
        qo_indptr.push_back(static_cast<uint32_t>(cmd.output_embedding_ids.size()));
        buffer->qo_indptr.copyFromHost(reinterpret_cast<const int32_t*>(qo_indptr.data()));
        
        // Set up last page lengths
        std::vector<uint32_t> last_page_lens;
        last_page_lens.push_back(cmd.last_block_len);
        buffer->kv_last_page_lens.copyFromHost(reinterpret_cast<const int32_t*>(last_page_lens.data()));
    }
}

void MetalModel::MetalModelImpl::handle_mask_block(const std::vector<MetalModel::MaskBlockCommand>& commands) {
    for (const auto& cmd : commands) {
        auto block_it = blocks.find(cmd.block_id);
        if (block_it != blocks.end()) {
            // Apply mask to block occupancy
            for (size_t i = 0; i < cmd.mask.size() && i < block_it->second.occupancy.size(); ++i) {
                if (!cmd.mask[i]) {
                    block_it->second.occupancy[i] = false;
                }
            }
        }
    }
}

void MetalModel::MetalModelImpl::handle_copy_block(const std::vector<MetalModel::CopyBlockCommand>& commands) {
    for (const auto& cmd : commands) {
        auto src_it = blocks.find(cmd.source_block_id);
        auto dst_it = blocks.find(cmd.destination_block_id);
        
        if (src_it != blocks.end() && dst_it != blocks.end()) {
            // Copy KV cache data from source to destination
            // This would involve Metal buffer operations in a full implementation
            
            // For now, just copy the metadata
            size_t copy_len = std::min(cmd.length, static_cast<uint32_t>(kv_page_size));
            size_t src_start = std::min(cmd.source_start, static_cast<uint32_t>(kv_page_size));
            size_t dst_start = std::min(cmd.destination_start, static_cast<uint32_t>(kv_page_size));
            
            for (size_t i = 0; i < copy_len; ++i) {
                if (src_start + i < kv_page_size && dst_start + i < kv_page_size) {
                    dst_it->second.position_ids[dst_start + i] = src_it->second.position_ids[src_start + i];
                    dst_it->second.occupancy[dst_start + i] = src_it->second.occupancy[src_start + i];
                }
            }
        }
    }
}

void MetalModel::MetalModelImpl::handle_decode_token_distribution(const std::vector<MetalModel::DecodeTokenDistributionCommand>& commands) {
    if (commands.empty()) return;
    
    // Setup buffer for inference
    buffer->num_tokens = commands.size();
    
    // Collect embedding data
    std::vector<int32_t> input_ids_temp, position_ids_temp;
    for (const auto& cmd : commands) {
        auto embed_it = embeds.find(cmd.embedding_id);
        if (embed_it != embeds.end()) {
            input_ids_temp.push_back(embed_it->second.token_id);
            position_ids_temp.push_back(embed_it->second.position_id);
        }
    }
    
    // Copy to buffer tensors
    buffer->input_ids.copyFromHost(input_ids_temp.data());
    buffer->position_ids.copyFromHost(position_ids_temp.data());
    
    // Run model inference
    MetalProfiler profiler_instance;
    auto& context = MetalContext::getInstance();
    id<MTLCommandBuffer> commandBuffer = [context.getCommandQueue() commandBuffer];
    auto profiler = profiler_instance.scope("decode_token_distribution", commandBuffer);
    auto [values, indices] = model->forward(profiler, *buffer, *kv_cache);
    
    // Store results in distribution objects
    for (size_t i = 0; i < commands.size(); ++i) {
        uint32_t dist_id = commands[i].distribution_id;
        auto dist_it = dists.find(dist_id);
        
        if (dist_it != dists.end()) {
            // Extract top-k results for this token
            size_t offset = i * dist_size;
            if (offset + dist_size <= values.size()) {
                std::copy(values.begin() + offset, values.begin() + offset + dist_size,
                         dist_it->second.probabilities.begin());
                std::copy(indices.begin() + offset, indices.begin() + offset + dist_size,
                         dist_it->second.token_ids.begin());
            }
        }
    }
}

std::vector<MetalModel::SampleTopKResult> MetalModel::MetalModelImpl::handle_sample_top_k(const std::vector<MetalModel::SampleTopKCommand>& commands) {
    std::vector<MetalModel::SampleTopKResult> results;
    results.reserve(commands.size());
    
    for (const auto& cmd : commands) {
        MetalModel::SampleTopKResult result;
        
        auto dist_it = dists.find(cmd.distribution_id);
        if (dist_it != dists.end()) {
            const auto& dist = dist_it->second;
            
            // Take top-k elements (they should already be sorted from model inference)
            uint32_t k = std::min(cmd.k, static_cast<uint32_t>(dist.token_ids.size()));
            
            result.token_ids.reserve(k);
            result.probabilities.reserve(k);
            
            for (uint32_t i = 0; i < k; ++i) {
                result.token_ids.push_back(dist.token_ids[i]);
                result.probabilities.push_back(dist.probabilities[i]);
            }
        }
        
        results.push_back(std::move(result));
    }
    
    return results;
}

// Implementation of MetalModel public interface
MetalModel::MetalModel(const AppConfig& config, const ModelMetadata& metadata) 
    : pimpl(std::make_unique<MetalModelImpl>()) {
    
    std::cout << "Initializing Metal Model..." << std::endl;
    
    // Initialize Metal context
    auto& metal_context = MetalContext::getInstance();
    if (!metal_context.getDevice()) {
        throw std::runtime_error("Metal device not available");
    }
    
    // Create command buffer
    pimpl->commandBuffer = metal_context.getCommandQueue().commandBuffer;
    
    // Store configuration
    pimpl->kv_page_size = config.kv_page_size;
    pimpl->dist_size = config.dist_size;
    
    // Load model
    pimpl->model = MetalModelUtils::load_model_internal<bfloat16_t>(config, metadata);
    if (!pimpl->model) {
        throw std::runtime_error("Failed to load Metal model");
    }
    
    // Initialize buffer and KV cache
    pimpl->buffer = std::make_unique<MetalL4maBuffer<bfloat16_t>>(
        metadata.config,        // L4maConfig
        config.kv_page_size,    // page_size
        config.dist_size,       // dist_size
        config.max_num_embeds * metadata.config.hidden_size * sizeof(bfloat16_t) // workspace_size
    );
    
    pimpl->kv_cache = std::make_unique<MetalL4maKVCache<bfloat16_t>>(
        metadata.config,        // L4maConfig
        config.max_num_kv_pages, // num_kv_pages
        config.kv_page_size     // page_size
    );
    
    std::cout << "Metal Model initialized successfully" << std::endl;
    std::cout << "  Model: " << metadata.model_name << std::endl;
    std::cout << "  Parameters: " << metadata.total_params << std::endl;
    std::cout << "  KV Page Size: " << config.kv_page_size << std::endl;
    std::cout << "  Max KV Pages: " << config.max_num_kv_pages << std::endl;
}

MetalModel::~MetalModel() = default;

void MetalModel::run() {
    // Main event loop - this would typically be called by the backend server
    std::cout << "Metal Model ready for inference requests" << std::endl;
}

// Handler method implementations (delegate to pimpl)
void MetalModel::handle_allocate(const std::vector<AllocateCommand>& commands) {
    pimpl->handle_allocate(commands);
}

void MetalModel::handle_deallocate(const std::vector<DeallocateCommand>& commands) {
    pimpl->handle_deallocate(commands);
}

void MetalModel::handle_embed_text(const std::vector<EmbedTextCommand>& commands) {
    pimpl->handle_embed_text(commands);
}

void MetalModel::handle_fill_block(const std::vector<FillBlockCommand>& commands) {
    pimpl->handle_fill_block(commands);
}

void MetalModel::handle_mask_block(const std::vector<MaskBlockCommand>& commands) {
    pimpl->handle_mask_block(commands);
}

void MetalModel::handle_copy_block(const std::vector<CopyBlockCommand>& commands) {
    pimpl->handle_copy_block(commands);
}

void MetalModel::handle_decode_token_distribution(const std::vector<DecodeTokenDistributionCommand>& commands) {
    pimpl->handle_decode_token_distribution(commands);
}

std::vector<MetalModel::SampleTopKResult> MetalModel::handle_sample_top_k(const std::vector<SampleTopKCommand>& commands) {
    return pimpl->handle_sample_top_k(commands);
}

std::vector<std::vector<MetalModel::Distribution>> MetalModel::handle_forward_text(const std::vector<ForwardTextCommand>& commands) {
    std::vector<std::vector<Distribution>> results;
    results.reserve(commands.size());
    
    for (const auto& cmd : commands) {
        std::vector<Distribution> item_distributions;
        
        // Setup buffer for this forward pass
        pimpl->buffer->num_tokens = cmd.token_ids.size();
        pimpl->buffer->input_ids.copyFromHost(reinterpret_cast<const int32_t*>(cmd.token_ids.data()));
        pimpl->buffer->position_ids.copyFromHost(reinterpret_cast<const int32_t*>(cmd.position_ids.data()));
        pimpl->buffer->kv_page_indices.copyFromHost(reinterpret_cast<const int32_t*>(cmd.kv_page_ids.data()));
        
        // Set up page pointers
        std::vector<uint32_t> page_indptr = {0, static_cast<uint32_t>(cmd.kv_page_ids.size())};
        pimpl->buffer->kv_page_indptr.copyFromHost(reinterpret_cast<const int32_t*>(page_indptr.data()));
        
        // Set up last page lengths
        std::vector<uint32_t> last_page_lens = {cmd.kv_page_last_len};
        pimpl->buffer->kv_last_page_lens.copyFromHost(reinterpret_cast<const int32_t*>(last_page_lens.data()));
        
        // Set up query output pointers based on output_indices
        std::vector<uint32_t> qo_indptr;
        qo_indptr.push_back(0);
        for (size_t i = 0; i < cmd.output_indices.size(); ++i) {
            qo_indptr.push_back(qo_indptr.back() + 1);
        }
        pimpl->buffer->qo_indptr.copyFromHost(reinterpret_cast<const int32_t*>(qo_indptr.data()));
        
        // Run inference
        MetalProfiler profiler_instance;
        auto& context = MetalContext::getInstance();
        id<MTLCommandBuffer> commandBuffer = [context.getCommandQueue() commandBuffer];
        auto profiler = profiler_instance.scope("forward_text", commandBuffer);
        auto [values, indices] = pimpl->model->forward(profiler, *pimpl->buffer, *pimpl->kv_cache);
        
        // Convert results to Distribution objects
        size_t top_k = 50; // Default top-k
        for (size_t out_idx = 0; out_idx < cmd.output_indices.size(); ++out_idx) {
            Distribution dist;
            
            // Extract top-k values for this output position
            size_t offset = out_idx * top_k;
            if (offset + top_k <= values.size()) {
                dist.probabilities.assign(values.begin() + offset, values.begin() + offset + top_k);
                
                // Convert int32_t indices to uint32_t token_ids
                for (size_t i = 0; i < top_k && offset + i < indices.size(); ++i) {
                    dist.token_ids.push_back(static_cast<uint32_t>(indices[offset + i]));
                }
            }
            
            item_distributions.push_back(std::move(dist));
        }
        
        results.push_back(std::move(item_distributions));
    }
    
    return results;
}

// Factory and utility implementations
namespace MetalModelLoader {
    
    template<typename T>
    std::unique_ptr<MetalL4maForCausalLM<T>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata) {
        return MetalModelUtils::load_model_internal<T>(config, metadata);
    }
    
    bool initialize_metal_backend() {
        try {
            auto& context = MetalContext::getInstance();
            return context.getDevice() != nil;
        } catch (...) {
            return false;
        }
    }
    
    void cleanup_metal_backend() {
        // Metal resources are automatically cleaned up
    }
    
    bool validate_model_metadata(const ModelMetadata& metadata) {
        return MetalModelUtils::validate_model_config(metadata.config);
    }
    
    // Explicit template instantiations
    template std::unique_ptr<MetalL4maForCausalLM<float>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata);
    template std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> load_model_internal(
        const AppConfig& config, const ModelMetadata& metadata);
}

namespace MetalModelFactory {
    
    std::unique_ptr<MetalModel> createModel(const AppConfig& config, const ModelMetadata& metadata) {
        if (!MetalModelLoader::initialize_metal_backend()) {
            throw std::runtime_error("Failed to initialize Metal backend");
        }
        
        if (!MetalModelLoader::validate_model_metadata(metadata)) {
            throw std::runtime_error("Invalid model metadata for Metal backend");
        }
        
        return std::make_unique<MetalModel>(config, metadata);
    }
    
    bool isMetalAvailable() {
        return MetalModelLoader::initialize_metal_backend();
    }
    
    MetalDeviceInfo getDeviceInfo() {
        MetalDeviceInfo info{};
        
        auto& context = MetalContext::getInstance();
        id<MTLDevice> device = context.getDevice();
        
        if (device) {
            info.name = [device.name UTF8String];
            info.max_buffer_length = device.maxBufferLength;
            info.max_threadgroup_memory_length = device.maxThreadgroupMemoryLength;
            info.supports_bfloat16 = true; // Assume modern Metal devices support bfloat16
            info.supports_function_pointers = [device supportsFamily:MTLGPUFamilyMac2];
        }
        
        return info;
    }
}

// Global profiler instance
namespace MetalModelProfiler {
    MetalProfiler globalProfiler;
    
    void enableProfiling(bool enabled) {
        globalProfiler.setEnabled(enabled);
    }
    
    void printProfilingReport() {
        globalProfiler.print_report();
    }
    
    void resetProfiling() {
        // Reset not implemented in MetalProfiler
        // Would need to clear timings_ vector
    }
}

namespace MetalModelDiagnostics {
    static ErrorInfo lastError;
    static bool verboseErrors = false;
    
    bool validateModelState(const MetalModel& model) {
        // Basic validation - could be expanded
        return true;
    }
    
    ErrorInfo getLastError() {
        return lastError;
    }
    
    void enableVerboseErrors(bool enabled) {
        verboseErrors = enabled;
    }
}