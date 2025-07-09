#include "model.hpp"

// All implementation-specific headers are safely included here
#include "l4ma.cuh"
#include "ztensor.hpp"

#include <iostream>
#include <set>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// --- Internal data structures ---

// Represents a block in the KV cache on the CPU.
struct Block {
    std::vector<uint32_t> position_ids;
    std::vector<bool> occupancy;

    Block() = default; // Default constructor
    Block(size_t kv_page_size)
        : position_ids(kv_page_size, 0), occupancy(kv_page_size, false) {}
};

// Represents a text embedding on the CPU.
struct TextEmbed {
    uint32_t token_id;
    uint32_t position_id;
};


// The actual implementation of the Server is hidden in this struct.
struct Model::ModelImpl {
    std::unique_ptr<L4maForCausalLM<__nv_bfloat16>> model;

    // --- State Management ---
    std::map<uint32_t, Block> blocks;
    std::map<uint32_t, TextEmbed> embeds;

    // Storage for results of embedding/inference, analogous to Python's embed_storage
    thrust::device_vector<__nv_bfloat16> embed_storage_p1;
    thrust::device_vector<int32_t> embed_storage_p2;

    // Configuration
    size_t kv_page_size;
    int dist_size;


    // --- Handler method declarations added to ModelImpl ---
    // These methods contain the core logic and have access to the model pointer.
    void handle_allocate(const std::vector<Model::AllocateCommand>& commands);
    void handle_deallocate(const std::vector<Model::DeallocateCommand>& commands);
    void handle_embed_text(const std::vector<Model::EmbedTextCommand>& commands);
    void handle_fill_block(const std::vector<Model::FillBlockCommand>& commands);
    void handle_mask_block(const std::vector<Model::MaskBlockCommand>& commands);
    void handle_copy_block(const std::vector<Model::CopyBlockCommand>& commands);
    void handle_decode_token_distribution(const std::vector<Model::DecodeTokenDistributionCommand>& commands);
    std::vector<Model::SampleTopKResult> handle_sample_top_k(const std::vector<Model::SampleTopKCommand>& commands);
};

namespace { 

template<typename T>
std::unique_ptr<L4maForCausalLM<T>> load_model_internal(const AppConfig& config, const ModelMetadata& metadata) {
    std::cout << "Instantiating model structure on device..." << std::endl;

    auto model_ptr = std::make_unique<L4maForCausalLM<T>>(metadata.architecture);

    auto params_map = model_ptr->get_parameters();
    std::cout << "Found " << params_map.size() << " parameter tensors in the model structure." << std::endl;

    const auto model_dir = config.cache_dir / config.model_name;
    std::set<std::string> loaded_keys;

    for (const auto& param_file : metadata.parameters) {
        std::filesystem::path weights_path = model_dir / param_file;
        std::cout << "Reading weights from: " << weights_path.string() << std::endl;

        try {
            ztensor::zTensorReader reader(weights_path.string());
            for (const auto& name : reader.list_tensors()) {
                if (params_map.count(name) && !loaded_keys.count(name)) {
                    const auto& info = reader.get_tensor_info(name);
                    thrust::device_vector<T>* target_tensor = params_map[name];

                    if (target_tensor->size() != info.num_elements()) {
                        std::cerr << "    Warning: Shape mismatch for tensor '" << name << "'. ZT: " << info.num_elements() << ", Model: " << target_tensor->size() << ". Skipping." << std::endl;
                        continue;
                    }

                    const T* host_ptr = static_cast<const T*>(reader.get_raw_tensor_pointer(name));
                    if (host_ptr) {
                        cudaMemcpy(thrust::raw_pointer_cast(target_tensor->data()), host_ptr, info.size, cudaMemcpyHostToDevice);
                        loaded_keys.insert(name);
                    }
                }
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "Warning: Could not read file " << weights_path.string() << ". Error: " << e.what() << std::endl;
        }
    }

    if (params_map.count("lm_head.weight") && params_map.count("model.embed_tokens.weight")) {
        *params_map["lm_head.weight"] = *params_map["model.embed_tokens.weight"];
        loaded_keys.insert("lm_head.weight");
    }
    
    if (loaded_keys.size() != params_map.size()) {
        std::cout << "\nWarning: Mismatch between loaded and expected parameter counts." << std::endl;
        std::cout << "Missing parameters:" << std::endl;
        for (const auto& pair : params_map) {
            if (loaded_keys.find(pair.first) == loaded_keys.end()) {
                std::cout << "  - " << pair.first << std::endl;
            }
        }
    }
    
    std::cout << "\nSuccessfully loaded " << loaded_keys.size() << " expected weights." << std::endl;

    return model_ptr;
}

} // anonymous namespace

// --- New: Placeholder implementations for handler methods ---
// These are the actual implementations within the ModelImpl struct.

void Model::ModelImpl::handle_allocate(const std::vector<Model::AllocateCommand>& commands) {
    std::cout << "  [ModelImpl] handle_allocate called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        if (cmd.kind == Model::ObjectKind::KV_BLOCK) {
            for (uint32_t i = 0; i < cmd.count; ++i) {
                uint32_t block_id = cmd.object_id_offset + i;
                blocks[block_id] = Block(kv_page_size);
            }
        }
    }
}

void Model::ModelImpl::handle_deallocate(const std::vector<Model::DeallocateCommand>& commands) {
    std::cout << "  [ModelImpl] handle_deallocate called with " << commands.size() << " items." << std::endl;
    // Currently a no-op, as in the Python implementation.
    // Blocks are cleared implicitly when the model is destroyed.
}

void Model::ModelImpl::handle_embed_text(const std::vector<Model::EmbedTextCommand>& commands) {
    std::cout << "  [ModelImpl] handle_embed_text called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        embeds[cmd.embedding_id] = {cmd.token_id, cmd.position_id};
    }
}

void Model::ModelImpl::handle_fill_block(const std::vector<Model::FillBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_fill_block called with " << commands.size() << " items." << std::endl;

    // --- Data Preparation ---
    // std::vector<int32_t> kv_page_indices_host;
    // std::vector<int32_t> kv_page_indptr_host = {0};
    // std::vector<int32_t> kv_last_page_lens_host;
    // std::vector<int32_t> qo_indptr_host = {0};
    // std::vector<bool> custom_masks_host;

    // std::vector<int32_t> new_token_ids_host;
    // std::vector<int32_t> new_position_ids_host;

    // struct OutputEmbedPostproc {
    //     size_t idx;
    //     uint32_t vec_id;
    // };
    // std::vector<OutputEmbedPostproc> output_embed_postproc;

    // for (const auto& cmd : commands) {
    //     kv_page_indices_host.insert(kv_page_indices_host.end(), cmd.context_block_ids.begin(), cmd.context_block_ids.end());
    //     kv_page_indptr_host.push_back(kv_page_indices_host.size());
    //     kv_last_page_lens_host.push_back(cmd.last_block_len);
        
    //     qo_indptr_host.push_back(qo_indptr_host.back() + cmd.input_embedding_ids.size());

    //     size_t total_ctx_tokens = kv_page_size * (cmd.context_block_ids.size() - 1) + cmd.last_block_len;

    //     std::vector<int32_t> inp_pos_ids;
    //     std::vector<bool> inp_occupancy;

    //     for (uint32_t embed_id : cmd.input_embedding_ids) {
    //         auto it = embeds.find(embed_id);
    //         if (it != embeds.end()) {
    //             const auto& embed = it->second;
    //             new_token_ids_host.push_back(embed.token_id);
    //             new_position_ids_host.push_back(embed.position_id);
    //             inp_pos_ids.push_back(embed.position_id);
    //             inp_occupancy.push_back(true);

    //             size_t token_offset = total_ctx_tokens - cmd.input_embedding_ids.size() + (inp_pos_ids.size() - 1);
    //             uint32_t tgt_block_idx = token_offset / kv_page_size;
    //             uint32_t tgt_block_offset = token_offset % kv_page_size;
    //             uint32_t tgt_block_id = cmd.context_block_ids[tgt_block_idx];
                
    //             blocks[tgt_block_id].occupancy[tgt_block_offset] = true;
    //             blocks[tgt_block_id].position_ids[tgt_block_offset] = embed.position_id;
    //         }
    //     }

    //     for (size_t i = 0; i < cmd.output_embedding_ids.size(); ++i) {
    //         output_embed_postproc.push_back({new_token_ids_host.size() - cmd.output_embedding_ids.size() + i, cmd.output_embedding_ids[i]});
    //     }

    //     std::vector<int32_t> ctx_pos_ids;
    //     std::vector<bool> ctx_occupancy;
    //     for(uint32_t block_id : cmd.context_block_ids) {
    //         const auto& block = blocks[block_id];
    //         ctx_pos_ids.insert(ctx_pos_ids.end(), block.position_ids.begin(), block.position_ids.end());
    //         ctx_occupancy.insert(ctx_occupancy.end(), block.occupancy.begin(), block.occupancy.end());
    //     }
    //     ctx_pos_ids.resize(total_ctx_tokens);
    //     ctx_occupancy.resize(total_ctx_tokens);

    //     for (size_t i = 0; i < inp_pos_ids.size(); ++i) {
    //         for (size_t j = 0; j < total_ctx_tokens; ++j) {
    //             bool casual_mask = ctx_pos_ids[j] <= inp_pos_ids[i];
    //             bool valid_mask = ctx_occupancy[j] && inp_occupancy[i];
    //             custom_masks_host.push_back(casual_mask && valid_mask);
    //         }
    //     }
    // }

    // // --- Copy data to device ---
    // thrust::device_vector<int32_t> kv_page_indices = kv_page_indices_host;
    // thrust::device_vector<int32_t> kv_page_indptr = kv_page_indptr_host;
    // thrust::device_vector<int32_t> kv_last_page_lens = kv_last_page_lens_host;
    // thrust::device_vector<int32_t> qo_indptr = qo_indptr_host;
    // thrust::device_vector<bool> custom_mask = custom_masks_host;
    // thrust::device_vector<int32_t> new_token_ids = new_token_ids_host;
    // thrust::device_vector<int32_t> new_position_ids = new_position_ids_host;

    // // --- Model Forward Pass ---
    // thrust::device_vector<__nv_bfloat16> hidden_states = model->forward(
    //     new_token_ids,
    //     new_position_ids,
    //     kv_page_indices,
    //     kv_page_indptr,
    //     kv_last_page_lens,
    //     qo_indptr,
    //     custom_mask,
    //     nullptr // Use internal stream
    // );

    // // --- Post-processing ---
    // thrust::device_vector<__nv_bfloat16> logits(hidden_states.size() / model->get_config().hidden_size * model->get_config().vocab_size);
    // model->lm_head(logits, hidden_states);

    // // TODO: Top-k selection and storing results in embed_storage
    // // This part is complex and requires a custom kernel or sophisticated use of Thrust.
    // // For now, we just acknowledge the logits are computed.
    
    // cudaDeviceSynchronize(); // Wait for all computations to finish
}

void Model::ModelImpl::handle_mask_block(const std::vector<Model::MaskBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_mask_block called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        auto it = blocks.find(cmd.block_id);
        if (it != blocks.end()) {
            Block& block = it->second;
            if (block.occupancy.size() == cmd.mask.size()) {
                block.occupancy = cmd.mask;
            } else {
                std::cerr << "Warning: Mask size mismatch for block " << cmd.block_id << std::endl;
            }
        } else {
            std::cerr << "Warning: Block not found for masking: " << cmd.block_id << std::endl;
        }
    }
}

void Model::ModelImpl::handle_copy_block(const std::vector<Model::CopyBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_copy_block called with " << commands.size() << " items." << std::endl;
    // TODO: Implement cudaMemcpy between different KV cache pages on the device.
    // This requires getting raw pointers to the device vectors for each layer in the KV cache.
}

void Model::ModelImpl::handle_decode_token_distribution(const std::vector<Model::DecodeTokenDistributionCommand>& commands) {
    std::cout << "  [ModelImpl] handle_decode_token_distribution called with " << commands.size() << " items." << std::endl;
    // This is a no-op in the provided python implementation.
    // The logic is integrated into fill_block where top-k results are computed and stored directly.
}

std::vector<Model::SampleTopKResult> Model::ModelImpl::handle_sample_top_k(const std::vector<Model::SampleTopKCommand>& commands) {
    std::cout << "  [ModelImpl] handle_sample_top_k called with " << commands.size() << " items." << std::endl;
    std::vector<Model::SampleTopKResult> results;
    results.reserve(commands.size());

    for (const auto& cmd : commands) {
        Model::SampleTopKResult res;
        
        // Determine the number of elements to copy
        uint32_t k = (cmd.k > 0 && cmd.k < static_cast<uint32_t>(dist_size)) ? cmd.k : dist_size;

        // Create host vectors to hold the results
        thrust::host_vector<__nv_bfloat16> topk_probs_host(k);
        thrust::host_vector<int32_t> topk_tokens_host(k);

        // Copy data from device to host
        cudaMemcpy(topk_probs_host.data(), thrust::raw_pointer_cast(embed_storage_p1.data()) + cmd.distribution_id * dist_size, k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(topk_tokens_host.data(), thrust::raw_pointer_cast(embed_storage_p2.data()) + cmd.distribution_id * dist_size, k * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        res.token_ids.assign(topk_tokens_host.begin(), topk_tokens_host.end());
        
        res.probabilities.resize(k);
        for(size_t i = 0; i < k; ++i) {
            res.probabilities[i] = static_cast<float>(topk_probs_host[i]);
        }

        results.push_back(res);
    }
    return results;
}

// --- Public Interface Implementation ---

Model::Model(const AppConfig& config,const ModelMetadata& out_metadata)
    : pimpl(std::make_unique<ModelImpl>()) {
    
    std::cout << "Starting service..." << std::endl;
    // Load the model and store it in the implementation object
    pimpl->model = load_model_internal<__nv_bfloat16>(config, out_metadata);
    std::cout << "Model loaded successfully and is resident on the GPU." << std::endl;

    // initialize kv cache
    pimpl->model->create_kv_device_vectors(config.max_num_kv_pages);

    // Initialize state
    //pimpl->kv_page_size = pimpl->model->get_kv_page_size();
    pimpl->dist_size = config.dist_size;
    pimpl->embed_storage_p1.resize(config.max_num_embeds * config.dist_size);
    pimpl->embed_storage_p2.resize(config.max_num_embeds * config.dist_size);
}

Model::~Model() = default;

void Model::run() {
    // This would contain the primary execution loop if the model ran continuously.
    // For a request/response server, it can remain empty.
}

// --- New: Public handler methods delegating to PIMPL ---
// These methods are the public API of your Model class. They simply
// forward the calls to the actual implementation in ModelImpl.

void Model::handle_allocate(const std::vector<AllocateCommand>& commands) {
    pimpl->handle_allocate(commands);
}

void Model::handle_deallocate(const std::vector<DeallocateCommand>& commands) {
    pimpl->handle_deallocate(commands);
}

void Model::handle_embed_text(const std::vector<EmbedTextCommand>& commands) {
    pimpl->handle_embed_text(commands);
}

void Model::handle_fill_block(const std::vector<FillBlockCommand>& commands) {
    pimpl->handle_fill_block(commands);
}

void Model::handle_mask_block(const std::vector<MaskBlockCommand>& commands) {
    pimpl->handle_mask_block(commands);
}

void Model::handle_copy_block(const std::vector<CopyBlockCommand>& commands) {
    pimpl->handle_copy_block(commands);
}

void Model::handle_decode_token_distribution(const std::vector<DecodeTokenDistributionCommand>& commands) {
    pimpl->handle_decode_token_distribution(commands);
}

std::vector<Model::SampleTopKResult> Model::handle_sample_top_k(const std::vector<SampleTopKCommand>& commands) {
    return pimpl->handle_sample_top_k(commands);
}