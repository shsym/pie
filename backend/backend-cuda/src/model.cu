#include "model.hpp"

// All implementation-specific headers are safely included here
#include "l4ma.cuh"
#include "ztensor.hpp"
#include "common.cuh"
#include "stack_allocator.cuh"
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
    Block(int32_t kv_page_size)
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
    int32_t kv_page_size;
    int32_t dist_size;

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
                    auto& target_tensor_ptr = params_map[name];

                    if (target_tensor_ptr->size() != info.num_elements()) {
                        std::cerr << "    Warning: Shape mismatch for tensor '" << name << "'. ZT: " << info.num_elements() << ", Model: " << target_tensor_ptr->size() << ". Skipping." << std::endl;
                        continue;
                    }

                    const T* host_ptr = static_cast<const T*>(reader.get_raw_tensor_pointer(name));
                    if (host_ptr) {
                        cudaMemcpy(thrust::raw_pointer_cast(target_tensor_ptr->data()), host_ptr, info.size, cudaMemcpyHostToDevice);
                        loaded_keys.insert(name);
                    }
                }
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "Warning: Could not read file " << weights_path.string() << ". Error: " << e.what() << std::endl;
        }
    }

    if (params_map.count("lm_head.weight") && params_map.count("model.embed_tokens.weight")) {
        params_map["lm_head.weight"] = params_map["model.embed_tokens.weight"];
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
// Assuming other necessary includes and the Model::ModelImpl class definition exist above

void Model::ModelImpl::handle_fill_block(const std::vector<Model::FillBlockCommand>& commands) {

    std::cout << "  [ModelImpl] handle_fill_block called with " << commands.size() << " items." << std::endl;

    // --- Host-side vector preparations ---
    std::vector<int32_t> kv_page_indices_host;
    std::vector<int32_t> kv_page_indptr_host = {0};
    std::vector<int32_t> kv_last_page_lens_host;
    std::vector<int32_t> qo_indptr_host = {0};
    std::vector<bool> custom_masks_host;
    std::vector<int32_t> mask_indptr_host = {0};
    std::vector<int32_t> kv_batch_indices_host;
    std::vector<int32_t> kv_positions_host;
    std::vector<uint32_t> new_token_ids_host;
    std::vector<int32_t> new_position_ids_host;

    struct OutputEmbedPostproc {
        size_t logit_row_idx;
        uint32_t dest_embed_id;
    };
    std::vector<OutputEmbedPostproc> output_embed_postproc;

    int batch_idx = 0;
    for (const auto& cmd : commands) {
        kv_page_indices_host.insert(kv_page_indices_host.end(), cmd.context_block_ids.begin(), cmd.context_block_ids.end());
        kv_page_indptr_host.push_back(kv_page_indices_host.size());
        kv_last_page_lens_host.push_back(cmd.last_block_len);

        int32_t num_new_tokens = cmd.input_embedding_ids.size();
        qo_indptr_host.push_back(qo_indptr_host.back() + num_new_tokens);

        size_t total_ctx_tokens = (cmd.context_block_ids.empty()) ? 0 :
                                  kv_page_size * (cmd.context_block_ids.size() - 1) + cmd.last_block_len;

        mask_indptr_host.push_back(mask_indptr_host.back() + (num_new_tokens * total_ctx_tokens));

        for (int32_t i = 0; i < num_new_tokens; ++i) {
            kv_batch_indices_host.push_back(batch_idx);
            kv_positions_host.push_back(total_ctx_tokens - cmd.last_block_len + i);
        }

        std::vector<uint32_t> inp_pos_ids_for_mask;

        for (size_t i = 0; i < cmd.input_embedding_ids.size(); ++i) {
            uint32_t embed_id = cmd.input_embedding_ids[i];
            auto it = embeds.find(embed_id);
            if (it != embeds.end()) {
                const auto& embed = it->second;
                new_token_ids_host.push_back(embed.token_id);
                new_position_ids_host.push_back(embed.position_id);
                inp_pos_ids_for_mask.push_back(embed.position_id);

                size_t token_abs_pos = total_ctx_tokens - num_new_tokens + i;
                uint32_t tgt_block_idx = token_abs_pos / kv_page_size;
                uint32_t tgt_block_offset = token_abs_pos % kv_page_size;

                // print tgt_block_idx and tgt_block_offset for debugging
                std::cout << "Processing token: " << embed.token_id 
                          << ", position: " << embed.position_id 
                          << ", token_abs_pos: " << token_abs_pos
                          << ", target block index: " << tgt_block_idx 
                          << ", target block offset: " << tgt_block_offset << std::endl;

                if (tgt_block_idx < cmd.context_block_ids.size()) {
                    uint32_t tgt_block_id = cmd.context_block_ids[tgt_block_idx];
                    auto block_it = blocks.find(tgt_block_id);
                    if (block_it != blocks.end()) {
                        block_it->second.occupancy[tgt_block_offset] = true;
                        block_it->second.position_ids[tgt_block_offset] = embed.position_id;
                    }
                }
            }
        }

        for (size_t i = 0; i < cmd.output_embedding_ids.size(); ++i) {
            size_t logit_row = new_token_ids_host.size() - cmd.output_embedding_ids.size() + i;
            output_embed_postproc.push_back({logit_row, cmd.output_embedding_ids[i]});
        }

        if (total_ctx_tokens > 0) {
            std::vector<uint32_t> ctx_pos_ids;
            std::vector<bool> ctx_occupancy;
            ctx_pos_ids.reserve(total_ctx_tokens);
            ctx_occupancy.reserve(total_ctx_tokens);

            for (size_t i = 0; i < cmd.context_block_ids.size(); ++i) {
                uint32_t block_id = cmd.context_block_ids[i];
                const auto& block = blocks.at(block_id);
                size_t len_to_copy = (i == cmd.context_block_ids.size() - 1) ? cmd.last_block_len : kv_page_size;
                ctx_pos_ids.insert(ctx_pos_ids.end(), block.position_ids.begin(), block.position_ids.begin() + len_to_copy);
                ctx_occupancy.insert(ctx_occupancy.end(), block.occupancy.begin(), block.occupancy.begin() + len_to_copy);
            }

            // print all ctx_pos_ids and ctx_occupancy for debugging
            std::cout << "ctx_pos_ids: ";
            for (const auto& pos_id : ctx_pos_ids) {
                std::cout << pos_id << " ";
            }
            std::cout << "\nctx_occupancy: ";
            for (const auto& occ : ctx_occupancy) {
                std::cout << (occ ? 1 : 0) << " ";
            }
            std::cout << std::endl;


            for (uint32_t inp_pos_id : inp_pos_ids_for_mask) {
                for (size_t j = 0; j < total_ctx_tokens; ++j) {
                    bool causal_mask = ctx_pos_ids[j] <= inp_pos_id;
                    bool valid_mask = ctx_occupancy[j];
                    custom_masks_host.push_back(causal_mask && valid_mask);
                }
            }
        }
        batch_idx++;
    }

    // print all host vectors for debugging
    std::cout << "kv_page_indices_host: ";
    for (const auto& idx : kv_page_indices_host) {
        std::cout << idx << " ";
    }
    std::cout << "\nkv_page_indptr_host: ";
    for (const auto& idx : kv_page_indptr_host) {
        std::cout << idx << " ";
    }
    std::cout << "\nkv_last_page_lens_host: ";
    for (const auto& len : kv_last_page_lens_host) {
        std::cout << len << " ";
    }
    std::cout << "\nqo_indptr_host: ";
    for (const auto& idx : qo_indptr_host) {
        std::cout << idx << " ";
    }
    std::cout << "\ncustom_masks_host: ";
    for (const auto& mask : custom_masks_host) {
        std::cout << static_cast<int>(mask) << " ";
    }
    std::cout << "\nmask_indptr_host: ";
    for (const auto& idx : mask_indptr_host) {
        std::cout << idx << " ";
    }
    std::cout << "\nnew_token_ids_host: ";
    for (const auto& token_id : new_token_ids_host) {
        std::cout << token_id << " ";
    }
    std::cout << "\nnew_position_ids_host: ";
    for (const auto& pos_id : new_position_ids_host) {
        std::cout << pos_id << " ";
    }
    std::cout << "\nkv_batch_indices_host: ";
    for (const auto& batch_idx : kv_batch_indices_host) {
        std::cout << batch_idx << " ";
    }
    std::cout << "\nkv_positions_host: ";
    for (const auto& pos : kv_positions_host) {
        std::cout << pos << " ";
    }
    std::cout << std::endl;

    std::vector<uint8_t> packed_mask_host = packbits_little(custom_masks_host);

    // --- Copy data to device ---
    thrust::device_vector<int32_t> kv_page_indices = kv_page_indices_host;
    thrust::device_vector<int32_t> kv_page_indptr = kv_page_indptr_host;
    thrust::device_vector<int32_t> kv_last_page_lens = kv_last_page_lens_host;
    thrust::device_vector<int32_t> qo_indptr = qo_indptr_host;
    thrust::device_vector<uint8_t> custom_mask = packed_mask_host;
    thrust::device_vector<int32_t> mask_indptr = mask_indptr_host;
    thrust::device_vector<uint32_t> new_token_ids = new_token_ids_host;
    thrust::device_vector<int32_t> new_position_ids = new_position_ids_host;
    thrust::device_vector<int32_t> kv_batch_indices = kv_batch_indices_host;
    thrust::device_vector<int32_t> kv_positions = kv_positions_host;

    // --- Allocate buffers ---
    size_t num_total_new_tokens = new_token_ids.size();
    if (num_total_new_tokens == 0) return;

    thrust::device_vector<__nv_bfloat16> logits(num_total_new_tokens * model->get_config().vocab_size);
    
    size_t workspace_size_bytes = model->get_workspace_size(num_total_new_tokens);

    StackAllocator allocator(workspace_size_bytes);

    cudaStream_t stream = 0;

    // --- Model Forward Pass ---
    model->forward(
        allocator,
        thrust::raw_pointer_cast(logits.data()),
        new_token_ids,
        new_position_ids,
        kv_page_indices,
        kv_page_indptr,
        kv_page_indptr_host,
        kv_last_page_lens,
        qo_indptr,
        qo_indptr_host,
        custom_mask,
        mask_indptr,
        stream,
        kv_page_size,
        kv_batch_indices,
        kv_positions
    );

    // --- Post-processing ---
    if (!output_embed_postproc.empty()) {
        std::vector<size_t> logit_indices_host;
        std::vector<uint32_t> dest_embed_ids_host;
        logit_indices_host.reserve(output_embed_postproc.size());
        dest_embed_ids_host.reserve(output_embed_postproc.size());
        for (const auto& p : output_embed_postproc) {
            logit_indices_host.push_back(p.logit_row_idx);
            dest_embed_ids_host.push_back(p.dest_embed_id);
        }
        thrust::device_vector<size_t> logit_indices_dev = logit_indices_host;
        thrust::device_vector<uint32_t> dest_embed_ids_dev = dest_embed_ids_host;

        topk_scatter(
            logits,
            logit_indices_dev,
            dest_embed_ids_dev,
            model->get_config().vocab_size,
            dist_size,
            embed_storage_p1,
            embed_storage_p2,
            stream
        );
    }
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
    pimpl->kv_page_size = config.kv_page_size;
    pimpl->dist_size = config.dist_size;
    pimpl->embed_storage_p1.resize(config.max_num_embeds * config.dist_size);
    pimpl->embed_storage_p2.resize(config.max_num_embeds * config.dist_size);
}

Model::~Model() = default;

void Model::run() {
    // This function is now used as a test routine for handle_fill_block.

    std::cout << "\n--- [START] Running Test Routine for handle_fill_block ---" << std::endl;

    // 1. Define test parameters: a random sequence of tokens and IDs.
    const std::vector<uint32_t> token_ids = {3513, 5331, 533, 11};
    const uint32_t block_id = 101; // A unique ID for our KV block
    const uint32_t embed_id_offset = 201; // Starting ID for our input embeddings
    const uint32_t dist_id = 301;         // ID for the output distribution object

    // Ensure the tokens fit within a single page.
    if (token_ids.size() > static_cast<size_t>(pimpl->kv_page_size)) {
        std::cerr << "Test Error: Number of tokens exceeds kv_page_size." << std::endl;
        return;
    }

    // 2. Call handle_allocate to allocate a page for the KV cache.
    std::cout << "\n[Step 1] Allocating KV Block..." << std::endl;
    Model::AllocateCommand alloc_cmd;
    alloc_cmd.kind = Model::ObjectKind::KV_BLOCK;
    alloc_cmd.object_id_offset = block_id;
    alloc_cmd.count = 1;
    handle_allocate({alloc_cmd});
    std::cout << "Allocated block with ID: " << block_id << std::endl;

    // 3. Call handle_embed_texts to create mappings for token and position IDs.
    std::cout << "\n[Step 2] Creating Text Embeddings..." << std::endl;
    std::vector<Model::EmbedTextCommand> embed_cmds;
    std::vector<uint32_t> input_embed_ids;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint32_t current_embed_id = embed_id_offset + i;
        input_embed_ids.push_back(current_embed_id);

        Model::EmbedTextCommand embed_cmd;
        embed_cmd.embedding_id = current_embed_id;
        embed_cmd.token_id = token_ids[i];
        embed_cmd.position_id = i; // Simple sequential positions 0, 1, 2, ...
        embed_cmds.push_back(embed_cmd);
    }
    handle_embed_text(embed_cmds);
    std::cout << "Created " << embed_cmds.size() << " embeddings." << std::endl;

    // 4. Call handle_fill_block to do a single forward pass.
    std::cout << "\n[Step 3] Calling handle_fill_block for a forward pass..." << std::endl;
    Model::FillBlockCommand fill_cmd;
    fill_cmd.last_block_len = token_ids.size(); // No previous context in the block
    fill_cmd.context_block_ids = {block_id}; // The block to fill with new KV data
    fill_cmd.input_embedding_ids = input_embed_ids;
    fill_cmd.output_embedding_ids = {dist_id}; // Store logits for the last token in this distribution
    handle_fill_block({fill_cmd});
    std::cout << "handle_fill_block completed." << std::endl;

    // 5. Verify the output by sampling the resulting distribution.
    std::cout << "\n[Step 4] Verifying output with handle_sample_top_k..." << std::endl;
    Model::SampleTopKCommand sample_cmd;
    sample_cmd.distribution_id = dist_id;
    sample_cmd.k = 5; // Get top 5 predictions
    auto results = handle_sample_top_k({sample_cmd});

    if (!results.empty()) {
        const auto& result = results[0];
        std::cout << "Successfully retrieved Top-" << result.token_ids.size() << " predicted next tokens:" << std::endl;
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            std::cout << "  - Token ID: " << result.token_ids[i]
                      << ", Probability: " << result.probabilities[i] << std::endl;
        }
    } else {
        std::cerr << "Test Error: Failed to get sampling results." << std::endl;
    }

    std::cout << "\n--- [END] Test Routine Finished ---\n" << std::endl;
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