#include "model.hpp"

// All implementation-specific headers are safely included here
#include "gpt.cuh"
#include "ztensor.hpp"

#include <iostream>
#include <set>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>


// The actual implementation of the Server is hidden in this struct.
struct Model::ModelImpl {
    std::unique_ptr<L4maForCausalLM<__nv_bfloat16>> model;

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
    // TODO: Implement logic to manage KV cache pages or other GPU resources.
    // This might involve interacting with a memory manager associated with `pimpl->model`.
}

void Model::ModelImpl::handle_deallocate(const std::vector<Model::DeallocateCommand>& commands) {
    std::cout << "  [ModelImpl] handle_deallocate called with " << commands.size() << " items." << std::endl;
    // TODO: Implement logic to free GPU resources.
}

void Model::ModelImpl::handle_embed_text(const std::vector<Model::EmbedTextCommand>& commands) {
    std::cout << "  [ModelImpl] handle_embed_text called with " << commands.size() << " items." << std::endl;
    // TODO: Prepare input tensors from commands and call the embedding layer of the model.
    // e.g., pimpl->model->embed_tokens(input_tensor);
}

void Model::ModelImpl::handle_fill_block(const std::vector<Model::FillBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_fill_block called with " << commands.size() << " items." << std::endl;
    // TODO: This is the main inference call.
    // 1. Prepare input tensors (position_ids, attention_mask, etc.) from the commands.
    // 2. Call the model's forward pass: pimpl->model->forward(prepared_inputs);
    // 3. Store the resulting output embeddings and update KV cache.
}

void Model::ModelImpl::handle_mask_block(const std::vector<Model::MaskBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_mask_block called with " << commands.size() << " items." << std::endl;
    // TODO: Apply attention masks to the KV cache pages.
}

void Model::ModelImpl::handle_copy_block(const std::vector<Model::CopyBlockCommand>& commands) {
    std::cout << "  [ModelImpl] handle_copy_block called with " << commands.size() << " items." << std::endl;
    // TODO: Perform cudaMemcpy between different KV cache pages on the device.
}

void Model::ModelImpl::handle_decode_token_distribution(const std::vector<Model::DecodeTokenDistributionCommand>& commands) {
    std::cout << "  [ModelImpl] handle_decode_token_distribution called with " << commands.size() << " items." << std::endl;
    // TODO: Take final hidden states (embeddings) and pass them through the language model head.
    // e.g., pimpl->model->lm_head(hidden_states_tensor);
    // Then store the resulting logits for sampling.
}

std::vector<Model::SampleTopKResult> Model::ModelImpl::handle_sample_top_k(const std::vector<Model::SampleTopKCommand>& commands) {
    std::cout << "  [ModelImpl] handle_sample_top_k called with " << commands.size() << " items." << std::endl;
    std::vector<Model::SampleTopKResult> results;
    results.reserve(commands.size());
    // TODO: Implement actual top-k sampling on the GPU from the distributions
    // generated by handle_decode_token_distribution.
    for (const auto& cmd : commands) {
        Model::SampleTopKResult res;
        res.token_ids = { 50256, 13, 220 }; // Dummy data: "<|endoftext|>", " a"
        res.probabilities = { 0.85f, 0.1f, 0.05f };
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