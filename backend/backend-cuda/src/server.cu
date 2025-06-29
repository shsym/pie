#include "server.hpp"

// All implementation-specific headers are safely included here
#include "gpt.cuh"
#include "ztensor.hpp"

#include <iostream>
#include <set>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>


// The actual implementation of the Server is hidden in this struct.
struct Server::ServerImpl {
    std::unique_ptr<L4maForCausalLM<__nv_bfloat16>> model;
    // You can add any other server state here (e.g., ZMQ context).
};

namespace { // Anonymous namespace for internal helper functions

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
    
    // (Verification logic would go here)
    std::cout << "\nSuccessfully loaded " << loaded_keys.size() << " expected weights." << std::endl;

    return model_ptr;
}

} // anonymous namespace


// --- Public Interface Implementation ---

Server::Server(const AppConfig& config, ModelMetadata& out_metadata)
    // Create the PIMPL object
    : pimpl(std::make_unique<ServerImpl>()) {
    
    std::cout << "Starting service..." << std::endl;


    // 2. Load the model and store it in the implementation object
    pimpl->model = load_model_internal<__nv_bfloat16>(config, out_metadata);
    std::cout << "Model loaded successfully and is resident on the GPU." << std::endl;
}

// By defining the destructor here, the compiler knows the full definition of ServerImpl
// and can correctly destroy the unique_ptr it contains.
Server::~Server() = default;

void Server::run() {
    std::cout << "Starting ZMQ server loop (placeholder)..." << std::endl;
    
}
