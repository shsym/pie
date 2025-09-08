#include "artifact_reader.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cstring>

// Use header-only JSON library
#include <nlohmann/json.hpp>

namespace metal_test {

// TensorData implementation
TensorData::TensorData(const std::vector<float>& data, const std::vector<size_t>& shape, const std::string& dtype)
    : data_(data), shape_(shape), dtype_(dtype) {
}

size_t TensorData::get_total_elements() const {
    size_t total = 1;
    for (size_t dim : shape_) {
        total *= dim;
    }
    return total;
}

// ArtifactReader implementation
ArtifactReader::ArtifactReader(const std::string& artifact_base_path)
    : artifact_path_(artifact_base_path), valid_(false) {

    // Check if base path exists
    if (!std::filesystem::exists(artifact_path_)) {
        std::cerr << "Artifact path does not exist: " << artifact_path_ << std::endl;
        return;
    }

    // Construct layer artifacts path
    layer_artifacts_path_ = artifact_path_ + "/layer_artifacts";

    if (!std::filesystem::exists(layer_artifacts_path_)) {
        std::cerr << "Layer artifacts path does not exist: " << layer_artifacts_path_ << std::endl;
        return;
    }

    // Try to load collection metadata
    if (!load_collection_metadata()) {
        std::cerr << "Failed to load collection metadata" << std::endl;
        return;
    }

    valid_ = true;
    std::cout << "Loaded CUDA artifacts from: " << artifact_path_ << std::endl;
    std::cout << "  Layers: " << collection_metadata_.num_layers << std::endl;
    std::cout << "  Hidden size: " << collection_metadata_.hidden_size << std::endl;
    std::cout << "  Sequence length: " << collection_metadata_.sequence_length << std::endl;
}

bool ArtifactReader::load_collection_metadata() {
    std::string metadata_file = layer_artifacts_path_ + "/collection_meta.json";
    return parse_collection_metadata(metadata_file);
}

std::unique_ptr<TensorData> ArtifactReader::load_tensor(const std::string& layer_name, const std::string& step_name) {
    if (!valid_) {
        std::cerr << "ArtifactReader is not valid" << std::endl;
        return nullptr;
    }

    // Load tensor metadata
    std::string metadata_file = get_metadata_file(layer_name, step_name);
    auto metadata = load_tensor_metadata(metadata_file);
    if (!metadata) {
        std::cerr << "Failed to load tensor metadata: " << metadata_file << std::endl;
        return nullptr;
    }

    // Load binary data
    std::string binary_file = get_layer_path(layer_name) + "/" + metadata->binary_file;
    auto binary_data = load_binary_data(binary_file, metadata->dtype, metadata->size);
    if (binary_data.empty() && metadata->size > 0) {
        std::cerr << "Failed to load binary data: " << binary_file << std::endl;
        return nullptr;
    }

    return std::make_unique<TensorData>(binary_data, metadata->shape, metadata->dtype);
}

std::unique_ptr<TensorData> ArtifactReader::load_final_tensor(const std::string& step_name) {
    if (!valid_) {
        std::cerr << "ArtifactReader is not valid" << std::endl;
        return nullptr;
    }

    // Load tensor metadata
    std::string metadata_file = get_final_metadata_file(step_name);
    auto metadata = load_tensor_metadata(metadata_file);
    if (!metadata) {
        std::cerr << "Failed to load final tensor metadata: " << metadata_file << std::endl;
        return nullptr;
    }

    // Load binary data
    std::string binary_file = get_final_path() + "/" + metadata->binary_file;
    auto binary_data = load_binary_data(binary_file, metadata->dtype, metadata->size);
    if (binary_data.empty() && metadata->size > 0) {
        std::cerr << "Failed to load final binary data: " << binary_file << std::endl;
        return nullptr;
    }

    return std::make_unique<TensorData>(binary_data, metadata->shape, metadata->dtype);
}

std::vector<std::string> ArtifactReader::get_available_layers() const {
    std::vector<std::string> layers;

    if (!valid_) {
        return layers;
    }

    try {
        for (int i = 0; i < collection_metadata_.num_layers; ++i) {
            std::string layer_name = "layer_" + std::to_string(i);
            std::string layer_path = get_layer_path(layer_name);
            if (std::filesystem::exists(layer_path)) {
                layers.push_back(layer_name);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error getting available layers: " << e.what() << std::endl;
    }

    return layers;
}

bool ArtifactReader::has_artifact(const std::string& layer_name, const std::string& step_name) const {
    if (!valid_) {
        return false;
    }

    std::string metadata_file = get_metadata_file(layer_name, step_name);
    return std::filesystem::exists(metadata_file);
}

bool ArtifactReader::has_final_artifact(const std::string& step_name) const {
    if (!valid_) {
        return false;
    }

    std::string metadata_file = get_final_metadata_file(step_name);
    return std::filesystem::exists(metadata_file);
}

// Private methods
bool ArtifactReader::parse_collection_metadata(const std::string& metadata_file) {
    try {
        std::ifstream file(metadata_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open collection metadata file: " << metadata_file << std::endl;
            return false;
        }

        nlohmann::json json_data;
        file >> json_data;

        collection_metadata_.case_id = json_data.at("case_id").get<std::string>();
        collection_metadata_.num_layers = json_data.at("num_layers").get<int>();
        collection_metadata_.hidden_size = json_data.at("hidden_size").get<int>();
        collection_metadata_.sequence_length = json_data.at("sequence_length").get<int>();
        collection_metadata_.timestamp = json_data.at("timestamp").get<int64_t>();

        // Parse steps per layer
        for (const auto& step : json_data.at("steps_per_layer")) {
            collection_metadata_.steps_per_layer.push_back(step.get<std::string>());
        }

        // Parse final steps
        for (const auto& step : json_data.at("final_steps")) {
            collection_metadata_.final_steps.push_back(step.get<std::string>());
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing collection metadata: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<TensorMetadata> ArtifactReader::load_tensor_metadata(const std::string& metadata_file) {
    try {
        std::ifstream file(metadata_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open tensor metadata file: " << metadata_file << std::endl;
            return nullptr;
        }

        nlohmann::json json_data;
        file >> json_data;

        auto metadata = std::make_unique<TensorMetadata>();
        metadata->name = json_data.at("name").get<std::string>();
        metadata->dtype = json_data.at("dtype").get<std::string>();
        metadata->size = json_data.at("size").get<size_t>();
        metadata->binary_file = json_data.at("binary_file").get<std::string>();

        // Parse shape
        for (const auto& dim : json_data.at("shape")) {
            metadata->shape.push_back(dim.get<size_t>());
        }

        return metadata;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing tensor metadata: " << e.what() << std::endl;
        return nullptr;
    }
}

std::vector<float> ArtifactReader::load_binary_data(const std::string& binary_file, const std::string& dtype, size_t expected_size) {
    std::vector<float> data;

    try {
        std::ifstream file(binary_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open binary file: " << binary_file << std::endl;
            return data;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (dtype == "float32") {
            // Direct read for float32
            size_t expected_bytes = expected_size * sizeof(float);
            if (file_size != expected_bytes) {
                std::cerr << "Binary file size mismatch. Expected: " << expected_bytes
                         << ", Got: " << file_size << std::endl;
                return data;
            }

            data.resize(expected_size);
            file.read(reinterpret_cast<char*>(data.data()), expected_bytes);

        } else {
            std::cerr << "Unsupported dtype for binary loading: " << dtype << std::endl;
            return data;
        }

        if (!file.good()) {
            std::cerr << "Error reading binary file: " << binary_file << std::endl;
            data.clear();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error loading binary data: " << e.what() << std::endl;
        data.clear();
    }

    return data;
}

// Path utilities
std::string ArtifactReader::get_layer_path(const std::string& layer_name) const {
    return layer_artifacts_path_ + "/" + layer_name;
}

std::string ArtifactReader::get_final_path() const {
    return layer_artifacts_path_ + "/final";
}

std::string ArtifactReader::get_metadata_file(const std::string& layer_name, const std::string& step_name) const {
    return get_layer_path(layer_name) + "/" + step_name + "_meta.json";
}

std::string ArtifactReader::get_binary_file(const std::string& layer_name, const std::string& step_name) const {
    return get_layer_path(layer_name) + "/" + step_name + ".bin";
}

std::string ArtifactReader::get_final_metadata_file(const std::string& step_name) const {
    return get_final_path() + "/" + step_name + "_meta.json";
}

std::string ArtifactReader::get_final_binary_file(const std::string& step_name) const {
    return get_final_path() + "/" + step_name + ".bin";
}

} // namespace metal_test