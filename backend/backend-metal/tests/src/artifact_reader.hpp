#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace metal_test {

// Tensor metadata from CUDA artifacts
struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    size_t size;
    std::string binary_file;
};

// Collection metadata from CUDA artifacts
struct CollectionMetadata {
    std::string case_id;
    int num_layers;
    int hidden_size;
    int sequence_length;
    std::vector<std::string> steps_per_layer;
    std::vector<std::string> final_steps;
    int64_t timestamp;
};

// Tensor data holder
class TensorData {
public:
    TensorData(const std::vector<float>& data, const std::vector<size_t>& shape, const std::string& dtype);
    ~TensorData() = default;

    const std::vector<float>& get_data() const { return data_; }
    const std::vector<size_t>& get_shape() const { return shape_; }
    const std::string& get_dtype() const { return dtype_; }
    size_t get_total_elements() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::string dtype_;
};

// Main artifact reader class
class ArtifactReader {
public:
    ArtifactReader(const std::string& artifact_base_path);
    ~ArtifactReader() = default;

    // Load collection metadata
    bool load_collection_metadata();
    const CollectionMetadata& get_collection_metadata() const { return collection_metadata_; }

    // Load specific layer and step artifact
    std::unique_ptr<TensorData> load_tensor(const std::string& layer_name, const std::string& step_name);

    // Load final step artifacts (logits, etc.)
    std::unique_ptr<TensorData> load_final_tensor(const std::string& step_name);

    // Get available layers
    std::vector<std::string> get_available_layers() const;

    // Check if artifact exists
    bool has_artifact(const std::string& layer_name, const std::string& step_name) const;
    bool has_final_artifact(const std::string& step_name) const;

    // Utility methods
    std::string get_artifact_path() const { return artifact_path_; }
    bool is_valid() const { return valid_; }

private:
    std::string artifact_path_;
    std::string layer_artifacts_path_;
    CollectionMetadata collection_metadata_;
    bool valid_;

    // Internal methods
    bool parse_collection_metadata(const std::string& metadata_file);
    std::unique_ptr<TensorMetadata> load_tensor_metadata(const std::string& metadata_file);
    std::vector<float> load_binary_data(const std::string& binary_file, const std::string& dtype, size_t expected_size);

    // Path utilities
    std::string get_layer_path(const std::string& layer_name) const;
    std::string get_final_path() const;
    std::string get_metadata_file(const std::string& layer_name, const std::string& step_name) const;
    std::string get_binary_file(const std::string& layer_name, const std::string& step_name) const;
    std::string get_final_metadata_file(const std::string& step_name) const;
    std::string get_final_binary_file(const std::string& step_name) const;
};

} // namespace metal_test