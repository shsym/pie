#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <nlohmann/json.hpp>

/**
 * Tool to create a valid zTensor file with test weight data
 * This will be used to test MetalTensor weight loading functionality
 */

struct TestTensorData {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    std::vector<float> data;

    size_t num_elements() const {
        size_t count = 1;
        for (int64_t dim : shape) {
            count *= dim;
        }
        return count;
    }
};

void create_test_ztensor_file(const std::string& filepath,
                              const std::vector<TestTensorData>& tensors) {
    std::ofstream file(filepath, std::ios::binary);

    // 1. Write magic number
    file.write("ZTEN0001", 8);

    // 2. Write tensor data
    uint64_t current_offset = 8;  // After magic

    // Keep track of tensor info for metadata
    std::vector<nlohmann::json> tensor_metadata;

    for (const auto& tensor : tensors) {
        // Write tensor data
        const char* data_ptr = reinterpret_cast<const char*>(tensor.data.data());
        size_t data_size = tensor.data.size() * sizeof(float);
        file.write(data_ptr, data_size);

        // Create metadata entry
        nlohmann::json meta;
        meta["name"] = tensor.name;
        meta["offset"] = current_offset;
        meta["size"] = data_size;
        meta["dtype"] = tensor.dtype;
        meta["shape"] = tensor.shape;
        meta["encoding"] = "raw";
        meta["layout"] = "dense";
        meta["data_endianness"] = "little";

        tensor_metadata.push_back(meta);
        current_offset += data_size;
    }

    // 3. Create CBOR metadata array
    nlohmann::json metadata_array = tensor_metadata;

    // Convert to CBOR
    std::vector<uint8_t> cbor_data = nlohmann::json::to_cbor(metadata_array);

    // Write CBOR data
    file.write(reinterpret_cast<const char*>(cbor_data.data()), cbor_data.size());

    // 4. Write CBOR size at the end
    uint64_t cbor_size = cbor_data.size();
    file.write(reinterpret_cast<const char*>(&cbor_size), sizeof(cbor_size));

    file.close();

    std::cout << "Created zTensor file: " << filepath << std::endl;
    std::cout << "  Tensors: " << tensors.size() << std::endl;
    std::cout << "  CBOR size: " << cbor_size << " bytes" << std::endl;
    std::cout << "  Total file size: " << (current_offset + cbor_size + sizeof(cbor_size)) << " bytes" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <output_file.ztensor>" << std::endl;
        return 1;
    }

    std::string output_file = argv[1];

    // Create test tensors that might appear in a small model
    std::vector<TestTensorData> test_tensors;

    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 1. Small embedding tensor (vocab_size=1000, hidden_size=64)
    {
        TestTensorData tensor;
        tensor.name = "model.embed_tokens.weight";
        tensor.shape = {1000, 64};
        tensor.dtype = "float32";
        tensor.data.resize(tensor.num_elements());

        for (auto& val : tensor.data) {
            val = dis(gen) * 0.1f;  // Small weights
        }

        test_tensors.push_back(tensor);
    }

    // 2. Small attention weight (64x64)
    {
        TestTensorData tensor;
        tensor.name = "model.layers.0.self_attn.q_proj.weight";
        tensor.shape = {64, 64};
        tensor.dtype = "float32";
        tensor.data.resize(tensor.num_elements());

        // Initialize as identity matrix with small noise
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                if (i == j) {
                    tensor.data[i * 64 + j] = 1.0f + dis(gen) * 0.01f;
                } else {
                    tensor.data[i * 64 + j] = dis(gen) * 0.01f;
                }
            }
        }

        test_tensors.push_back(tensor);
    }

    // 3. Bias vector
    {
        TestTensorData tensor;
        tensor.name = "model.layers.0.self_attn.q_proj.bias";
        tensor.shape = {64};
        tensor.dtype = "float32";
        tensor.data.resize(tensor.num_elements());

        for (auto& val : tensor.data) {
            val = dis(gen) * 0.01f;  // Small bias
        }

        test_tensors.push_back(tensor);
    }

    // 4. Layer norm weights
    {
        TestTensorData tensor;
        tensor.name = "model.layers.0.input_layernorm.weight";
        tensor.shape = {64};
        tensor.dtype = "float32";
        tensor.data.resize(tensor.num_elements());

        for (auto& val : tensor.data) {
            val = 1.0f + dis(gen) * 0.01f;  // Near 1.0 for layer norm
        }

        test_tensors.push_back(tensor);
    }

    try {
        create_test_ztensor_file(output_file, test_tensors);
        std::cout << "✅ Successfully created test zTensor file!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error creating zTensor file: " << e.what() << std::endl;
        return 1;
    }
}