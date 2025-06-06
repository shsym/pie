// #include "ztensor.hpp"
// #include <iostream>
// #include <fstream>
// #include <vector>

// // --- Dummy CUDA function stubs for demonstration ---
// // In a real application, you would include <cuda_runtime.h>
// enum cudaMemcpyKind { cudaMemcpyHostToDevice };
// void cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
//     // This is a fake stub to illustrate the API call.
//     std::cout << "  (INFO) Faking cudaMemcpy: " << count << " bytes from host to device." << std::endl;
// }
// // --- End stubs ---


// // Helper function to create a dummy zTensor file for testing.
// // In a real scenario, this file would already exist.
// void create_dummy_ztensor_file(const std::string& filename) {
//     std::cout << "Creating dummy file: " << filename << std::endl;
    
//     // Tensors
//     std::vector<float> tensor1_data = {1.0f, 2.0f, 3.0f, 4.0f}; // 16 bytes
//     // Will be compressed
//     std::vector<int32_t> tensor2_data = {10, 20, 30, 40, 50, 60}; // 24 bytes

//     // Compress tensor2 with zstd
//     size_t const c_buffer_size = ZSTD_compressBound(tensor2_data.size() * sizeof(int32_t));
//     std::vector<char> compressed_buffer(c_buffer_size);
//     size_t const c_size = ZSTD_compress(
//         compressed_buffer.data(), c_buffer_size, 
//         tensor2_data.data(), tensor2_data.size() * sizeof(int32_t), 
//         1 /* compression level */);
//     compressed_buffer.resize(c_size);

//     // Offsets must be 64-byte aligned
//     uint64_t tensor1_offset = 64;
//     uint64_t tensor2_offset = 128;

//     // Metadata
//     nlohmann::json metadata = nlohmann::json::array();
//     metadata.push_back({
//         {"name", "dense_float_raw"},
//         {"offset", tensor1_offset},
//         {"size", tensor1_data.size() * sizeof(float)},
//         {"dtype", "float32"},
//         {"shape", {2, 2}},
//         {"encoding", "raw"}
//     });
//     metadata.push_back({
//         {"name", "dense_int_zstd"},
//         {"offset", tensor2_offset},
//         {"size", c_size},
//         {"dtype", "int32"},
//         {"shape", {6}},
//         {"encoding", "zstd"}
//     });
    
//     // Serialize metadata to CBOR
//     std::vector<uint8_t> cbor_data = nlohmann::json::to_cbor(metadata);
//     uint64_t cbor_size = cbor_data.size();

//     std::ofstream outfile(filename, std::ios::binary);
    
//     // Magic number
//     outfile.write("ZTEN0001", 8);

//     // Write tensor blobs at aligned offsets
//     outfile.seekp(tensor1_offset);
//     outfile.write(reinterpret_cast<const char*>(tensor1_data.data()), tensor1_data.size() * sizeof(float));

//     outfile.seekp(tensor2_offset);
//     outfile.write(compressed_buffer.data(), c_size);
    
//     // Go to end to write metadata
//     outfile.seekp(0, std::ios::end);
//     auto current_pos = outfile.tellp(); // Find where blobs ended
    
//     outfile.write(reinterpret_cast<const char*>(cbor_data.data()), cbor_size);
//     outfile.write(reinterpret_cast<const char*>(&cbor_size), sizeof(cbor_size));
    
//     outfile.close();
//     std::cout << "Dummy file created." << std::endl;
// }


// int main() {
//     const std::string filename = "test.ztensor";
//     create_dummy_ztensor_file(filename);

//     try {
//         ztensor::zTensorReader reader(filename);
        
//         std::cout << "\n--- Tensors in file ---" << std::endl;
//         for (const auto& name : reader.list_tensors()) {
//             std::cout << "- " << name << std::endl;
//         }

//         // --- 1. Zero-Copy GPU Upload Example ---
//         std::cout << "\n--- Handling 'dense_float_raw' (Zero-Copy Path) ---" << std::endl;
//         const auto& raw_info = reader.get_tensor_info("dense_float_raw");
//         const void* raw_ptr = reader.get_raw_tensor_pointer("dense_float_raw");

//         if (raw_ptr) {
//             std::cout << "Successfully got raw pointer: " << raw_ptr << std::endl;
//             std::cout << "This pointer points directly into the memory-mapped file." << std::endl;
//             std::cout << "It can now be used with cudaMemcpy for a zero-copy upload." << std::endl;
            
//             // Example: Pass this pointer directly to CUDA
//             // void* device_ptr;
//             // cudaMalloc(&device_ptr, raw_info.size);
//             cudaMemcpy(/* device_ptr */ nullptr, raw_ptr, raw_info.size, cudaMemcpyHostToDevice);

//             // You can also cast and access the data on the CPU
//             const float* data = static_cast<const float*>(raw_ptr);
//             std::cout << "  Data on CPU: [" << data[0] << ", " << data[1] << ", ...]" << std::endl;

//         } else {
//             std::cerr << "Failed to get raw pointer (tensor might be compressed or have wrong endianness)." << std::endl;
//         }


//         // --- 2. Decompression Example ---
//         std::cout << "\n--- Handling 'dense_int_zstd' (Decompression Path) ---" << std::endl;
//         const auto& zstd_info = reader.get_tensor_info("dense_int_zstd");
//         std::cout << "Tensor is zstd compressed (on-disk size: " << zstd_info.size << " bytes)." << std::endl;
//         std::cout << "Reading and decompressing..." << std::endl;

//         std::vector<std::byte> decompressed_data = reader.read_tensor_data("dense_int_zstd");
        
//         std::cout << "Decompressed successfully into an owned buffer of size: " << decompressed_data.size() << " bytes." << std::endl;
        
//         // This owned buffer can now be uploaded to the GPU
//         cudaMemcpy(/* device_ptr */ nullptr, decompressed_data.data(), decompressed_data.size(), cudaMemcpyHostToDevice);
        
//         // Accessing the data
//         const int32_t* data = reinterpret_cast<const int32_t*>(decompressed_data.data());
//         std::cout << "  Data on CPU: [" << data[0] << ", " << data[1] << ", ...]" << std::endl;


//     } catch (const std::exception& e) {
//         std::cerr << "An error occurred: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }