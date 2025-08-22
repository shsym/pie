#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <regex>
#include <cstring>

namespace ops {

enum class DType {
    BF16,
    FP16, 
    FP32
};

struct DTypeInfo {
    DType dtype;
    bool success;
    std::string dtype_str;
};

/**
 * Parse meta.json to detect the target dtype for the operation
 * Returns DTypeInfo with success=true if dtype found, false otherwise
 */
inline DTypeInfo detect_dtype_from_meta(const std::string& op_name, const std::string& case_id) {
    DTypeInfo result{DType::BF16, false, "bf16"};
    
    try {
        std::filesystem::path cuda_base_dir;
        if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
            cuda_base_dir = std::filesystem::path(envp);
        } else {
            std::filesystem::path this_file = __FILE__;
            auto tests_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
            cuda_base_dir = tests_dir;
        }
        
        std::filesystem::path meta_path = cuda_base_dir / op_name / case_id / "meta.json";
        if (std::filesystem::exists(meta_path)) {
            std::ifstream fin(meta_path);
            std::string meta((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            
            // Check for fp32 first (most specific)
            if (meta.find("\"fp32\"") != std::string::npos || meta.find("\"float32\"") != std::string::npos) {
                result = {DType::FP32, true, "fp32"};
            }
            // Then fp16  
            else if (meta.find("\"fp16\"") != std::string::npos) {
                result = {DType::FP16, true, "fp16"};
            }
            // Finally bf16 (includes "bf16" detection)
            else if (meta.find("\"bf16\"") != std::string::npos) {
                result = {DType::BF16, true, "bf16"};
            }
            // If no explicit dtype found but meta exists, assume bf16 (backward compatibility)
            else {
                result = {DType::BF16, true, "bf16"};
            }
        }
    } catch (...) {
        // On any error, return failure - caller should handle missing meta appropriately
        result.success = false;
    }
    
    return result;
}

/**
 * Get string representation of DType enum
 */
inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::BF16: return "bf16";
        case DType::FP16: return "fp16"; 
        case DType::FP32: return "fp32";
        default: return "unknown";
    }
}

/**
 * Load binary data from CUDA reference artifacts based on metadata
 * Returns vector of bytes - caller needs to cast to appropriate type
 */
inline std::vector<uint8_t> load_cuda_tensor_bytes(const std::string& op_name, const std::string& case_id, const std::string& tensor_name) {
    std::vector<uint8_t> result;
    
    try {
        std::filesystem::path cuda_base_dir;
        if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
            cuda_base_dir = std::filesystem::path(envp);
        } else {
            std::filesystem::path this_file = __FILE__;
            auto tests_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
            cuda_base_dir = tests_dir;
        }
        
        std::filesystem::path tensor_path = cuda_base_dir / op_name / case_id / (tensor_name + ".bin");
        if (std::filesystem::exists(tensor_path)) {
            std::ifstream file(tensor_path, std::ios::binary);
            file.seekg(0, std::ios::end);
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            result.resize(size);
            file.read(reinterpret_cast<char*>(result.data()), size);
        }
    } catch (...) {
        // Return empty vector on any error
        result.clear();
    }
    
    return result;
}

/**
 * Extract tensor dtype from metadata JSON
 */
inline std::string get_tensor_dtype(const std::string& op_name, const std::string& case_id, const std::string& tensor_name) {
    try {
        std::filesystem::path cuda_base_dir;
        if (const char* envp = std::getenv("PIE_CUDA_ARTIFACTS_DIR")) {
            cuda_base_dir = std::filesystem::path(envp);
        } else {
            std::filesystem::path this_file = __FILE__;
            auto tests_dir = this_file.parent_path().parent_path() / "tests" / "artifacts";
            cuda_base_dir = tests_dir;
        }
        
        std::filesystem::path meta_path = cuda_base_dir / op_name / case_id / "meta.json";
        if (std::filesystem::exists(meta_path)) {
            std::ifstream fin(meta_path);
            std::string meta((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            
            // Simple regex to extract dtype for specific tensor
            // Looking for pattern: "tensor_name": "dtype"
            std::regex dtype_regex("\"" + tensor_name + "\"\\s*:\\s*\"([^\"]+)\"");
            std::smatch match;
            if (std::regex_search(meta, match, dtype_regex)) {
                return match[1].str();
            }
        }
    } catch (...) {
        // Return empty string on any error
    }
    
    return "";
}

/**
 * Load tensor as specific type from CUDA reference
 * Template specializations handle type conversion based on CUDA dtype
 */
template<typename T>
inline std::vector<T> load_cuda_tensor(const std::string& op_name, const std::string& case_id, const std::string& tensor_name) {
    std::vector<T> result;
    
    auto bytes = load_cuda_tensor_bytes(op_name, case_id, tensor_name);
    if (bytes.empty()) {
        return result;
    }
    
    std::string dtype = get_tensor_dtype(op_name, case_id, tensor_name);
    
    // Handle different CUDA dtypes
    if (dtype == "fp32" || dtype == "float32") {
        const float* data = reinterpret_cast<const float*>(bytes.data());
        size_t count = bytes.size() / sizeof(float);
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = static_cast<T>(data[i]);
        }
    } else if (dtype == "fp16") {
        const uint16_t* data = reinterpret_cast<const uint16_t*>(bytes.data());
        size_t count = bytes.size() / sizeof(uint16_t);
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            // Convert fp16 to T (assuming T has appropriate conversion)
            result[i] = static_cast<T>(data[i]); // This needs proper fp16 conversion
        }
    } else if (dtype == "bf16") {
        const uint16_t* data = reinterpret_cast<const uint16_t*>(bytes.data());
        size_t count = bytes.size() / sizeof(uint16_t);
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            // Convert bf16 to T (assuming T has appropriate conversion)
            result[i] = static_cast<T>(data[i]); // This needs proper bf16 conversion
        }
    } else if (dtype == "s32" || dtype == "int32") {
        const int32_t* data = reinterpret_cast<const int32_t*>(bytes.data());
        size_t count = bytes.size() / sizeof(int32_t);
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = static_cast<T>(data[i]);
        }
    } else {
        // Unknown dtype - treat as raw bytes cast to T
        size_t count = bytes.size() / sizeof(T);
        result.resize(count);
        std::memcpy(result.data(), bytes.data(), count * sizeof(T));
    }
    
    return result;
}

} // namespace ops