#ifndef ZTENSOR_HPP
#define ZTENSOR_HPP

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <numeric>
#include <span>
#include <optional>
#include <algorithm>
#include <cstddef> // For std::byte

// Required third-party headers
#include <nlohmann/json.hpp>
#include <zstd.h>

// Platform-specific headers for memory mapping
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace ztensor {

// Forward declaration
class zTensorReader;

/**
 * @struct TensorInfo
 * @brief Holds the metadata for a single tensor.
 */
struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::string dtype;
    std::vector<int64_t> shape;
    std::string encoding;
    std::string layout;
    std::string data_endianness = "little"; // Default as per spec

    /**
     * @brief Calculates the total number of elements in the tensor.
     * @return The total number of elements. Returns 0 if shape is empty.
     */
    uint64_t num_elements() const {
        if (shape.empty()) return 1; // Scalar
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<uint64_t>());
    }
};

/**
 * @brief Helper to check the endianness of the current system.
 * @return True if the system is little-endian, false otherwise.
 */
inline bool is_system_little_endian() {
    const uint32_t i = 1;
    return *reinterpret_cast<const uint8_t*>(&i) == 1;
}

/**
 * @class zTensorReader
 * @brief Reads zTensor files and provides efficient, zero-copy access to tensor data.
 *
 * This class memory-maps the zTensor file into memory. For dense tensors stored
 * with "raw" encoding and matching system endianness, it provides a direct, non-owning
 * pointer into the mapped memory. This pointer can be used in CUDA calls for
 * high-performance, zero-copy data uploads to a GPU.
 *
 * For compressed tensors, it provides a method to decompress the data into an
 * owned buffer. Sparse tensors are not yet supported.
 */
class zTensorReader {
public:
    /**
     * @brief Constructs a zTensorReader and memory-maps the file.
     * @param path Path to the .ztensor file.
     * @throws std::runtime_error on file I/O or parsing errors.
     */
    explicit zTensorReader(const std::string& path);

    /**
     * @brief Destructor. Unmaps the memory and closes the file.
     */
    ~zTensorReader();

    // Disable copy and move semantics to prevent issues with resource management.
    zTensorReader(const zTensorReader&) = delete;
    zTensorReader& operator=(const zTensorReader&) = delete;
    zTensorReader(zTensorReader&&) = delete;
    zTensorReader& operator=(zTensorReader&&) = delete;

    /**
     * @brief Lists the names of all tensors in the file.
     * @return A vector of tensor names.
     */
    std::vector<std::string> list_tensors() const;

    /**
     * @brief Retrieves the metadata for a specific tensor.
     * @param name The name of the tensor.
     * @return A constant reference to the TensorInfo struct.
     * @throws std::out_of_range if the tensor name does not exist.
     */
    const TensorInfo& get_tensor_info(const std::string& name) const;

    /**
     * @brief Gets a raw, non-owning pointer to a tensor's data for zero-copy access.
     *
     * This is the primary method for high-performance access. The pointer is valid
     * as long as the zTensorReader object exists.
     *
     * @param name The name of the tensor.
     * @return A `const void*` pointing directly to the tensor data in the memory-mapped file.
     * This pointer is suitable for use with `cudaMemcpy` and can be registered
     * with `cudaHostRegister` for pinned memory transfers.
     * Returns `nullptr` if the tensor is not eligible for zero-copy access
     * (e.g., it's compressed, has non-native endianness, or is sparse).
     */
    const void* get_raw_tensor_pointer(const std::string& name) const;

    /**
     * @brief Reads tensor data into an owned buffer, decompressing if necessary.
     *
     * Use this method for compressed tensors or when a mutable copy of the data is needed.
     *
     * @param name The name of the tensor.
     * @return A `std::vector<std::byte>` containing the tensor data.
     * @throws std::runtime_error if decompression fails.
     * @throws std::logic_error for unsupported layouts (e.g., "sparse").
     */
    std::vector<std::byte> read_tensor_data(const std::string& name) const;


private:
    void map_file(const std::string& path);
    void unmap_file();
    void parse_metadata();

    std::string _filepath;
    const char* _mmap_ptr = nullptr;
    uint64_t _file_size = 0;
    std::unordered_map<std::string, TensorInfo> _tensors;

#ifdef _WIN32
    HANDLE _hFile = INVALID_HANDLE_VALUE;
    HANDLE _hMapping = NULL;
#else
    int _fd = -1;
#endif
};

// --- Implementation ---

inline zTensorReader::zTensorReader(const std::string& path) : _filepath(path) {
    map_file(path);

    // 1. Check magic number
    const std::string magic_number("ZTEN0001");
    if (_file_size < magic_number.size() ||
        std::string(_mmap_ptr, magic_number.size()) != magic_number) {
        throw std::runtime_error("Invalid zTensor magic number.");
    }

    // 2. A valid file must be at least 17 bytes (magic + empty CBOR array + size)
    if (_file_size < 17) {
        // Handle zero-tensor case
        if (_file_size == 17 && *(_mmap_ptr + 8) == static_cast<char>(0x80)) {
            return; // Valid zero-tensor file
        }
        throw std::runtime_error("File size is too small to be a valid zTensor file.");
    }
    
    parse_metadata();
}

inline zTensorReader::~zTensorReader() {
    unmap_file();
}

inline void zTensorReader::map_file(const std::string& path) {
#ifdef _WIN32
    _hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (_hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    LARGE_INTEGER fs;
    if (!GetFileSizeEx(_hFile, &fs)) {
        CloseHandle(_hFile);
        throw std::runtime_error("Failed to get file size.");
    }
    _file_size = fs.QuadPart;

    _hMapping = CreateFileMapping(_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (_hMapping == NULL) {
        CloseHandle(_hFile);
        throw std::runtime_error("Failed to create file mapping.");
    }

    _mmap_ptr = static_cast<const char*>(MapViewOfFile(_hMapping, FILE_MAP_READ, 0, 0, _file_size));
    if (_mmap_ptr == nullptr) {
        CloseHandle(_hMapping);
        CloseHandle(_hFile);
        throw std::runtime_error("Failed to map view of file.");
    }
#else
    _fd = open(path.c_str(), O_RDONLY);
    if (_fd == -1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    struct stat sb;
    if (fstat(_fd, &sb) == -1) {
        close(_fd);
        throw std::runtime_error("Failed to get file size.");
    }
    _file_size = sb.st_size;

    _mmap_ptr = static_cast<const char*>(mmap(NULL, _file_size, PROT_READ, MAP_PRIVATE, _fd, 0));
    if (_mmap_ptr == MAP_FAILED) {
        close(_fd);
        throw std::runtime_error("Failed to memory map file.");
    }
#endif
}

inline void zTensorReader::unmap_file() {
    if (_mmap_ptr == nullptr) return;

#ifdef _WIN32
    UnmapViewOfFile(_mmap_ptr);
    CloseHandle(_hMapping);
    CloseHandle(_hFile);
#else
    munmap(const_cast<char*>(_mmap_ptr), _file_size);
    close(_fd);
#endif
    _mmap_ptr = nullptr;
}


inline void zTensorReader::parse_metadata() {
    // 1. Read CBOR array size from the last 8 bytes
    uint64_t cbor_size;
    std::memcpy(&cbor_size, _mmap_ptr + _file_size - sizeof(uint64_t), sizeof(uint64_t));
    
    // Basic validation
    if (cbor_size == 0 || (cbor_size + sizeof(uint64_t) + 8) > _file_size) {
        throw std::runtime_error("Invalid CBOR metadata size.");
    }

    // 2. Locate and parse the CBOR metadata array
    const auto cbor_start = reinterpret_cast<const uint8_t*>(_mmap_ptr + _file_size - sizeof(uint64_t) - cbor_size);
    
    nlohmann::json metadata_array;
    try {
         metadata_array = nlohmann::json::from_cbor(cbor_start, cbor_start + cbor_size);
    } catch(const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse CBOR metadata: " + std::string(e.what()));
    }

    if (!metadata_array.is_array()) {
        throw std::runtime_error("Metadata block is not a valid CBOR array.");
    }

    // 3. Populate tensor info map
    for (const auto& item : metadata_array) {
        TensorInfo info;
        info.name = item.at("name").get<std::string>();
        info.offset = item.at("offset").get<uint64_t>();
        info.size = item.at("size").get<uint64_t>();
        info.dtype = item.at("dtype").get<std::string>();
        info.shape = item.at("shape").get<std::vector<int64_t>>();
        info.encoding = item.at("encoding").get<std::string>();
        info.layout = item.value("layout", "dense");
        info.data_endianness = item.value("data_endianness", "little");

        // Basic validation for blob location
        if (info.offset + info.size > _file_size) {
            throw std::runtime_error("Tensor '" + info.name + "' data is out of file bounds.");
        }
        
        _tensors[info.name] = std::move(info);
    }
}

inline std::vector<std::string> zTensorReader::list_tensors() const {
    std::vector<std::string> names;
    names.reserve(_tensors.size());
    for (const auto& pair : _tensors) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

inline const TensorInfo& zTensorReader::get_tensor_info(const std::string& name) const {
    return _tensors.at(name);
}

inline const void* zTensorReader::get_raw_tensor_pointer(const std::string& name) const {
    const auto& info = get_tensor_info(name);

    // Zero-copy is only possible for raw, dense tensors with matching endianness
    bool is_eligible = info.encoding == "raw" &&
                       info.layout == "dense" &&
                       (info.data_endianness == "little") == is_system_little_endian();

    if (!is_eligible) {
        return nullptr;
    }

    return static_cast<const void*>(_mmap_ptr + info.offset);
}


inline std::vector<std::byte> zTensorReader::read_tensor_data(const std::string& name) const {
    const auto& info = get_tensor_info(name);

    if (info.layout == "sparse") {
        throw std::logic_error("Sparse tensors are not yet supported.");
    }

    const void* blob_ptr = _mmap_ptr + info.offset;

    if (info.encoding == "raw") {
        // For raw, we just copy the data from the memory map.
        const auto* data_start = reinterpret_cast<const std::byte*>(blob_ptr);
        std::vector<std::byte> data(data_start, data_start + info.size);
        
        // Handle endianness swap if necessary
        bool endian_mismatch = (info.data_endianness == "little") != is_system_little_endian();
        if (endian_mismatch) {
            size_t element_size = 0;
            if (info.dtype.find("64") != std::string::npos) element_size = 8;
            else if (info.dtype.find("32") != std::string::npos) element_size = 4;
            else if (info.dtype.find("16") != std::string::npos) element_size = 2;
            
            if (element_size > 1) {
                for (size_t i = 0; i < data.size(); i += element_size) {
                    std::reverse(data.begin() + i, data.begin() + i + element_size);
                }
            }
        }
        return data;

    } else if (info.encoding == "zstd") {
        // Decompress zstd-encoded data
        unsigned long long const decompressed_size = ZSTD_getFrameContentSize(blob_ptr, info.size);
        if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Cannot get decompressed size for tensor '" + name + "'.");
        }

        std::vector<std::byte> decompressed_buffer(decompressed_size);
        size_t const actual_size = ZSTD_decompress(
            decompressed_buffer.data(),
            decompressed_buffer.size(),
            blob_ptr,
            info.size);
        
        if (ZSTD_isError(actual_size) || actual_size != decompressed_size) {
             throw std::runtime_error("Zstd decompression failed for tensor '" + name + "'. Error: " + ZSTD_getErrorName(actual_size));
        }

        return decompressed_buffer;
    }

    throw std::logic_error("Unsupported encoding: " + info.encoding);
}


} // namespace ztensor

#endif // ZTENSOR_HPP