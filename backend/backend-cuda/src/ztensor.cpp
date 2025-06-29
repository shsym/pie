#include "ztensor.hpp"

#include <algorithm>
#include <cstring> // For strcmp, memcpy
#include <iostream>
#include <memory>

// Required third-party headers
#include <cbor.h>
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

zTensorReader::zTensorReader(const std::string& path) : _filepath(path) {
    map_file(path);

    // 1. Check magic number
    const std::string magic_number("ZTEN0001");
    if (_file_size < magic_number.size() ||
        std::string(_mmap_ptr, magic_number.size()) != magic_number) {
        throw std::runtime_error("Invalid zTensor magic number.");
    }

    // 2. A valid file must be at least 17 bytes (magic + empty CBOR array + size)
    if (_file_size < 17) {
        // Handle zero-tensor case (8B magic + 1B empty CBOR array 0x80 + 8B size)
        if (_file_size == 17 && *(_mmap_ptr + 8) == static_cast<char>(0x80)) {
            return; // Valid zero-tensor file
        }
        throw std::runtime_error("File size is too small to be a valid zTensor file.");
    }
    
    parse_metadata();
}

zTensorReader::~zTensorReader() {
    unmap_file();
}

// --- Private Methods ---

void zTensorReader::map_file(const std::string& path) {
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

void zTensorReader::unmap_file() {
    if (_mmap_ptr == nullptr) return;

#ifdef _WIN32
    UnmapViewOfFile(_mmap_ptr);
    if (_hMapping != NULL) CloseHandle(_hMapping);
    if (_hFile != INVALID_HANDLE_VALUE) CloseHandle(_hFile);
#else
    munmap(const_cast<char*>(_mmap_ptr), _file_size);
    if (_fd != -1) close(_fd);
#endif
    _mmap_ptr = nullptr;
}

void zTensorReader::parse_metadata() {
    // 1. Read CBOR array size from the last 8 bytes
    uint64_t cbor_size;
    std::memcpy(&cbor_size, _mmap_ptr + _file_size - sizeof(uint64_t), sizeof(uint64_t));
    
    if (cbor_size == 0 || (cbor_size + sizeof(uint64_t) + 8) > _file_size) {
        throw std::runtime_error("Invalid CBOR metadata size.");
    }

    // 2. Locate the CBOR metadata start
    const auto cbor_start = reinterpret_cast<const unsigned char*>(_mmap_ptr + _file_size - sizeof(uint64_t) - cbor_size);

    // 3. Parse the CBOR metadata using libcbor
    cbor_item_t* root;
    struct cbor_load_result result;
    root = cbor_load(cbor_start, cbor_size, &result);

    if (result.error.code != CBOR_ERR_NONE) {
        throw std::runtime_error("Failed to parse CBOR metadata. Error at offset " + std::to_string(result.error.position));
    }
    
    // Use a unique_ptr with a custom deleter for RAII-style resource management
    auto cbor_deleter = [](cbor_item_t* p) { if (p) cbor_decref(&p); };
    std::unique_ptr<cbor_item_t, decltype(cbor_deleter)> root_ptr(root, cbor_deleter);

    if (!cbor_isa_array(root)) {
        throw std::runtime_error("Metadata block is not a valid CBOR array.");
    }

    size_t num_tensors = cbor_array_size(root);
    cbor_item_t** tensor_maps = cbor_array_handle(root);

    for (size_t i = 0; i < num_tensors; ++i) {
        cbor_item_t* map_item = tensor_maps[i];
        if (!cbor_isa_map(map_item)) {
            throw std::runtime_error("Metadata array item is not a map.");
        }

        TensorInfo info;
        info.layout = "dense"; // Default
        info.data_endianness = "little"; // Default

        size_t num_pairs = cbor_map_size(map_item);
        struct cbor_pair* pairs = cbor_map_handle(map_item);

        for (size_t j = 0; j < num_pairs; ++j) {
            cbor_item_t* key_item = pairs[j].key;
            cbor_item_t* val_item = pairs[j].value;

            if (!cbor_isa_string(key_item)) continue;

            const char* key = reinterpret_cast<const char*>(cbor_string_handle(key_item));
            
            if (strcmp(key, "name") == 0 && cbor_isa_string(val_item)) {
                info.name.assign(reinterpret_cast<const char*>(cbor_string_handle(val_item)), cbor_string_length(val_item));
            } else if (strcmp(key, "offset") == 0 && cbor_is_int(val_item)) {
                info.offset = cbor_get_int(val_item);
            } else if (strcmp(key, "size") == 0 && cbor_is_int(val_item)) {
                info.size = cbor_get_int(val_item);
            } else if (strcmp(key, "dtype") == 0 && cbor_isa_string(val_item)) {
                info.dtype.assign(reinterpret_cast<const char*>(cbor_string_handle(val_item)), cbor_string_length(val_item));
            } else if (strcmp(key, "encoding") == 0 && cbor_isa_string(val_item)) {
                info.encoding.assign(reinterpret_cast<const char*>(cbor_string_handle(val_item)), cbor_string_length(val_item));
            } else if (strcmp(key, "layout") == 0 && cbor_isa_string(val_item)) {
                info.layout.assign(reinterpret_cast<const char*>(cbor_string_handle(val_item)), cbor_string_length(val_item));
            } else if (strcmp(key, "data_endianness") == 0 && cbor_isa_string(val_item)) {
                info.data_endianness.assign(reinterpret_cast<const char*>(cbor_string_handle(val_item)), cbor_string_length(val_item));
            } else if (strcmp(key, "shape") == 0 && cbor_isa_array(val_item)) {
                size_t shape_size = cbor_array_size(val_item);
                cbor_item_t** shape_elements = cbor_array_handle(val_item);
                info.shape.reserve(shape_size);
                for (size_t k = 0; k < shape_size; ++k) {
                    if (cbor_is_int(shape_elements[k])) {
                        info.shape.push_back(cbor_get_int(shape_elements[k]));
                    }
                }
            }
        }

        if (info.name.empty()) {
            throw std::runtime_error("Found tensor with no name in metadata.");
        }
        
        if (info.offset + info.size > _file_size) {
            throw std::runtime_error("Tensor '" + info.name + "' data is out of file bounds.");
        }
        
        _tensors[info.name] = std::move(info);
    }
}


// --- Public Methods ---

std::vector<std::string> zTensorReader::list_tensors() const {
    std::vector<std::string> names;
    names.reserve(_tensors.size());
    for (const auto& pair : _tensors) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

const TensorInfo& zTensorReader::get_tensor_info(const std::string& name) const {
    return _tensors.at(name);
}

const void* zTensorReader::get_raw_tensor_pointer(const std::string& name) const {
    const auto& info = get_tensor_info(name);

    bool is_eligible = info.encoding == "raw" &&
                       info.layout == "dense" &&
                       (info.data_endianness == "little") == is_system_little_endian();

    if (!is_eligible) {
        return nullptr;
    }

    return static_cast<const void*>(_mmap_ptr + info.offset);
}

std::vector<std::byte> zTensorReader::read_tensor_data(const std::string& name) const {
    const auto& info = get_tensor_info(name);

    if (info.layout == "sparse") {
        throw std::logic_error("Sparse tensors are not yet supported.");
    }

    const void* blob_ptr = _mmap_ptr + info.offset;

    if (info.encoding == "raw") {
        const auto* data_start = reinterpret_cast<const std::byte*>(blob_ptr);
        std::vector<std::byte> data(data_start, data_start + info.size);
        
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