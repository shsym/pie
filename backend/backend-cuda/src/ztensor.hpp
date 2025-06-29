#ifndef ZTENSOR_HPP
#define ZTENSOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <numeric>
#include <cstddef> // For std::byte

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
    std::string data_endianness;

    /**
     * @brief Calculates the total number of elements in the tensor.
     * @return The total number of elements. Returns 1 for a scalar.
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
     * @param name The name of the tensor.
     * @return A `const void*` pointing directly to the tensor data in the memory-mapped file.
     * Returns `nullptr` if the tensor is not eligible for zero-copy access.
     */
    const void* get_raw_tensor_pointer(const std::string& name) const;

    /**
     * @brief Reads tensor data into an owned buffer, decompressing if necessary.
     *
     * @param name The name of the tensor.
     * @return A `std::vector<std::byte>` containing the tensor data.
     * @throws std::runtime_error if decompression fails.
     * @throws std::logic_error for unsupported layouts or encodings.
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
    void* _hFile = (void*) -1; // Use void* to avoid including <windows.h> in header
    void* _hMapping = nullptr;
#else
    int _fd = -1;
#endif
};

} // namespace ztensor

#endif // ZTENSOR_HPP