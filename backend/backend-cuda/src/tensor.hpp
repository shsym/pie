#pragma once

#include <vector>
#include <memory>
#include <type_traits>
#include <cstdint> // For uint8_t
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declare Tensor<uint8_t> so the new constructor can use it.
template <typename T> class Tensor;
using ByteTensor = Tensor<uint8_t>;

template <typename T>
class Tensor {

    template <typename U>
    friend class Tensor;
public:
    // --- Constructors and Lifetime ---
    explicit Tensor(size_t size);
    Tensor(const Tensor& other, size_t offset); // Same-type view constructor

    /**
     * @brief Reinterpreting View Constructor.
     * Creates a typed Tensor view from a raw byte buffer.
     * @param byte_buffer The source tensor of raw bytes.
     * @param byte_offset The offset in bytes from the start of the buffer.
     * @param count The number of elements of type T in this view.
     */
    Tensor(const ByteTensor& byte_buffer, size_t byte_offset, size_t count);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    // --- Accessors ---
    size_t size() const;
    T* data() const;

    // --- Host/Device Data Transfer ---
    std::vector<T> to_vector() const;
    void from_vector(const std::vector<T>& data);
    void from_pointer(const T* data, size_t count);

    // --- Debugging Methods ---
    float mean() const requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>;
    void print(size_t start_index = 0, size_t count = -1) const;

private:
    // This constructor is used internally by the reinterpreting constructor
    Tensor(std::shared_ptr<T> data, size_t size, size_t offset);

    struct CudaDeleter {
        void operator()(void* ptr) const;
    };
    std::shared_ptr<T> data_;
    size_t size_;
    size_t offset_;
};
