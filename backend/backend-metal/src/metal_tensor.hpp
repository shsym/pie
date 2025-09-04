#pragma once

#include "metal_common.hpp"
#include <Metal/Metal.h>
#include <vector>
#include <memory>
#include <initializer_list>

/**
 * @brief Metal tensor wrapper that matches the CUDA Tensor<T> interface
 *
 * This class provides a Metal equivalent of the CUDA Tensor class from tensor.hpp,
 * maintaining the same interface for compatibility while using Metal buffers underneath.
 */
template<typename T>
class MetalTensor {
public:
    // Default constructor
    MetalTensor();

    // Constructor with shape
    explicit MetalTensor(const std::vector<size_t>& shape);
    MetalTensor(std::initializer_list<size_t> shape);

    // Constructor with data and shape
    MetalTensor(const T* data, const std::vector<size_t>& shape);
    MetalTensor(const T* data, std::initializer_list<size_t> shape);

    // Move constructor and assignment
    MetalTensor(MetalTensor&& other) noexcept;
    MetalTensor& operator=(MetalTensor&& other) noexcept;

    // Delete copy constructor and assignment
    MetalTensor(const MetalTensor&) = delete;
    MetalTensor& operator=(const MetalTensor&) = delete;

    ~MetalTensor() = default;

    // Shape and size queries
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return total_size_; }
    size_t byteSize() const { return total_size_ * sizeof(T); }
    bool is_empty() const { return total_size_ == 0; }

    // Data access
    T* data() { return buffer_.data(); }
    const T* data() const { return buffer_.data(); }

    // Metal-specific access
    id<MTLBuffer> getMetalBuffer() const { return buffer_.getBuffer(); }

    // Initialize from host pointer (matches CUDA Tensor::from_pointer)
    void from_pointer(const T* host_data, size_t count);

    // Copy operations
    void copyFromHost(const T* host_data);
    void copyToHost(T* host_data) const;
    void copyFromDevice(const MetalTensor<T>& other);

    // Copy from memory-mapped file data (for weight loading)
    void copyFromMappedMemory(const void* mapped_data, size_t num_elements);

    // Copy from memory-mapped file data with type conversion
    template<typename SrcType>
    void copyFromMappedMemory(const SrcType* mapped_data, size_t num_elements);

    // Reshape operations
    void reshape(const std::vector<size_t>& new_shape);
    MetalTensor<T> view(const std::vector<size_t>& new_shape) const;

    // Slicing operations
    MetalTensor<T> slice(size_t start, size_t end) const;
    MetalTensor<T> slice(const std::vector<std::pair<size_t, size_t>>& ranges) const;

    // Fill operations
    void fill(const T& value);
    void zero();

    // Utility methods
    bool isContiguous() const { return true; } // Metal tensors are always contiguous
    void synchronize() const; // Wait for Metal operations to complete

private:
    void calculateStrides();
    size_t calculateTotalSize() const;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_size_;
    MetalBuffer<T> buffer_;
};

/**
 * @brief Factory functions for creating Metal tensors
 */
namespace MetalTensorFactory {

    template<typename T>
    MetalTensor<T> zeros(const std::vector<size_t>& shape);

    template<typename T>
    MetalTensor<T> ones(const std::vector<size_t>& shape);

    template<typename T>
    MetalTensor<T> full(const std::vector<size_t>& shape, const T& value);

    template<typename T>
    MetalTensor<T> fromData(const T* data, const std::vector<size_t>& shape);

    // Create tensor view of existing buffer (no copy)
    template<typename T>
    MetalTensor<T> view(id<MTLBuffer> buffer, const std::vector<size_t>& shape, size_t offset = 0);
}

/**
 * @brief Tensor operations namespace
 * Provides common tensor operations using Metal compute
 */
namespace MetalTensorOps {

    // Element-wise operations
    template<typename T>
    void add(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result);

    template<typename T>
    void mul(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result);

    template<typename T>
    void addScalar(const MetalTensor<T>& input, T scalar, MetalTensor<T>& result);

    template<typename T>
    void mulScalar(const MetalTensor<T>& input, T scalar, MetalTensor<T>& result);

    // Reduction operations
    template<typename T>
    T sum(const MetalTensor<T>& tensor);

    template<typename T>
    T mean(const MetalTensor<T>& tensor);

    template<typename T>
    T max(const MetalTensor<T>& tensor);

    template<typename T>
    T min(const MetalTensor<T>& tensor);

    // Matrix operations
    template<typename T>
    void matmul(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result);

    // Transpose operations
    template<typename T>
    void transpose(const MetalTensor<T>& input, MetalTensor<T>& result);

    template<typename T>
    void transpose2D(const MetalTensor<T>& input, MetalTensor<T>& result);
}

// Template implementations

template<typename T>
MetalTensor<T>::MetalTensor() : total_size_(0) {
}

template<typename T>
MetalTensor<T>::MetalTensor(const std::vector<size_t>& shape)
    : shape_(shape), total_size_(calculateTotalSize()) {
    if (total_size_ > 0) {
        buffer_ = MetalBuffer<T>(total_size_);
        calculateStrides();
    }
}

template<typename T>
MetalTensor<T>::MetalTensor(std::initializer_list<size_t> shape)
    : MetalTensor(std::vector<size_t>(shape)) {
}

template<typename T>
MetalTensor<T>::MetalTensor(const T* data, const std::vector<size_t>& shape)
    : MetalTensor(shape) {
    if (total_size_ > 0) {
        copyFromHost(data);
    }
}

template<typename T>
MetalTensor<T>::MetalTensor(const T* data, std::initializer_list<size_t> shape)
    : MetalTensor(data, std::vector<size_t>(shape)) {
}

template<typename T>
MetalTensor<T>::MetalTensor(MetalTensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , total_size_(other.total_size_)
    , buffer_(std::move(other.buffer_)) {
    other.total_size_ = 0;
}

template<typename T>
MetalTensor<T>& MetalTensor<T>::operator=(MetalTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        total_size_ = other.total_size_;
        buffer_ = std::move(other.buffer_);
        other.total_size_ = 0;
    }
    return *this;
}

template<typename T>
void MetalTensor<T>::from_pointer(const T* host_data, size_t count) {
    if (count != total_size_) {
        // Reshape to match count
        shape_ = {count};
        total_size_ = count;
        strides_ = {1};
        buffer_ = MetalBuffer<T>(count);
    }

    if (count > 0 && host_data) {
        buffer_.copyFromHost(host_data, count);
    }
}

template<typename T>
void MetalTensor<T>::copyFromHost(const T* host_data) {
    if (total_size_ > 0 && host_data) {
        buffer_.copyFromHost(host_data, total_size_);
    }
}

template<typename T>
void MetalTensor<T>::copyToHost(T* host_data) const {
    if (total_size_ > 0 && host_data) {
        buffer_.copyToHost(host_data, total_size_);
    }
}

template<typename T>
void MetalTensor<T>::copyFromDevice(const MetalTensor<T>& other) {
    if (total_size_ != other.total_size_) {
        throw std::runtime_error("Tensor size mismatch in copyFromDevice");
    }

    if (total_size_ > 0) {
        auto& context = MetalContext::getInstance();
        MetalMemory::copyBuffer(context.getCommandQueue(),
                               other.buffer_.getBuffer(), 0,
                               buffer_.getBuffer(), 0,
                               total_size_ * sizeof(T));
    }
}

template<typename T>
void MetalTensor<T>::copyFromMappedMemory(const void* mapped_data, size_t num_elements) {
    if (!mapped_data) {
        throw std::runtime_error("Null mapped data pointer");
    }

    if (num_elements != total_size_) {
        throw std::runtime_error("Element count mismatch in copyFromMappedMemory");
    }

    if (total_size_ > 0) {
        // Direct memory copy from mapped memory to Metal buffer
        // This assumes the mapped data is the same type as T
        buffer_.copyFromHost(static_cast<const T*>(mapped_data), total_size_);
    }
}

template<typename T>
template<typename SrcType>
void MetalTensor<T>::copyFromMappedMemory(const SrcType* mapped_data, size_t num_elements) {
    if (!mapped_data) {
        throw std::runtime_error("Null mapped data pointer");
    }

    if (num_elements != total_size_) {
        throw std::runtime_error("Element count mismatch in copyFromMappedMemory");
    }

    if (total_size_ > 0) {
        // Type conversion from SrcType to T
        // For efficient conversion, we'll create a temporary buffer
        std::vector<T> temp_buffer(total_size_);

        for (size_t i = 0; i < total_size_; ++i) {
            temp_buffer[i] = static_cast<T>(mapped_data[i]);
        }

        // Copy converted data to Metal buffer
        buffer_.copyFromHost(temp_buffer.data(), total_size_);
    }
}

template<typename T>
void MetalTensor<T>::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }

    if (new_size != total_size_) {
        throw std::runtime_error("Reshape size mismatch");
    }

    shape_ = new_shape;
    calculateStrides();
}

template<typename T>
void MetalTensor<T>::fill(const T& value) {
    if (total_size_ > 0) {
        // For now, fill on CPU - should be replaced with Metal kernel
        T* data_ptr = buffer_.data();
        for (size_t i = 0; i < total_size_; ++i) {
            data_ptr[i] = value;
        }
    }
}

template<typename T>
void MetalTensor<T>::zero() {
    if (total_size_ > 0) {
        auto& context = MetalContext::getInstance();
        MetalMemory::zeroBuffer(context.getCommandQueue(),
                               buffer_.getBuffer(),
                               total_size_ * sizeof(T));
    }
}

template<typename T>
void MetalTensor<T>::synchronize() const {
    // For Metal, we might need to wait for command buffer completion
    // For now, this is a no-op as Metal operations are synchronous in our current implementation
}

template<typename T>
void MetalTensor<T>::calculateStrides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

template<typename T>
size_t MetalTensor<T>::calculateTotalSize() const {
    size_t size = 1;
    for (size_t dim : shape_) {
        size *= dim;
    }
    return size;
}

// Factory function implementations
namespace MetalTensorFactory {

    template<typename T>
    MetalTensor<T> zeros(const std::vector<size_t>& shape) {
        MetalTensor<T> tensor(shape);
        tensor.zero();
        return tensor;
    }

    template<typename T>
    MetalTensor<T> ones(const std::vector<size_t>& shape) {
        MetalTensor<T> tensor(shape);
        tensor.fill(T(1));
        return tensor;
    }

    template<typename T>
    MetalTensor<T> full(const std::vector<size_t>& shape, const T& value) {
        MetalTensor<T> tensor(shape);
        tensor.fill(value);
        return tensor;
    }

    template<typename T>
    MetalTensor<T> fromData(const T* data, const std::vector<size_t>& shape) {
        return MetalTensor<T>(data, shape);
    }
}