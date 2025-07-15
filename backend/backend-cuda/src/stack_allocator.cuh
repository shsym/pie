#pragma once

#include "common.cuh"
#include "tensor.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdint>

inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

class StackAllocator {
public:
    explicit StackAllocator(size_t total_size_bytes)
        : buffer_(total_size_bytes), 
          offset_(0) {}

    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    /**
     * @brief Resets the allocator's offset, effectively "freeing" all memory.
     */
    void reset() {
        offset_ = 0;
    }

    /**
     * @brief Allocates a Tensor view for one or more elements.
     * @tparam T The type of the elements to allocate.
     * @param count The number of elements of type T to allocate.
     * @return A Tensor<T> view into the workspace buffer.
     */
    template<typename T>
    Tensor<T> allocate(size_t count = 1) {
        constexpr size_t alignment = std::max(alignof(T), (size_t)256);
        size_t aligned_offset = align_up(offset_, alignment);
        size_t bytes_to_alloc = count * sizeof(T);

        if (aligned_offset + bytes_to_alloc > buffer_.size()) {
            throw std::runtime_error("StackAllocator out of memory.");
        }

        offset_ = aligned_offset + bytes_to_alloc;
        
        // Use the new reinterpreting constructor to create a typed view
        return Tensor<T>(buffer_, aligned_offset, count);
    }


    /**
     * @brief Allocates and copies data from a host vector to the device asynchronously.
     * @tparam U The data type of the vector elements.
     * @param host_vector The host-side std::vector to copy from.
     * @param stream The CUDA stream to use for the asynchronous copy.
     * @return A device pointer of type U* to the newly allocated and populated memory.
     */
    template<typename T>
    Tensor<T> allocate_and_copy_async(const std::vector<T>& host_vector, cudaStream_t stream) {
        if (host_vector.empty()) {
            throw std::runtime_error("empty vector provided to StackAllocator::allocate_and_copy_async.");
        }

        size_t count = host_vector.size();
        constexpr size_t alignment = std::max(alignof(T), (size_t)256);
        size_t aligned_offset = align_up(offset_, alignment);
        size_t bytes_to_alloc = count * sizeof(T);

        if (aligned_offset + bytes_to_alloc > buffer_.size()) {
            throw std::runtime_error("StackAllocator::allocate_and_copy_async out of memory.");
        }

        // Get the typed device pointer at the calculated offset
        T* device_ptr = reinterpret_cast<T*>(buffer_.data() + aligned_offset);

        // Update the stack offset to reserve the space
        offset_ = aligned_offset + bytes_to_alloc;

        // Perform the asynchronous copy from host to device
        CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_vector.data(), bytes_to_alloc, cudaMemcpyHostToDevice, stream));

        return Tensor<T>(buffer_, aligned_offset, count);
    }

    /**
     * @brief Allocates the rest of the available memory in the buffer.
     * Useful for operations like cuBLAS that can use a variable amount of workspace.
     * @return A Tensor<uint8_t> view representing all remaining memory.
     */
    Tensor<uint8_t> allocate_rest() {
        constexpr size_t alignment = 256;
        size_t aligned_offset = align_up(offset_, alignment);
        size_t remaining_bytes = buffer_.size() - aligned_offset;

        if (remaining_bytes == 0) {
            return Tensor<uint8_t>(buffer_, aligned_offset, 0);
        }

        offset_ = buffer_.size();
        return Tensor<uint8_t>(buffer_, aligned_offset, remaining_bytes);
    }

    /**
     * @brief Deallocates the Tensor view, which must be the most recent allocation.
     * @tparam T The type of the elements in the Tensor.
     * @param tensor The Tensor view returned by the last corresponding allocate call.
     */
    template<typename T>
    void deallocate(const Tensor<T>& tensor) {
        if (tensor.size() == 0) {
            return;
        }
        
        uint8_t* ptr = reinterpret_cast<uint8_t*>(tensor.data());
        size_t bytes = tensor.size() * sizeof(T);
        
        // This is the core check for LIFO order, restored from your original code.
        // It confirms that the block being freed is exactly at the top of the stack.
        size_t allocation_offset = ptr - buffer_.data();
        if (align_up(allocation_offset, 256) + bytes != offset_ && allocation_offset + bytes != offset_) {
            // This fallback check is what your original code simplified to.
            // It's a good sanity check if the alignment logic gets complex.
            if (ptr + bytes != buffer_.data() + offset_) {
                 throw std::runtime_error("StackAllocator deallocation error: The provided tensor does not match the most recent allocation (LIFO violation).");
            }
        }
        
        // Rewind the stack pointer to the start of the deallocated block.
        offset_ = allocation_offset;
    }

    size_t get_used_size() const { return offset_; }
    size_t get_total_size() const { return buffer_.size(); }

private:
    ByteTensor buffer_;
    size_t offset_;
};