#pragma once

#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <algorithm> // For std::max

// A helper function to align pointers/offsets upwards.
inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @class StackAllocator
 * @brief Manages a device memory buffer with a stack discipline (LIFO) and 256-byte alignment.
 */
class StackAllocator {
public:
    /**
     * @brief Constructs the allocator with a given device buffer.
     * @param buffer Pointer to the start of the device memory buffer.
     * @param total_size The total size of the buffer in bytes.
     */
    StackAllocator(void* buffer, size_t total_size)
        : buffer_start_(static_cast<char*>(buffer)), 
          total_size_(total_size), 
          offset_(0) {
        if (!buffer && total_size > 0) {
            throw std::invalid_argument("StackAllocator cannot be initialized with a null buffer and non-zero size.");
        }
    }

    // Disable copy and move semantics.
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    /**
     * @brief Allocates a block of memory in bytes with 256-byte alignment.
     * @param bytes The number of bytes to allocate.
     * @param alignment The required alignment, defaulting to 256.
     * @return A void pointer to the allocated memory.
     */
    void* allocate_bytes(size_t bytes, size_t alignment = 256) {
        if (bytes == 0) {
            return nullptr;
        }
        size_t aligned_offset = align_up(offset_, alignment);
        if (aligned_offset + bytes > total_size_) {
            throw std::runtime_error(
                "StackAllocator out of memory. Requested " + std::to_string(bytes) +
                " bytes (aligned from " + std::to_string(offset_) + " to " + std::to_string(aligned_offset) + 
                "), but only " + std::to_string(total_size_ - offset_) + " bytes available."
            );
        }
        offset_ = aligned_offset + bytes;
        return buffer_start_ + aligned_offset;
    }

    /**
     * @brief Allocates memory for elements, ensuring at least 256-byte alignment.
     * @tparam T The type of the elements to allocate.
     * @param count The number of elements of type T to allocate.
     * @return A pointer of type T* to the allocated memory.
     */
    template<typename T>
    T* allocate(size_t count = 1) {
        // Enforce a minimum of 256-byte alignment, but respect larger alignment requirements of the type if any.
        constexpr size_t alignment = std::max(alignof(T), (size_t)256);
        return static_cast<T*>(allocate_bytes(count * sizeof(T), alignment));
    }

    /**
     * @brief Deallocates the most recently allocated block of memory.
     * @param ptr The pointer returned by the last allocation call.
     * @param bytes The size of the last allocation in bytes.
     */
    void deallocate_bytes(void* ptr, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        
        char* char_ptr = static_cast<char*>(ptr);
        size_t allocation_offset = char_ptr - buffer_start_;
        
        // This is the core check for LIFO order. It ensures the block being freed
        // is exactly at the top of the stack.
        if (align_up(allocation_offset, 256) + bytes != offset_ && allocation_offset + bytes != offset_) {
             // The check needs to account for the original unaligned offset before this allocation.
             // This is complex. A simpler and more robust deallocation is just to rewind.
             // The user is responsible for passing the correct size.
             // Let's simplify and just check the end pointer.
             if (char_ptr + bytes != buffer_start_ + offset_) {
                 throw std::runtime_error("StackAllocator deallocation error: The provided pointer/size does not match the most recent allocation.");
             }
        }
        
        offset_ = allocation_offset;
    }
    
    /**
     * @brief Deallocates the most recently allocated block of typed elements.
     * @tparam T The type of the elements to deallocate.
     * @param ptr The pointer returned by the last corresponding allocate<T> call.
     * @param count The number of elements that were allocated.
     */
    template<typename T>
    void deallocate(T* ptr, size_t count = 1) {
        deallocate_bytes(ptr, count * sizeof(T));
    }

    size_t get_used_size() const { return offset_; }
    size_t get_total_size() const { return total_size_; }

private:
    char* const buffer_start_;
    const size_t total_size_;
    size_t offset_;
};