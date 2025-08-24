#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include <Metal/Metal.h>
#include <vector>
#include <memory>

// Forward declarations
struct L4maConfig;

/**
 * @brief Metal equivalent of CUDA StackAllocator
 * 
 * Provides stack-based memory allocation for temporary buffers during model execution.
 * This matches the interface of the CUDA StackAllocator from stack_allocator.cuh.
 */
class MetalStackAllocator {
public:
    explicit MetalStackAllocator(size_t total_size);
    ~MetalStackAllocator() = default;
    
    // Allocate aligned memory
    template<typename T>
    MetalTensor<T> allocate(size_t count);
    
    // Allocate remaining memory as buffer
    MetalBuffer<uint8_t> allocate_rest();
    
    // Deallocate (stack-based, so just moves pointer back)
    template<typename T>
    void deallocate(MetalTensor<T>& tensor);
    
    // Reset allocator to beginning
    void reset();
    
    // Query methods
    size_t total_size() const { return total_size_; }
    size_t allocated_size() const { return current_offset_; }
    size_t available_size() const { return total_size_ - current_offset_; }

private:
    size_t align_offset(size_t offset, size_t alignment) const;
    
    MetalBuffer<uint8_t> buffer_;
    size_t total_size_;
    size_t current_offset_;
};

/**
 * @brief Metal equivalent of L4maBuffer from l4ma.cuh
 * 
 * Consolidates workspace memory, intermediate computations, and index pointers
 * for a single forward pass using Metal compute resources.
 */
template <typename T>
class MetalL4maBuffer {
public:
    // Device-side tensors, valid after plan() is called
    MetalTensor<int32_t> input_ids;
    MetalTensor<int32_t> position_ids;
    MetalTensor<int32_t> kv_page_indices;
    MetalTensor<int32_t> kv_page_indptr;
    MetalTensor<int32_t> kv_last_page_lens;
    MetalTensor<int32_t> qo_indptr;
    MetalTensor<uint8_t> custom_mask;
    MetalTensor<int32_t> mask_indptr;
    MetalTensor<int32_t> kv_batch_indices;
    MetalTensor<int32_t> kv_positions;
    MetalTensor<int32_t> output_indices_src;
    
    // Configuration and context
    const L4maConfig& config;
    const int32_t page_size;
    const int32_t dist_size;
    size_t num_tokens;
    size_t batch_size;
    
    // Metal command buffer for operations
    id<MTLCommandBuffer> commandBuffer;
    
    // Constructor/Destructor
    MetalL4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size);
    ~MetalL4maBuffer() = default;
    
    // Deleted copy/move operations
    MetalL4maBuffer(const MetalL4maBuffer&) = delete;
    MetalL4maBuffer& operator=(const MetalL4maBuffer&) = delete;
    MetalL4maBuffer(MetalL4maBuffer&&) = delete;
    MetalL4maBuffer& operator=(MetalL4maBuffer&&) = delete;
    
    // Static method for workspace size calculation
    static size_t get_workspace_size(
        const L4maConfig& config,
        size_t max_num_tokens,
        size_t max_batch_size,
        size_t max_kv_seqlens,
        size_t dist_size
    );
    
    // Plan buffer allocation and data transfer
    void plan(
        id<MTLCommandBuffer> command_buffer,
        std::vector<int32_t>& input_ids_host,
        std::vector<int32_t>& position_ids_host,
        std::vector<int32_t>& kv_page_indices_host,
        std::vector<int32_t>& kv_page_indptr_host,
        std::vector<int32_t>& kv_last_page_lens_host,
        std::vector<int32_t>& qo_indptr_host,
        std::vector<bool>& packed_custom_mask_host,
        std::vector<int32_t>& mask_indptr_host,
        std::vector<int32_t>& kv_batch_indices_host,
        std::vector<int32_t>& kv_positions_host,
        std::vector<int32_t>& output_indices_src_host
    );
    
    // Allocator wrappers (matching CUDA interface)
    template <typename U> 
    MetalTensor<U> allocate(size_t count);
    
    MetalBuffer<uint8_t> allocate_rest();
    
    template <typename U> 
    void deallocate(MetalTensor<U>& tensor);

private:
    size_t buffer_size_;
    std::unique_ptr<MetalStackAllocator> allocator_;
};

/**
 * @brief Metal batch prefill handler
 * 
 * Equivalent to flashinfer::BatchPrefillHandler for Metal compute.
 * Manages batch attention computation with paged KV cache.
 */
class MetalBatchPrefillHandler {
public:
    MetalBatchPrefillHandler();
    ~MetalBatchPrefillHandler() = default;
    
    // Initialize handler with workspace
    bool initialize(size_t workspace_size);
    
    // Plan batch prefill operation
    template<typename DTypeQ, typename DTypeKV, typename DTypeO>
    bool plan(
        id<MTLCommandBuffer> commandBuffer,
        const std::vector<int32_t>& qo_indptr_host,
        const std::vector<int32_t>& kv_page_indices_host,
        const std::vector<int32_t>& kv_page_indptr_host,
        const std::vector<int32_t>& kv_last_page_lens_host,
        size_t num_qo_heads,
        size_t num_kv_heads,
        size_t head_dim,
        size_t page_size
    );
    
    // Execute batch prefill attention
    template<typename DTypeQ, typename DTypeKV, typename DTypeO>
    void forward(
        id<MTLCommandBuffer> commandBuffer,
        const DTypeQ* q_data,
        const DTypeKV* kv_data,
        DTypeO* o_data,
        const std::vector<bool>& custom_mask,
        const std::vector<int32_t>& mask_indptr
    );

private:
    bool initialized_;
    size_t workspace_size_;
    MetalBuffer<uint8_t> workspace_;
};

// Template implementations

template<typename T>
MetalTensor<T> MetalStackAllocator::allocate(size_t count) {
    size_t required_bytes = count * sizeof(T);
    size_t aligned_offset = align_offset(current_offset_, alignof(T));
    
    if (aligned_offset + required_bytes > total_size_) {
        throw std::runtime_error("MetalStackAllocator: Insufficient memory");
    }
    
    // Create view of the buffer at the allocated offset
    id<MTLBuffer> metal_buffer = buffer_.getBuffer();
    MetalTensor<T> tensor({count});
    
    // Copy data from the main buffer to the tensor's buffer at the right offset
    auto& context = MetalContext::getInstance();
    MetalMemory::copyBuffer(context.getCommandQueue(),
                           metal_buffer, aligned_offset,
                           tensor.getMetalBuffer(), 0,
                           required_bytes);
    
    current_offset_ = aligned_offset + required_bytes;
    return tensor;
}

template<typename T>
void MetalStackAllocator::deallocate(MetalTensor<T>& tensor) {
    // Stack allocator - deallocate by moving pointer back
    size_t tensor_bytes = tensor.byteSize();
    if (current_offset_ >= tensor_bytes) {
        current_offset_ -= tensor_bytes;
    }
    // Note: In a real implementation, we'd need to track allocations more carefully
}

template <typename T>
MetalL4maBuffer<T>::MetalL4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size)
    : config(cfg), page_size(page_size), dist_size(dist_size), buffer_size_(workspace_size) {
    
    // Initialize the stack allocator
    allocator_ = std::make_unique<MetalStackAllocator>(workspace_size);
    
    // Initialize Metal command buffer
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        throw std::runtime_error("Metal context not initialized");
    }
}

template <typename T>
size_t MetalL4maBuffer<T>::get_workspace_size(
    const L4maConfig& config,
    size_t max_num_tokens,
    size_t max_batch_size,
    size_t max_kv_seqlens,
    size_t dist_size
) {
    // Calculate workspace size requirements
    // This should match the CUDA version's calculations
    
    size_t tensor_sizes = 0;
    
    // Input/position IDs
    tensor_sizes += max_num_tokens * sizeof(int32_t) * 2;
    
    // KV page management
    tensor_sizes += max_kv_seqlens * sizeof(int32_t) * 3; // indices, indptr, lens
    
    // Batch management  
    tensor_sizes += max_batch_size * sizeof(int32_t) * 2; // qo_indptr, batch_indices
    
    // Masks
    tensor_sizes += max_num_tokens * max_kv_seqlens / 8; // Custom mask (bits)
    tensor_sizes += max_batch_size * sizeof(int32_t); // mask_indptr
    
    // Intermediate computations (Q, K, V projections, etc.)
    size_t hidden_size = config.hidden_size;
    size_t intermediate_size = config.intermediate_size;
    size_t head_size = config.head_size;
    size_t num_heads = config.num_query_heads;
    size_t num_kv_heads = config.num_key_value_heads;
    
    // Q, K, V projections
    tensor_sizes += max_num_tokens * (num_heads + 2 * num_kv_heads) * head_size * sizeof(T);
    
    // MLP intermediate
    tensor_sizes += max_num_tokens * intermediate_size * sizeof(T) * 2; // gate + up
    
    // Attention output
    tensor_sizes += max_num_tokens * hidden_size * sizeof(T);
    
    // Distribution storage
    tensor_sizes += max_num_tokens * dist_size * (sizeof(T) + sizeof(int32_t)); // values + indices
    
    // Add some padding for alignment
    tensor_sizes += 4096;
    
    return tensor_sizes;
}

template <typename T>
void MetalL4maBuffer<T>::plan(
    id<MTLCommandBuffer> command_buffer,
    std::vector<int32_t>& input_ids_host,
    std::vector<int32_t>& position_ids_host,
    std::vector<int32_t>& kv_page_indices_host,
    std::vector<int32_t>& kv_page_indptr_host,
    std::vector<int32_t>& kv_last_page_lens_host,
    std::vector<int32_t>& qo_indptr_host,
    std::vector<bool>& packed_custom_mask_host,
    std::vector<int32_t>& mask_indptr_host,
    std::vector<int32_t>& kv_batch_indices_host,
    std::vector<int32_t>& kv_positions_host,
    std::vector<int32_t>& output_indices_src_host
) {
    commandBuffer = command_buffer;
    
    // Reset allocator
    allocator_->reset();
    
    // Set sizes
    num_tokens = input_ids_host.size();
    batch_size = qo_indptr_host.size() - 1;
    
    // Allocate and copy tensors
    if (!input_ids_host.empty()) {
        input_ids = allocator_->allocate<int32_t>(input_ids_host.size());
        input_ids.copyFromHost(input_ids_host.data());
    }
    
    if (!position_ids_host.empty()) {
        position_ids = allocator_->allocate<int32_t>(position_ids_host.size());
        position_ids.copyFromHost(position_ids_host.data());
    }
    
    if (!kv_page_indices_host.empty()) {
        kv_page_indices = allocator_->allocate<int32_t>(kv_page_indices_host.size());
        kv_page_indices.copyFromHost(kv_page_indices_host.data());
    }
    
    if (!kv_page_indptr_host.empty()) {
        kv_page_indptr = allocator_->allocate<int32_t>(kv_page_indptr_host.size());
        kv_page_indptr.copyFromHost(kv_page_indptr_host.data());
    }
    
    if (!kv_last_page_lens_host.empty()) {
        kv_last_page_lens = allocator_->allocate<int32_t>(kv_last_page_lens_host.size());
        kv_last_page_lens.copyFromHost(kv_last_page_lens_host.data());
    }
    
    if (!qo_indptr_host.empty()) {
        qo_indptr = allocator_->allocate<int32_t>(qo_indptr_host.size());
        qo_indptr.copyFromHost(qo_indptr_host.data());
    }
    
    // Handle custom mask (convert bool vector to uint8_t)
    if (!packed_custom_mask_host.empty()) {
        size_t mask_bytes = (packed_custom_mask_host.size() + 7) / 8; // Round up to bytes
        custom_mask = allocator_->allocate<uint8_t>(mask_bytes);
        
        // Pack bool vector to bytes
        std::vector<uint8_t> packed_mask(mask_bytes, 0);
        for (size_t i = 0; i < packed_custom_mask_host.size(); ++i) {
            if (packed_custom_mask_host[i]) {
                packed_mask[i / 8] |= (1 << (i % 8));
            }
        }
        custom_mask.copyFromHost(packed_mask.data());
    }
    
    if (!mask_indptr_host.empty()) {
        mask_indptr = allocator_->allocate<int32_t>(mask_indptr_host.size());
        mask_indptr.copyFromHost(mask_indptr_host.data());
    }
    
    if (!kv_batch_indices_host.empty()) {
        kv_batch_indices = allocator_->allocate<int32_t>(kv_batch_indices_host.size());
        kv_batch_indices.copyFromHost(kv_batch_indices_host.data());
    }
    
    if (!kv_positions_host.empty()) {
        kv_positions = allocator_->allocate<int32_t>(kv_positions_host.size());
        kv_positions.copyFromHost(kv_positions_host.data());
    }
    
    if (!output_indices_src_host.empty()) {
        output_indices_src = allocator_->allocate<int32_t>(output_indices_src_host.size());
        output_indices_src.copyFromHost(output_indices_src_host.data());
    }
}

template <typename T>
template <typename U>
MetalTensor<U> MetalL4maBuffer<T>::allocate(size_t count) {
    return allocator_->allocate<U>(count);
}

template <typename T>
MetalBuffer<uint8_t> MetalL4maBuffer<T>::allocate_rest() {
    return allocator_->allocate_rest();
}

template <typename T>
template <typename U>
void MetalL4maBuffer<T>::deallocate(MetalTensor<U>& tensor) {
    allocator_->deallocate(tensor);
}